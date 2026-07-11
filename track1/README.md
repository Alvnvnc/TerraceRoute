# TerraceRoute — Token-Efficient Routing Agent

**AMD Developer Hackathon (ACT II) · Track 1**

A **batch** agent that solves natural-language tasks across the 8 official categories while
spending **as few Fireworks API tokens as possible** — local inference is free, so the
winning move is to answer locally whenever we can *prove* the answer is right, and escalate
to the remote model only when a data-driven policy says it pays off.

> 🇮🇩 Versi Bahasa Indonesia: [`README.id.md`](README.id.md)

## The harness contract (what is graded)

- Read `/input/tasks.json` (`[{task_id, prompt}]`) → write `/output/results.json`
  (`[{task_id, answer}]`), exit `0`. Malformed output scores zero.
- Injected env: `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL` (every remote call must go through
  this URL), `ALLOWED_MODELS` (read at runtime — never hardcoded).
- Scoring environment: **4 GB RAM, 2 vCPU, CPU-only, 10-minute** total budget.
- Scoring: an accuracy gate (LLM-Judge, unseen prompt variants); entries that pass are then
  **ranked by ascending total Fireworks tokens** (input + output). **Local tokens count as
  zero**, so a zero-API entry that clears the gate is the theoretical optimum.

## How it works — terrace routing

Each task walks down a "terrace" of increasingly expensive checks and exits as early as it can:

```
classify category (regex, free)
  ├─ letter counting ("how many r in strawberry"; non-code prompts only) → count in Python  (DETERMINISTIC, 0 tokens)
  ├─ math    → solve locally + recompute with Python
  ├─ code    → solve locally + `ast.parse` syntax check in a subprocess  (catches unloadable code)
  └─ other   → solve locally (3B model)
        ↓ escalation policy (ESCALATION_LEVEL knob, 0..4)
   verify-failed / empty local answer → escalate (level ≥ 1)
   unverifiable + category on this rung → escalate (levels 2/3)
        ↓
   remote: terse prompt (final answer only) + per-category max_tokens cap → minimal tokens
```

**Key insight (from calibration).** A small model's *internal* confidence signals —
perplexity, self-consistency, even "Python it wrote to check itself" — **anti-correlate**
with correctness: the model is often *confidently, consistently wrong*. So the only thing
that counts as **verified** is a genuinely **independent** check (a syntax check on generated
code, counting letters, arithmetic we parse ourselves), never the model agreeing with itself. Everything
else is escalated according to the token budget the accuracy gate allows.

## Escalation ladder (data-driven)

The rung order is not intuition — it comes from a Monte-Carlo decision model over the
measured per-category accuracies (see [`docs/escalation-math.md`](docs/escalation-math.md)).
Marginal efficiency (ΔAccuracy per token) ranks **math ≈ factual > sentiment ≈ logical ≫**
the categories the local model already nails (NER, summarisation, code — escalating those is
a pure loss).

| `ESCALATION_LEVEL` | escalates |
|---|---|
| 0 | never (zero-API) |
| 1 | only `empty` / `verify-failed` (high-precision signals, near-free) |
| 2 | + unverifiable `math` & `factual` |
| 3 | + `sentiment` & `logical` |
| 4 | + everything unverifiable |

A rejected idea worth noting: **"verify-then-fix"** (ask the remote to grade the local answer
YES/NO, regenerate only on NO) is *net negative in every category* — because input tokens are
counted, the check costs about as much as just answering. The math is in the doc above.

## Reliability engineering

Infra failures score zero, so the runner is defensive:

- **Progressive atomic write** — every task is flushed immediately; a crash or timeout still
  leaves a valid partial `results.json`.
- **Watchdog** — flushes and exits cleanly near the 10-minute limit.
- **Preemptive + adaptive throttle** — shrinks `LOCAL_MAX_TOKENS` when the projected runtime
  would blow the budget (CPU generation time is ~linear in output length).
- **Parallel escalation** — remote calls run in a thread pool, overlapping their network
  latency with local work on the next task (the 2 vCPUs stay on llama.cpp).
- **Early/late tail-to-remote** — if the local queue cannot finish in time, the tail is sent
  to the remote model instead of being left blank. Gated on *credentials*, not
  `ESCALATION_LEVEL`: a blank answer fails the gate outright, whereas tokens are only a
  ranking penalty. On a normal-sized task set this path is never touched (still 0 tokens).

## Repository layout

```
agent/
  run.py        entrypoint: batch I/O, watchdog, progressive write, parallel escalation, throttle
  solve.py      per-category terrace + unified escalation decision + free format coercion
  classify.py   8-category classification (regex, free)
  verify.py     deterministic verification: arithmetic, sandboxed Python execution, counting
  local_llm.py  local model: ollama (dev) / llama-cpp in-process (prod)
  fireworks.py  escalation client (reads ALLOWED_MODELS at runtime, all calls via BASE_URL)
  config.py     all settings from env; http.py = urllib (no network dependencies)
Dockerfile      multi-stage, CPU-only, self-contained model (~1.8 GB compressed)
requirements.txt
scripts/build_and_push.sh   build linux/amd64 + push to a public registry
eval/           reproducible evaluation harness (see below)
docs/           escalation-math.md — the design in one doc (decision math, ladder, measured results)
```

## Running

### Dev (host Ollama — fast iteration)

```bash
cp .env.example .env            # then edit as needed
pip install -r requirements.txt
ollama pull qwen2.5:3b-instruct
INPUT_PATH=eval/practice_tasks.json OUTPUT_PATH=/tmp/results.json \
  LOCAL_BACKEND=ollama python -m agent.run
```

### Prod (Docker — mirrors the scoring environment)

```bash
docker build -t terraceroute:latest .
docker run --rm --cpus=2 --memory=4g --network none \
  -v "$PWD/eval":/input:ro -v /tmp/out:/output \
  terraceroute:latest        # zero-API run, offline
```

The image bundles the GGUF model, so it needs no network at scoring time. Build and push a
public `linux/amd64` image with `REGISTRY=ghcr.io/<user> ./scripts/build_and_push.sh v1`.

## Evaluation

`eval/` mirrors the 8 official categories with deterministic grading (code is graded by
**executing it against asserts**, facts by boundary-aware matching, summaries by an
independent gemma-3-12B judge):

```bash
python -m eval.gen_tasks_v5                        # regenerate eval/tasks_v5.jsonl (124 tasks, ~16/cat)
python -m eval.validate_tasks --tasks eval/tasks_v5.jsonl   # LLM-free: prove the set is well-formed
OLLAMA_HOST=<host> python -m eval.agent_eval --tasks eval/tasks_v5.jsonl   # run the production solver
python -m eval.escalation_math --results eval/agent_eval_results.jsonl     # recompute the ladder from data
```

`validate_tasks` runs every code task's reference solution against its own asserts, so a mistyped
test can never silently bias the accuracy estimate. `escalation_math --results` recomputes the whole
policy ladder from a *measured* run (falling back to the baked-in v4 numbers with no flag). The larger
v5 set exists because v4 (66 tasks) was too small to estimate the per-category rates reliably.

Measured on an AMD Radeon eval rig (v4): **zero-API accuracy ≈ 0.85–0.91**, level-2 ≈ 0.91 with
~1.1–1.4k remote tokens over 66 tasks. Full measured results are in
[`docs/escalation-math.md`](docs/escalation-math.md) §11.

## Submission / leaderboard protocol

The leaderboard is the real calibration oracle (10 submissions/hour). Because ranking is
ascending by tokens among gate-passers, the optimal play is to find the **lowest rung that
still passes**:

1. Submit **level 0** (zero-API). Passes → done, freeze (0 tokens is unbeatable).
2. Fails → jump to **level 3** (high pass probability, moderate tokens); then walk *down*
   to the cheapest rung that still passes and freeze there.

The full probing protocol is in [`docs/escalation-math.md`](docs/escalation-math.md) §8.

## Docs

- [`docs/escalation-math.md`](docs/escalation-math.md) — the design in one document: the escalation
  decision theory, the data-driven rung order, the leaderboard probing protocol, and the measured
  evaluation results (§11).
