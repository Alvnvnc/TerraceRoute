# TerraceGate — a local-first infrastructure agent (Track 3)

> Natural language in, verified Cloudflare Tunnel + DNS out; a deterministic
> self-healing watchdog; and an uncertainty gate that decides when **not** to act.
>
> Built for self-hosters: your Cloudflare token, DNS, and network topology never
> leave the machine, because the reasoning runs on a **local model on your AMD
> Radeon GPU**. That is what makes the local model load-bearing rather than a
> checkbox.

Full design and rationale: [`plan.md`](plan.md).
No Docker image is required for Track 3; the deliverables are the repository,
a slide deck, and a demo.

## Why it is different

Most "AI + infra" demos generate a YAML file. TerraceGate does not let the model
touch Cloudflare at all. The LLM only emits a **typed plan**; deterministic typed
tools apply it, verify it **from the public edge**, and roll it back automatically
if verification fails. Failure diagnosis is pure rules. Risky actions are gated by
**cross-model disagreement** — the empirically correct error signal (a model that
is confidently wrong looks certain to itself; a second model of a different family
diverges).

## Phase 2/3 results (measured on AMD Radeon, gfx1100 / ROCm)

Two local model families run on the GPU at once (`gemma3:12b` planner +
`qwen2.5:3b-instruct` verifier); the Cloudflare token never leaves the machine.
Full write-up + raw report: [`results/PHASE23_RESULTS.md`](results/PHASE23_RESULTS.md).

- **NL → plan**: **100%** op accuracy and **100%** args accuracy on 34 labeled,
  mixed Indonesian/English requests; 0 omissions; 57 tok/s on the planner.
- **Safety gate**: **0% false-act** and **0% false-refuse** across 12 safe /
  destructive / ambiguous / dangerous prompts. On vague or dangerous requests the
  two model families diverge (disagreement 1.0) and the gate **refuses** — the
  divergence, not any single model's confidence, is the signal.

```
python3 -m agent.cli plan "hapus semua DNS yang kelihatannya tidak dipakai"
  planner  gemma3:12b          → unexpose   verifier qwen2.5:3b → diagnose
  disagreement 1.00 · blast radius destructive · → DECISION: REFUSE
```

## What works today (Phase 0 + Phase 1, no LLM)

- **Typed Cloudflare tools** — create/configure/delete a remotely-managed tunnel,
  create/find/delete the proxied CNAME (`agent/tools/cloudflare.py`).
- **Expose transaction** — plan → apply → external probe → **automatic rollback**
  on failure, with a JSONL audit journal + LIFO undo stack (`agent/operations.py`,
  `agent/journal.py`).
- **Self-heal diagnosis** — an 8-mode failure taxonomy driven off real signals
  (`/ready`, `/metrics`, edge HTTP/1xxx code, Cloudflare API status). The key
  discriminator `1033 = connector down` vs `502 = origin down` is encoded and
  tested (`agent/heal/taxonomy.py`).
- **Safety gate** — blast-radius × disagreement → act / ask / refuse
  (`agent/brain/gate.py`).
- **CLI** — `expose`, `teardown`, `diagnose`, `status`, and `plan` (`agent/cli.py`).

## Phase 2/3 code (local LLM brain + gate)

- **NL → plan** — constrained-JSON planner over local Ollama, reasoning-first
  schema + in-distribution few-shot + omission retry (`agent/brain/llm.py`,
  `agent/brain/planner.py`).
- **Two-model gate** — `dual_plan` runs both families and feeds the disagreement
  into the blast-radius matrix (`agent/brain/gate.py`).
- **Intent-coverage guard** — a deterministic backstop that catches destructive
  or multi-intent requests both models truncate identically (which disagreement
  alone can't see): if the raw text carries destructive language the chosen op
  doesn't reflect, it never auto-applies (`agent/brain/intent.py`). Found via an
  adversarial pass; closed a real 10% false-act to 0% (see results).
- **End-to-end command** — `agent/cli.py agent "<request>"` runs the full loop
  NL → two plans → gate → execute (or refuse); the gate governs whether the typed
  op actually touches Cloudflare (`--execute`, `--yes`, `--dry-run`).
- **Eval harness** — labeled + adversarial NL→plan and refuse sets + a runner
  reporting the numbers above (`eval/*.jsonl`, `eval/run_brain_eval.py`).

## Quick start

No dependencies to install — Phase 0/1 is standard-library only (Python 3.10+).

```bash
# Exercise the whole expose flow with no network / no credentials:
python3 -m agent.cli expose --host media.example.com --port 8096 --dry-run

# Diagnose a running cloudflared connector (read-only):
python3 -m agent.cli diagnose --metrics 127.0.0.1:20241

# Real expose (needs a TEST zone in .env — never production):
cp .env.example .env      # then fill in CF_ACCOUNT_ID / CF_ZONE_ID / CF_API_TOKEN
python3 -m agent.cli expose --host media.<your-test-zone> --port 8096
```

## Tests

Standard-library `unittest`, no pytest required:

```bash
python3 -m unittest discover -t . -s eval -p 'test_*.py'
```

67 tests cover the taxonomy (every failure mode + the 1033/502 discriminator),
the gate matrix, plan parsing against messy model output, the journal/rollback
semantics, edge classification, cloudflared log parsers, and the end-to-end
expose transaction (dry-run).

## Safety

`expose`/`teardown` change real infrastructure. Always use a dedicated **test**
domain and a zone-scoped token (Account: Cloudflare Tunnel Edit + Zone: DNS Edit
on that one zone). `diagnose`/`status` are read-only.
