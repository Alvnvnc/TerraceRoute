# TerraceRoute — Escalation Math & Method

> The decision theory behind *when the agent pays for a remote (Fireworks) token*.
> Reproduce the Monte-Carlo numbers with `python3 eval/escalation_math.py` (20k samples).
> Empirical base: eval **v4** (66 tasks) and **v5** (124 tasks), strict grader — measured results in §11–§12.

---

## 1. The objective is a constraint, not a trade-off

Track 1 does **not** score `loss + λ·tokens`. It applies an accuracy **gate** (an LLM judge),
then ranks everything that passes by *ascending* remote-token count. Local compute is free; only
Fireworks input+output tokens count. So the problem is a **threshold-constrained cost minimization**:

```math
\min_{\pi}\ \; \mathbb{E}_{x\sim\mathcal{D}}\big[\,T_{\text{remote}}(x,\pi)\,\big]
\qquad\text{subject to}\qquad
\mathbb{E}_{x\sim\mathcal{D}}\big[\,\mathrm{Acc}(y_\pi(x),\,y^{*}(x))\,\big]\ \ge\ \alpha
```

Two consequences drive the whole design:

1. **Push every unit of work that can be free to the local side** (drafting, verification, arithmetic,
   letter-counting). Remote is spent only to protect accuracy.
2. The decision reduces to a **single question per task**: *is the local answer good enough, or must we
   escalate?* — not an elaborate difficulty score. A calibrated one-threshold cascade is provably
   cost-optimal for small-vs-large routing (UCCI, arXiv 2605.18796), so we do not build a fancy router.

The gate threshold `α` is **unknown before launch**; we recover it from the leaderboard (§8).

---

## 2. Escalation is a two-sided bet

Escalating means *replacing* the local answer with the remote one. That helps when the local answer was
wrong and remote is right, but it **hurts** when the local answer was already right and remote corrupts it.
With local per-task correctness probability `p` and remote correctness `q`:

```math
\Delta\mathrm{Acc}_{\text{task}} \;=\; \underbrace{(1-p)\,q}_{\text{fix a local error}} \;-\; \underbrace{p\,(1-q)}_{\text{break a local win}}
```

This is why escalating an already-strong category is not free accuracy — it can be a net loss (§6).

### Accuracy model (honest about small samples)

- Local per-category correctness: `p_c ~ Beta(correct+1, wrong+1)` — a Bayesian posterior, so a perfect
  8/8 category is scored ≈ 0.9, **not** 1.0. This shrinkage is why the modeled zero-API accuracy
  (`E[acc]=0.834`) sits below the raw point estimate (~0.91).
- Remote correctness (Gemma-3-class): `q ~ Beta(37, 3)`, mean **0.925** (assumed — not yet measured).
- Hidden set assumed balanced: **64 tasks, 8 per category**.

---

## 3. Marginal efficiency per category

For each category we escalate, the *value* is `ΔAcc/task` and the *cost* is the mean remote tokens per
escalated task. The ranking knob is **accuracy bought per 1000 remote tokens**:

```math
\mathrm{Eff}_c \;=\; \frac{\Delta\mathrm{Acc}_{\text{task},c}}{T_c}\times 1000
\qquad\text{with}\qquad
\Delta\mathrm{Acc}_{\text{task},c} = (1-p_c)\,q - p_c\,(1-q)
```

| category      | local `p_c` | ΔAcc/task | tokens/task `T_c` | efficiency (‰ = ΔAcc per 1000 tok) |
|---------------|------------:|----------:|------------------:|-----------------------------------:|
| sentiment     | 0.800 | +0.125 |  55 | **2.27** \* |
| math          | 0.750 | +0.175 |  81 | **2.16** |
| factual       | 0.750 | +0.175 |  92 | **1.90** |
| logical       | 0.800 | +0.125 |  86 | 1.45 |
| summarisation | 0.875 | +0.050 | 169 | 0.30 |
| ner           | 0.900 | +0.025 |  94 | 0.27 |
| code_debug    | 0.900 | +0.025 | 194 | 0.13 |
| code_gen      | 0.900 | +0.025 | 214 | 0.12 |

\* Sentiment looks most efficient **only** because its output is ~3 tokens, and its single observed error
is a defensible "mixed" review a lenient judge may accept — a one-datapoint signal. That is why it sits on
rung 3, not rung 2.

**Read-off:** the frontier is `math ≈ factual ≫ logical ≫ everything else`. Escalating code/NER/summ —
already ~100% local — buys ≈ +0.025/task (pure prior shrinkage) at the **highest** token cost, while
risking the `p·(1−q)` corruption term. It is worth it only if the gate is above ~0.92.

---

## 4. Policy comparison (strict grader, Monte-Carlo)

Each policy escalates a growing set of categories. `P≥x` is the probability the policy clears a gate at `x`.

| policy | escalates | E[acc] | E[remote tok] | P≥.75 | P≥.80 | P≥.85 | P≥.90 |
|--------|-----------|-------:|--------------:|------:|------:|------:|------:|
| **L0** | nothing (zero-API) | 0.834 | **0** | .93 | .71 | .41 | .14 |
| P1 | math | 0.856 | ~650 | .97 | .83 | .56 | .23 |
| P2 | math + logical | 0.872 | ~1340 | .99 | .90 | .68 | .32 |
| **P3** | **math + factual** | **0.878** | ~1380 | .99 | .91 | .71 | .37 |
| P4 | math + factual + logical | 0.894 | ~2070 | 1.00 | .95 | .81 | .50 |
| P5 | + sentiment | 0.910 | ~2510 | 1.00 | .98 | .89 | .63 |
| P6 | everything | 0.925 | ~7880 | 1.00 | .97 | .90 | .74 |

**Finding 1 — P3 dominates P2 (v4 only; see §12 caveat).** On v4, higher accuracy at essentially equal
token cost: factual errors are hallucinations invisible to any *local* check, whereas many logical errors
are already caught for free by trap-routing in the classifier. **This does not replicate on the larger v5
set** (P2 ≈ P3 within noise) — the durable reason to keep `factual` on rung 2 is not that it dominates, but
that its errors are *deterministic and tool-invisible* (§12), not the noisy `logical` numbers.

---

## 5. Finding 2 — "verify-then-fix" is mathematically dead

The tempting idea: send the local answer to remote for a 1-token YES/NO check, regenerate only on NO.
Because **input tokens are billed**, the verification prompt (task + local answer + instruction) already
costs about as much as answering outright. Net result: *more* expensive in **every** category
(−25 to −39 tokens/task versus just answering). Do not build it.

```math
T^{\text{verify}}_{\text{remote}} = \underbrace{T_{\text{in}}(x) + T_{\text{in}}(\text{answer}) + T_{\text{in}}(\text{instr})}_{\approx\ T_{\text{in}}\text{ of a direct answer}} + \underbrace{T_{\text{out}}(\text{verdict})}_{\ge 1} \;>\; T^{\text{direct}}_{\text{remote}}
```

The only escalation shape that saves money is a **compressed re-answer with a capped output**, which is
what the ladder does.

---

## 6. Finding 3 — escalating strong categories is a net loss

For a near-perfect category (`p_c ≈ 0.9`, `q ≈ 0.925`):

```math
\Delta\mathrm{Acc}_{\text{task}} = (1-0.9)(0.925) - (0.9)(1-0.925) = 0.0925 - 0.0675 = +0.025
```

+0.025/task is almost entirely prior shrinkage, and it carries the largest token bill (code 194–214 tok).
Rung 4 (escalate everything) only makes sense if the gate turns out to be > ~0.92.

---

## 7. The implemented ladder

`ESCALATION_LEVEL` (env, read at runtime) selects the rung. Order is the efficiency ranking of §3–§4,
not intuition. Implemented in `agent/solve.py::_LEVEL_CATEGORIES`.

| `ESCALATION_LEVEL` | escalates | rationale |
|:------------------:|-----------|-----------|
| **0** | never (zero-API) | 0 tokens is unbeatable if it clears the gate |
| 1 | only `empty` / `verify_failed` | high-precision signals, almost free |
| 2 | + `unverifiable` in {math, factual} | the efficiency frontier (P3) |
| 3 | + {sentiment, logical} | mid-efficiency; needs a higher gate |
| 4 | all `unverifiable` | only worth it if gate > ~0.92 |

A local answer is `verified` (never escalated) only when an **independent** check confirms it — code
execution, deterministic letter-counting, or safe arithmetic. Model self-agreement is *not* verification
(§9).

---

## 8. Leaderboard probing protocol (1-bit oracle, 10 submits/hour)

Only the *last* submitted config counts, passing configs can always be re-submitted, and the gate `α` is
hidden. The optimal search is therefore **find the lowest rung that still passes**:

1. Submit **L0**. If it passes → **done** (0 tokens is unbeatable; freeze).
2. If it fails → submit **level 3** (P5: high pass-probability, still moderate tokens).
   - Passes → walk **down** one rung (level 2); passes again → try level 1; on a failure, step back up to
     the last passing rung and freeze. (Safe walk-down: a passing config can always be re-submitted.)
   - Fails → try level 4. If that also fails, the bottleneck is remote/strong-category accuracy, not the
     policy — read the judge feedback, don't probe blindly.
3. Log every probe (image tag, level, status, score). Each result tightens the interval estimate
   of the gate `α`.

Expected probes to converge: **≤ 5** — well under the rate limit.

---

## 9. Why the escalation signal is a *category*, not a confidence score

We measured the correlation between candidate uncertainty signals and *actual* local error
(point-biserial `corr(u, wrong)`; more positive = more useful) on a hard 30-task set with an independent
gemma3:12b judge:

| signal | `corr(u, wrong)` | why |
|--------|:----------------:|-----|
| perplexity / logprob (`u_ppl`) | **−0.09 … −0.16** | the model is *confidently wrong* on reasoning traps |
| self-consistency (`u_sc`) | **−0.12 … −0.21** | the error is *consistent* across samples — variance can't see a bias |
| cross-model disagreement (`u_cross`) | **+0.03 … +0.30** | the only positive signal: a different-family verifier makes different mistakes |

**Internal confidence anti-correlates with correctness**, because perplexity and self-consistency reuse the
*same* model — when it is both wrong and sure, there is no internal signal. Only disagreement with a
*different-family* model catches it. Practical corollaries proven in that experiment:

- The verifier must be **strong and different-family.** Adding a weak same-family model (qwen2.5:3b) *lowered*
  the correlation (+0.30 → +0.12) — it dissents even when the primary is right (noise).
- The verifier must be allowed to **reason** (verbose). Forcing terse answers destroyed the signal
  (+0.30 → +0.03).
- A **universal blind spot** remains: tasks where *all* models are wrong-and-agree (e.g. counting the r's
  in "strawberry") cannot be routed by disagreement — they need a tool / code execution.

This is exactly why TerraceRoute routes by **category + deterministic tool verification**, not by a model's
self-reported confidence: the free, trustworthy signals are the independent tools; the paid signal is the
remote model itself.

---

## 10. Capacity / timing budget

The scoring box is fixed: **2 vCPU, 4 GB RAM, CPU-only, 10-minute wall clock**. GPU access does not change
it. Measured on that box with qwen2.5:3b-instruct Q4_K_M (in-process llama.cpp):

```math
\text{local capacity} \approx 85\ \text{tasks}\ /\ 8.5\ \text{min}
\qquad(\text{watchdog fires at } 510\text{s},\ \text{flushes a valid schema, exits 0})
```

The bottleneck is **CPU tokens/s of decode**, not output length — throttling `LOCAL_MAX_TOKENS` alone does
not rescue an oversized set. So `run.py` adds two guards, gated on **credentials** (not
`ESCALATION_LEVEL`), because an empty answer fails the gate for certain while a few tokens are only a
ranking penalty:

- **Early tail-to-remote:** when the projected finish exceeds the budget, the queue's tail is sent to
  remote *now* (overlapping local solves) instead of being dumped in the final seconds.
- **Late tail-to-remote:** a fallback if the watchdog cuts in before the early path triggers.

On a normal-sized set neither path is touched and the run stays at 0 remote tokens.

---

## 11. Measured results (eval v4)

Setup: the production Solver (`agent/`, unchanged) + qwen2.5:3b-instruct Q4_K_M on an AMD Radeon
instance, independent gemma3:12b judge, task set `tasks_v4.jsonl` (66 tasks mirroring the 8 official
categories). Grading is as deterministic as possible: exact / contains-all / code-execution+assert;
an LLM judge is used only for summarisation. `ESCALATION_LEVEL=0` (pure zero-API).

### Per-category local accuracy (zero-API, after the classify patch)

| category      | accuracy | note |
|---------------|---------:|------|
| ner           | 1.00 | perfect |
| summarisation | 1.00 | perfect (gemma judge) |
| code_debug    | 1.00 | perfect (passes execution + assert) |
| code_gen      | 1.00 | perfect (passes execution + assert) |
| sentiment     | 0.88 | lone error is an ambiguous "mixed" review |
| factual       | 0.80 | pure hallucination — invisible to local checks |
| math          | 0.80 | rose from 0.70 via verbal-math routing |
| logical       | 0.88 | rose from 0.62 via riddle/trap routing |
| **TOTAL**     | **≈0.91** | zero-API, 0 Fireworks tokens |

Two classifier patches produced the "rose from" gains at zero token cost: `_RE_LOGIC` now catches
riddle/trap markers ("all but", "are left", relative dates, syllogisms, knights/knaves) and `_RE_MATH`
catches verbal math ("multiplied", "divisor", "factorial", "speed") — classification 89% → 98.5%, so
level-2 escalation actually lands on the intended categories.

### Empirical ladder validation (L0 vs L2)

Two back-to-back full runs, identical conditions (remote AMD, gemma3:12b judge, retry-on-empty):

| | L0 zero-API | L2 (math+factual) | MC prediction |
|---|---:|---:|---|
| TOTAL | 0.85 | 0.91 | L0 .834, P3 .878 — within noise ✓ |
| factual | 0.80 | **1.00** | hallucinations swept by escalation (P3 rationale confirmed) |
| math | 0.80 | 0.90 | the one remaining fail also fails on gemma-12B |
| logical | 0.62 | 0.62 | not escalated at L2 (by design); volatile across runs |
| remote tokens | 0 | ~21 tasks ≈ 1.1–1.4k | model: ~1.4k ✓ |

Format coercion makes grading more honest: a logical answer is now judged on its final `Answer:` line
(the model's actual conclusion), not on a correct number that merely "passed through" its reasoning.

### Container harness-mirror tests (`--cpus=2 --memory=4g`)

| scenario | result |
|---|---|
| 9 practice tasks, zero-API, `--network none` | 71s, all answered, exit 0 |
| malformed input | `[]` + exit 0 (safe from RUNTIME_ERROR) |
| 66 tasks, zero-API offline | 494s (< 510s watchdog), 66/66 answered |
| 132 tasks (2× capacity), zero-API offline | watchdog fires, valid schema; local capacity ≈ 85 tasks / 8.5 min |
| 66 tasks, forced overload, credentials, level 0 | early tail-to-remote → **0 empty answers** |
| L2 escalation in-container (stand-in Fireworks) | parallel escalation works, 164 tokens / 3 tasks, exit 0 |

### AMD compute evidence

`/api/ps` shows the model fully resident in Radeon VRAM (`size_vram` = 3.36 GB). Measured throughput:
qwen2.5:3b ≈ 147 tok/s decode; gemma3:12b ≈ 65 tok/s — the full eval + judge loop (66 tasks) ≈ 4 minutes.

---

## 12. Expanded results (eval v5, 124 tasks)

v4's 66 tasks (~8/category) were too few to estimate per-category rates reliably. v5 doubles the set to
**124 tasks (~16/category)** and over-samples the trap cases (`eval/gen_tasks_v5.py`;
`eval/validate_tasks.py` proves every code task's asserts are satisfiable offline). Two full zero-API runs
on the AMD Radeon rig, before and after a classifier fix:

| category | acc (before) | acc (after) | classify (before → after) |
|---|---:|---:|---|
| code_debug | 0.94 | 0.94 | 1.00 → 1.00 |
| code_gen | 0.94 | 0.94 | 1.00 → 1.00 |
| factual | 0.88 | 0.88 | 0.94 → 0.94 |
| logical | 0.81 | 0.75 | **0.56 → 1.00** |
| math | 0.88 | 0.94 | 0.94 → 0.94 |
| ner | 0.88 | 0.88 | 1.00 → 1.00 |
| sentiment | 0.81 | 0.88 | 1.00 → 1.00 |
| summarisation | 0.83 | 0.83 | 1.00 → 1.00 |
| **TOTAL** | **0.87** | **0.88** | 0.93 → **0.98** |

Zero-API, **0 Fireworks tokens**, ~420–450 s wall time.

**Classifier fix.** v5 exposed `logical` classification at **0.56** — seven logical puzzles (syllogisms,
comparison/ordering, race-position, family-relation riddles) were misrouting to `factual`/`math`. General
puzzle patterns were added to `_RE_LOGIC` (not prompt-specific), lifting logical classification to **1.00**
(overall 0.98) with **no accuracy regression** (total 0.87 → 0.88). One casualty: "you overtake 2nd place →
what place?" now routes to the logical path, whose step-by-step prompt makes the model *overthink* a
lateral-thinking gotcha it answered correctly as a terse factual (True → False). The routing fix still
matters because escalation is category-gated: without it, logical traps escalate under the wrong rung.

**The per-category efficiency ranking is noise-dominated at n=16 — do not over-tune to it.** With `temp=0`
Ollama nondeterminism, categories swing ±1 task/run (math, sentiment, and logical all moved between the two
runs above with *no* routing change). Consequently the marginal-efficiency ordering is unstable: `math` was
the top escalation target in v4 but near-worthless in the after-fix v5 run (it happened to score 0.94), while
`logical` jumped to the top. The data-driven ladder recompute (`escalation_math --results`) therefore
reorders run-to-run, and the v4 "**P3 (math+factual) dominates P2 (math+logical)**" finding (§4) does **not**
replicate on the larger set — there P2 ≈ P3 within noise.

**What *is* stable, and what it implies:**

- **Zero-API ≈ 0.87–0.88 at 0 tokens** — the bankable number, now on a larger sample. If the gate is
  ≤ ~0.85, L0 wins outright.
- **`factual` is the one deterministic error source:** the same two hallucinations (Berlin Wall → "1945",
  Canberra's water body) fail every run and are invisible to any local check. That makes `factual` the most
  *reliable* escalation target even when its instantaneous efficiency isn't the highest — which is why the
  shipped rung-2 keeps `{math, factual}` rather than chasing the noisy `logical`/`sentiment` numbers.
- **n=16/category is still too small** to stabilize the ranking; the true per-category rates (and the gate)
  are settled by the leaderboard, not by more local eval. The ladder order is left unchanged pending that
  signal.
