# Track 2 — Video Captioning Agent: Adaptation Plan

Theory of adaptation: Track 2 reuses Track 1's *reliability architecture* unchanged and
replaces its *economy policy* with a *quality policy*. This document maps each Track 1
module to its Track 2 counterpart and states what transfers, what inverts, and what is new.

## 1. What the two tracks share (verified rules)

| Dimension | Track 1 | Track 2 |
|---|---|---|
| Harness I/O | `/input/tasks.json` → `/output/results.json` | identical shape, different fields |
| Runtime cap | 10 minutes | 10 minutes |
| Failure modes | malformed JSON = 0, missing answer = 0 | malformed JSON = 0, **missing style = 0 for that clip** |
| Image | public, linux/amd64, ≤10GB | identical |
| Scoring | accuracy gate → rank by tokens | LLM-judge: caption accuracy (0–1) + style match (0–1) |
| Credentials | injected (`FIREWORKS_*`) | **none injected — bring your own** |

The **objective inverts**: Track 1 is an economy problem (pass the gate, minimize spend);
Track 2 is a quality problem (maximize two judge scores, spend is unconstrained). Everything
in Track 1 that exists to *save tokens* is dropped; everything that exists to *never score
zero* and to *not trust a single model output* transfers directly.

## 2. Module-by-module mapping

| Track 1 module | Fate | Track 2 counterpart |
|---|---|---|
| `run.py` (pre-seed skeleton, atomic progressive flush, watchdog, adaptive budget, parallel executor) | **transfers ~verbatim** | `run.py` — skeleton = every `task_id` × every requested style pre-filled with a safe fallback caption; watchdog + atomic flush identical; parallelism now spans clips (all work is I/O-bound) |
| `config.py` (env-only config) | transfers | env-only, but keys are **ours** (baked, spending-capped) since the harness injects nothing |
| `classify.py` (category router) | **deleted** | unnecessary — the "category" axis becomes the *style* axis, and styles are given in the input |
| `solve.py` `_SYS` per-category prompts | transforms | **style cards**: per-style definition + do/don't + 2–3 few-shot exemplars (formal / sarcastic / humorous_tech / humorous_non_tech) |
| `solve.py` status machine (`verified` / `verify_failed` / `empty` / `unverifiable`) | **transfers as concept** | caption states after verification; `verify_failed` → regenerate once with checker feedback (the analog of escalation) |
| `verify.py` (free deterministic checks) | transforms | **style lint** (deterministic, zero-cost): `humorous_non_tech` must contain no tech jargon (blocklist); `humorous_tech` must contain ≥1 tech reference (allowlist); `formal` bans slang/emoji/first person; length bounds for all |
| `fireworks.py` + `http.py` (OpenAI-compatible client, retries, never-raise) | transfers | same client shape pointed at a vision-capable provider; **cross-provider fallback** replaces local→remote fallback |
| escalation policy (`ESCALATION_LEVEL`, token caps, tail-to-remote) | **deleted** | no token ranking exists; the only "escalation" is regenerate-with-feedback |
| `eval/agent_eval.py` (offline judge eval driving the tuning knob) | **transfers ~1:1** | `eval/caption_eval.py` (built) — judge scores each caption on the *exact published rubric* (accuracy 0–1, style 0–1), judge from a **different model family** than the generator (cross-model calibration finding from T1/T3); supports A/B against a baseline results.json. Default judge is the local qwen2.5vl (same model as the checker), so absolute scores are optimistic — use deltas; point `JUDGE_MODEL`/`JUDGE_BASE_URL` at a third family for calibration |
| Dockerfile (CPU-only slim, self-contained) | transfers | much simpler: no local model, no llama.cpp build — ffmpeg + Python + HTTP clients |

## 3. Pipeline (new domain logic)

The one genuinely new module is **ingest**; everything downstream is Track 1 patterns.

```
Stage 0  INGEST     Download the clip (official clips are only 8–19 MB despite UHD —
                    full download takes seconds and is more robust than HTTP-range
                    streaming, which segfaulted the static ffmpeg build), then sample
                    frames ADAPTIVELY: one cheap ffmpeg pass produces a dense pool of
                    tiny gray thumbnails (~2/s); thumbnail differencing yields a motion
                    score, scene-cut positions, and a near-duplicate measure from the
                    same deterministic signal. Simple single-scene clips get ~6-9
                    frames, clips with many transitions up to 20, distributed over
                    scene segments by duration+motion weight, deduped, and capped at
                    30 images / <10 MB payload (Fireworks limits). Frames keep their
                    timestamps. No audio (official clips carry no audio stream).
Stage 1  EVIDENCE   ONE VLM call: timestamp-labelled frames → TEMPORAL EVIDENCE GRAPH,
                    a timestamped JSON (entities, timeline, scene_transitions,
                    important_events, uncertainties). Each image is preceded by its
                    "[frame at t=Xs]" text label — without per-image labels the model
                    mis-binds frame order to time (measured: transitions drifted ~6s;
                    with labels they land within one sampling interval of ground truth).
                    This graph is the single source of truth — the analog of Track 1's
                    "independent evidence".
Stage 2  STYLIZE    ONE schema-constrained TEXT call: evidence graph + every style card →
                    a complete caption map. One call prevents partial per-style failures
                    and grounds all styles in the same visual evidence.
Stage 3  VERIFY     Confidence is never the generator's self-assessment. Three signals:
                    (a) deterministic style lint (free);
                    (b) a different-family VLM checker (qwen2.5vl) scoring against the
                    judge's own rubric and its own view of the original frames (caption
                    checked against evidence, never against itself — the T1 calibration
                    lesson);
                    (c) deterministic evidence-graph gaps: empty timeline / no entities
                    flag EVERY style for repair; ≥3 uncertainties or a timeline that
                    stops mid-clip ride along as repair context.
                    A single frame-grounded repair rewrites all flagged styles with the
                    reviewer feedback. When configured, Fireworks Gemma is preferred for
                    this repair; Qwen remains the independent checker.
Stage 4  EMIT       progressive atomic flush after every clip; all requested styles are
                    always present (pre-seeded fallback survives any crash).
```

Why the two-stage split (perceive → stylize) instead of one call per style:
1. 1 vision call instead of 4 (vision calls are the slow/expensive/fragile part);
2. all 4 captions are grounded in the *same* facts → consistent accuracy scores;
3. the description doubles as the verification reference;
4. humor grounded in actual video content scores on BOTH judge dimensions —
   a generic joke maxes style but forfeits accuracy.

### Selective visual repair theory

The factual description is deliberately an information bottleneck: it makes the first
caption pass cheap and internally consistent, but may omit a small visual detail that the
judge cares about. The repair step is a **selective-prediction** policy: accept captions
that clear the deterministic lint and Qwen's visual score; spend an extra VLM call only
for captions rejected by either signal. The repair model sees the raw frames again, not
just the lossy description, so it can recover omitted evidence instead of paraphrasing the
same mistake. This is a generator-verifier loop with **cross-family verification**:
Gemma proposes or repairs, Qwen judges against the frames, and no model accepts its own
output as proof. JSON Schema on Fireworks constrains the repair response to the exact set
of flagged styles, avoiding a valid-looking but incomplete caption map.

## 4. Time-budget theory (Track 1 math, new constants)

Hidden clip count `n` is unknown; clips are 30s–2min. Per-clip serial cost ≈ ingest
5–15s + perceive 5–15s + stylize ~5s (parallel) + verify/regen ~10s ≈ 30–45s. All of it
is network-bound → run clips concurrently (4–6 workers), per-clip budget =
`(WATCHDOG_S − overhead) / ceil(n / workers)`.

Degradation ladder (analog of Track 1's preemptive throttle, quality-first order):
1. full pipeline (perceive → stylize → verify → regen);
2. drop regeneration (keep verification for logging only);
3. drop the checker call, keep the free style lint;
4. single combined call: frames → all 4 captions in one structured response;
5. fallback caption from whatever evidence exists (even 1 frame beats a blank —
   a missing style is a guaranteed 0, a mediocre caption is not).

## 5. Reliability requirements (inherited verbatim from T1)

- Pre-seed `/output/results.json` with every `task_id` × style before any work.
- Atomic write (`.tmp` + `os.replace`) + flush after every clip.
- Watchdog thread hard-exits 0 at `WATCHDOG_S` with whatever is done.
- Every network call: bounded retries, never raises, cross-provider fallback.
- Always exit 0 after writing valid JSON.

## 6. Risks specific to Track 2

- **Credentials at runtime**: do not bake keys into a public image. Pass the Track 1
  Fireworks key through the runtime environment and use a dedicated Gemma vision
  deployment; the pipeline degrades to fallback captions (never crashes) if it is dead.
- **Grading hardware unstated** (4GB/2vCPU is only written under Track 1 rules):
  assume CPU-only and API-driven; nothing here needs a GPU.
- **UHD downloads**: never download full files; ffmpeg-from-URL with a hard per-clip
  ingest timeout.
- **Rate limit 10 submissions/hour**: tune offline with `eval/caption_eval.py` on the
  3 published example clips + self-gathered stock clips of other scene types.
- **No fine-tuning — a deliberate decision, not a gap.** (a) There is no training
  signal: the judge is an LLM with an unpublished prompt and we hold 3 example clips
  with no gold captions, so the only possible dataset is our own pipeline's outputs
  scored by our own checker — distilling the pipeline into weights adds nothing at
  inference time. (b) Tuning Gemma on captions Qwen scored highly couples the
  generator to its verifier; correlated errors destroy exactly the cross-family
  disagreement signal this architecture is built on (the T1/T3 calibration finding).
  (c) Mechanically, a QLoRA pass on gemma3-12B-vision under ROCm/gfx1100 plus
  merge → GGUF convert → requantize → re-serve does not fit the submission window,
  and the W7900 is busy serving inference for two tracks. Prompt-level tuning
  (style cards) + evidence grounding + selective repair is the highest-EV use of the
  same hours.

## 7. Measured evidence (2026-07-11, Radeon instance hf-321-ee2a015c)

Adaptive-sampler + evidence-graph pipeline, `gemma3:12b` (generator) and
`qwen2.5vl:7b` (checker) served by Ollama on the remote Radeon W7900:

| Stage | Measured (official kitten clip) | Note |
|---|---|---|
| Adaptive ingest (download + analysis + 8 × 512px frames) | 6.8 s | motion 4.8, 1 scene → 8 frames, 507 KB payload |
| Evidence graph (8 labelled frames → timestamped JSON) | 8.8 s | timeline covers the full clip, zero gaps |
| Stylize (all 4 styles, ONE structured call) | 3.8 s | `response_format: json_object`, grounded styles |
| Qwen checker (6 spread frames, judge rubric) | 9.3 s | flags → one batched visual repair |

≈ 29–35 s per clip serial including verification. Full harness run on the three
official clips (docker image, 3 workers): **98.9 s total, 12/12 captions present**,
repair path exercised on every clip.

Sampler validation on a real multi-scene composite (kitten 0–12s → traffic 12–18s →
office 18–30s, concatenated from the official clips): detected `scenes=3`, allocated
13 frames across all three segments (vs 8 for single-scene clips), and the evidence
graph placed both scene transitions within one sampling interval of ground truth —
but ONLY with per-image timestamp labels interleaved in the message content; with a
bare header list of timestamps the model shifted events by ~6 s (one whole scene).

Unplanned chaos test: the Radeon instance was recycled by the hosting platform
mid-session ("Instance service not found"). A full harness run against the dead
backend finished in 26.7 s with exit 0 and a valid results.json — every style filled
by the pre-seeded fallback path. The zero-proofing layer holds under total backend
loss; the operational lesson stands: keep the instance alive during grading and
restart Ollama after any platform restart.

Open decision: the scoring container's VLM backend — (a) this Radeon instance as a
self-hosted API (AMD load-bearing narrative; requires the instance alive during grading;
needs a cloud fallback) vs (b) a cloud VLM API primary. Either way the pipeline code is
identical (OpenAI-compatible endpoint + model name in env).

## 8. Narrative (monorepo thesis)

One thesis across all three tracks: *never trust a single model's unverified output.*
- Track 1: deterministic verification (code execution, counting, arithmetic).
- Track 2: judge-aligned verification — captions checked against extracted visual
  evidence plus deterministic style lint, regenerated on failure.
- Track 3: cross-model disagreement gate before any action.
