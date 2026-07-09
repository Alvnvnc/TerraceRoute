# TerraceRoute — Submission Form Copy (Track 1)

## Submission Title (5–50 chars)
```
TerraceRoute — Token-Efficient Routing Agent
```
(44 chars)

## Short Description (50–255 chars)
```
A CPU-only hybrid routing agent that answers most tasks with free local models and escalates to Fireworks AI only when a cheaper path would fail — clearing the accuracy gate at the lowest possible token cost.
```
(206 chars)

## Long Description (600–2000 chars, min 100 words)
```
TerraceRoute is a hybrid, token-efficient routing agent for Track 1. It reads a batch of natural-language tasks and answers each one using the fewest billed Fireworks tokens possible, without dropping below the accuracy gate.

The core insight came from our own calibration experiments: a small model's internal confidence signals (perplexity, self-consistency) actually anti-correlate with correctness — models are often confidently wrong on reasoning traps. What works instead is cross-model disagreement and, wherever possible, deterministic verification.

So TerraceRoute routes every task down a terrace of increasingly expensive checks, exiting as early as it safely can:
1. Classify each task into one of the eight capability domains.
2. Solve it locally with a bundled 3B model — local tokens cost zero.
3. Verify for free where the domain allows: math expressions are evaluated in Python, generated code is executed against self-tests, and counting tasks use real tools instead of the LLM.
4. Escalate to Fireworks AI only when local verification fails or the task lands in a known trap zone (logical/deductive puzzles), sending the most compact prompt that still clears the gate.

Everything runs inside the 4 GB / 2 vCPU, CPU-only, 10-minute scoring environment. A watchdog writes results progressively and guarantees every task is answered, so the container never times out or ships an empty slot. Escalation aggressiveness is a single knob we calibrate against the live leaderboard — the true oracle for the hidden accuracy threshold.
```
(~1,500 chars)

## Categories / Event Tracks
Track 1 — General-Purpose AI Agent (Hybrid Token-Efficient Routing Agent)

## Technologies Used
