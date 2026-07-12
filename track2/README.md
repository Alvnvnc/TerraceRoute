# Track 2 — Video Captioning Agent

An agent that watches a video clip and captions it in four styles (`formal`,
`sarcastic`, `humorous_tech`, `humorous_non_tech`), scored by an LLM-judge on
**caption accuracy** and **style match**.

Captions are generated from extracted visual evidence, checked against the sampled
frames, and regenerated when they fail. The default backend uses a restricted public
proxy to a self-hosted `gemma3:12b` vision model; all backend settings remain overridable
through the runtime environment.

## Pipeline

```
ingest    download clip → ADAPTIVE sampling: one ffmpeg pass builds a dense pool of
          tiny gray thumbnails; thumbnail differencing gives a motion score, scene-cut
          positions, and dedupe from one deterministic signal. Simple clips → ~6-9
          frames @512px, many transitions → up to 20; hard caps 30 images / <10 MB
          payload. Frames keep their timestamps.
evidence  gemma3:12b vision → TEMPORAL EVIDENCE GRAPH: timestamped JSON (entities,
          timeline, scene_transitions, important_events, uncertainties). Each frame
          is preceded by its "[frame at t=Xs]" label — measured: without per-image
          labels the model binds frames to the wrong times by a whole scene.  (~9-19 s)
stylize   ONE structured JSON call from that one graph → all requested styles  (~4 s)
verify    confidence never comes from the generator's self-assessment:
          (a) free deterministic style lint (jargon blocklist for humorous_non_tech,
              tech-reference requirement for humorous_tech, formality rules)
          (b) visual checker re-scores each caption against the
              actual frames on the judge's own rubric (accuracy + style match)
          (c) deterministic evidence-graph gaps (empty timeline / no entities flag
              every style; many uncertainties ride along as repair context)
repair    ONE frame-grounded rewrite covering all flagged styles (Fireworks Gemma
          preferred when configured), accepted only if it clears the lint
```

Reliability layer inherited from Track 1: results are **pre-seeded** (every task ×
every style has a caption from second zero — a missing style scores 0), written
**atomically and progressively** after every clip, and a **watchdog** flushes and
exits 0 before the 10-minute limit. Clips are processed in parallel (all stages are
network-bound).

Measured end-to-end on the three official example clips (adaptive sampler + evidence
graph + checker + repair): **94 s total**, all 12 captions present, repair path
exercised. Sampler validation on a real multi-scene composite: 3 scenes detected at
the true cut positions, 13 frames allocated across all segments vs 8 for single-scene
clips (details in `plan.md` §7).

## Run

Track 2 is published independently from Track 1. The submission reference has this form:

```text
ghcr.io/alvnvnc/terraceroute:track2-<immutable-version>
```

Build the Track 2 component from this directory for local development:

```bash
docker build -t track2-captioner .
docker run --rm \
  -v $PWD/input:/input -v $PWD/output:/output \
  track2-captioner
```

From the repository root, publish a public `linux/amd64` image with:

```bash
REGISTRY=ghcr.io/alvnvnc ./scripts/build_and_push.sh 2 track2-v2
```

I/O per the harness spec: reads `/input/tasks.json`
(`[{task_id, video_url, styles:[...]}]`), writes `/output/results.json`
(`[{task_id, captions:{style: text}}]`).

Backend configuration (all optional, defaults target the restricted self-hosted proxy):
`VLM_BASE_URL`, `VLM_TOKEN`, `VLM_MODEL`, and `CHECKER_MODEL`. The default public
proxy only permits the pinned submission model and caps request size/output tokens.

### Fireworks Gemma fallback

The Track 1 `FIREWORKS_API_KEY` can be supplied to Track 2 at runtime; it is never
copied into the image. Create an image-capable Gemma on-demand deployment in Fireworks,
then set its deployment ID as `FIREWORKS_GEMMA_MODEL` (for example,
`accounts/<account>/deployments/<deployment>`). The agent uses it as a fallback if the
default generator is unavailable and preferentially for a frame-grounded repair after
the visual checker flags a caption. Checker calls never use the Fireworks generator
fallback, so a failed checker cannot silently approve its own generated caption.

```bash
docker run --rm --env-file ../track1/.env \
  -e FIREWORKS_GEMMA_MODEL=accounts/<account>/deployments/<deployment> \
  -v "$PWD/input:/input" -v "$PWD/output:/output" track2-captioner
```

`FB_BASE_URL` / `FB_API_KEY` / `FB_MODEL` remain an optional final fallback for generator
calls. Configure `FB_CHECKER_MODEL` separately if the checker needs a fallback; do not
point it to the caption generator.

See [`plan.md`](plan.md) for the full architecture rationale and measured evidence.
