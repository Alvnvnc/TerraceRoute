# TerraceRoute - AMD Hackathon Submissions

Track 1 and Track 2 live in one repository but are built and submitted as independent
containers. This keeps each scored image minimal and removes schema-dispatch ambiguity.
The root unified container remains available for local integration testing only.

> 🇮🇩 Indonesian version: [`README.id.md`](README.id.md)

## Tracks

| Track | Folder | Title | Status |
|-------|--------|-------|--------|
| 1 | [`track1/`](track1/) | **TerraceRoute** — token-efficient routing agent | ✅ Complete |
| 2 | [`track2/`](track2/) | Video captioning in four required styles | ✅ Implemented |
| 3 | [`track3/`](track3/) | Unicorn — self-hosted AMD agent infrastructure | 📝 Planning |

## Event Containers

Build each image from its track directory:

```bash
docker build --platform linux/amd64 -t terraceroute-track1:test track1
docker build --platform linux/amd64 -t terraceroute-track2:test track2
```

Publish each track under a distinct immutable tag:

```bash
REGISTRY=ghcr.io/alvnvnc ./scripts/build_and_push.sh 1 track1-v2
REGISTRY=ghcr.io/alvnvnc ./scripts/build_and_push.sh 2 track2-v2
```

Paste the matching bare image reference into each track submission. Do not prefix it
with `http://` or `https://`:

```text
Track 1: ghcr.io/alvnvnc/terraceroute:track1-v2
Track 2: ghcr.io/alvnvnc/terraceroute:track2-v2
```

Both images must be public and expose a `linux/amd64` manifest. Do not overwrite a tag
after submitting it; publish a new tag and re-save the new reference so the grader cannot
reuse a cached manifest.

## Local Unified Container

The optional root image detects the input schema and is useful for exercising both runners:

```bash
docker build -t terraceroute:unified-test .
```

Do not use this larger image as the primary event reference unless a single-image
submission is explicitly required.

## Track 1 - TerraceRoute

A batch agent that answers natural-language tasks across 8 categories while spending
**as few Fireworks API tokens as possible** — local inference is free, and the leaderboard
ranks passing entries by ascending token count. A local 3B model plus deterministic
tools and syntax checks reaches
**~85–91% accuracy at zero API tokens**, escalating to the remote model only when a
data-driven policy says the marginal accuracy is worth the token cost.

See [`track1/README.md`](track1/README.md) for the full architecture, evaluation, and
build/run instructions.

## Layout

```
entrypoint.py     ← optional local Track 1/Track 2 schema dispatcher
Dockerfile       ← optional unified integration image
requirements.txt ← unified runtime dependencies
track1/          ← token-efficient text agent and scored Dockerfile
track2/          ← video captioning agent and scored Dockerfile
track3/          ← separate Unicorn-track project
```
