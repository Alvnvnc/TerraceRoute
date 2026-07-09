# AMD Developer Hackathon (ACT II) — Submissions

Monorepo holding my submissions for the AMD Developer Hackathon (ACT II).
Each track is a self-contained folder with its own README, code, and docs.

> 🇮🇩 Indonesian version: [`README.id.md`](README.id.md)

## Tracks

| Track | Folder | Title | Status |
|-------|--------|-------|--------|
| 1 | [`track1/`](track1/) | **TerraceRoute** — token-efficient routing agent | ✅ Complete |
| 2 | [`track2/`](track2/) | — | ⏳ Placeholder |
| 3 | [`track3/`](track3/) | Unicorn — self-hosted AMD agent infrastructure | 📝 Planning |

## Track 1 — TerraceRoute (highlight)

A batch agent that answers natural-language tasks across 8 categories while spending
**as few Fireworks API tokens as possible** — local inference is free, and the leaderboard
ranks passing entries by ascending token count. A local 3B model plus **independent
deterministic verification** (code execution, letter counting, arithmetic) reaches
**~85–91% accuracy at zero API tokens**, escalating to the remote model only when a
data-driven policy says the marginal accuracy is worth the token cost.

See [`track1/README.md`](track1/README.md) for the full architecture, evaluation, and
build/run instructions.

## Layout

```
README.md        ← you are here (repo landing)
README.id.md     ← Indonesian landing
track1/          ← Track 1 submission (complete)
track2/          ← Track 2 (placeholder)
track3/          ← Track 3 (placeholder)
```
