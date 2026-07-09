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
- **CLI** — `expose`, `teardown`, `diagnose`, `status` (`agent/cli.py`).

Coming next: Phase 2 (local-model NL → plan via Ollama constrained decoding) and
Phase 3 (second model + live disagreement gate). See `plan.md` §7.

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
