"""TerraceGate CLI — the Phase 0/1 surface (no LLM yet).

Commands:
  expose    Create tunnel + DNS for a hostname and verify it live (with rollback).
  teardown  Remove an exposed hostname's DNS record and tunnel.
  diagnose  Observe a running tunnel and print a deterministic root-cause.
  status    Print the Cloudflare-side status of a tunnel.

Add --dry-run to expose/teardown to exercise the flow with no network/credentials.
The natural-language front-end (Phase 2) and the two-model safety gate (Phase 3)
build on top of these typed operations.
"""

from __future__ import annotations

import argparse
import sys

from .brain.schemas import Plan
from .config import load_config
from .heal.taxonomy import diagnose as diagnose_signals
from .journal import Journal
from .operations import expose, teardown
from .tools import connector as conn
from .tools.cloudflare import CloudflareClient, CloudflareError
from .tools.connector import ConnectorManager
from .tools.probe import probe
from .types import SignalVector


def _client(cfg, dry_run: bool) -> CloudflareClient:
    return CloudflareClient(cfg.account_id, cfg.zone_id, cfg.api_token, dry_run=dry_run)


def cmd_expose(args) -> int:
    cfg = load_config()
    if not args.dry_run and not cfg.has_credentials:
        print("error: missing Cloudflare credentials (set .env or use --dry-run)",
              file=sys.stderr)
        return 2
    plan = Plan(op="expose", hostname=args.host, port=args.port,
                service_scheme=args.scheme)
    journal = Journal(path=cfg.journal_path)
    connector = None if args.dry_run else ConnectorManager()
    verifier = (lambda h: True) if args.dry_run else None

    print(f"→ expose {plan.hostname} → {plan.origin_service()}"
          f"{'  [dry-run]' if args.dry_run else ''}")
    result = expose(_client(cfg, args.dry_run), plan, journal,
                    connector=connector, verifier=verifier)
    for step in result.steps:
        print(f"   • {step}")
    if result.ok:
        print(f"✓ live at {result.url}")
        return 0
    print(f"✗ failed: {result.error} (rolled back)", file=sys.stderr)
    return 1


def cmd_teardown(args) -> int:
    cfg = load_config()
    if not args.dry_run and not cfg.has_credentials:
        print("error: missing Cloudflare credentials", file=sys.stderr)
        return 2
    cf = _client(cfg, args.dry_run)
    journal = Journal(path=cfg.journal_path)
    dns_id = args.dns_id or (cf.find_dns_record(args.host) or "")
    result = teardown(cf, args.host, args.tunnel_id, dns_id, journal)
    for step in result.steps:
        print(f"   • {step}")
    print("✓ torn down" if result.ok else f"✗ {result.error}")
    return 0 if result.ok else 1


def cmd_diagnose(args) -> int:
    cfg = load_config()
    # Gather the signal vector from every surface we can reach.
    ready_code, ha = conn.query_ready(args.metrics)
    if ha is None:
        ha = conn.query_ha_connections(args.metrics)
    metrics_open = ready_code is not None or ha is not None

    edge_code = None
    if args.host:
        edge = probe(f"https://{args.host}/")
        edge_code = edge.effective_code

    api_status = None
    if args.tunnel_id and cfg.has_credentials:
        try:
            api_status = _client(cfg, False).get_tunnel_status(args.tunnel_id)
        except CloudflareError as exc:
            print(f"   (tunnel status unavailable: {exc})", file=sys.stderr)

    signals = SignalVector(
        metrics_port_open=metrics_open,
        ready_code=ready_code,
        ha_connections=ha,
        edge_code=edge_code,
        api_status=api_status,
    )
    d = diagnose_signals(signals)
    print(f"diagnosis: {d.mode.value}  (confidence {d.confidence:.2f})")
    print(f"  root cause: {d.root_cause}")
    print(f"  suggested repair: {d.repair.value}")
    print(f"  evidence: {', '.join(d.evidence) or '(none observed)'}")
    return 0


def cmd_status(args) -> int:
    cfg = load_config()
    if not cfg.has_credentials:
        print("error: missing Cloudflare credentials", file=sys.stderr)
        return 2
    status = _client(cfg, False).get_tunnel_status(args.tunnel_id)
    print(f"tunnel {args.tunnel_id}: {status}")
    return 0


def cmd_plan(args) -> int:
    """NL -> two local plans -> disagreement gate (Phase 2/3, needs Ollama)."""
    from .brain.planner import DEFAULT_PLANNER, DEFAULT_VERIFIER, dual_plan
    from .brain.llm import LLMError, OllamaClient

    client = OllamaClient(args.endpoint)
    try:
        dp = dual_plan(args.text, planner_model=args.planner,
                       verifier_model=args.verifier, client=client)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    def _fmt(res):
        p = res.plan
        body = f"{p.op} {p.hostname or '-'} :{p.port}/{p.service_scheme}" if p else "(no usable plan)"
        return f"{body}   [{res.tokens_per_s:.0f} tok/s{', retried' if res.retried else ''}]"

    print(f'"{args.text}"\n')
    print(f"  planner  {args.planner:22} → {_fmt(dp.planner)}")
    print(f"  verifier {args.verifier:22} → {_fmt(dp.verifier)}")
    g = dp.gate
    print(f"\n  blast radius : {g.blast_radius.name.lower()}")
    print(f"  disagreement : {g.disagreement:.2f}"
          + (f"  ({'; '.join(g.conflicts)})" if g.conflicts else ""))
    print(f"  → DECISION   : {g.decision.value.upper()}")
    print(f"    {g.rationale}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="terracegate", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("expose", help="expose localhost:PORT at HOST over HTTPS")
    e.add_argument("--host", required=True)
    e.add_argument("--port", type=int, required=True)
    e.add_argument("--scheme", default="http", choices=["http", "https"])
    e.add_argument("--dry-run", action="store_true")
    e.set_defaults(func=cmd_expose)

    t = sub.add_parser("teardown", help="remove an exposed hostname")
    t.add_argument("--host", required=True)
    t.add_argument("--tunnel-id", default="")
    t.add_argument("--dns-id", default="")
    t.add_argument("--dry-run", action="store_true")
    t.set_defaults(func=cmd_teardown)

    d = sub.add_parser("diagnose", help="observe a tunnel and print a root cause")
    d.add_argument("--host", default="")
    d.add_argument("--metrics", default="127.0.0.1:20241")
    d.add_argument("--tunnel-id", default="")
    d.set_defaults(func=cmd_diagnose)

    s = sub.add_parser("status", help="print Cloudflare-side tunnel status")
    s.add_argument("--tunnel-id", required=True)
    s.set_defaults(func=cmd_status)

    pl = sub.add_parser("plan", help="NL request → two local plans → safety gate")
    pl.add_argument("text", help="the natural-language request, in quotes")
    pl.add_argument("--endpoint", default="http://localhost:11434")
    pl.add_argument("--planner", default="gemma3:12b")
    pl.add_argument("--verifier", default="qwen2.5:3b-instruct")
    pl.set_defaults(func=cmd_plan)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
