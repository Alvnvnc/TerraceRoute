"""Orchestrated infrastructure transactions: plan -> apply -> verify -> rollback.

These are the typed tools composed into one atomic-ish operation. Every step
that creates something pushes its reverse onto the journal's undo stack, so a
failure at *any* later step (including the external probe never going green)
unwinds everything created so far. Success calls ``journal.commit()``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .brain.schemas import Plan
from .journal import Journal
from .tools.cloudflare import CloudflareClient
from .tools.connector import ConnectorManager
from .tools.probe import is_live, probe

# A verifier answers "is this hostname live at the edge?"; injectable for tests.
Verifier = Callable[[str], bool]


@dataclass
class ExposeResult:
    ok: bool
    hostname: str
    url: str = ""
    tunnel_id: str = ""
    dns_id: str = ""
    error: str = ""
    steps: list[str] = field(default_factory=list)


def _default_verifier(hostname: str) -> bool:
    return is_live(probe(f"https://{hostname}/"))


def expose(
    cf: CloudflareClient,
    plan: Plan,
    journal: Journal,
    *,
    connector: Optional[ConnectorManager] = None,
    verifier: Optional[Verifier] = None,
    attempts: int = 10,
    delay: float = 3.0,
    sleep: Callable[[float], None] = time.sleep,
) -> ExposeResult:
    """Expose ``localhost:{plan.port}`` at ``plan.hostname`` over HTTPS.

    Rolls back automatically if the hostname is not live within
    ``attempts * delay`` seconds.
    """
    verifier = verifier or _default_verifier
    result = ExposeResult(ok=False, hostname=plan.hostname)
    tunnel_name = f"terracegate-{plan.hostname.replace('.', '-')}"

    try:
        # 1. Create the tunnel; register its full teardown as the undo.
        tunnel = cf.create_tunnel(tunnel_name)
        tunnel_id = tunnel["id"]
        result.tunnel_id = tunnel_id
        result.steps.append(f"created tunnel {tunnel_id}")

        def _undo_tunnel() -> None:
            if connector is not None:
                connector.stop()
            cf.delete_tunnel_connections(tunnel_id)
            cf.delete_tunnel(tunnel_id)

        journal.record("create_tunnel", detail={"id": tunnel_id},
                       undo=_undo_tunnel, undo_label=f"delete_tunnel:{tunnel_id}")

        # 2. Start the connector with the run token (skipped in pure dry runs).
        if connector is not None:
            connector.run_token(tunnel.get("token") or cf.get_tunnel_token(tunnel_id))
            result.steps.append("started connector")
            journal.record("start_connector", detail={"tunnel": tunnel_id})

        # 3. Push ingress config (idempotent full replace).
        ingress = cf.build_ingress(plan.hostname, plan.origin_service())
        cf.put_configuration(tunnel_id, ingress)
        result.steps.append("put ingress config")
        journal.record("put_configuration", detail={"ingress": ingress})

        # 4. Create the proxied CNAME; register its deletion as the undo.
        dns_id = cf.create_dns_cname(plan.hostname, tunnel_id, proxied=True)
        result.dns_id = dns_id
        result.steps.append(f"created DNS record {dns_id}")
        journal.record("create_dns", detail={"id": dns_id, "host": plan.hostname},
                       undo=lambda: cf.delete_dns_record(dns_id),
                       undo_label=f"delete_dns:{dns_id}")

        # 5. Verify from the edge before declaring success.
        for i in range(attempts):
            if verifier(plan.hostname):
                result.ok = True
                result.url = f"https://{plan.hostname}/"
                result.steps.append(f"verified live after {i + 1} probe(s)")
                journal.record("verify_live", detail={"host": plan.hostname})
                journal.commit()
                return result
            if i < attempts - 1:
                sleep(delay)

        raise TimeoutError(
            f"{plan.hostname} did not go live within {attempts * delay:.0f}s"
        )

    except Exception as exc:  # noqa: BLE001 - any failure triggers rollback
        result.error = str(exc)
        result.steps.append(f"ERROR: {exc} -> rolling back")
        journal.record("expose_failed", status="error", detail={"error": str(exc)})
        journal.rollback()
        return result


def teardown(
    cf: CloudflareClient,
    hostname: str,
    tunnel_id: str,
    dns_id: str,
    journal: Journal,
    *,
    connector: Optional[ConnectorManager] = None,
) -> ExposeResult:
    """Explicitly tear down an exposed hostname (the DESTRUCTIVE op)."""
    result = ExposeResult(ok=False, hostname=hostname, tunnel_id=tunnel_id, dns_id=dns_id)
    try:
        if dns_id:
            cf.delete_dns_record(dns_id)
            journal.record("delete_dns", detail={"id": dns_id})
            result.steps.append("deleted DNS record")
        if connector is not None:
            connector.stop()
        if tunnel_id:
            cf.delete_tunnel_connections(tunnel_id)
            cf.delete_tunnel(tunnel_id)
            journal.record("delete_tunnel", detail={"id": tunnel_id})
            result.steps.append("deleted tunnel")
        result.ok = True
        return result
    except Exception as exc:  # noqa: BLE001
        result.error = str(exc)
        journal.record("teardown_failed", status="error", detail={"error": str(exc)})
        return result
