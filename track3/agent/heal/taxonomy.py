"""The self-heal decision core.

Given a :class:`SignalVector` observed by the watchdog, return a single
:class:`Diagnosis` (failure mode + repair action + evidence). This is the
deterministic heart of the "heal" demo: no LLM, fully testable.

Design notes
------------
The rules are ordered and first-match-wins. Ordering encodes precedence:

1. A dead process (metrics port refuses TCP) is unambiguous — check first.
2. Specific log signatures (auth, TLS) are more informative than generic
   connection-count signals, so they outrank the edge/HA rules.
3. "Tunnel healthy + edge code" rules (origin down / ingress / slow) are
   mutually exclusive with the "not attached to edge" rules by construction:
   the former require ``ready_code == 200`` and ``ha_connections > 0``, the
   latter require ``ready_code == 503`` / ``ha_connections == 0``.

The key discriminator from the research: at the edge, ``1033`` means the
*connector* is down while ``502`` means the connector is up but the *origin*
is dead. We never treat 521/523 as origin-down for a tunnel — that path is 502.
"""

from __future__ import annotations

from typing import Callable

from ..types import Diagnosis, FailureMode, RepairAction, SignalVector

# Edge codes that mean "the connector itself is not serving".
_CONNECTOR_DOWN_EDGE_CODES = {1033, 530}


def _tunnel_looks_healthy(s: SignalVector) -> bool:
    """True when the connector is attached to the edge (so a failure is
    downstream of the tunnel — origin, ingress, or timing)."""
    if s.ready_code == 200:
        return True
    if s.ready_code is None and s.ha_connections is not None and s.ha_connections > 0:
        return True
    return False


def _process_dead(s: SignalVector) -> bool:
    # Metrics port refuses TCP => the cloudflared process is not running.
    if s.metrics_port_open is False:
        return True
    # No local signals at all, but the edge says "no healthy connector".
    if (
        s.metrics_port_open is None
        and s.ready_code is None
        and s.edge_code in _CONNECTOR_DOWN_EDGE_CODES
        and s.api_status in (None, "down", "inactive")
    ):
        return True
    return False


def _auth_invalid(s: SignalVector) -> bool:
    if {"unauthorized", "credentials_missing"} & s.log_signatures:
        return True
    # Never came up and never registered, but the process is running:
    # points at a bad/expired token or missing credentials, not a crash.
    if (
        s.ever_registered is False
        and s.api_status == "inactive"
        and s.metrics_port_open is not False
    ):
        return True
    return False


def _origin_tls(s: SignalVector) -> bool:
    return "x509" in s.log_signatures


def _origin_down(s: SignalVector) -> bool:
    if not _tunnel_looks_healthy(s):
        return False
    return s.edge_code == 502 or "connection_refused" in s.log_signatures


def _ingress_mismatch(s: SignalVector) -> bool:
    return _tunnel_looks_healthy(s) and s.edge_code == 404


def _origin_slow(s: SignalVector) -> bool:
    return _tunnel_looks_healthy(s) and s.edge_code in (524, 520)


def _edge_detached(s: SignalVector) -> bool:
    if s.ready_code == 503:
        return True
    if s.ready_code != 200 and s.ha_connections == 0:
        return True
    return False


def _partial_degraded(s: SignalVector) -> bool:
    if s.api_status == "degraded":
        return True
    if s.ha_connections is not None and 1 <= s.ha_connections <= 3:
        return True
    return False


def _healthy(s: SignalVector) -> bool:
    # A known non-healthy API status or a below-full HA count is not healthy —
    # it is degradation, handled by _partial_degraded.
    if s.api_status is not None and s.api_status != "healthy":
        return False
    if s.ha_connections is not None and s.ha_connections < 4:
        return False
    edge_ok = s.edge_code is None or 200 <= s.edge_code < 400
    return _tunnel_looks_healthy(s) and edge_ok and not s.log_signatures


# Ordered rules: (predicate, mode, repair, root_cause, confidence).
_RULES: list[tuple[Callable[[SignalVector], bool], FailureMode, RepairAction, str, float]] = [
    (_process_dead, FailureMode.PROCESS_DEAD, RepairAction.RESTART_CONNECTOR,
     "The cloudflared connector process is not running (metrics port refuses "
     "connections / edge reports no healthy connector).", 0.95),
    (_auth_invalid, FailureMode.AUTH_INVALID, RepairAction.REISSUE_TOKEN,
     "The tunnel never authenticated — the token or credentials are invalid, "
     "missing, or expired.", 0.9),
    (_origin_tls, FailureMode.ORIGIN_TLS, RepairAction.FIX_ORIGIN_TLS,
     "The tunnel is healthy but the origin's TLS certificate was rejected "
     "(x509 error).", 0.9),
    (_origin_down, FailureMode.ORIGIN_DOWN, RepairAction.RESTART_ORIGIN,
     "The tunnel is healthy but the local origin service refused the connection "
     "(edge 502) — your app is down or on the wrong port, not the tunnel.", 0.92),
    (_ingress_mismatch, FailureMode.INGRESS_MISMATCH, RepairAction.FIX_INGRESS,
     "The tunnel is healthy but no ingress rule matched the request "
     "(served by the catch-all, edge 404).", 0.88),
    (_origin_slow, FailureMode.ORIGIN_SLOW, RepairAction.TUNE_ORIGIN_TIMEOUT,
     "The origin accepts connections but is too slow to respond "
     "(edge 524/520).", 0.75),
    (_edge_detached, FailureMode.EDGE_DETACHED, RepairAction.CHECK_EGRESS_RESTART,
     "The connector is running but is not attached to the Cloudflare edge "
     "(/ready 503, 0 HA connections) — likely egress/UDP 7844 blocked.", 0.85),
    (_partial_degraded, FailureMode.PARTIAL_DEGRADED, RepairAction.MONITOR,
     "The tunnel is up but running with fewer than the usual edge connections "
     "(degraded) — often self-heals.", 0.7),
]


def diagnose(s: SignalVector) -> Diagnosis:
    """Map an observed :class:`SignalVector` to a single :class:`Diagnosis`."""
    if _healthy(s):
        return Diagnosis(
            mode=FailureMode.HEALTHY,
            repair=RepairAction.NONE,
            root_cause="Tunnel attached to the edge and serving normally.",
            evidence=_collect_evidence(s),
            confidence=0.9,
        )

    for predicate, mode, repair, root_cause, confidence in _RULES:
        if predicate(s):
            return Diagnosis(
                mode=mode,
                repair=repair,
                root_cause=root_cause,
                evidence=_collect_evidence(s),
                confidence=confidence,
            )

    return Diagnosis(
        mode=FailureMode.UNKNOWN,
        repair=RepairAction.ESCALATE_HUMAN,
        root_cause="Signals do not match a known failure mode; escalating with "
                   "the raw evidence for a human or the LLM to interpret.",
        evidence=_collect_evidence(s),
        confidence=0.3,
    )


def _collect_evidence(s: SignalVector) -> list[str]:
    ev: list[str] = []
    if s.metrics_port_open is not None:
        ev.append(f"metrics_port_open={s.metrics_port_open}")
    if s.ready_code is not None:
        ev.append(f"ready={s.ready_code}")
    if s.ha_connections is not None:
        ev.append(f"ha_connections={s.ha_connections}")
    if s.edge_code is not None:
        ev.append(f"edge={s.edge_code}")
    if s.api_status is not None:
        ev.append(f"api_status={s.api_status}")
    if s.ever_registered is not None:
        ev.append(f"ever_registered={s.ever_registered}")
    if s.log_signatures:
        ev.append("logs=" + ",".join(sorted(s.log_signatures)))
    return ev
