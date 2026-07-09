"""Shared, dependency-free types for the whole agent.

Everything here is stdlib-only (dataclasses + enums) so the core logic —
failure taxonomy, safety gate, plan schema — is testable without a network,
without Cloudflare credentials, and without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Optional


# --------------------------------------------------------------------------- #
# Blast radius — how much damage an action can do if it is wrong.
# Ordered: a larger value is a bigger blast radius.
# --------------------------------------------------------------------------- #
class BlastRadius(IntEnum):
    READ_ONLY = 0     # inspect status, probe an endpoint
    ADDITIVE = 1      # create a new tunnel / DNS record (nothing pre-existing changes)
    MUTATING = 2      # change ingress on an existing tunnel, update a record
    DESTRUCTIVE = 3   # delete a tunnel / DNS record, change an Access policy


# --------------------------------------------------------------------------- #
# Failure taxonomy (plan.md section 5). Each mode is meant to be uniquely
# identifiable from the signal vector below.
# --------------------------------------------------------------------------- #
class FailureMode(Enum):
    HEALTHY = "healthy"
    PROCESS_DEAD = "process_dead"          # 1: cloudflared not running
    EDGE_DETACHED = "edge_detached"        # 2: up but 0 edge connections
    PARTIAL_DEGRADED = "partial_degraded"  # 3: some of the HA connections lost
    ORIGIN_DOWN = "origin_down"            # 4: tunnel healthy, local service refused
    INGRESS_MISMATCH = "ingress_mismatch"  # 5: no ingress rule matched (edge 404)
    AUTH_INVALID = "auth_invalid"          # 6: token/credentials invalid or expired
    ORIGIN_TLS = "origin_tls"              # 7: origin TLS/cert failure
    ORIGIN_SLOW = "origin_slow"            # 8: origin connects but hangs (edge 524)
    UNKNOWN = "unknown"                    # hand to the LLM narrator


class RepairAction(Enum):
    NONE = "none"
    RESTART_CONNECTOR = "restart_connector"
    CHECK_EGRESS_RESTART = "check_egress_then_restart"
    MONITOR = "monitor"                    # transient; watch before acting
    RESTART_ORIGIN = "restart_origin"
    FIX_INGRESS = "fix_ingress"
    REISSUE_TOKEN = "reissue_token"
    FIX_ORIGIN_TLS = "fix_origin_tls"
    TUNE_ORIGIN_TIMEOUT = "tune_origin_timeout"
    ESCALATE_HUMAN = "escalate_human"      # UNKNOWN: raw signals to a human/LLM


@dataclass
class SignalVector:
    """Everything the watchdog observes in one poll. All fields optional so a
    partial observation still yields a best-effort diagnosis.

    metrics_port_open : can we TCP-connect to cloudflared's local metrics port?
                        False is a strong "process is dead" signal.
    ready_code        : HTTP status from the connector's /ready endpoint
                        (200 = attached to edge, 503 = no edge connections).
    ha_connections    : cloudflared_tunnel_ha_connections gauge (4 = healthy, 0 = down).
    edge_code         : status/error code seen when probing the public hostname
                        from outside (200, 404, 502, 524, 1033, ...).
    api_status        : Cloudflare API tunnel status
                        (healthy | degraded | down | inactive).
    ever_registered   : has the connector ever successfully registered? False +
                        never-healthy points at auth/config rather than a crash.
    log_signatures    : set of parsed markers from cloudflared JSON logs, e.g.
                        {"connection_refused", "x509", "unauthorized",
                         "credentials_missing"}.
    """

    metrics_port_open: Optional[bool] = None
    ready_code: Optional[int] = None
    ha_connections: Optional[int] = None
    edge_code: Optional[int] = None
    api_status: Optional[str] = None
    ever_registered: Optional[bool] = None
    log_signatures: set[str] = field(default_factory=set)


@dataclass
class Diagnosis:
    mode: FailureMode
    repair: RepairAction
    root_cause: str          # human-readable, shown to the user / narrated by LLM
    evidence: list[str]      # which signals fired, for the audit journal
    confidence: float        # 0..1; UNKNOWN is low

    @property
    def healthy(self) -> bool:
        return self.mode is FailureMode.HEALTHY


@dataclass
class EdgeResult:
    """Result of probing the public hostname from outside the tunnel."""

    reachable: bool
    http_status: Optional[int] = None   # HTTP status line, if any
    cf_error_code: Optional[int] = None  # parsed Cloudflare 1xxx code (e.g. 1033)
    body_snippet: str = ""

    @property
    def effective_code(self) -> Optional[int]:
        """The single code the taxonomy keys on: a Cloudflare 1xxx error takes
        precedence over the wrapping HTTP status (e.g. 530 wrapping 1033)."""
        return self.cf_error_code if self.cf_error_code is not None else self.http_status


# --------------------------------------------------------------------------- #
# Safety gate (plan.md section 2).
# --------------------------------------------------------------------------- #
class GateDecision(Enum):
    AUTO_APPLY = "auto_apply"            # low risk + agreement: just do it (show diff)
    CONFIRM = "confirm"                  # one confirmation
    CONFIRM_PER_ITEM = "confirm_per_item"  # explicit confirmation for each field/item
    REFUSE = "refuse"                    # destructive + disagreement: refuse outright


@dataclass
class GateResult:
    decision: GateDecision
    blast_radius: BlastRadius
    disagreement: float       # 0 = plans identical, 1 = fully divergent
    rationale: str
    conflicts: list[str] = field(default_factory=list)  # fields the models disagreed on
