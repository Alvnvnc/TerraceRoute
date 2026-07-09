"""The safety gate: blast radius x cross-model disagreement -> act/ask/refuse.

This is the "knows when not to act" core. It never inspects the *quality* of a
plan (that is the model's job); it decides how much human confirmation an action
needs based on (a) how much damage it could do and (b) whether two independent
local models agreed on it.

Cross-model disagreement — not a single model's self-confidence — is the signal,
per the project's calibration finding and the literature (SAC^3, PoLL): a model
that is confidently and consistently wrong looks certain to itself, but a second
model of a different family tends to diverge.
"""

from __future__ import annotations

from typing import Optional

from ..types import BlastRadius, GateDecision, GateResult
from .schemas import Plan, normalize_for_compare

# Blast radius per operation. ``expose`` onto an already-live hostname mutates an
# existing record rather than adding one, so it escalates.
_BASE_BLAST = {
    "status": BlastRadius.READ_ONLY,
    "diagnose": BlastRadius.READ_ONLY,
    "heal": BlastRadius.MUTATING,       # restarts a process / rewrites ingress
    "expose": BlastRadius.ADDITIVE,
    "unexpose": BlastRadius.DESTRUCTIVE,
}


def blast_radius_of(plan: Plan, target_exists: bool = False) -> BlastRadius:
    br = _BASE_BLAST.get(plan.op, BlastRadius.MUTATING)
    if plan.op == "expose" and target_exists:
        # Re-pointing an existing hostname changes live routing.
        return BlastRadius.MUTATING
    return br


def disagreement(plan_a: Optional[Plan], plan_b: Optional[Plan]) -> tuple[float, list[str]]:
    """Score how much two plans diverge on their decision-relevant fields.

    Returns ``(score, conflicts)`` where score is 0 (identical) .. 1 (fully
    divergent). A missing plan (parse failure / omission) counts as maximum
    disagreement — we never let a silent failure read as agreement.
    """
    if plan_a is None or plan_b is None:
        return 1.0, ["one model produced no usable plan"]

    a = normalize_for_compare(plan_a)
    b = normalize_for_compare(plan_b)

    if a.get("op") != b.get("op"):
        return 1.0, [f"op: {a.get('op')!r} vs {b.get('op')!r}"]

    keys = set(a) | set(b)
    conflicts = [
        f"{k}: {a.get(k)!r} vs {b.get(k)!r}" for k in sorted(keys) if a.get(k) != b.get(k)
    ]
    score = len(conflicts) / len(keys) if keys else 0.0
    return score, conflicts


def decide(
    plan_a: Optional[Plan],
    plan_b: Optional[Plan],
    *,
    target_exists: bool = False,
    disagreement_threshold: float = 0.0,
    text: Optional[str] = None,
) -> GateResult:
    """Combine blast radius and disagreement into a gate decision.

    ``disagreement_threshold`` is the score above which two plans count as
    "disagreeing". Default 0.0: any conflict on a decision-relevant field is a
    disagreement, because for infrastructure the cost of a wrong destructive act
    is high and the fields are few and exact.

    ``text`` (the raw request) enables a deterministic backstop: if it carries
    destructive/compound intent the chosen op does not reflect, the request is
    never auto-applied even when both models agree — this closes the "both models
    dropped the same destructive clause" hole that disagreement alone can't see.
    """
    # The primary plan drives the blast-radius classification; if it failed to
    # parse, fall back to the secondary so we still gate conservatively.
    primary = plan_a or plan_b
    if primary is None:
        return GateResult(
            decision=GateDecision.REFUSE,
            blast_radius=BlastRadius.DESTRUCTIVE,
            disagreement=1.0,
            rationale="Neither model produced a usable plan; refusing to act.",
            conflicts=["both models failed to produce a plan"],
        )

    br = blast_radius_of(primary, target_exists=target_exists)
    score, conflicts = disagreement(plan_a, plan_b)
    disagree = score > disagreement_threshold

    decision = _matrix(br, disagree)

    # Deterministic intent-coverage backstop.
    text_reason = ""
    if text is not None:
        from .intent import text_risk
        risky, text_reason = text_risk(text, primary.op)
        if risky and decision == GateDecision.AUTO_APPLY:
            decision = GateDecision.CONFIRM
            conflicts = conflicts + [f"intent guard: {text_reason}"]

    rationale = _rationale(br, disagree, decision)
    if text_reason and decision != GateDecision.AUTO_APPLY:
        rationale += f" (intent guard: {text_reason})"
    return GateResult(
        decision=decision,
        blast_radius=br,
        disagreement=score,
        rationale=rationale,
        conflicts=conflicts,
    )


def _matrix(br: BlastRadius, disagree: bool) -> GateDecision:
    """The decision matrix from plan.md section 2."""
    if br <= BlastRadius.ADDITIVE:
        return GateDecision.CONFIRM if disagree else GateDecision.AUTO_APPLY
    if br == BlastRadius.MUTATING:
        return GateDecision.CONFIRM_PER_ITEM if disagree else GateDecision.CONFIRM
    # DESTRUCTIVE
    return GateDecision.REFUSE if disagree else GateDecision.CONFIRM_PER_ITEM


def _rationale(br: BlastRadius, disagree: bool, decision: GateDecision) -> str:
    agree_txt = "the two models disagreed" if disagree else "the two models agreed"
    return (
        f"Blast radius is {br.name.lower()} and {agree_txt}; "
        f"gate decision: {decision.value}."
    )
