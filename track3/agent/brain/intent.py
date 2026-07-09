"""Deterministic intent-coverage guard — a backstop the disagreement gate needs.

Cross-model disagreement catches the case where the two models *diverge*. It is
blind to the case where both models make the *same* omission: a compound request
like "expose git.example.com AND delete everything else" can be truncated by both
models to the benign "expose" sub-intent, so they agree and the gate would
auto-apply — silently dropping the destructive clause.

This module closes that hole without an LLM: it scans the raw request for
destructive language and for multiple distinct actions. If the text carries
destructive intent that the chosen plan's op does not reflect, the request is
flagged so the gate refuses to auto-apply (escalates to confirmation). Pure
rules, in the spirit of plan.md section 5 — diagnosis and safety limits are
deterministic; the LLM only proposes.
"""

from __future__ import annotations

import re

# Destructive verbs/phrases in English + Indonesian. Word-boundary matched.
_DESTRUCTIVE = re.compile(
    r"\b(delete|remove|wipe|destroy|purge|tear\s*down|take\s*down|drop|"
    r"reset|clear\s*out|clean\s*up|tidy\s*up|"
    r"hapus|hilangkan|bersihkan|rapikan|matikan\s*semua|singkirkan|buang)\b",
    re.IGNORECASE,
)

# "Everything / all / the rest" scope words that make a destructive verb broad.
_BROAD_SCOPE = re.compile(
    r"\b(all|every|everything|the\s+rest|semua|semuanya|seluruh|sisanya|"
    r"yang\s+lain|lainnya)\b",
    re.IGNORECASE,
)

# Conjunctions that join a second action onto the first.
_CONJUNCTION = re.compile(
    r"\b(and\s+(also\s+)?(delete|remove|then|take|tear)|then|also|plus|"
    r"lalu|kemudian|terus|serta|dan\s+(juga\s+)?(hapus|matikan|take))\b",
    re.IGNORECASE,
)

# Ops that already *are* destructive — a destructive verb is expected there and
# is not an uncovered clause.
_DESTRUCTIVE_OPS = {"unexpose"}


def has_destructive_language(text: str) -> bool:
    return bool(_DESTRUCTIVE.search(text))


def has_broad_scope(text: str) -> bool:
    return bool(_BROAD_SCOPE.search(text))


def looks_compound(text: str) -> bool:
    """True when the request appears to ask for more than one action."""
    return bool(_CONJUNCTION.search(text))


def uncovered_destructive_intent(text: str, op: str) -> bool:
    """The plan's op doesn't account for destructive language present in the text.

    e.g. op == "expose" but the sentence also says "delete everything else" — the
    model dropped the destructive clause; do not auto-apply.
    """
    if op in _DESTRUCTIVE_OPS:
        return False
    return has_destructive_language(text)


def text_risk(text: str, op: str) -> tuple[bool, str]:
    """Overall deterministic risk for a request/op pair.

    Returns ``(risky, reason)``. ``risky`` means "must not auto-apply" — the gate
    downgrades AUTO_APPLY to CONFIRM even if the two models agreed.
    """
    if uncovered_destructive_intent(text, op):
        scope = " (broad scope)" if has_broad_scope(text) else ""
        return True, f"request contains destructive language not reflected in op '{op}'{scope}"
    if looks_compound(text) and op != "unexpose":
        return True, "request appears to bundle multiple actions"
    return False, ""
