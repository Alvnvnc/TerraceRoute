"""Phase 2/3 brain: natural language -> typed plan, with a second opinion.

Two responsibilities, kept separate from transport (:mod:`agent.brain.llm`) and
from the gate (:mod:`agent.brain.gate`):

* :func:`plan_from_nl` — one model turns a sentence into a :class:`Plan`.
* :func:`dual_plan` — two models of *different families* plan independently so
  the gate can measure cross-model disagreement (the empirically correct error
  signal; a confidently-wrong model looks certain to itself, a different-family
  model diverges — plan.md sections 1 and 6).

The two models run on the same local Ollama, so both inferences happen on the
AMD GPU and no request ever leaves the machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..types import GateResult
from .gate import decide
from .llm import LLMResult, OllamaClient
from .schemas import Plan

# Two different model families = the diversity the disagreement signal needs.
# Whatever is resident on this AMD box; overridable per call.
DEFAULT_PLANNER = "gemma3:12b"          # family: gemma3
DEFAULT_VERIFIER = "qwen2.5:3b-instruct"  # family: qwen2 (different lineage)


def plan_from_nl(
    text: str,
    *,
    model: str = DEFAULT_PLANNER,
    client: Optional[OllamaClient] = None,
) -> LLMResult:
    """Turn one natural-language request into a typed Plan via one local model."""
    client = client or OllamaClient()
    return client.plan(model, text)


@dataclass
class DualPlan:
    """Both models' plans plus the gate's verdict — the full Phase 3 decision."""

    text: str
    planner: LLMResult
    verifier: LLMResult
    gate: GateResult

    @property
    def plan(self) -> Optional[Plan]:
        """The primary plan (planner's, falling back to verifier's)."""
        return self.planner.plan or self.verifier.plan

    @property
    def tokens_per_s(self) -> float:
        return max(self.planner.tokens_per_s, self.verifier.tokens_per_s)


def dual_plan(
    text: str,
    *,
    planner_model: str = DEFAULT_PLANNER,
    verifier_model: str = DEFAULT_VERIFIER,
    client: Optional[OllamaClient] = None,
    target_exists: bool = False,
) -> DualPlan:
    """Plan with two models and run the safety gate on their (dis)agreement."""
    client = client or OllamaClient()
    a = client.plan(planner_model, text)
    b = client.plan(verifier_model, text)
    gate = decide(a.plan, b.plan, target_exists=target_exists, text=text)
    return DualPlan(text=text, planner=a, verifier=b, gate=gate)
