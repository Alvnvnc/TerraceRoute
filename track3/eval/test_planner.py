"""Phase 2/3 planner wiring — offline (a fake Ollama, no GPU needed).

The live accuracy numbers come from ``run_brain_eval`` on the AMD box; these
tests lock the plumbing: constrained-output parsing, omission retry, and that
``dual_plan`` feeds two independent plans into the gate.
"""

import unittest

from agent.brain.llm import LLMResult, OllamaClient
from agent.brain.planner import dual_plan, plan_from_nl
from agent.brain.schemas import Plan
from agent.types import GateDecision


class FakeClient(OllamaClient):
    """Returns canned raw strings per model without touching the network."""

    def __init__(self, replies):
        self._replies = replies  # model -> raw string (or list for retry)

    def plan(self, model, user_text, *, constrain=True):
        from agent.brain.schemas import parse_plan
        raw = self._replies[model]
        if isinstance(raw, list):
            raw = raw.pop(0)
        return LLMResult(plan=parse_plan(raw), raw=raw, model=model, latency_s=0.0)


EXPOSE = '{"reasoning":"r","op":"expose","hostname":"media.example.com","port":8096,"service_scheme":"http"}'
UNEXPOSE_A = '{"reasoning":"r","op":"unexpose","hostname":"a.example.com","port":0,"service_scheme":"http"}'
DIAGNOSE = '{"reasoning":"r","op":"diagnose","hostname":"","port":0,"service_scheme":"http"}'


class TestPlanner(unittest.TestCase):
    def test_plan_from_nl_parses_constrained_output(self):
        c = FakeClient({"m": EXPOSE})
        res = plan_from_nl("expose it", model="m", client=c)
        self.assertIsInstance(res.plan, Plan)
        self.assertEqual(res.plan.op, "expose")
        self.assertEqual(res.plan.port, 8096)

    def test_agreement_auto_applies(self):
        c = FakeClient({"p": EXPOSE, "v": EXPOSE})
        dp = dual_plan("expose it", planner_model="p", verifier_model="v", client=c)
        self.assertEqual(dp.gate.decision, GateDecision.AUTO_APPLY)
        self.assertEqual(dp.gate.disagreement, 0.0)

    def test_destructive_disagreement_refuses(self):
        # planner wants to delete, verifier only wants to look — the danger signal.
        c = FakeClient({"p": UNEXPOSE_A, "v": DIAGNOSE})
        dp = dual_plan("clean up stuff", planner_model="p", verifier_model="v", client=c)
        self.assertEqual(dp.gate.disagreement, 1.0)
        self.assertEqual(dp.gate.decision, GateDecision.REFUSE)

    def test_omission_returns_no_plan(self):
        c = FakeClient({"m": "I cannot help with that."})
        res = plan_from_nl("gibberish", model="m", client=c)
        self.assertIsNone(res.plan)


if __name__ == "__main__":
    unittest.main()
