"""The safety gate: blast radius x disagreement -> act/ask/refuse."""

import unittest

from agent.brain.gate import blast_radius_of, decide, disagreement
from agent.brain.schemas import Plan
from agent.types import BlastRadius, GateDecision


def expose(host="media.example.com", port=8096, scheme="http"):
    return Plan(op="expose", hostname=host, port=port, service_scheme=scheme)


def unexpose(host="media.example.com"):
    return Plan(op="unexpose", hostname=host)


class TestBlastRadius(unittest.TestCase):
    def test_status_is_read_only(self):
        self.assertEqual(blast_radius_of(Plan(op="status")), BlastRadius.READ_ONLY)

    def test_expose_new_is_additive(self):
        self.assertEqual(blast_radius_of(expose()), BlastRadius.ADDITIVE)

    def test_expose_existing_escalates_to_mutating(self):
        self.assertEqual(
            blast_radius_of(expose(), target_exists=True), BlastRadius.MUTATING
        )

    def test_unexpose_is_destructive(self):
        self.assertEqual(blast_radius_of(unexpose()), BlastRadius.DESTRUCTIVE)


class TestDisagreement(unittest.TestCase):
    def test_identical_plans_agree(self):
        score, conflicts = disagreement(expose(), expose())
        self.assertEqual(score, 0.0)
        self.assertEqual(conflicts, [])

    def test_different_port_conflicts(self):
        score, conflicts = disagreement(expose(port=8096), expose(port=8080))
        self.assertGreater(score, 0.0)
        self.assertTrue(any("port" in c for c in conflicts))

    def test_different_op_is_total_disagreement(self):
        score, _ = disagreement(expose(), unexpose())
        self.assertEqual(score, 1.0)

    def test_missing_plan_is_total_disagreement(self):
        score, conflicts = disagreement(expose(), None)
        self.assertEqual(score, 1.0)
        self.assertTrue(conflicts)

    def test_reasoning_difference_does_not_count(self):
        a = expose()
        b = expose()
        a.reasoning = "user wants jellyfin exposed"
        b.reasoning = "expose the media server"
        self.assertEqual(disagreement(a, b)[0], 0.0)


class TestGateMatrix(unittest.TestCase):
    def test_additive_agreement_auto_applies(self):
        r = decide(expose(), expose())
        self.assertEqual(r.decision, GateDecision.AUTO_APPLY)

    def test_additive_disagreement_asks(self):
        r = decide(expose(port=8096), expose(port=8080))
        self.assertEqual(r.decision, GateDecision.CONFIRM)

    def test_mutating_agreement_confirms_once(self):
        r = decide(expose(), expose(), target_exists=True)
        self.assertEqual(r.decision, GateDecision.CONFIRM)

    def test_mutating_disagreement_confirms_per_item(self):
        r = decide(expose(port=8096), expose(port=8080), target_exists=True)
        self.assertEqual(r.decision, GateDecision.CONFIRM_PER_ITEM)

    def test_destructive_agreement_confirms_per_item(self):
        r = decide(unexpose(), unexpose())
        self.assertEqual(r.decision, GateDecision.CONFIRM_PER_ITEM)

    def test_destructive_disagreement_refuses(self):
        # The headline "knows when not to act" case.
        r = decide(unexpose("a.example.com"), unexpose("b.example.com"))
        self.assertEqual(r.decision, GateDecision.REFUSE)

    def test_both_plans_missing_refuses(self):
        r = decide(None, None)
        self.assertEqual(r.decision, GateDecision.REFUSE)

    def test_one_plan_missing_on_destructive_refuses(self):
        # A destructive op where a model failed to plan must not slip through.
        r = decide(unexpose(), None)
        self.assertEqual(r.decision, GateDecision.REFUSE)


if __name__ == "__main__":
    unittest.main()
