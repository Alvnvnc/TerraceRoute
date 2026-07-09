"""The deterministic intent-coverage guard (closes the compound-request hole)."""

import unittest

from agent.brain.gate import decide
from agent.brain.intent import looks_compound, text_risk, uncovered_destructive_intent
from agent.brain.schemas import Plan
from agent.types import GateDecision


def expose(host="git.example.com", port=3000):
    return Plan(op="expose", hostname=host, port=port)


class TestIntentSignals(unittest.TestCase):
    def test_uncovered_destructive_en(self):
        self.assertTrue(uncovered_destructive_intent(
            "expose git.example.com and delete everything else", "expose"))

    def test_uncovered_destructive_id(self):
        self.assertTrue(uncovered_destructive_intent(
            "bikin git.example.com online terus hapus semua yang lain", "expose"))

    def test_destructive_op_not_flagged(self):
        # "delete old.example.com" with op=unexpose is *covered*, not a hidden clause.
        self.assertFalse(uncovered_destructive_intent(
            "delete old.example.com", "unexpose"))

    def test_plain_expose_not_flagged(self):
        self.assertFalse(uncovered_destructive_intent(
            "expose media.example.com port 8096", "expose"))

    def test_compound_detection(self):
        self.assertTrue(looks_compound("expose A and then remove B"))
        self.assertTrue(looks_compound("bikin A online terus hapus B"))
        self.assertFalse(looks_compound("expose media.example.com port 8096"))


class TestGateWithText(unittest.TestCase):
    def test_compound_agreement_no_longer_autoapplies(self):
        # Both models agreed on the benign expose; the raw text carries a dropped
        # destructive clause -> guard downgrades AUTO_APPLY to CONFIRM.
        r = decide(expose(), expose(),
                   text="bikin git.example.com online port 3000 terus hapus semua yang lain")
        self.assertEqual(r.decision, GateDecision.CONFIRM)
        self.assertTrue(any("intent guard" in c for c in r.conflicts))

    def test_clean_expose_still_autoapplies(self):
        r = decide(expose(), expose(), text="expose git.example.com port 3000")
        self.assertEqual(r.decision, GateDecision.AUTO_APPLY)

    def test_backwards_compatible_without_text(self):
        r = decide(expose(), expose())
        self.assertEqual(r.decision, GateDecision.AUTO_APPLY)


if __name__ == "__main__":
    unittest.main()
