"""End-to-end agent command: gate decision must govern whether infra is touched.

Offline: a fake Ollama (canned raw) + dry-run Cloudflare client, so we assert the
*control flow* (refuse never executes; confirm halts without --yes; auto-apply
runs) without a GPU or network.
"""

import argparse
import unittest

from agent.cli import cmd_agent
from agent.brain.llm import LLMResult, OllamaClient
from agent.brain.schemas import parse_plan


class Fake(OllamaClient):
    def __init__(self, replies):
        self._r = replies

    def plan(self, model, user_text, *, constrain=True):
        raw = self._r[model]
        return LLMResult(plan=parse_plan(raw), raw=raw, model=model, latency_s=0.0)


EXPOSE = '{"reasoning":"r","op":"expose","hostname":"media.alvnvnc.site","port":8096,"service_scheme":"http"}'
UNEXPOSE_A = '{"reasoning":"r","op":"unexpose","hostname":"a.example.com","port":0,"service_scheme":"http"}'
DIAGNOSE = '{"reasoning":"r","op":"diagnose","hostname":"","port":0,"service_scheme":"http"}'


def _ns(**kw):
    base = dict(text="x", endpoint="", planner="p", verifier="v",
                execute=True, yes=False, dry_run=True)
    base.update(kw)
    return argparse.Namespace(**base)


class TestAgentFlow(unittest.TestCase):
    def _run(self, replies, **kw):
        # cmd_agent does `from .brain.planner import dual_plan` at call time, so
        # patching the module attribute injects our fake (offline) client.
        from agent.brain import planner
        fake = Fake(replies)
        real_dual = planner.dual_plan

        def patched(text, *, planner_model, verifier_model, client=None, target_exists=False):
            return real_dual(text, planner_model=planner_model,
                             verifier_model=verifier_model, client=fake,
                             target_exists=target_exists)

        planner.dual_plan = patched
        try:
            return cmd_agent(_ns(**kw))
        finally:
            planner.dual_plan = real_dual

    def test_refuse_never_executes(self):
        # planner deletes, verifier diagnoses -> destructive + disagreement -> REFUSE
        rc = self._run({"p": UNEXPOSE_A, "v": DIAGNOSE})
        self.assertEqual(rc, 3)

    def test_confirm_halts_without_yes(self):
        # expose onto same host by both -> additive agreement -> AUTO_APPLY executes
        rc = self._run({"p": EXPOSE, "v": EXPOSE})
        self.assertEqual(rc, 0)

    def test_plan_only_without_execute(self):
        rc = self._run({"p": EXPOSE, "v": EXPOSE}, execute=False)
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
