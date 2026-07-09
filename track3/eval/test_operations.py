"""End-to-end expose flow with a dry-run Cloudflare client (no network).

Proves the transaction shape: success commits with no pending undo; a failed
external verification rolls everything back in LIFO order.
"""

import unittest

from agent.brain.schemas import Plan
from agent.journal import Journal
from agent.operations import expose
from agent.tools.cloudflare import CloudflareClient


def dry_cf():
    return CloudflareClient("acct", "zone", "token", dry_run=True)


def a_plan():
    return Plan(op="expose", hostname="media.example.com", port=8096)


class TestExpose(unittest.TestCase):
    def test_success_commits(self):
        j = Journal()
        r = expose(dry_cf(), a_plan(), j,
                   verifier=lambda host: True, sleep=lambda _: None)
        self.assertTrue(r.ok)
        self.assertEqual(r.url, "https://media.example.com/")
        self.assertTrue(r.tunnel_id)
        self.assertTrue(r.dns_id)
        self.assertEqual(j.pending_undo, [])  # committed: nothing to roll back

    def test_verify_failure_rolls_back(self):
        j = Journal()
        r = expose(dry_cf(), a_plan(), j,
                   verifier=lambda host: False, attempts=3, delay=0.0,
                   sleep=lambda _: None)
        self.assertFalse(r.ok)
        self.assertIn("did not go live", r.error)
        self.assertEqual(j.pending_undo, [])  # undo stack drained by rollback

        actions = [e["action"] for e in j.entries()]
        dns_undo = "undo:delete_dns:dryrun-dns-media.example.com"
        tunnel_undo = "undo:delete_tunnel:dryrun-terracegate-media-example-com"
        # Both created resources were undone...
        self.assertIn(dns_undo, actions)
        self.assertIn(tunnel_undo, actions)
        # ...and DNS (created last) was undone before the tunnel (LIFO).
        self.assertLess(actions.index(dns_undo), actions.index(tunnel_undo))

    def test_verifier_eventually_succeeds(self):
        j = Journal()
        calls = {"n": 0}

        def verifier(host):
            calls["n"] += 1
            return calls["n"] >= 2  # fails once, then live

        r = expose(dry_cf(), a_plan(), j, verifier=verifier,
                   attempts=5, delay=0.0, sleep=lambda _: None)
        self.assertTrue(r.ok)
        self.assertEqual(calls["n"], 2)


if __name__ == "__main__":
    unittest.main()
