"""Journal audit trail + rollback semantics."""

import os
import tempfile
import unittest

from agent.journal import Journal


class TestJournal(unittest.TestCase):
    def test_records_to_jsonl(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "audit.jsonl")
            j = Journal(path=path)
            j.record("create_tunnel", detail={"id": "abc"})
            self.assertTrue(os.path.exists(path))
            with open(path) as fh:
                lines = fh.read().strip().splitlines()
            self.assertEqual(len(lines), 1)

    def test_rollback_is_lifo(self):
        j = Journal()
        order = []
        j.record("create_tunnel", undo=lambda: order.append("del_tunnel"),
                 undo_label="del_tunnel")
        j.record("create_dns", undo=lambda: order.append("del_dns"),
                 undo_label="del_dns")
        j.rollback()
        # DNS created last must be undone first.
        self.assertEqual(order, ["del_dns", "del_tunnel"])

    def test_rollback_continues_on_error(self):
        j = Journal()
        done = []

        def boom():
            raise RuntimeError("cannot delete: active connections")

        j.record("create_tunnel", undo=lambda: done.append("del_tunnel"),
                 undo_label="del_tunnel")
        j.record("create_dns", undo=boom, undo_label="del_dns")
        results = j.rollback()
        # The failing DNS undo is logged as error, but the tunnel undo still runs.
        self.assertIn("del_tunnel", done)
        self.assertTrue(any(r["status"] == "error" for r in results))

    def test_commit_clears_undo_stack(self):
        j = Journal()
        j.record("create_tunnel", undo=lambda: None, undo_label="del_tunnel")
        self.assertEqual(j.pending_undo, ["del_tunnel"])
        j.commit()
        self.assertEqual(j.pending_undo, [])
        self.assertEqual(j.rollback(), [])  # nothing left to undo


if __name__ == "__main__":
    unittest.main()
