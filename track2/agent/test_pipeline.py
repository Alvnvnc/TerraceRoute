"""Focused tests for frame-grounded caption repair."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from .ingest import Clip
from .pipeline import _regenerate


class VisualRepairTest(unittest.TestCase):
    def test_repair_uses_original_frames_and_prefers_fireworks(self) -> None:
        clip = Clip(task_id="clip-1", duration_s=8.0, frames_b64=["aGVsbG8="])
        with patch("agent.pipeline.vlm.chat", return_value=(
            '{"formal":"A person walks through a sunny park."}'
        )) as chat:
            repaired = _regenerate(
                clip,
                '{"timeline": [{"t": "0-8s", "event": "a person walks through a sunny park"}]}',
                {"formal": "A person walks outside."},
                {"formal": "caption needs more visual detail"},
                ["timeline does not cover the second half of the clip"],
            )

        self.assertEqual(repaired, {"formal": "A person walks through a sunny park."})
        self.assertEqual(chat.call_args.kwargs["images_b64"], ["aGVsbG8="])
        self.assertTrue(chat.call_args.kwargs["prefer_fireworks"])
        self.assertEqual(chat.call_args.kwargs["json_schema"]["required"], ["formal"])


if __name__ == "__main__":
    unittest.main()
