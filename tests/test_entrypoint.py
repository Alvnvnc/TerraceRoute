from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import entrypoint


class DispatcherTests(unittest.TestCase):
    def test_detects_track1_prompt(self):
        tasks = [{"task_id": "1", "prompt": "What is the capital of France?"}]
        self.assertEqual(entrypoint.detect_track(tasks), 1)

    def test_detects_track2_signed_url_without_extension(self):
        tasks = [{"task_id": "2", "video_url": "https://example.test/signed?id=42"}]
        self.assertEqual(entrypoint.detect_track(tasks), 2)

    def test_detects_track2_from_official_styles(self):
        tasks = [{"task_id": "2", "styles": ["formal", "sarcastic"]}]
        self.assertEqual(entrypoint.detect_track(tasks), 2)

    def test_rejects_mixed_input_without_override(self):
        tasks = [
            {"task_id": "1", "prompt": "Summarize this."},
            {"task_id": "2", "video_url": "https://example.test/a.mp4"},
        ]
        with self.assertRaisesRegex(ValueError, "mixed"):
            entrypoint.detect_track(tasks)
        self.assertEqual(entrypoint.detect_track(tasks, "track2"), 2)

    def test_normalizes_video_uri_for_track2_runner(self):
        tasks = [{"task_id": "2", "video_uri": "https://example.test/a.mp4"}]
        normalized = entrypoint.normalize_tasks(tasks, 2)
        self.assertEqual(normalized[0]["video_url"], tasks[0]["video_uri"])
        self.assertNotIn("video_url", tasks[0])

    def test_main_dispatches_wrapped_track2_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "tasks.json"
            output_path = root / "results.json"
            input_path.write_text(
                json.dumps(
                    {
                        "tasks": [
                            {
                                "task_id": "clip-1",
                                "video_uri": "https://example.test/clip",
                                "styles": ["formal"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            def fake_run(command, *, env, check):
                self.assertEqual(command[-1], "track2.agent.run")
                self.assertFalse(check)
                child_tasks = json.loads(Path(env["INPUT_PATH"]).read_text(encoding="utf-8"))
                self.assertEqual(child_tasks[0]["video_url"], "https://example.test/clip")
                Path(env["OUTPUT_PATH"]).write_text("[]", encoding="utf-8")
                return SimpleNamespace(returncode=0)

            environment = {
                "INPUT_PATH": str(input_path),
                "OUTPUT_PATH": str(output_path),
            }
            with mock.patch.dict(os.environ, environment, clear=True), mock.patch(
                "entrypoint.subprocess.run", side_effect=fake_run
            ):
                self.assertEqual(entrypoint.main(), 0)

            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), [])

    def test_malformed_input_still_writes_valid_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "tasks.json"
            output_path = root / "results.json"
            input_path.write_text("not json", encoding="utf-8")
            environment = {
                "INPUT_PATH": str(input_path),
                "OUTPUT_PATH": str(output_path),
            }
            with mock.patch.dict(os.environ, environment, clear=True), mock.patch(
                "entrypoint.subprocess.run"
            ) as run:
                self.assertEqual(entrypoint.main(), 0)
                run.assert_not_called()
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), [])


if __name__ == "__main__":
    unittest.main()
