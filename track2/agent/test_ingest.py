"""Deterministic tests for the adaptive temporal sampler (no network, no ffmpeg)."""
from __future__ import annotations

import unittest

from .config import config
from .ingest import _TH, _TW, Clip, _enforce_payload_caps, _plan_samples

_FSZ = _TW * _TH


def _flat(level: int) -> bytes:
    return bytes([level]) * _FSZ


def _noisy(level: int, seed: int) -> bytes:
    """Same scene + small motion: a few pixels differ per frame."""
    buf = bytearray(_flat(level))
    for k in range(40):
        buf[(seed * 131 + k * 37) % _FSZ] = (level + 60) % 256
    return bytes(buf)


class PlanSamplesTest(unittest.TestCase):
    def test_static_clip_gets_few_frames(self) -> None:
        thumbs = [_flat(100) for _ in range(60)]  # 30s tripod shot, zero motion
        times, n_scenes, motion = _plan_samples(thumbs, 2.0, config.max_frames)
        self.assertEqual(n_scenes, 1)
        self.assertLessEqual(len(times), config.min_frames)
        self.assertGreaterEqual(len(times), 3)

    def test_multi_scene_clip_gets_more_frames(self) -> None:
        # Three hard cuts: black -> gray -> white -> black, small motion within scenes.
        thumbs = ([_noisy(20, i) for i in range(20)]
                  + [_noisy(110, i) for i in range(20)]
                  + [_noisy(200, i) for i in range(20)]
                  + [_noisy(60, i) for i in range(20)])
        times, n_scenes, _ = _plan_samples(thumbs, 2.0, config.max_frames)
        static_times, _, _ = _plan_samples([_flat(100)] * 80, 2.0, config.max_frames)
        self.assertEqual(n_scenes, 4)
        self.assertGreater(len(times), len(static_times))
        # Every scene contributes at least one timestamp (10s segments at 2 fps).
        for lo, hi in ((0, 10), (10, 20), (20, 30), (30, 40)):
            self.assertTrue(any(lo <= t < hi for t in times),
                            f"no frame from segment {lo}-{hi}s in {times}")

    def test_respects_max_frames_cap(self) -> None:
        thumbs = [_noisy(20 + (i // 6) * 25 % 200, i) for i in range(96)]  # cut every 3s
        times, _, _ = _plan_samples(thumbs, 2.0, 8)
        self.assertLessEqual(len(times), 8)


class PayloadCapsTest(unittest.TestCase):
    def test_drops_closest_frames_first_under_image_cap(self) -> None:
        clip = Clip(task_id="t", frames_b64=["x" * 10] * 5,
                    frame_times=[0.0, 1.0, 1.2, 5.0, 9.0])
        old_cap = config.max_images
        config.max_images = 4
        try:
            _enforce_payload_caps(clip)
        finally:
            config.max_images = old_cap
        self.assertEqual(len(clip.frames_b64), 4)
        # The densest neighborhood (1.0s / 1.2s pair) loses one of its two frames.
        self.assertFalse({1.0, 1.2} <= set(clip.frame_times))
        self.assertIn(9.0, clip.frame_times)  # sparse anchors survive

    def test_payload_budget_enforced(self) -> None:
        big = "x" * (2 * 1024 * 1024)  # 2MB each, 8 frames = 16MB > 10MB cap
        clip = Clip(task_id="t", frames_b64=[big] * 8,
                    frame_times=[float(i) for i in range(8)])
        _enforce_payload_caps(clip)
        self.assertLessEqual(sum(len(f) for f in clip.frames_b64),
                             int(config.max_payload_mb * 1024 * 1024 * 0.95))
        self.assertGreaterEqual(len(clip.frames_b64), 3)


class GraphGapsTest(unittest.TestCase):
    def test_gap_signals(self) -> None:
        from .pipeline import _graph_gaps
        clip = Clip(task_id="t", duration_s=30.0)
        self.assertTrue(_graph_gaps({}, clip)[0].startswith("no evidence graph"))
        gaps = _graph_gaps({"timeline": [], "entities": ["a cat"],
                            "uncertainties": ["a", "b", "c"]}, clip)
        self.assertIn("timeline is empty", gaps)
        self.assertTrue(any(g.startswith("3 uncertainties") for g in gaps))
        # Healthy graph covering the full clip → no gaps.
        healthy = {"entities": ["a cat"], "scene_transitions": [],
                   "timeline": [{"t": "0-15s", "event": "cat walks"},
                                {"t": "15-30s", "event": "cat sits"}],
                   "important_events": ["cat walks"], "uncertainties": []}
        self.assertEqual(_graph_gaps(healthy, clip), [])
        # Timeline stopping at the clip's first half is a gap.
        half = dict(healthy, timeline=[{"t": "0-10s", "event": "cat walks"}])
        self.assertTrue(any("second half" in g for g in _graph_gaps(half, clip)))


if __name__ == "__main__":
    unittest.main()
