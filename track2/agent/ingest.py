"""Clip ingest: download + ADAPTIVE temporal frame sampling.

Measured on the official clips (plan.md §7): files are 8–19 MB despite UHD resolution,
so a full download (seconds) is simpler and more robust than HTTP-range streaming —
which segfaulted the static ffmpeg build outright. The official clips carry NO audio
stream, so audio is not part of the pipeline.

Sampling is adaptive instead of uniform: one cheap ffmpeg pass produces a dense pool
of tiny grayscale thumbnails; frame-to-frame differences on that pool give (a) a
motion score, (b) scene-cut positions, and (c) a near-duplicate measure — all from
the same deterministic signal, no extra model calls. Simple clips get few frames
(min_frames), clips with many transitions get more (up to max_frames), and the result
is always capped at max_images / max_payload_mb so the frame set can ride on a single
Fireworks request.
"""
from __future__ import annotations

import base64
import hashlib
import os
import re
import subprocess
import time
from bisect import bisect_right
from dataclasses import dataclass, field

from .config import config
from .http import download

_DURATION = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def _safe_stem(task_id: str) -> str:
    """Filesystem-safe, collision-free work-file stem for an arbitrary task_id.
    The hash suffix keeps two ids that sanitize identically (or contain ../) apart."""
    clean = _UNSAFE.sub("_", task_id)[:64].strip("._") or "task"
    return f"{clean}-{hashlib.sha1(task_id.encode()).hexdigest()[:8]}"

# Thumbnail geometry for the analysis pass (16:9, gray). 1296 bytes per frame —
# pure-Python differencing over ~100 candidates is instantaneous.
_TW, _TH = 48, 27


@dataclass
class Clip:
    task_id: str
    path: str = ""
    duration_s: float = 0.0
    frames_b64: list[str] = field(default_factory=list)   # jpeg, base64
    frame_times: list[float] = field(default_factory=list)  # seconds, same order
    frame_scenes: list[int] = field(default_factory=list)   # scene id, same order
    sampler: str = "none"      # adaptive | uniform (fallback)
    n_scenes: int = 1          # detected scene segments
    motion: float = 0.0        # mean thumbnail MAD (0-255 scale)

    @property
    def ok(self) -> bool:
        return bool(self.frames_b64)


def _probe_duration(path: str) -> float:
    """Parse the clip duration from ffmpeg -i stderr (no ffprobe dependency)."""
    proc = subprocess.run([config.ffmpeg_bin, "-i", path],
                          capture_output=True, text=True, timeout=30)
    m = _DURATION.search(proc.stderr)
    if not m:
        return 0.0
    h, mnt, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
    return h * 3600 + mnt * 60 + s


def _extract_frame(path: str, t: float, out_path: str) -> bool:
    proc = subprocess.run(
        [config.ffmpeg_bin, "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", path,
         "-frames:v", "1", "-vf", f"scale={config.frame_width}:-1", "-q:v", "4", out_path],
        capture_output=True, timeout=60)
    return proc.returncode == 0 and os.path.exists(out_path)


# --- Analysis pass -------------------------------------------------------------------

def _thumbnails(path: str, dur: float, timeout_s: float = 180.0) -> tuple[list[bytes], float]:
    """One decode pass → dense pool of tiny gray thumbnails. Returns (thumbs, fps)."""
    cap = max(8, config.sampler_candidates)
    fps = min(2.0, cap / max(dur, 1.0)) if dur > 0 else 1.0
    proc = subprocess.run(
        [config.ffmpeg_bin, "-loglevel", "error", "-i", path,
         "-vf", f"fps={fps:.4f},scale={_TW}:{_TH}", "-pix_fmt", "gray",
         "-f", "rawvideo", "-"],
        capture_output=True, timeout=max(10.0, timeout_s))
    raw = proc.stdout
    fsz = _TW * _TH
    thumbs = [raw[i * fsz:(i + 1) * fsz] for i in range(len(raw) // fsz)]
    return thumbs, fps


def _mad(a: bytes, b: bytes) -> float:
    """Mean absolute difference between two equal-size gray thumbnails (0-255)."""
    return sum(x - y if x >= y else y - x for x, y in zip(a, b)) / len(a)


def _scene_cuts(diffs: list[float]) -> list[int]:
    """Indices where a new scene starts. Threshold is relative to the clip's own
    within-scene motion (p75), so handheld footage does not read as a cut storm."""
    if not diffs:
        return []
    srt = sorted(diffs)
    p75 = srt[int(0.75 * (len(srt) - 1))]
    thr = max(18.0, 3.0 * p75)
    cuts, last = [], -10
    for i, d in enumerate(diffs):
        if d > thr and i - last > 1:  # merge adjacent detections (fades/wipes)
            cuts.append(i + 1)
            last = i
    return cuts[:8]  # montage guard: beyond ~9 segments extra cuts add no frame budget


def _pick_indices(diffs: list[float], cuts: list[int], n_cand: int,
                  n_target: int) -> list[int]:
    """Distribute n_target picks over scene segments, weighted by duration + motion;
    within a segment picks are evenly spaced (midpoint rule — avoids cut-blur)."""
    bounds = [0, *cuts, n_cand]
    segs = [(bounds[j], bounds[j + 1]) for j in range(len(bounds) - 1)
            if bounds[j + 1] > bounds[j]]
    total_len = sum(b - a for a, b in segs)
    total_mot = sum(sum(diffs[max(a - 1, 0):b - 1]) for a, b in segs) or 1.0

    weights = [0.5 * (b - a) / total_len
               + 0.5 * sum(diffs[max(a - 1, 0):b - 1]) / total_mot
               for a, b in segs]

    if n_target < len(segs):  # rare: many cuts, tiny budget — keep heaviest segments
        order = sorted(range(len(segs)), key=lambda j: -weights[j])[:n_target]
        segs = [segs[j] for j in sorted(order)]
        alloc = [1] * len(segs)
    else:
        alloc = [1] * len(segs)
        rest = n_target - len(segs)
        # largest-remainder distribution of the remaining picks
        quota = [w * rest for w in weights]
        alloc = [a + int(q) for a, q in zip(alloc, quota)]
        rema = sorted(range(len(segs)), key=lambda j: -(quota[j] - int(quota[j])))
        for j in rema[:rest - sum(int(q) for q in quota)]:
            alloc[j] += 1

    picks: list[int] = []
    for (a, b), k in zip(segs, alloc):
        picks += [min(b - 1, a + int((b - a) * (j + 0.5) / k)) for j in range(k)]
    return sorted(set(picks))


def _plan_samples(thumbs: list[bytes], fps: float,
                  max_frames: int) -> tuple[list[float], list[int], int, float]:
    """Adaptive plan: (timestamps, scene id per timestamp, n_scenes, motion).
    Empty timestamps → caller falls back to uniform sampling."""
    n = len(thumbs)
    if n < 4:
        return [], [], 1, 0.0
    diffs = [_mad(thumbs[i - 1], thumbs[i]) for i in range(1, n)]
    cuts = _scene_cuts(diffs)
    motion = sum(diffs) / len(diffs)

    # Complexity → frame count: base for a single quiet scene, +2 per transition,
    # + up to 6 for sustained motion. Clamped to the configured + hard caps.
    n_target = round(config.min_frames + 2 * len(cuts) + min(6.0, motion / 3.0))
    n_target = max(3, min(n_target, max_frames, config.max_images, n))

    picks = _pick_indices(diffs, cuts, n, n_target)

    # Dedupe: drop a pick when it is visually ~identical to the previous kept one,
    # unless it anchors a distant part of the timeline (static clips keep coverage).
    kept: list[int] = []
    gap = max(8.0, (n / fps) / 4.0) * fps
    for idx in picks:
        if not kept:
            kept.append(idx)
            continue
        prev = kept[-1]
        if _mad(thumbs[prev], thumbs[idx]) >= config.dedupe_mad or idx - prev >= gap:
            kept.append(idx)
    if len(kept) < 3:  # degenerate (still tripod shot): keep sparse uniform anchors
        kept = sorted(set([0, n // 2, n - 1]))
    scene_ids = [bisect_right(cuts, i) for i in kept]
    return [i / fps for i in kept], scene_ids, len(cuts) + 1, motion


# --- Ingest --------------------------------------------------------------------------

def ingest(task_id: str, video_url: str, max_frames: int = 0,
           budget_s: float = 0.0) -> Clip:
    """Download the clip and sample frames adaptively. Never raises.

    A partially successful ingest (fewer frames, even a single one) is still usable:
    one frame of evidence beats a blank caption. `budget_s` (0 = unbounded) soft-bounds
    the whole ingest: download and analysis timeouts shrink to the remaining budget and
    frame extraction stops early rather than overrun the clip's share of the wall clock.
    """
    clip = Clip(task_id=task_id)
    cap = max_frames or config.max_frames
    deadline = (time.monotonic() + budget_s) if budget_s > 0 else None

    def remaining(default: float) -> float:
        return default if deadline is None else min(default, deadline - time.monotonic())

    os.makedirs(config.work_dir, exist_ok=True)
    stem = _safe_stem(task_id)
    path = os.path.join(config.work_dir, f"{stem}.mp4")
    try:
        if not download(video_url, path,
                        timeout=max(5.0, remaining(config.download_timeout_s))):
            return clip
        clip.path = path
        clip.duration_s = _probe_duration(path)
        dur = clip.duration_s

        times: list[float] = []
        scenes: list[int] = []
        try:
            if remaining(180.0) > 10.0:
                thumbs, fps = _thumbnails(path, dur, timeout_s=remaining(180.0))
                times, scenes, clip.n_scenes, clip.motion = _plan_samples(thumbs, fps, cap)
                clip.sampler = "adaptive"
        except Exception:  # noqa: BLE001 — analysis is an optimization, not a need
            times = []
        if not times:
            # Fallback: uniform sampling (or fixed probes when duration is unknown).
            n = min(config.frames, cap)
            times = ([dur * (i + 0.5) / n for i in range(n)] if dur > 0
                     else [0.5, 2.0, 5.0, 10.0])
            scenes = [0] * len(times)
            clip.sampler = "uniform"

        for i, t in enumerate(times):
            if deadline is not None and deadline - time.monotonic() < 3.0:
                break  # partial frame set still beats a blank caption
            fp = os.path.join(config.work_dir, f"{stem}_{i}.jpg")
            try:
                if _extract_frame(path, t, fp):
                    with open(fp, "rb") as f:
                        clip.frames_b64.append(base64.b64encode(f.read()).decode())
                    clip.frame_times.append(t)
                    clip.frame_scenes.append(scenes[i] if i < len(scenes) else 0)
                    os.remove(fp)
            except Exception:  # noqa: BLE001
                continue

        _enforce_payload_caps(clip)
    except Exception:  # noqa: BLE001
        pass
    finally:
        if clip.path and os.path.exists(clip.path):
            try:
                os.remove(clip.path)  # keep the workdir small on long task lists
            except OSError:
                pass
    return clip


def _enforce_payload_caps(clip: Clip) -> None:
    """Hard caps for any provider: ≤ max_images frames, base64 total < max_payload_mb.
    Drops the temporally closest (least informative) frames first, never below 3."""
    budget = int(config.max_payload_mb * 1024 * 1024 * 0.95)
    while len(clip.frames_b64) > 3 and (
            len(clip.frames_b64) > config.max_images
            or sum(len(f) for f in clip.frames_b64) > budget):
        gaps = [clip.frame_times[i + 1] - clip.frame_times[i - 1]
                for i in range(1, len(clip.frame_times) - 1)]
        drop = 1 + min(range(len(gaps)), key=gaps.__getitem__) if gaps else 1
        del clip.frames_b64[drop], clip.frame_times[drop]
        if drop < len(clip.frame_scenes):
            del clip.frame_scenes[drop]
