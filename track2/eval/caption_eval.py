"""Offline judge eval — score a results.json against the published Track 2 rubric.

Drives offline tuning without burning the 10-submissions/hour rate limit:
    cd track2
    PYTHONPATH=. python3 eval/caption_eval.py --tasks input/tasks.json \
        --results output/results.json [--baseline old/results.json] [--out report.json]

The judge is an LLM proxy (default: the local qwen2.5vl — a different family from the
gemma generator, but the SAME model as the pipeline's checker). Absolute numbers are
therefore optimistic for checker-approved captions; use them as a RELATIVE signal
between two runs (A/B via --baseline), not as a leaderboard prediction. Point
JUDGE_MODEL / JUDGE_BASE_URL at an unrelated third-family VLM for calibration runs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import vlm  # noqa: E402
from agent.config import config  # noqa: E402
from agent.ingest import Clip, ingest  # noqa: E402
from agent.pipeline import _checker_frame_idx, _labels, _parse_json  # noqa: E402

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", config.checker_model)

_RUBRIC = (
    "You are the official judge for a video-captioning contest. You see frames "
    "sampled from a clip (each preceded by its timestamp) and candidate captions, "
    "one per requested style. Score EACH caption on the contest rubric:\n"
    "- accuracy (0-1): does the caption faithfully describe what actually happens "
    "in the clip? Hallucinated subjects or actions score low.\n"
    "- style_match (0-1): does it deliver the requested tone? "
    "formal = professional, objective, no slang/jokes; "
    "sarcastic = dry, ironic, mocking wit; "
    "humorous_tech = genuinely funny AND hinges on a technology/programming "
    "reference; humorous_non_tech = genuinely funny with NO technical vocabulary.\n"
    "Score strictly and independently per style."
)


def _judge_clip(clip: Clip, captions: dict[str, str]) -> dict[str, dict[str, float]]:
    """One judge call for all styles of one clip. Returns style -> {accuracy, style_match}."""
    prompt = (
        f"{_RUBRIC}\n\nCAPTIONS:\n{json.dumps(captions, ensure_ascii=False)}\n\n"
        "Reply with ONLY JSON mapping each style name to "
        '{"accuracy": <0-1>, "style_match": <0-1>}.'
    )
    idx = _checker_frame_idx(clip, config.checker_frames)
    out = vlm.chat(prompt, images_b64=[clip.frames_b64[i] for i in idx],
                   image_labels=_labels(clip, idx), model=JUDGE_MODEL,
                   temperature=0.0, max_tokens=400, json_mode=True,
                   allow_fireworks=False)
    parsed = _parse_json(out or "")
    scores: dict[str, dict[str, float]] = {}
    for style, v in parsed.items():
        if style not in captions or not isinstance(v, dict):
            continue
        try:
            acc, sm = float(v.get("accuracy")), float(v.get("style_match"))
        except (TypeError, ValueError):
            continue
        if 0.0 <= acc <= 1.0 and 0.0 <= sm <= 1.0:
            scores[style] = {"accuracy": acc, "style_match": sm}
    return scores


def _load(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _score_run(tasks: list[dict], results_path: str) -> dict:
    results = {str(r["task_id"]): r.get("captions", {}) for r in _load(results_path)}
    per_clip: dict[str, dict] = {}
    for t in tasks:
        tid = str(t["task_id"])
        captions = results.get(tid) or {}
        if not captions:
            per_clip[tid] = {"error": "missing from results"}
            continue
        clip = ingest(f"eval-{tid}", str(t.get("video_url", "")))
        if not clip.ok:
            per_clip[tid] = {"error": "ingest failed"}
            continue
        per_clip[tid] = _judge_clip(clip, captions)
        print(f"[eval] {tid}: " + " ".join(
            f"{s}={v['accuracy']:.2f}/{v['style_match']:.2f}"
            for s, v in per_clip[tid].items()), file=sys.stderr)
    scored = [v for clip_scores in per_clip.values()
              for v in clip_scores.values() if isinstance(v, dict) and "accuracy" in v]
    n = len(scored) or 1
    return {
        "results": results_path,
        "judge": JUDGE_MODEL,
        "clips": per_clip,
        "mean_accuracy": sum(v["accuracy"] for v in scored) / n,
        "mean_style_match": sum(v["style_match"] for v in scored) / n,
        "scored_captions": len(scored),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--baseline", help="second results.json to A/B against")
    ap.add_argument("--out", help="write the full report JSON here")
    args = ap.parse_args()

    tasks = [t for t in _load(args.tasks) if isinstance(t, dict) and "task_id" in t]
    report = {"run": _score_run(tasks, args.results)}
    if args.baseline:
        report["baseline"] = _score_run(tasks, args.baseline)

    for name in ("run", "baseline"):
        if name in report:
            r = report[name]
            print(f"{name:9} acc={r['mean_accuracy']:.3f} "
                  f"style={r['mean_style_match']:.3f} "
                  f"({r['scored_captions']} captions, judge={r['judge']})")
    if "baseline" in report:
        d_acc = report["run"]["mean_accuracy"] - report["baseline"]["mean_accuracy"]
        d_sm = (report["run"]["mean_style_match"]
                - report["baseline"]["mean_style_match"])
        print(f"delta     acc={d_acc:+.3f} style={d_sm:+.3f}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
        print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
