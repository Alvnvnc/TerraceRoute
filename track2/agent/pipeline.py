"""Per-clip pipeline: evidence graph → stylize → verify → selective visual repair.

Stage 1 no longer produces free prose: Gemma builds a TEMPORAL EVIDENCE GRAPH —
timestamped JSON (entities, timeline, scene_transitions, important_events,
uncertainties). Every style caption is generated from that one graph, so all four
share the same visual facts.

Confidence never comes from the generator's self-assessment. Three independent
signals decide whether a caption is accepted or repaired:
  (a) deterministic style lint (free),
  (b) the cross-family Qwen checker scoring against the original frames,
  (c) deterministic incompleteness of the evidence graph itself (empty timeline,
      missing entities, pile of uncertainties).
The caption is checked against evidence, never against itself — Track 1's
calibration lesson. Every stage degrades gracefully; the caller always receives a
caption for every requested style (a mediocre caption is recoverable, a missing
style is a guaranteed 0).
"""
from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field

from . import styles as S
from . import vlm
from .config import config
from .ingest import Clip, ingest

_JSON_BLOCK = re.compile(r"\{.*\}", re.S)
_NUM = re.compile(r"\d+(?:\.\d+)?")


def _parse_json(text: str) -> dict:
    """Best-effort JSON object extraction (models sometimes wrap it in prose/fences)."""
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_BLOCK.search(text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def _caption_schema(styles: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {style: {"type": "string"} for style in styles},
        "required": styles,
        "additionalProperties": False,
    }


def _verdict_schema(styles: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            style: {
                "type": "object",
                "properties": {
                    "accuracy": {"type": "number"},
                    "style_match": {"type": "number"},
                    "hint": {"type": "string"},
                },
                "required": ["accuracy", "style_match", "hint"],
                "additionalProperties": False,
            }
            for style in styles
        },
        "required": styles,
        "additionalProperties": False,
    }


_EVIDENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {"type": "array", "items": {"type": "string"}},
        "timeline": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"t": {"type": "string"}, "event": {"type": "string"}},
                "required": ["t", "event"],
                "additionalProperties": False,
            },
        },
        "scene_transitions": {"type": "array", "items": {"type": "string"}},
        "important_events": {"type": "array", "items": {"type": "string"}},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["entities", "timeline", "scene_transitions",
                 "important_events", "uncertainties"],
    "additionalProperties": False,
}


@dataclass
class ClipResult:
    task_id: str
    captions: dict[str, str] = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)   # the temporal evidence graph
    description: str = ""                           # serialized evidence (repair context)
    gaps: list[str] = field(default_factory=list)   # deterministic graph gaps
    routes: dict[str, str] = field(default_factory=dict)  # style -> pipeline route tag


# Neutral last-resort caption per style — used only when every model call failed.
# Grounded in nothing, but a plausible caption scores above a blank one.
_LAST_RESORT = {
    S.FORMAL: "A short video clip depicting a scene recorded in a real-world setting.",
    S.SARCASTIC: "Ah yes, a video clip. Truly no other footage like it anywhere.",
    S.HUMOROUS_TECH: "This clip loaded faster than my last software update installed.",
    S.HUMOROUS_NON_TECH: "Somewhere out there, someone filmed this and said: perfect, no notes.",
}


def _spread_idx(n: int, k: int) -> list[int]:
    """Up to k indices evenly spread over range(n) (keeps first and last)."""
    if n <= k:
        return list(range(n))
    step = (n - 1) / (k - 1)
    return [round(i * step) for i in range(k)]


def _labels(clip: Clip, idx: list[int] | None = None) -> list[str]:
    """Per-image timestamp labels, interleaved before each frame in the request."""
    idx = idx if idx is not None else range(len(clip.frame_times))
    return [f"[frame at t={clip.frame_times[i]:.1f}s]" for i in idx]


# --- Stage 1: temporal evidence graph -------------------------------------------------

def _evidence(clip: Clip) -> tuple[dict, str]:
    """ONE VLM call: frames → timestamped evidence graph. Returns (graph, serialized).
    On unparseable output the raw text still serves as (weaker) caption evidence."""
    prompt = (
        f"These are {len(clip.frames_b64)} frames sampled in order from ONE video clip "
        f"({clip.duration_s:.0f}s long). Each frame is preceded by its timestamp label.\n"
        "Build a temporal evidence graph of the clip. Reply with ONLY this JSON object:\n"
        '{"entities": ["each visible subject/object + short visual description"], '
        '"timeline": [{"t": "<second range, e.g. 0-4s>", "event": "what happens"}], '
        '"scene_transitions": ["<t>s: <from> to <to>"], '
        '"important_events": ["the 1-3 moments a caption must mention"], '
        '"uncertainties": ["what cannot be determined from these frames"]}\n'
        "Use the frame timestamps for every t. Cover the whole clip from start to end. "
        "Facts only — no interpretation, no story."
    )
    out = vlm.chat(prompt, images_b64=clip.frames_b64, image_labels=_labels(clip),
                   temperature=0.2, max_tokens=config.evidence_max_tokens,
                   json_mode=True, json_schema=_EVIDENCE_SCHEMA)
    graph = _parse_json(out or "")
    if graph:
        return graph, json.dumps(graph, ensure_ascii=False)
    return {}, (out or "").strip()


# Gaps that mean the graph cannot support a grounded caption at all — these flag
# every style for visual repair, not just the ones the checker rejects.
_SEVERE = ("no evidence graph", "timeline is empty", "no entities")


def _graph_gaps(graph: dict, clip: Clip) -> list[str]:
    """Deterministic incompleteness signals — the third confidence input.
    Never asks any model how confident it feels."""
    if not graph:
        return ["no evidence graph — perception output was not valid JSON"]
    gaps: list[str] = []
    if not graph.get("timeline"):
        gaps.append("timeline is empty")
    if not graph.get("entities"):
        gaps.append("no entities identified")
    unc = [str(u) for u in graph.get("uncertainties") or []]
    if len(unc) >= 3:
        gaps.append(f"{len(unc)} uncertainties: " + "; ".join(unc[:3]))
    ts = [float(m.group()) for e in graph.get("timeline") or []
          if isinstance(e, dict) and (m := _NUM.search(str(e.get("t", ""))))]
    if clip.duration_s > 8 and ts and max(ts) < 0.5 * clip.duration_s:
        gaps.append("timeline does not cover the second half of the clip")
    return gaps


# --- Stage 2: stylize ------------------------------------------------------------------

def _stylize(evidence: str, req_styles: list[str]) -> dict[str, str]:
    """ONE structured call produces every requested style from the same graph."""
    cards = "\n\n".join(f"### {s}\n{S.CARDS[s]}" for s in req_styles if s in S.CARDS)
    schema = ", ".join(f'"{s}": "..."' for s in req_styles)
    prompt = (
        "You write video captions. Below is a temporal evidence graph extracted from a "
        "video clip (entities, timestamped timeline, scene transitions, important "
        "events), then the requested styles. Write ONE caption of 1-2 sentences per "
        "style. Every caption must be grounded in this evidence — reflect what actually "
        "happens across the clip, and cover the important events. Jokes must be about "
        "what is really in the clip. Never mention timestamps, frames, or the evidence "
        "graph itself.\n\n"
        f"EVIDENCE GRAPH:\n{evidence}\n\nSTYLES:\n{cards}\n\n"
        f"Reply with ONLY a JSON object: {{{schema}}}"
    )
    out = vlm.chat(prompt, temperature=0.7, max_tokens=config.stylize_max_tokens,
                   json_mode=True, json_schema=_caption_schema(req_styles))
    parsed = _parse_json(out or "")
    return {s: str(parsed.get(s, "")).strip() for s in req_styles}


# --- Stage 3: verification -------------------------------------------------------------

def _check(clip: Clip, captions: dict[str, str]) -> dict[str, str]:
    """Cross-family checker (qwen2.5vl vs gemma3): score each caption on the judge's own
    rubric, grounded in the actual frames. Returns style -> feedback for failures only."""
    cap_json = json.dumps(captions, ensure_ascii=False)
    prompt = (
        "You are a strict caption judge. You see frames from a video clip and candidate "
        "captions, one per style. For each style give: accuracy (0-1, does the caption "
        "faithfully reflect the video content) and style_match (0-1, does it match the "
        "requested tone: formal=professional/objective; sarcastic=dry ironic mocking; "
        "humorous_tech=funny WITH a technology reference; humorous_non_tech=funny with "
        "NO technical vocabulary), plus a one-line fix hint if either score is below 0.7.\n\n"
        f"CAPTIONS:\n{cap_json}\n\n"
        "Reply with ONLY JSON mapping each style name to an object with keys "
        "accuracy (number), style_match (number), hint (write an actual one-line fix "
        "in your own words, or an empty string when both scores are 0.7+)."
    )
    idx = _spread_idx(len(clip.frames_b64), config.checker_frames)
    out = vlm.chat(prompt, images_b64=[clip.frames_b64[i] for i in idx],
                   image_labels=_labels(clip, idx), model=config.checker_model,
                   temperature=0.0, max_tokens=450, json_mode=True,
                   json_schema=_verdict_schema(list(captions)), allow_fireworks=False)
    verdicts = _parse_json(out or "")
    feedback: dict[str, str] = {}
    for style, v in verdicts.items():
        if style not in captions or not isinstance(v, dict):
            continue
        try:
            acc = float(v.get("accuracy", 1.0))
            sm = float(v.get("style_match", 1.0))
        except (TypeError, ValueError):
            continue
        if acc < 0.7 or sm < 0.7:
            hint = str(v.get("hint") or "").strip()
            if hint in ("", "...", "…"):  # models sometimes echo the schema placeholder
                hint = (f"accuracy={acc:.1f} style_match={sm:.1f} — improve grounding "
                        "in the visible content and match the requested tone")
            feedback[style] = hint[:200]
    return feedback


def _regenerate(clip: Clip, evidence: str, captions: dict[str, str],
                problems: dict[str, str], gaps: list[str]) -> dict[str, str]:
    """Repair every flagged style in one frame-grounded, schema-constrained call."""
    styles = list(problems)
    cards = "\n\n".join(f"### {style}\n{S.CARDS.get(style, '')}" for style in styles)
    candidates = json.dumps({style: captions.get(style, "") for style in styles}, ensure_ascii=False)
    feedback = json.dumps(problems, ensure_ascii=False)
    gap_note = ("\n\nKNOWN EVIDENCE GAPS (the graph below may be missing this — trust "
                "the frames):\n- " + "\n- ".join(gaps)) if gaps else ""
    prompt = (
        "You repair video captions that a strict visual reviewer flagged. Inspect the original "
        "frames yourself; the evidence graph is supporting context, not a substitute for "
        "what you see. Rewrite only the requested styles. Each caption must be 1-2 sentences, "
        "grounded in the clip, and comply with its style card.\n\n"
        f"EVIDENCE GRAPH:\n{evidence}{gap_note}\n\n"
        f"CURRENT CAPTIONS:\n{candidates}\n\n"
        f"REVIEWER FEEDBACK:\n{feedback}\n\n"
        f"STYLE CARDS:\n{cards}\n\n"
        "Reply with ONLY a JSON object whose keys are the requested styles."
    )
    out = vlm.chat(prompt, images_b64=clip.frames_b64, image_labels=_labels(clip),
                   temperature=0.4, max_tokens=config.stylize_max_tokens, json_mode=True,
                   json_schema=_caption_schema(styles), prefer_fireworks=True)
    parsed = _parse_json(out or "")
    return {style: str(parsed.get(style, "")).strip() for style in styles}


# --- Orchestration ---------------------------------------------------------------------

def process_clip(task_id: str, video_url: str, req_styles: list[str],
                 budget_s: float) -> ClipResult:
    """Full pipeline for one clip within a soft time budget. Never raises."""
    start = time.perf_counter()
    res = ClipResult(task_id=task_id)

    def left() -> float:
        return budget_s - (time.perf_counter() - start)

    # Stage 0: adaptive ingest (fewest frames when the budget is tight).
    clip = ingest(task_id, video_url,
                  max_frames=(0 if budget_s > 60 else config.min_frames))
    if clip.ok:
        print(f"[pipeline] {task_id} sampler={clip.sampler} frames={len(clip.frames_b64)} "
              f"scenes={clip.n_scenes} motion={clip.motion:.1f}", file=sys.stderr)

    # Stage 1: evidence graph. Without frames there is no evidence — last-resort path.
    if clip.ok:
        res.evidence, res.description = _evidence(clip)
    res.gaps = _graph_gaps(res.evidence, clip) if clip.ok else []

    # Stage 2: stylize (one structured call, all styles from the same graph).
    if res.description:
        res.captions = _stylize(res.description, req_styles)
        res.routes = {s: "stylized" for s in req_styles}

    # Stage 3a: free deterministic lint (always runs — it costs nothing).
    problems: dict[str, str] = {}
    for s in req_styles:
        p = S.lint(s, res.captions.get(s, ""))
        if p:
            problems[s] = p

    # Stage 3b: cross-family checker — only when the budget allows (degradation ladder).
    if res.description and left() > config.verify_min_budget_s:
        try:
            for s, hint in _check(clip, res.captions).items():
                problems.setdefault(s, hint)
        except Exception as exc:  # noqa: BLE001
            print(f"[pipeline] checker failed for {task_id}: {exc}", file=sys.stderr)

    # Stage 3c: evidence-completeness signal. A severely incomplete graph means every
    # caption was built on weak evidence — flag them all for a frame-grounded rewrite.
    if res.description and any(g.startswith(_SEVERE) for g in res.gaps):
        note = "evidence was incomplete (" + "; ".join(res.gaps[:2]) + \
               ") — re-inspect the frames and describe what you actually see"
        for s in req_styles:
            problems.setdefault(s, note)

    # Stage 3d: one visual repair round for every flagged style. Re-showing the frames
    # closes the information bottleneck from the perception summary before accepting a fix.
    if res.description and problems and left() >= 10.0:
        for s, new in _regenerate(clip, res.description, res.captions,
                                  problems, res.gaps).items():
            # Accept the rewrite only if it clears the free lint.
            if new and S.lint(s, new) is None:
                res.captions[s] = new
                res.routes[s] = "regenerated"

    # Final guarantee: every requested style has a non-empty caption.
    for s in req_styles:
        if not res.captions.get(s, "").strip():
            res.captions[s] = _LAST_RESORT.get(s, _LAST_RESORT[S.FORMAL])
            res.routes[s] = "last_resort"
    return res
