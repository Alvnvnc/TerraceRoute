"""Unified TerraceRoute dispatcher for the Track 1 and Track 2 harnesses."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_INPUT_PATH = "/input/tasks.json"
DEFAULT_OUTPUT_PATH = "/output/results.json"
TRACK_OVERRIDE_ENV = "TERRACEROUTE_TRACK"

_VIDEO_FIELDS = (
    "video_url",
    "video_uri",
    "video",
    "video_path",
    "video_file",
    "media_url",
)
_VIDEO_MARKERS = (".mp4", ".webm", ".mov", ".mkv", "video")
_VIDEO_STYLES = {
    "formal",
    "sarcastic",
    "humorous_tech",
    "humorous_non_tech",
}


def load_input(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_items(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = next(
            (data[key] for key in ("tasks", "items", "data") if isinstance(data.get(key), list)),
            None,
        )
        if items is None:
            raise ValueError("unsupported input object; expected tasks, items, or data")
    else:
        raise ValueError("unsupported input structure; expected an array or object")

    if not all(isinstance(item, dict) for item in items):
        raise ValueError("every task must be an object")
    return items


def looks_like_video_task(task: dict[str, Any]) -> bool:
    if any(task.get(field) not in (None, "") for field in _VIDEO_FIELDS):
        return True

    generic_url = str(task.get("url", "")).lower()
    if generic_url and any(marker in generic_url for marker in _VIDEO_MARKERS):
        return True

    styles = task.get("styles")
    if isinstance(styles, list) and _VIDEO_STYLES.intersection(map(str, styles)):
        return True
    return False


def detect_track(tasks: list[dict[str, Any]], override: str = "") -> int:
    selected = override.strip().lower().replace("track", "")
    if selected:
        if selected not in {"1", "2"}:
            raise ValueError(f"{TRACK_OVERRIDE_ENV} must be 1 or 2")
        return int(selected)

    flags = [looks_like_video_task(task) for task in tasks]
    if any(flags) and not all(flags):
        raise ValueError(
            f"mixed Track 1/Track 2 input; set {TRACK_OVERRIDE_ENV}=1 or 2 explicitly"
        )
    return 2 if flags and all(flags) else 1


def normalize_tasks(tasks: list[dict[str, Any]], track: int) -> list[dict[str, Any]]:
    if track != 2:
        return tasks

    normalized = []
    for task in tasks:
        item = dict(task)
        if not item.get("video_url"):
            for field in (*_VIDEO_FIELDS[1:], "url"):
                if item.get(field):
                    item["video_url"] = item[field]
                    break
        normalized.append(item)
    return normalized


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    os.replace(temporary, path)


def main() -> int:
    input_path = Path(os.environ.get("INPUT_PATH", DEFAULT_INPUT_PATH))
    output_path = Path(os.environ.get("OUTPUT_PATH", DEFAULT_OUTPUT_PATH))
    _write_json(output_path, [])

    try:
        tasks = get_items(load_input(input_path))
        track = detect_track(tasks, os.environ.get(TRACK_OVERRIDE_ENV, ""))
    except Exception as exc:  # noqa: BLE001 - malformed harness input must still exit cleanly
        print(f"[dispatcher] input error: {exc}", file=sys.stderr)
        return 0

    normalized = normalize_tasks(tasks, track)
    module = f"track{track}.agent.run"
    child_env = os.environ.copy()
    child_env["OUTPUT_PATH"] = str(output_path)

    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".json", delete=False
        ) as handle:
            json.dump(normalized, handle, ensure_ascii=False)
            temporary_path = handle.name
        child_env["INPUT_PATH"] = temporary_path

        print(f"[dispatcher] selected Track {track}", file=sys.stderr)
        completed = subprocess.run(
            [sys.executable, "-B", "-m", module],
            env=child_env,
            check=False,
        )
        if completed.returncode:
            print(
                f"[dispatcher] Track {track} exited with code {completed.returncode}; "
                "keeping its latest valid output",
                file=sys.stderr,
            )
    except Exception as exc:  # noqa: BLE001 - preserve the pre-seeded valid output
        print(f"[dispatcher] Track {track} failed: {exc}", file=sys.stderr)
    finally:
        if temporary_path:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
