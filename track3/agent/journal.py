"""Append-only audit journal + in-session undo stack.

Every action the agent takes is recorded to a JSONL file (the audit trail a
self-hoster can inspect) and, when it is reversible, pushed onto an undo stack.
If a multi-step operation fails partway (e.g. tunnel created, DNS created, then
the external probe never goes green), :meth:`Journal.rollback` unwinds the
completed steps in reverse order — the "automatic rollback" from the plan.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class _Undo:
    label: str
    fn: Callable[[], None]


@dataclass
class Journal:
    path: Optional[str] = None
    _undo_stack: list[_Undo] = field(default_factory=list)
    _entries: list[dict] = field(default_factory=list)

    def record(
        self,
        action: str,
        status: str = "ok",
        detail: Optional[dict] = None,
        undo: Optional[Callable[[], None]] = None,
        undo_label: str = "",
    ) -> dict:
        """Record one action. If ``undo`` is given, push it onto the undo stack.

        ``undo_label`` documents the reverse action in the audit trail even
        though the callable itself is not serializable.
        """
        entry = {
            "ts": time.time(),
            "action": action,
            "status": status,
            "detail": detail or {},
            "reversible": undo is not None,
            "undo": undo_label,
        }
        self._entries.append(entry)
        self._append(entry)
        if undo is not None:
            self._undo_stack.append(_Undo(label=undo_label or action, fn=undo))
        return entry

    def rollback(self) -> list[dict]:
        """Execute the undo stack LIFO. Records each undo; keeps going even if
        one undo raises, so a single stubborn step cannot strand the rest."""
        results: list[dict] = []
        while self._undo_stack:
            u = self._undo_stack.pop()
            try:
                u.fn()
                results.append(self.record(f"undo:{u.label}", status="ok"))
            except Exception as exc:  # noqa: BLE001 - audit the failure, continue
                results.append(
                    self.record(f"undo:{u.label}", status="error",
                                detail={"error": str(exc)})
                )
        return results

    def commit(self) -> None:
        """Mark the current undo stack as final (a successful operation): the
        steps are no longer candidates for rollback."""
        self._undo_stack.clear()

    @property
    def pending_undo(self) -> list[str]:
        return [u.label for u in self._undo_stack]

    def entries(self) -> list[dict]:
        return list(self._entries)

    def _append(self, entry: dict[str, Any]) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
