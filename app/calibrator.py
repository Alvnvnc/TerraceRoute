"""Kalibrator uncertainty -> P(local salah) via isotonic regression.

Ini INTI pembeda TerraceRoute. Sebelum ada data kalibrasi, fallback ke identitas
(p_wrong = u), sehingga sistem tetap jalan namun belum "calibrated".
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression


class Calibrator:
    def __init__(self):
        self._iso: Optional[IsotonicRegression] = None

    @property
    def is_fitted(self) -> bool:
        return self._iso is not None

    def fit(self, uncertainties: list[float], local_wrong: list[int]) -> "Calibrator":
        """local_wrong[i] = 1 jika jawaban lokal salah untuk sampel i, else 0."""
        x = np.asarray(uncertainties, dtype=float)
        y = np.asarray(local_wrong, dtype=float)
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(x, y)
        self._iso = iso
        return self

    def predict(self, u: float) -> float:
        if self._iso is None:
            return max(0.0, min(1.0, u))          # fallback identitas
        return float(self._iso.predict([u])[0])

    # ---- persistence (JSON, tanpa pickle biar aman & portable) ----
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if self._iso is None:
            payload = {"fitted": False}
        else:
            payload = {
                "fitted": True,
                "x": self._iso.X_thresholds_.tolist(),
                "y": self._iso.y_thresholds_.tolist(),
            }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Calibrator":
        c = cls()
        if not os.path.exists(path):
            return c
        with open(path) as f:
            payload = json.load(f)
        if not payload.get("fitted"):
            return c
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        # Rekonstruksi dari titik-titik threshold yang tersimpan.
        iso.fit(payload["x"], payload["y"])
        c._iso = iso
        return c
