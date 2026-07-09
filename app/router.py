"""Mesin kebijakan TerraceRoute: calibrated risk -> route -> eksekusi.

Alur (lihat formula.md):
  cache? -> draft lokal + uncertainty -> p_wrong terkalibrasi
  p<τ1 & !high_risk         -> LOCAL_ONLY
  τ1<=p<τ2 (atau high_risk) -> LOCAL_DRAFT + REMOTE_PATCH (cap output remote)
  p>=τ2                     -> REMOTE_COMPRESSED (full)
"""
from __future__ import annotations

import re
import time
from typing import Optional

from .cache import SemanticCache
from .calibrator import Calibrator
from .compressor import compress
from .config import settings
from .fireworks_client import FireworksClient
from .ollama_client import OllamaClient
from .schemas import AnswerResponse, Mode, Route, RouteDecision
from .tokens import estimate_tokens
from .uncertainty import self_consistency_uncertainty, uncertainty_from_logprob

_HIGH_RISK = re.compile(
    r"\b(legal|lawsuit|contract|medical|diagnos|patient|dosage|financial|invest|tax|"
    r"security|exploit|vulnerab|password|secret|compliance|gdpr|hipaa)\b",
    re.IGNORECASE,
)

_PATCH_SYSTEM = (
    "You verify and minimally correct a draft answer. If the draft fully and correctly "
    "answers the task, reply with exactly: OK. Otherwise return ONLY the corrected answer, "
    "as short as possible, with no explanation."
)


class Router:
    def __init__(self, calibrator: Optional[Calibrator] = None):
        self.local = OllamaClient()
        self.remote = FireworksClient()
        self.calibrator = calibrator or Calibrator()
        self.cache = SemanticCache(self.local)
        # Verifier beda-keluarga utk cross-model check (sinyal routing utama).
        self.verifiers = [OllamaClient(model=m.strip())
                          for m in settings.verifier_models.split(",") if m.strip()]

    def high_risk(self, text: str) -> bool:
        return bool(_HIGH_RISK.search(text))

    async def _draft_and_uncertainty(self, text: str):
        """Draft lokal + sinyal uncertainty CROSS-MODEL.

        Temuan v2 (lihat memori): perplexity & self-consistency ANTI-korelasi dgn error
        (model confidently-consistent wrong). Yang bekerja = ketidaksepakatan antar-model
        keluarga berbeda. Verifier menjawab VERBOSE (bernalar) — memaksa terse malah
        merusak akurasi verifier & menghapus sinyal. Draft primary dipakai ulang sbg jawaban.
        """
        gen = await self.local.generate(text)                       # draft verbose (jawaban) + logprobs
        if settings.cross_model and self.verifiers:
            answers = [gen.text]                                    # jawaban primary (reuse draft)
            for v in self.verifiers:
                va = await v.generate(text, temperature=0.0, want_logprobs=False,
                                      num_predict=768)
                answers.append(va.text)
            u, method = self_consistency_uncertainty(answers)
            return gen.text, u, f"cross_model:{method}"
        # Fallback (tanpa verifier): perplexity — lemah, hanya bila cross-model dimatikan.
        u = uncertainty_from_logprob(gen.avg_logprob, gen.min_logprob,
                                     settings.logprob_min_weight) or 0.0
        return gen.text, u, "logprob"

    async def _self_check(self, text: str, draft: str):
        """Perbaiki draft secara lokal (gratis) + hitung ulang uncertainty."""
        prompt = (
            f"Task:\n{text}\n\nDraft answer:\n{draft}\n\n"
            "Find any factual or logical errors in the draft, then output ONLY the "
            "corrected final answer (no critique, no preamble)."
        )
        gen = await self.local.generate(prompt, temperature=0.1)
        u = uncertainty_from_logprob(gen.avg_logprob, gen.min_logprob,
                                     settings.logprob_min_weight)
        return gen.text, (u if u is not None else 1.0)

    def decide(self, p_wrong: float, high_risk: bool, mode: Mode) -> tuple[Route, str]:
        if mode == Mode.local:
            return Route.local_only, "forced_local"
        if mode == Mode.remote:
            return Route.remote_compressed, "forced_remote"

        # Override high-risk: severity asimetris.
        if high_risk and p_wrong >= settings.tau1 * 0.5:
            if p_wrong >= settings.tau2:
                return Route.remote_compressed, "high_risk_full"
            return Route.local_draft_remote_patch, "high_risk_patch"

        if p_wrong < settings.tau1:
            return Route.local_only, "confident_local"
        if p_wrong < settings.tau2:
            return Route.local_draft_remote_patch, "medium_patch"
        return Route.remote_compressed, "low_confidence_full"

    async def answer(self, text: str, mode: Mode = Mode.auto) -> AnswerResponse:
        t0 = time.perf_counter()

        # 0. Forced remote: langsung compress -> remote full, tanpa draft lokal mubazir.
        if mode == Mode.remote:
            comp = await compress(text, self.local)
            rg = await self.remote.generate(comp.text, temperature=0.2, max_tokens=1024)
            saved = max(0, estimate_tokens(text) - estimate_tokens(comp.text))
            await self.cache.put(text, rg.text)
            return self._resp(Route.remote_compressed, 0.0, 1.0, "forced_remote", rg.text,
                              remote_used=True, tin=rg.prompt_tokens, tout=rg.completion_tokens,
                              saved=saved, ratio=comp.ratio, t0=t0)

        # 1. Cache presisi tinggi (gratis)
        if mode == Mode.auto:
            cached = await self.cache.get(text)
            if cached is not None:
                return self._resp(Route.cache_hit, 0.0, 0.0, "cache", cached,
                                  remote_used=False, tin=0, tout=0,
                                  saved=estimate_tokens(text) + estimate_tokens(cached),
                                  ratio=None, t0=t0)

        # 2. Draft lokal + uncertainty terkalibrasi
        draft, u, signal = await self._draft_and_uncertainty(text)
        p_wrong = self.calibrator.predict(u)
        hr = self.high_risk(text)
        route, _reason = self.decide(p_wrong, hr, mode)

        # 3. Eksekusi rute
        if route == Route.local_only:
            await self.cache.put(text, draft)
            saved = estimate_tokens(text) + estimate_tokens(draft)  # perkiraan biaya remote yg dihindari
            return self._resp(route, u, p_wrong, signal, draft,
                              remote_used=False, tin=0, tout=0, saved=saved, ratio=None, t0=t0)

        # 2b. Local self-check (gratis): coba perbaiki draft sebelum bayar token remote.
        if route == Route.local_draft_remote_patch and settings.self_check:
            revised, u2 = await self._self_check(text, draft)
            p2 = self.calibrator.predict(u2)
            if p2 < settings.tau1:
                await self.cache.put(text, revised)
                saved = estimate_tokens(text) + estimate_tokens(revised)
                return self._resp(Route.local_self_check, u2, p2, "self_check", revised,
                                  remote_used=False, tin=0, tout=0, saved=saved,
                                  ratio=None, t0=t0)
            draft = revised   # pakai draft yang sudah diperbaiki untuk patch remote

        # Compress sebelum remote (dengan guardrail)
        comp = await compress(text, self.local)

        if route == Route.local_draft_remote_patch:
            patch_prompt = (
                f"Task (compressed): {comp.text}\n\n"
                f"Draft answer from a local model:\n{draft}"
            )
            rg = await self.remote.generate(patch_prompt, system=_PATCH_SYSTEM,
                                            temperature=0.0, max_tokens=512)
            final = draft if rg.text.strip().upper() == "OK" else rg.text.strip()
            # Saved = perkiraan output full-remote (proxy: sepanjang draft) - output patch aktual
            saved = max(0, estimate_tokens(draft) - rg.completion_tokens)
            await self.cache.put(text, final)
            return self._resp(route, u, p_wrong, signal, final,
                              remote_used=True, tin=rg.prompt_tokens, tout=rg.completion_tokens,
                              saved=saved, ratio=comp.ratio, t0=t0)

        # REMOTE_COMPRESSED (full)
        rg = await self.remote.generate(comp.text, temperature=0.2, max_tokens=1024)
        # Saved = token input yang dihemat oleh kompresi
        saved = max(0, estimate_tokens(text) - estimate_tokens(comp.text))
        await self.cache.put(text, rg.text)
        return self._resp(route, u, p_wrong, signal, rg.text,
                          remote_used=True, tin=rg.prompt_tokens, tout=rg.completion_tokens,
                          saved=saved, ratio=comp.ratio, t0=t0)

    def _resp(self, route, u, p_wrong, signal, answer, *, remote_used, tin, tout,
              saved, ratio, t0) -> AnswerResponse:
        return AnswerResponse(
            route=route,
            p_wrong=round(p_wrong, 4),
            uncertainty=round(u, 4),
            signal=signal,
            local_model=settings.local_model,
            remote_model=settings.remote_model if remote_used else None,
            remote_used=remote_used,
            remote_tokens_in=tin,
            remote_tokens_out=tout,
            estimated_tokens_saved=saved,
            compression_ratio=round(ratio, 3) if ratio is not None else None,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            answer=answer,
        )

    async def route_decision(self, text: str, mode: Mode = Mode.auto) -> RouteDecision:
        """Keputusan routing tanpa eksekusi remote (untuk POST /route)."""
        draft, u, signal = await self._draft_and_uncertainty(text)
        p_wrong = self.calibrator.predict(u)
        hr = self.high_risk(text)
        route, reason = self.decide(p_wrong, hr, mode)
        return RouteDecision(
            route=route,
            uncertainty=round(u, 4),
            p_wrong=round(p_wrong, 4),
            high_risk=hr,
            signal=signal,
            reason=reason,
        )
