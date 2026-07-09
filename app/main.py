"""FastAPI app: /route, /answer, /metrics, /health."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from .calibrator import Calibrator
from .config import settings
from .metrics import metrics
from .router import Router
from .schemas import AnswerRequest, AnswerResponse, RouteDecision

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    calibrator = Calibrator.load(settings.calibrator_path)
    state["router"] = Router(calibrator=calibrator)
    state["calibrated"] = calibrator.is_fitted
    yield
    state.clear()


app = FastAPI(title="TerraceRoute", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "local_model": settings.local_model,
        "remote_model": settings.remote_model,
        "calibrated": state.get("calibrated", False),
        "tau1": settings.tau1,
        "tau2": settings.tau2,
    }


@app.post("/route", response_model=RouteDecision)
async def route(req: AnswerRequest):
    return await state["router"].route_decision(req.input, req.mode)


@app.post("/answer", response_model=AnswerResponse)
async def answer(req: AnswerRequest):
    resp = await state["router"].answer(req.input, req.mode)
    metrics.record(
        resp.route, resp.remote_used, resp.remote_tokens_in,
        resp.remote_tokens_out, resp.estimated_tokens_saved, resp.latency_ms,
    )
    return resp


@app.get("/metrics")
async def get_metrics():
    return metrics.snapshot()
