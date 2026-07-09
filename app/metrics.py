"""Agregasi metrik in-memory untuk endpoint /metrics dan demo 'ekonomi token'."""
from __future__ import annotations

from dataclasses import dataclass, field

from .schemas import Route


@dataclass
class Metrics:
    total_requests: int = 0
    by_route: dict = field(default_factory=dict)
    remote_calls: int = 0
    remote_tokens_in: int = 0
    remote_tokens_out: int = 0
    estimated_tokens_saved: int = 0
    latency_ms_sum: int = 0

    def record(self, route: Route, remote_used: bool, tin: int, tout: int,
               saved: int, latency_ms: int) -> None:
        self.total_requests += 1
        self.by_route[route.value] = self.by_route.get(route.value, 0) + 1
        if remote_used:
            self.remote_calls += 1
        self.remote_tokens_in += tin
        self.remote_tokens_out += tout
        self.estimated_tokens_saved += saved
        self.latency_ms_sum += latency_ms

    def snapshot(self) -> dict:
        n = max(1, self.total_requests)
        return {
            "total_requests": self.total_requests,
            "by_route": self.by_route,
            "remote_calls": self.remote_calls,
            "remote_call_rate": round(self.remote_calls / n, 3),
            "remote_tokens_in": self.remote_tokens_in,
            "remote_tokens_out": self.remote_tokens_out,
            "remote_tokens_total": self.remote_tokens_in + self.remote_tokens_out,
            "estimated_tokens_saved": self.estimated_tokens_saved,
            "avg_latency_ms": round(self.latency_ms_sum / n, 1),
        }


metrics = Metrics()
