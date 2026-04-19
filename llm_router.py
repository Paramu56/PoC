from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ProviderHealth:
    ok_count: int = 0
    err_count: int = 0
    last_latency_ms: float = 0.0
    cooled_until: float = 0.0

    @property
    def score(self) -> float:
        base = (self.ok_count + 1.0) / (self.err_count + 1.0)
        latency_penalty = 1.0 / (1.0 + (self.last_latency_ms / 1000.0))
        return base * latency_penalty


class ProviderRouter:
    def __init__(self, provider_names: List[str]) -> None:
        self.provider_names = [p for p in provider_names if p]
        self.health: Dict[str, ProviderHealth] = {p: ProviderHealth() for p in self.provider_names}

    def pick_order(self) -> List[str]:
        now = time.time()
        live = [p for p in self.provider_names if self.health[p].cooled_until <= now]
        cool = [p for p in self.provider_names if self.health[p].cooled_until > now]
        live.sort(key=lambda p: self.health[p].score, reverse=True)
        cool.sort(key=lambda p: self.health[p].cooled_until)
        return live + cool

    def mark_success(self, provider: str, latency_ms: float) -> None:
        h = self.health.setdefault(provider, ProviderHealth())
        h.ok_count += 1
        h.last_latency_ms = latency_ms
        h.cooled_until = 0.0

    def mark_failure(self, provider: str, cool_s: int = 20) -> None:
        h = self.health.setdefault(provider, ProviderHealth())
        h.err_count += 1
        h.cooled_until = time.time() + max(1, cool_s)

