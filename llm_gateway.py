from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List

from chat_config import ChatConfig
from llm_cache import TTLCache, make_cache_key
from llm_providers import BaseProvider, GeminiProvider, OpenAIProvider
from llm_queue import QueueClient
from llm_router import ProviderRouter


@dataclass
class LLMResult:
    text: str
    provider: str
    meta: Dict[str, Any]


class LLMGateway:
    def __init__(self, providers: List[str] | None = None, cfg: ChatConfig | None = None) -> None:
        self.cfg = cfg or ChatConfig()
        provider_names = providers or [p.strip() for p in self.cfg.providers_csv.split(",") if p.strip()]
        self.providers: Dict[str, BaseProvider] = {}
        for name in provider_names:
            try:
                if name == "gemini":
                    self.providers[name] = GeminiProvider()
                elif name == "openai":
                    self.providers[name] = OpenAIProvider()
            except Exception:
                continue
        if not self.providers:
            # Last-resort hard requirement for this app.
            self.providers["gemini"] = GeminiProvider()
        self.router = ProviderRouter(list(self.providers.keys()))
        self.cache = TTLCache()
        self.queue = QueueClient(redis_url=self.cfg.redis_url)

    def generate(self, messages: List[Dict[str, str]], task_type: str = "chat") -> LLMResult:
        cache_key = make_cache_key(
            "llm",
            {"task_type": task_type, "messages": messages, "providers": list(self.providers.keys())},
        )
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        if self.cfg.queue_enabled and self.queue.enabled:
            # Hybrid mode: enqueue audit trail while still trying sync first.
            self.queue.enqueue("poc:llm:requests", {"messages": messages, "task_type": task_type})

        last_err: Exception | None = None
        for provider_name in self.router.pick_order():
            provider = self.providers[provider_name]
            t0 = time.perf_counter()
            try:
                out = provider.generate(messages=messages, task_type=task_type)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                self.router.mark_success(provider_name, elapsed_ms)
                result = LLMResult(text=out.text, provider=provider_name, meta=out.meta)
                self.cache.set(cache_key, result, ttl_s=self.cfg.llm_cache_ttl_s)
                return result
            except Exception as e:
                last_err = e
                self.router.mark_failure(provider_name)
                continue
        raise RuntimeError(f"All LLM providers failed. Last error: {last_err}")

