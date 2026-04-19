from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ChatConfig:
    llm_timeout_s: float = float(os.environ.get("POC_LLM_TIMEOUT_S", "45"))
    llm_cache_ttl_s: int = int(os.environ.get("POC_LLM_CACHE_TTL_S", "600"))
    retrieval_cache_ttl_s: int = int(os.environ.get("POC_RETRIEVAL_CACHE_TTL_S", "600"))
    queue_enabled: bool = os.environ.get("POC_LLM_QUEUE_ENABLED", "").strip().lower() in ("1", "true", "yes")
    redis_url: str = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    providers_csv: str = os.environ.get("POC_LLM_PROVIDERS", "gemini,openai")

