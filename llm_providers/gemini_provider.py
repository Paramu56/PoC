from __future__ import annotations

import os
import re
import time
from typing import Dict, List

from google import genai

from llm_providers.base import BaseProvider, ProviderResult


def _is_429_rate_or_quota(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "resource_exhausted" in msg or "quota" in msg


def _looks_like_daily_free_tier_cap(exc: BaseException) -> bool:
    """Free tier per-day caps won't recover with short backoff; fail fast with a clear hint."""
    s = str(exc)
    return "PerDay" in s or ("FreeTier" in s and "GenerateRequestsPerDay" in s)


class GeminiProvider(BaseProvider):
    name = "gemini"

    def __init__(self, model: str | None = None) -> None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        self.client = genai.Client(api_key=api_key)
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-flash-latest")

    def generate(self, messages: List[Dict[str, str]], task_type: str) -> ProviderResult:
        prompt = self._to_prompt(messages, task_type)
        max_attempts = int(os.environ.get("GEMINI_GATEWAY_MAX_RETRIES", "5"))
        base_wait = float(os.environ.get("GEMINI_429_BACKOFF_S", "2.0"))
        for attempt in range(max_attempts):
            try:
                resp = self.client.models.generate_content(model=self.model, contents=prompt)
                text = getattr(resp, "text", "") or ""
                if not text and getattr(resp, "candidates", None):
                    cands = getattr(resp, "candidates") or []
                    if cands and getattr(cands[0], "content", None) and getattr(cands[0].content, "parts", None):
                        text = getattr(cands[0].content.parts[0], "text", "") or ""
                return ProviderResult(text=text, meta={"model": self.model})
            except Exception as e:
                if _looks_like_daily_free_tier_cap(e):
                    api_model = ""
                    mquota = re.search(r"model['\"]:\s*['\"]([^'\"]+)['\"]", str(e))
                    if mquota:
                        api_model = mquota.group(1).strip()
                    alias_note = ""
                    if api_model and api_model != self.model:
                        alias_note = (
                            f" Your setting `{self.model}` is billed/quota’d as **`{api_model}`** on the free tier "
                            "(aliases like `*-latest` share that model’s 20/day cap). "
                        )
                    raise RuntimeError(
                        "Gemini free-tier daily request limit for this model family is exhausted (often 20 requests/day "
                        "per model on the free tier). "
                        f"{alias_note}"
                        "Try **today** without waiting until tomorrow: set **Gemini model** to a **different** model id "
                        "that has its own counter, e.g. `gemini-2.0-flash`, `gemini-1.5-flash`, or "
                        "`gemini-2.5-flash-preview` (names depend on what Google exposes in your region). "
                        "Or enable billing / wait for the daily reset. "
                        f"Configured model: {self.model}. Details: {e}"
                    ) from e
                if not _is_429_rate_or_quota(e) or attempt >= max_attempts - 1:
                    raise
                # Short burst limits: honor "retry in Xs" when present.
                m = re.search(r"retry in ([0-9.]+)s", str(e), re.I)
                delay = float(m.group(1)) + 0.5 if m else min(45.0, base_wait * (2**attempt))
                time.sleep(delay)
        raise RuntimeError("Gemini generate_content: unexpected end of retry loop")

    @staticmethod
    def _to_prompt(messages: List[Dict[str, str]], task_type: str) -> str:
        rows = [f"TASK TYPE: {task_type}", ""]
        for m in messages:
            role = (m.get("role") or "user").upper()
            rows.append(f"{role}: {m.get('content', '')}")
        return "\n".join(rows)

