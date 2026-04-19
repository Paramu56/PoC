from __future__ import annotations

import os
from typing import Dict, List

from llm_providers.base import BaseProvider, ProviderResult


class OpenAIProvider(BaseProvider):
    name = "openai"

    def __init__(self, model: str | None = None) -> None:
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError("openai package is not installed.") from e
        self.client = OpenAI(api_key=self.api_key)

    def generate(self, messages: List[Dict[str, str]], task_type: str) -> ProviderResult:
        payload = [{"role": "system", "content": f"Task type: {task_type}"}] + [
            {"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages
        ]
        resp = self.client.chat.completions.create(model=self.model, messages=payload)
        text = (resp.choices[0].message.content or "").strip() if resp.choices else ""
        return ProviderResult(text=text, meta={"model": self.model})

