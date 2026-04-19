from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ProviderResult:
    text: str
    meta: Dict[str, Any]


class BaseProvider:
    name: str = "base"

    def generate(self, messages: List[Dict[str, str]], task_type: str) -> ProviderResult:
        raise NotImplementedError()

