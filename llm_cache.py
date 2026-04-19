from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, Optional, Tuple


class TTLCache:
    def __init__(self) -> None:
        self._store: Dict[str, Tuple[float, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        row = self._store.get(key)
        if not row:
            return None
        expires_at, value = row
        if time.time() > expires_at:
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any, ttl_s: int) -> None:
        self._store[key] = (time.time() + max(1, int(ttl_s)), value)


def make_cache_key(prefix: str, payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"

