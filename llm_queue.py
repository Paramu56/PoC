from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, Optional


class QueueClient:
    """
    Redis-backed queue when redis package is available.
    Falls back to no-op mode when Redis is unavailable.
    """

    def __init__(self, redis_url: str) -> None:
        self.redis_url = redis_url
        self.enabled = False
        self._redis = None
        try:
            import redis  # type: ignore

            self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
            self._redis.ping()
            self.enabled = True
        except Exception:
            self.enabled = False

    def enqueue(self, queue_name: str, payload: Dict[str, Any]) -> str:
        task_id = str(uuid.uuid4())
        body = {"task_id": task_id, "enqueued_at": time.time(), "payload": payload}
        if self.enabled and self._redis is not None:
            self._redis.rpush(queue_name, json.dumps(body))
        return task_id

    def pop(self, queue_name: str, timeout_s: int = 1) -> Optional[Dict[str, Any]]:
        if not self.enabled or self._redis is None:
            return None
        row = self._redis.blpop(queue_name, timeout=timeout_s)
        if not row:
            return None
        _key, value = row
        try:
            return json.loads(value)
        except Exception:
            return None

