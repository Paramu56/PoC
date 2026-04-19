from __future__ import annotations

from typing import Any, Dict

from llm_gateway import LLMGateway
from llm_queue import QueueClient


def run_worker_once(*, redis_url: str, queue_name: str = "poc:llm:requests") -> Dict[str, Any]:
    q = QueueClient(redis_url=redis_url)
    item = q.pop(queue_name, timeout_s=1)
    if not item:
        return {"status": "idle"}
    payload = item.get("payload") or {}
    messages = payload.get("messages") or []
    task_type = payload.get("task_type") or "chat"
    gateway = LLMGateway()
    out = gateway.generate(messages=messages, task_type=task_type)
    return {"status": "processed", "task_id": item.get("task_id"), "provider": out.provider}

