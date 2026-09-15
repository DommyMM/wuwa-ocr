"""Structured JSON logging for Railway

Railway renders `message` from JSON log lines, so an event without it ships blank and matches no search
Routing every event through log_event() makes the field required instead of something each call site remembers
"""

from __future__ import annotations

import json
from typing import Any, Literal

LogLevel = Literal["info", "warning", "error"]


def log_event(
    event: str,
    message: str,
    level: LogLevel = "info",
    **fields: Any,
) -> None:
    """Emit one structured log line"""
    payload: dict[str, Any] = {"event": event, "message": message, "level": level}
    payload.update(fields)
    print(json.dumps(payload, separators=(",", ":"), ensure_ascii=False), flush=True)
