"""Structured JSON logging for Railway.

Railway parses any JSON log line as a structured log and renders `message`.
An event without `message` ships as a blank line and matches no log search,
which once hid every r2_result in production. Routing every event through
log_event() makes that field structural rather than something each call site
has to remember to include.
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
    """Emit one structured log line. `message` is required, never optional."""

    payload: dict[str, Any] = {"event": event, "message": message, "level": level}
    payload.update(fields)
    print(json.dumps(payload, separators=(",", ":"), ensure_ascii=False), flush=True)
