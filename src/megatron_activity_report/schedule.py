"""Application-level schedule and duplicate-trigger gate."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .config import ReportConfig
from .pipeline import ReportPipeline
from .publisher import is_published_cutoff
from .window import scheduled_window


def run_scheduled(
    config: ReportConfig,
    *,
    now: datetime | None = None,
    publish: bool = False,
    progress=print,
) -> dict[str, Any]:
    window = scheduled_window(timezone_name=config.timezone, now=now)
    if window is None:
        return {"action": "not_due"}
    if publish and is_published_cutoff(config, window):
        return {
            "action": "already_published",
            "window": window.key,
            "final": window.final,
        }
    result = ReportPipeline(config, progress=progress).run(window, publish=publish)
    return {"action": "published" if publish else "drafted", **result}
