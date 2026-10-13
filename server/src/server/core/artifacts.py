"""Small, atomic helpers for persisted pipeline and training metadata."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ArtifactNotFoundError(FileNotFoundError):
    """Raised when an operation needs an artifact that does not exist yet."""


class JobAlreadyRunningError(RuntimeError):
    """Raised when a non-concurrent pipeline or training job is requested."""


def utc_run_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically publish JSON so readers never observe a partial manifest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ArtifactNotFoundError(f"Artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def latest_json(directory: Path) -> tuple[Path, dict[str, Any]]:
    files = sorted(directory.glob("*.json"), reverse=True)
    if not files:
        raise ArtifactNotFoundError(f"No JSON artifacts found in: {directory}")
    latest = files[0]
    return latest, read_json(latest)
