"""Dataset status and preprocessing service functions."""

from __future__ import annotations

import zipfile
from typing import Any

from server.api.schemas import PipelineRequest
from server.core.artifacts import ArtifactNotFoundError, latest_json
from server.core.config import Settings
from server.core.locks import exclusive_pipeline_job
from server.pipeline.ingestion import prepare_dataset


def dataset_status(settings: Settings) -> dict[str, Any]:
    archive = settings.raw_archive
    result: dict[str, Any] = {
        "archive": {
            "path": str(archive),
            "exists": archive.exists(),
            "bytes": archive.stat().st_size if archive.exists() else 0,
        },
        "extracted": {
            "path": str(settings.extracted_root),
            "ratings_csv_exists": (settings.extracted_root / "ratings.csv").exists(),
            "movies_csv_exists": (settings.extracted_root / "movies.csv").exists(),
        },
    }
    if archive.exists():
        with zipfile.ZipFile(archive) as source:
            result["archive"]["members"] = [
                item.filename for item in source.infolist() if item.filename.endswith(".csv")
            ]
    try:
        _, result["latest_processing"] = latest_json(settings.manifests_root)
    except ArtifactNotFoundError:
        result["latest_processing"] = None
    return result


def processing_status(settings: Settings) -> dict[str, Any]:
    try:
        _, manifest = latest_json(settings.manifests_root)
        return {"status": "ready", "manifest": manifest}
    except ArtifactNotFoundError:
        return {"status": "not_processed", "manifest": None}


def run_preprocessing(settings: Settings, request: PipelineRequest) -> dict[str, Any]:
    with exclusive_pipeline_job():
        return prepare_dataset(settings, request)
