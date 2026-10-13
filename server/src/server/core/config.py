"""Centralized runtime configuration for repeatable local runs."""

import os
from dataclasses import dataclass
from pathlib import Path


def _project_root() -> Path:
    configured_root = os.getenv("MOVIE_RECOMMENDER_ROOT")
    if configured_root:
        return Path(configured_root).expanduser().resolve()
    return Path(__file__).resolve().parents[4]


@dataclass(frozen=True, slots=True)
class Settings:
    """Filesystem paths and integration settings shared by all services."""

    project_root: Path
    spark_master: str
    spark_driver_memory: str
    redis_url: str | None
    cors_origins: tuple[str, ...]

    @classmethod
    def from_environment(cls) -> "Settings":
        origins = os.getenv("CORS_ORIGINS", "http://localhost:3000")
        return cls(
            project_root=_project_root(),
            spark_master=os.getenv("SPARK_MASTER", "local[2]"),
            spark_driver_memory=os.getenv("SPARK_DRIVER_MEMORY", "4g"),
            redis_url=os.getenv("REDIS_URL") or None,
            cors_origins=tuple(origin.strip() for origin in origins.split(",") if origin),
        )

    @property
    def data_root(self) -> Path:
        return self.project_root / "data"

    @property
    def raw_archive(self) -> Path:
        return self.data_root / "raw" / "ml-32m.zip"

    @property
    def extracted_root(self) -> Path:
        return self.data_root / "extracted" / "ml-32m"

    @property
    def processed_root(self) -> Path:
        return self.data_root / "processed"

    @property
    def artifacts_root(self) -> Path:
        return self.data_root / "artifacts"

    @property
    def manifests_root(self) -> Path:
        return self.artifacts_root / "manifests"

    @property
    def training_root(self) -> Path:
        return self.artifacts_root / "training"
