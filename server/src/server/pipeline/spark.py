"""Spark session construction kept out of API import time."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager, suppress

from server.core.config import Settings


class SparkUnavailableError(RuntimeError):
    """Raised with a useful action when PySpark or Java is unavailable."""


def ensure_supported_python_version(version: tuple[int, int]) -> None:
    """Reject interpreters outside the project's tested PySpark runtime."""
    if version != (3, 12):
        raise SparkUnavailableError(
            "This project requires Python 3.12 for its PySpark 3.5 runtime; "
            f"found Python {version[0]}.{version[1]}. "
            "Recreate the environment with `uv venv --clear --python 3.12` "
            "and run `uv sync --group dev`."
        )


@contextmanager
def spark_session(settings: Settings, application_name: str) -> Iterator[object]:
    ensure_supported_python_version(sys.version_info[:2])

    try:
        from pyspark.sql import SparkSession
    except ImportError as error:
        raise SparkUnavailableError(
            "PySpark is not installed. Run `uv sync --group dev` in server/."
        ) from error

    try:
        session = (
            SparkSession.builder.appName(application_name)
            .master(settings.spark_master)
            .config("spark.driver.memory", settings.spark_driver_memory)
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.shuffle.partitions", "200")
            .getOrCreate()
        )
    except Exception as error:
        raise SparkUnavailableError(
            "Spark could not start. Install a compatible Java runtime and check SPARK_MASTER."
        ) from error

    session.sparkContext.setLogLevel("WARN")
    try:
        yield session
    finally:
        with suppress(Exception):
            session.stop()
