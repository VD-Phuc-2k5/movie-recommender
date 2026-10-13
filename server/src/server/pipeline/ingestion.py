"""Schema-first MovieLens extraction, quality checks, and temporal splitting."""

from __future__ import annotations

import zipfile
from pathlib import Path
from shutil import copyfileobj
from typing import Any

from server.api.schemas import PipelineRequest
from server.core.artifacts import utc_now, utc_run_id, write_json
from server.core.config import Settings
from server.pipeline.spark import spark_session

_ARCHIVE_MEMBERS = frozenset({"movies.csv", "ratings.csv", "tags.csv", "links.csv"})


def _extract_archive(settings: Settings) -> Path:
    """Extract only expected CSV members and reject unsafe archive paths."""

    if not settings.raw_archive.exists():
        raise FileNotFoundError(f"MovieLens archive not found: {settings.raw_archive}")

    destination = settings.extracted_root
    required = [destination / name for name in ("movies.csv", "ratings.csv")]
    if all(path.exists() for path in required):
        return destination

    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(settings.raw_archive) as archive:
        for info in archive.infolist():
            member_path = Path(info.filename)
            if member_path.parts[:1] != ("ml-32m",):
                continue
            if member_path.name not in _ARCHIVE_MEMBERS:
                continue
            target = destination / member_path.name
            target.resolve().relative_to(destination.resolve())
            with archive.open(info) as source, target.open("wb") as output:
                copyfileobj(source, output)
    return destination


def _null_counts(frame: object, columns: list[str]) -> dict[str, int]:
    from pyspark.sql import functions as functions

    row = frame.select(
        *[
            functions.sum(functions.col(column).isNull().cast("long")).alias(column)
            for column in columns
        ]
    ).first()
    return {column: int(row[column] or 0) for column in columns}


def prepare_dataset(settings: Settings, request: PipelineRequest) -> dict[str, Any]:
    """Create immutable Parquet splits and an auditable preprocessing manifest."""

    csv_root = _extract_archive(settings)
    run_id = utc_run_id("preprocess")
    output_root = settings.processed_root / run_id
    ratings_output = output_root / "ratings"
    movies_output = output_root / "movies"

    with spark_session(settings, "movie-recommender-preprocess") as spark:
        from pyspark import StorageLevel
        from pyspark.sql import functions as functions
        from pyspark.sql.types import (
            DoubleType,
            LongType,
            StringType,
            StructField,
            StructType,
        )

        ratings_schema = StructType(
            [
                StructField("userId", LongType(), nullable=True),
                StructField("movieId", LongType(), nullable=True),
                StructField("rating", DoubleType(), nullable=True),
                StructField("timestamp", LongType(), nullable=True),
            ]
        )
        movies_schema = StructType(
            [
                StructField("movieId", LongType(), nullable=True),
                StructField("title", StringType(), nullable=True),
                StructField("genres", StringType(), nullable=True),
            ]
        )

        ratings_raw = (
            spark.read.option("header", "true")
            .option("mode", "PERMISSIVE")
            .schema(ratings_schema)
            .csv(str(csv_root / "ratings.csv"))
            .persist(StorageLevel.DISK_ONLY)
        )
        movies_raw = (
            spark.read.option("header", "true")
            .option("mode", "PERMISSIVE")
            .schema(movies_schema)
            .csv(str(csv_root / "movies.csv"))
            .persist(StorageLevel.MEMORY_AND_DISK)
        )

        raw_rating_count = ratings_raw.count()
        raw_movie_count = movies_raw.count()
        rating_nulls = _null_counts(ratings_raw, ["userId", "movieId", "rating", "timestamp"])
        movie_nulls = _null_counts(movies_raw, ["movieId", "title", "genres"])

        rating_is_valid = (
            functions.col("userId").isNotNull()
            & (functions.col("userId") > 0)
            & functions.col("movieId").isNotNull()
            & (functions.col("movieId") > 0)
            & functions.col("rating").between(0.5, 5.0)
            & functions.col("timestamp").isNotNull()
            & (functions.col("timestamp") > 0)
        )
        valid_ratings = ratings_raw.filter(rating_is_valid).persist(StorageLevel.DISK_ONLY)
        valid_rating_count = valid_ratings.count()
        deduplicated_ratings = valid_ratings.dropDuplicates(["userId", "movieId", "timestamp"])
        deduplicated_count = deduplicated_ratings.count()
        clean_movies = movies_raw.filter(
            functions.col("movieId").isNotNull()
            & (functions.col("movieId") > 0)
            & functions.col("title").isNotNull()
            & (functions.length(functions.trim("title")) > 0)
        ).dropDuplicates(["movieId"])
        clean_movie_count = clean_movies.count()

        known_movie_ids = clean_movies.select("movieId")
        orphan_count = deduplicated_ratings.join(known_movie_ids, "movieId", "left_anti").count()
        cleaned = (
            deduplicated_ratings.join(known_movie_ids, "movieId", "inner")
            .withColumn("event_time", functions.to_timestamp(functions.from_unixtime("timestamp")))
            .persist(StorageLevel.DISK_ONLY)
        )
        cleaned_count = cleaned.count()
        lower_boundary, upper_boundary = cleaned.approxQuantile(
            "timestamp",
            [request.train_ratio, request.train_ratio + request.validation_ratio],
            request.quantile_relative_error,
        )
        if lower_boundary >= upper_boundary:
            raise ValueError(
                "Temporal split boundaries collapsed; lower quantile error or inspect timestamps."
            )

        split_ratings = cleaned.withColumn(
            "split",
            functions.when(functions.col("timestamp") < lower_boundary, "train")
            .when(functions.col("timestamp") < upper_boundary, "validation")
            .otherwise("test"),
        ).persist(StorageLevel.DISK_ONLY)
        split_counts = {
            row["split"]: int(row["count"])
            for row in split_ratings.groupBy("split").count().collect()
        }
        if set(split_counts) != {"train", "validation", "test"}:
            raise ValueError(f"Temporal split is incomplete: {split_counts}")

        split_ratings.write.mode("errorifexists").partitionBy("split").parquet(str(ratings_output))
        clean_movies.write.mode("errorifexists").parquet(str(movies_output))

        manifest: dict[str, Any] = {
            "run_id": run_id,
            "created_at": utc_now(),
            "dataset": "MovieLens 32M",
            "source": {"archive": str(settings.raw_archive), "csv_root": str(csv_root)},
            "schema": {
                "ratings": (
                    "userId: long, movieId: long, rating: double, timestamp: long, "
                    "event_time: timestamp, split: string"
                ),
                "movies": "movieId: long, title: string, genres: string",
            },
            "parameters": request.model_dump(mode="json"),
            "quality": {
                "raw_rating_count": raw_rating_count,
                "raw_movie_count": raw_movie_count,
                "rating_null_counts": rating_nulls,
                "movie_null_counts": movie_nulls,
                "invalid_rating_count": raw_rating_count - valid_rating_count,
                "duplicate_rating_count": valid_rating_count - deduplicated_count,
                "orphan_rating_count": orphan_count,
                "clean_rating_count": cleaned_count,
                "clean_movie_count": clean_movie_count,
            },
            "temporal_split": {
                "lower_timestamp_inclusive": lower_boundary,
                "upper_timestamp_inclusive": upper_boundary,
                "counts": split_counts,
            },
            "paths": {"ratings": str(ratings_output), "movies": str(movies_output)},
        }
        write_json(settings.manifests_root / f"{run_id}.json", manifest)

        for frame in (ratings_raw, movies_raw, valid_ratings, cleaned, split_ratings):
            frame.unpersist()
        return manifest
