"""Train Spark ALS models and persist their model cards and metric artifacts."""

from __future__ import annotations

from typing import Any

from server.api.schemas import TrainingRequest
from server.core.artifacts import (
    ArtifactNotFoundError,
    latest_json,
    read_json,
    utc_now,
    utc_run_id,
    write_json,
)
from server.core.config import Settings
from server.pipeline.spark import spark_session
from server.training.evaluation import (
    evaluate_als_ranking,
    evaluate_global_mean_baseline,
    evaluate_popularity_ranking,
    evaluate_regression,
)


def latest_preprocessing_manifest(settings: Settings) -> dict[str, Any]:
    _, manifest = latest_json(settings.manifests_root)
    return manifest


def list_training_runs(settings: Settings) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in sorted(settings.training_root.glob("*/run.json"), reverse=True):
        try:
            runs.append(read_json(path))
        except ValueError:
            continue
    return runs


def load_training_run(settings: Settings, run_id: str | None = None) -> dict[str, Any]:
    if run_id:
        return read_json(settings.training_root / run_id / "run.json")
    runs = list_training_runs(settings)
    if not runs:
        raise ArtifactNotFoundError("No training run exists. Run preprocessing and ALS training first.")
    return runs[0]


def _als(config: TrainingRequest) -> object:
    from pyspark.ml.recommendation import ALS

    return ALS(
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        rank=config.rank,
        maxIter=config.max_iter,
        regParam=config.reg_param,
        seed=config.seed,
        nonnegative=config.nonnegative,
        coldStartStrategy="drop",
    )


def train_als(settings: Settings, request: TrainingRequest) -> dict[str, Any]:
    """Validate an ALS configuration, train final model, and persist all outputs."""

    manifest = latest_preprocessing_manifest(settings)
    run_id = utc_run_id("als")
    run_root = settings.training_root / run_id
    model_path = run_root / "model"
    config = request.model_dump(mode="json")

    with spark_session(settings, "movie-recommender-als-training") as spark:
        from pyspark.sql import functions as functions

        ratings = spark.read.parquet(manifest["paths"]["ratings"])
        movies = spark.read.parquet(manifest["paths"]["movies"])
        train = ratings.filter("split = 'train'").select("userId", "movieId", "rating")
        validation = ratings.filter("split = 'validation'").select("userId", "movieId", "rating")
        test = ratings.filter("split = 'test'").select("userId", "movieId", "rating")

        train_mean = float(train.agg(functions.avg("rating").alias("mean")).first()["mean"])
        selection_model = _als(request).fit(train)
        validation_metrics = {
            "als": {
                "regression": evaluate_regression(selection_model, validation),
                "ranking": evaluate_als_ranking(selection_model, validation, train, movies, config),
            },
            "baseline": {
                "global_mean": evaluate_global_mean_baseline(validation, train_mean),
                "popularity": evaluate_popularity_ranking(validation, train, movies, config),
            },
        }

        final_training = train.unionByName(validation)
        final_training_mean = float(
            final_training.agg(functions.avg("rating").alias("mean")).first()["mean"]
        )
        final_model = _als(request).fit(final_training)
        test_metrics = {
            "als": {
                "regression": evaluate_regression(final_model, test),
                "ranking": evaluate_als_ranking(final_model, test, final_training, movies, config),
            },
            "baseline": {
                "global_mean": evaluate_global_mean_baseline(test, final_training_mean),
                "popularity": evaluate_popularity_ranking(test, final_training, movies, config),
            },
        }
        final_model.write().overwrite().save(str(model_path))

    run: dict[str, Any] = {
        "run_id": run_id,
        "created_at": utc_now(),
        "algorithm": "Spark MLlib ALS explicit-feedback",
        "preprocessing_run_id": manifest["run_id"],
        "model_path": str(model_path),
        "parameters": config,
        "training_data": {"splits": ["train", "validation"], "global_mean": final_training_mean},
        "metrics": {"validation": validation_metrics, "test": test_metrics},
        "ranking_protocol": {
            "positive_definition": f"rating >= {request.positive_rating_threshold}",
            "candidate_set": "held-out unseen positives + deterministic sampled unseen negatives",
            "candidate_negative_samples": request.negative_samples,
            "maximum_evaluated_users": request.ranking_user_limit,
        },
    }
    write_json(run_root / "run.json", run)
    return run
