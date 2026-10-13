"""Actual model serving with optional Redis cache and ANN candidate retrieval."""

from __future__ import annotations

from typing import Any

from server.api.schemas import PredictionRequest, RecommendationRequest
from server.core.config import Settings
from server.pipeline.spark import spark_session
from server.retrieval.redis_vector import (
    RedisCache,
    RedisVectorIndex,
    VectorSearchUnavailableError,
)
from server.training.als import load_training_run


def _manifest_for_training_run(settings: Settings, run: dict[str, Any]) -> dict[str, Any]:
    from server.core.artifacts import read_json

    return read_json(settings.manifests_root / f"{run['preprocessing_run_id']}.json")


def _dot_product_udf() -> object:
    from pyspark.sql import functions as functions

    return functions.udf(
        lambda item_features, user_features: float(
            sum(float(item) * float(user) for item, user in zip(item_features, user_features, strict=True))
        ),
        "double",
    )


def predict_rating(settings: Settings, request: PredictionRequest) -> dict[str, Any]:
    run = load_training_run(settings, request.training_run_id)
    cache = RedisCache(settings.redis_url)
    cache_key = f"movie-recommender:prediction:{run['run_id']}:{request.user_id}:{request.movie_id}"
    cached = cache.get_json(cache_key)
    if cached is not None:
        cached["cache"] = {"hit": True}
        return cached

    manifest = _manifest_for_training_run(settings, run)
    with spark_session(settings, "movie-recommender-prediction") as spark:
        from pyspark.ml.recommendation import ALSModel
        from pyspark.sql import functions as functions

        model = ALSModel.load(run["model_path"])
        input_frame = spark.createDataFrame([(request.user_id, request.movie_id)], ["userId", "movieId"])
        prediction = model.transform(input_frame).select("prediction").first()
        movie = (
            spark.read.parquet(manifest["paths"]["movies"])
            .filter(functions.col("movieId") == request.movie_id)
            .select("title", "genres")
            .first()
        )

    result: dict[str, Any] = {
        "training_run_id": run["run_id"],
        "user_id": request.user_id,
        "movie_id": request.movie_id,
        "movie": dict(movie.asDict()) if movie else None,
        "prediction": float(prediction["prediction"]) if prediction and prediction["prediction"] is not None else None,
        "cold_start": prediction is None or prediction["prediction"] is None,
        "cache": {"hit": False, "stored": False},
    }
    result["cache"]["stored"] = cache.set_json(cache_key, result)
    return result


def _score_movies(
    model: object,
    movie_ids: list[int] | None,
    user_id: int,
    known_interactions: object,
    movies: object,
    k: int,
) -> tuple[list[dict[str, Any]], bool]:
    from pyspark.sql import functions as functions

    user_factor = model.userFactors.filter(functions.col("id") == user_id).select("features").first()
    if user_factor is None:
        return [], True
    item_factors = model.itemFactors.select(
        functions.col("id").alias("movieId"), functions.col("features").alias("item_features")
    )
    if movie_ids is not None:
        item_factors = item_factors.filter(functions.col("movieId").isin(movie_ids))
    unseen_items = item_factors.join(known_interactions, "movieId", "left_anti")
    user_frame = item_factors.sparkSession.createDataFrame(
        [(list(user_factor["features"]),)], ["user_features"]
    )
    dot_product = _dot_product_udf()
    ranked = (
        unseen_items.crossJoin(user_frame)
        .withColumn("score", dot_product("item_features", "user_features"))
        .orderBy(functions.col("score").desc(), functions.col("movieId").asc())
        .limit(k)
        .join(movies, "movieId", "left")
        .select("movieId", "title", "genres", "score")
        .collect()
    )
    return [
        {
            "movie_id": int(row["movieId"]),
            "title": row["title"],
            "genres": row["genres"],
            "score": float(row["score"]),
        }
        for row in ranked
    ], False


def recommend_movies(settings: Settings, request: RecommendationRequest) -> dict[str, Any]:
    run = load_training_run(settings, request.training_run_id)
    cache = RedisCache(settings.redis_url)
    cache_key = f"movie-recommender:recommendations:{run['run_id']}:{request.user_id}:{request.k}:{request.strategy}"
    cached = cache.get_json(cache_key)
    if cached is not None:
        cached["cache"] = {"hit": True}
        return cached

    manifest = _manifest_for_training_run(settings, run)
    source = "exact-als"
    ann_detail: str | None = None
    with spark_session(settings, "movie-recommender-recommendation") as spark:
        from pyspark.ml.recommendation import ALSModel
        from pyspark.sql import functions as functions

        model = ALSModel.load(run["model_path"])
        processed_ratings = spark.read.parquet(manifest["paths"]["ratings"])
        known_interactions = (
            processed_ratings.filter(
                (functions.col("userId") == request.user_id)
                & functions.col("split").isin("train", "validation")
            )
            .select("movieId")
            .dropDuplicates()
        )
        movies = spark.read.parquet(manifest["paths"]["movies"])
        candidate_ids: list[int] | None = None
        if request.strategy != "exact":
            try:
                user_factor = model.userFactors.filter(functions.col("id") == request.user_id).first()
                if user_factor is None:
                    recommendations, cold_start = [], True
                else:
                    index = RedisVectorIndex(
                        settings.redis_url, run["run_id"], int(run["parameters"]["rank"])
                    )
                    candidate_ids = index.query(list(user_factor["features"]), max(request.k * 10, 100))
                    recommendations, cold_start = _score_movies(
                        model, candidate_ids, request.user_id, known_interactions, movies, request.k
                    )
                    source = "redis-ann-reranked-by-als"
            except VectorSearchUnavailableError as error:
                if request.strategy == "ann":
                    raise
                ann_detail = str(error)
                recommendations, cold_start = _score_movies(
                    model, None, request.user_id, known_interactions, movies, request.k
                )
            else:
                if user_factor is None:
                    source = "redis-ann-reranked-by-als"
        else:
            recommendations, cold_start = _score_movies(
                model, None, request.user_id, known_interactions, movies, request.k
            )

    result: dict[str, Any] = {
        "training_run_id": run["run_id"],
        "user_id": request.user_id,
        "strategy": source,
        "ann_detail": ann_detail,
        "cold_start": cold_start,
        "recommendations": recommendations,
        "cache": {"hit": False, "stored": False},
    }
    result["cache"]["stored"] = cache.set_json(cache_key, result)
    return result


def serving_status(settings: Settings) -> dict[str, Any]:
    status: dict[str, Any] = {"cache": RedisCache(settings.redis_url).status()}
    try:
        run = load_training_run(settings)
        retrieval_path = settings.training_root / run["run_id"] / "retrieval.json"
        status["vector_ann"] = {
            "training_run_id": run["run_id"],
            "artifact_exists": retrieval_path.exists(),
            "artifact_path": str(retrieval_path),
        }
    except Exception as error:
        status["vector_ann"] = {"training_run_id": None, "artifact_exists": False, "detail": str(error)}
    return status
