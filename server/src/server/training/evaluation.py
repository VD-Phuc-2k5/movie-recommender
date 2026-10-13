"""Regression and sampled-candidate ranking evaluation for recommender models."""

from __future__ import annotations

from typing import Any


def evaluate_regression(model: object, ratings: object) -> dict[str, float | int]:
    """Evaluate a Spark recommender model after dropping cold-start predictions."""

    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.sql import functions as functions

    predictions = model.transform(ratings).dropna(subset=["prediction"])
    evaluated_rows = predictions.count()
    if evaluated_rows == 0:
        raise ValueError("No non-cold-start predictions are available for regression evaluation.")

    rmse = RegressionEvaluator(
        labelCol="rating", predictionCol="prediction", metricName="rmse"
    ).evaluate(predictions)
    mae_row = predictions.select(
        functions.avg(functions.abs(functions.col("prediction") - functions.col("rating"))).alias(
            "mae"
        )
    ).first()
    return {"rmse": float(rmse), "mae": float(mae_row["mae"]), "rows": evaluated_rows}


def evaluate_global_mean_baseline(
    ratings: object, training_mean: float
) -> dict[str, float | int]:
    """A transparent rating baseline: every prediction is the train-set mean."""

    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.sql import functions as functions

    predictions = ratings.withColumn("prediction", functions.lit(training_mean))
    evaluated_rows = predictions.count()
    rmse = RegressionEvaluator(
        labelCol="rating", predictionCol="prediction", metricName="rmse"
    ).evaluate(predictions)
    mae_row = predictions.select(
        functions.avg(functions.abs(functions.col("prediction") - functions.col("rating"))).alias(
            "mae"
        )
    ).first()
    return {"rmse": float(rmse), "mae": float(mae_row["mae"]), "rows": evaluated_rows}


def build_ranking_candidates(
    ratings: object,
    history: object,
    movies: object,
    positive_rating_threshold: float,
    user_limit: int,
    negative_samples: int,
    seed: int,
) -> object:
    """Build a bounded, deterministic relevance-vs-negative candidate set.

    Full user-by-catalog ranking is infeasible at MovieLens 32M scale. The evaluator
    scores held-out positive interactions against deterministic, unseen catalogue
    negatives for a reproducible sampled-candidate Recall@K/NDCG@K estimate.
    """

    from pyspark.sql import Window
    from pyspark.sql import functions as functions

    relevant = (
        ratings.filter(functions.col("rating") >= positive_rating_threshold)
        .select("userId", "movieId")
        .dropDuplicates()
        .join(history.select("userId", "movieId").dropDuplicates(), ["userId", "movieId"], "left_anti")
        .withColumn("relevant", functions.lit(1))
    )
    sampled_users = (
        relevant.select("userId")
        .distinct()
        .orderBy(functions.hash("userId", functions.lit(seed)))
        .limit(user_limit)
    )
    relevant = relevant.join(sampled_users, "userId", "inner")
    relevant_count = relevant.count()
    if relevant_count == 0:
        raise ValueError("No unseen held-out positive ratings are available for ranking evaluation.")

    catalogue = movies.select("movieId").dropDuplicates()
    catalogue_count = catalogue.count()
    if catalogue_count == 0:
        raise ValueError("Cannot evaluate ranking without catalogue movies.")
    catalogue = catalogue.withColumn(
        "catalogue_index", functions.row_number().over(Window.orderBy("movieId")) - 1
    )

    sampled_indices = (
        sampled_users.withColumn(
            "sample_slot",
            functions.explode(functions.sequence(functions.lit(0), functions.lit(negative_samples - 1))),
        )
        .withColumn(
            "catalogue_index",
            functions.pmod(
                functions.hash("userId", "sample_slot", functions.lit(seed)),
                functions.lit(catalogue_count),
            ).cast("long"),
        )
        .select("userId", "catalogue_index")
    )
    negatives = (
        sampled_indices.join(catalogue, "catalogue_index", "inner")
        .select("userId", "movieId")
        .dropDuplicates()
        .join(history.select("userId", "movieId").dropDuplicates(), ["userId", "movieId"], "left_anti")
        .join(relevant.select("userId", "movieId"), ["userId", "movieId"], "left_anti")
        .withColumn("relevant", functions.lit(0))
    )
    return relevant.unionByName(negatives).dropDuplicates(["userId", "movieId"])


def ranking_metrics_from_scored_candidates(
    scored_candidates: object, ranking_k: int
) -> dict[str, float | int]:
    """Compute user-averaged Recall@K and binary NDCG@K from scored candidates."""

    from pyspark.sql import Window
    from pyspark.sql import functions as functions

    rankings = scored_candidates.dropna(subset=["prediction"]).withColumn(
        "rank",
        functions.row_number().over(
            Window.partitionBy("userId").orderBy(
                functions.col("prediction").desc(), functions.col("movieId").asc()
            )
        ),
    )
    evaluated_user_count = rankings.select("userId").distinct().count()
    if evaluated_user_count == 0:
        raise ValueError("No candidates could be scored for ranking evaluation.")

    relevance_counts = rankings.filter(functions.col("relevant") == 1).groupBy("userId").agg(
        functions.sum("relevant").alias("relevant_count")
    )
    top_k = rankings.filter(functions.col("rank") <= ranking_k)
    top_metrics = top_k.groupBy("userId").agg(
        functions.sum("relevant").alias("hits"),
        functions.sum(
            functions.when(
                functions.col("relevant") == 1,
                functions.lit(1.0) / functions.log2(functions.col("rank") + 1.0),
            ).otherwise(functions.lit(0.0))
        ).alias("dcg"),
    )
    per_user = (
        relevance_counts.join(top_metrics, "userId", "left")
        .fillna({"hits": 0, "dcg": 0.0})
        .withColumn("recall", functions.col("hits") / functions.col("relevant_count"))
        .withColumn(
            "ideal_dcg",
            functions.expr(
                f"aggregate(sequence(1, least(relevant_count, {ranking_k})), "
                "CAST(0.0 AS DOUBLE), (acc, position) -> "
                "acc + 1.0 / log2(position + 1.0))"
            ),
        )
        .withColumn("ndcg", functions.col("dcg") / functions.col("ideal_dcg"))
    )
    summary = per_user.agg(
        functions.avg("recall").alias("recall_at_k"),
        functions.avg("ndcg").alias("ndcg_at_k"),
    ).first()
    return {
        "recall_at_k": float(summary["recall_at_k"]),
        "ndcg_at_k": float(summary["ndcg_at_k"]),
        "users": evaluated_user_count,
        "k": ranking_k,
    }


def evaluate_als_ranking(
    model: object,
    ratings: object,
    history: object,
    movies: object,
    config: dict[str, Any],
) -> dict[str, float | int]:
    candidates = build_ranking_candidates(
        ratings=ratings,
        history=history,
        movies=movies,
        positive_rating_threshold=float(config["positive_rating_threshold"]),
        user_limit=int(config["ranking_user_limit"]),
        negative_samples=int(config["negative_samples"]),
        seed=int(config["seed"]),
    )
    return ranking_metrics_from_scored_candidates(
        model.transform(candidates), int(config["ranking_k"])
    )


def evaluate_popularity_ranking(
    ratings: object,
    history: object,
    movies: object,
    config: dict[str, Any],
) -> dict[str, float | int]:
    """Compare against a non-personalized popularity recommender."""

    from pyspark.sql import functions as functions

    candidates = build_ranking_candidates(
        ratings=ratings,
        history=history,
        movies=movies,
        positive_rating_threshold=float(config["positive_rating_threshold"]),
        user_limit=int(config["ranking_user_limit"]),
        negative_samples=int(config["negative_samples"]),
        seed=int(config["seed"]),
    )
    popularity = history.groupBy("movieId").count().withColumnRenamed("count", "rating_count")
    scored = (
        candidates.join(popularity, "movieId", "left")
        .fillna({"rating_count": 0})
        .withColumn("prediction", functions.log(functions.col("rating_count") + 1.0))
    )
    return ranking_metrics_from_scored_candidates(scored, int(config["ranking_k"]))
