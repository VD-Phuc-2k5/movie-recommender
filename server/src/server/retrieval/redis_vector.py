"""Optional Redis cache and Redis Stack vector index over ALS factor embeddings."""

from __future__ import annotations

from array import array
from typing import Any

from server.core.artifacts import write_json
from server.core.config import Settings
from server.pipeline.spark import spark_session
from server.training.als import load_training_run


class VectorSearchUnavailableError(RuntimeError):
    """Raised when Redis is absent or does not include the RediSearch module."""


class RedisCache:
    """Best-effort JSON cache; cache failures never block a recommendation request."""

    def __init__(self, redis_url: str | None) -> None:
        self._redis_url = redis_url

    def _client(self) -> object | None:
        if not self._redis_url:
            return None
        try:
            import redis

            return redis.Redis.from_url(self._redis_url, decode_responses=True)
        except ImportError:
            return None

    def get_json(self, key: str) -> dict[str, Any] | None:
        client = self._client()
        if client is None:
            return None
        try:
            import json

            value = client.get(key)
            return json.loads(value) if value else None
        except Exception:
            return None

    def set_json(self, key: str, value: dict[str, Any], ttl_seconds: int = 300) -> bool:
        client = self._client()
        if client is None:
            return False
        try:
            import json

            client.set(key, json.dumps(value, ensure_ascii=False), ex=ttl_seconds)
            return True
        except Exception:
            return False

    def status(self) -> dict[str, Any]:
        client = self._client()
        if client is None:
            return {"configured": False, "available": False, "detail": "REDIS_URL is not configured."}
        try:
            client.ping()
            return {"configured": True, "available": True, "detail": "Redis cache is reachable."}
        except Exception as error:
            return {"configured": True, "available": False, "detail": str(error)}


class RedisVectorIndex:
    """Stores ALS item factors as float32 embeddings in a Redis Stack HNSW index."""

    def __init__(self, redis_url: str | None, training_run_id: str, dimensions: int) -> None:
        self._redis_url = redis_url
        safe_run_id = training_run_id.replace("-", "_")
        self.index_name = f"idx:movie_recommender:{safe_run_id}"
        self.key_prefix = f"movie_recommender:{training_run_id}:item:"
        self.dimensions = dimensions

    def _client(self) -> object:
        if not self._redis_url:
            raise VectorSearchUnavailableError("REDIS_URL is not configured.")
        try:
            import redis

            return redis.Redis.from_url(self._redis_url, decode_responses=False)
        except ImportError as error:
            raise VectorSearchUnavailableError("The redis Python package is not installed.") from error

    def _ensure_search_module(self, client: object) -> None:
        try:
            client.execute_command("FT._LIST")
        except Exception as error:
            raise VectorSearchUnavailableError(
                "Redis Stack with the RediSearch module is required for vector ANN."
            ) from error

    def _ensure_index(self, client: object) -> bool:
        try:
            client.execute_command("FT.INFO", self.index_name)
            return False
        except Exception:
            client.execute_command(
                "FT.CREATE",
                self.index_name,
                "ON",
                "HASH",
                "PREFIX",
                "1",
                self.key_prefix,
                "SCHEMA",
                "movie_id",
                "TAG",
                "embedding",
                "VECTOR",
                "HNSW",
                "6",
                "TYPE",
                "FLOAT32",
                "DIM",
                str(self.dimensions),
                "DISTANCE_METRIC",
                "COSINE",
            )
            return True

    @staticmethod
    def _float32_bytes(features: list[float]) -> bytes:
        return array("f", (float(value) for value in features)).tobytes()

    def build(self, item_factors: object) -> dict[str, Any]:
        client = self._client()
        self._ensure_search_module(client)
        created = self._ensure_index(client)
        pipeline = client.pipeline(transaction=False)
        indexed_count = 0
        for row in item_factors.toLocalIterator():
            pipeline.hset(
                f"{self.key_prefix}{row['id']}",
                mapping={
                    "movie_id": str(row["id"]),
                    "embedding": self._float32_bytes(row["features"]),
                },
            )
            indexed_count += 1
            if indexed_count % 1000 == 0:
                pipeline.execute()
        pipeline.execute()
        return {
            "index_name": self.index_name,
            "key_prefix": self.key_prefix,
            "dimensions": self.dimensions,
            "indexed_items": indexed_count,
            "created_index": created,
        }

    def query(self, user_features: list[float], candidate_limit: int) -> list[int]:
        client = self._client()
        self._ensure_search_module(client)
        raw_result = client.execute_command(
            "FT.SEARCH",
            self.index_name,
            f"*=>[KNN {candidate_limit} @embedding $query_vector AS distance]",
            "PARAMS",
            "2",
            "query_vector",
            self._float32_bytes(user_features),
            "SORTBY",
            "distance",
            "ASC",
            "RETURN",
            "1",
            "movie_id",
            "DIALECT",
            "2",
        )
        movie_ids: list[int] = []
        for position in range(1, len(raw_result), 2):
            fields = raw_result[position + 1]
            parsed = {
                fields[index].decode(): fields[index + 1].decode()
                for index in range(0, len(fields), 2)
            }
            movie_ids.append(int(parsed["movie_id"]))
        return movie_ids


def build_vector_index(settings: Settings, training_run_id: str | None = None) -> dict[str, Any]:
    """Materialize a Redis Stack HNSW index from a persisted ALS model's item factors."""

    run = load_training_run(settings, training_run_id)
    index = RedisVectorIndex(settings.redis_url, run["run_id"], int(run["parameters"]["rank"]))
    with spark_session(settings, "movie-recommender-vector-index"):
        from pyspark.ml.recommendation import ALSModel

        model = ALSModel.load(run["model_path"])
        result = index.build(model.itemFactors)
    payload = {"training_run_id": run["run_id"], "status": "ready", **result}
    write_json(settings.training_root / run["run_id"] / "retrieval.json", payload)
    return payload
