"""FastAPI endpoints exposing only persisted pipeline and model results."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException

from server.api.schemas import (
    PipelineRequest,
    PredictionRequest,
    RecommendationRequest,
    RetrievalIndexRequest,
    TrainingRequest,
)
from server.core.artifacts import ArtifactNotFoundError, JobAlreadyRunningError
from server.core.config import Settings
from server.core.locks import exclusive_training_job
from server.pipeline.spark import SparkUnavailableError
from server.retrieval.redis_vector import (
    VectorSearchUnavailableError,
    build_vector_index,
)
from server.services.datasets import (
    dataset_status,
    processing_status,
    run_preprocessing,
)
from server.services.recommendations import (
    predict_rating,
    recommend_movies,
    serving_status,
)
from server.training.als import list_training_runs, load_training_run, train_als

router = APIRouter(prefix="/api/v1")
settings = Settings.from_environment()


def _call(operation: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return operation()
    except ArtifactNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except JobAlreadyRunningError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except (SparkUnavailableError, VectorSearchUnavailableError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "movie-recommender-api", **serving_status(settings)}


@router.get("/datasets")
def get_datasets() -> dict[str, Any]:
    return dataset_status(settings)


@router.post("/datasets/prepare")
def prepare(request: PipelineRequest) -> dict[str, Any]:
    return _call(lambda: run_preprocessing(settings, request))


@router.get("/processing/latest")
def get_processing() -> dict[str, Any]:
    return processing_status(settings)


@router.post("/training")
def train(request: TrainingRequest) -> dict[str, Any]:
    def operation() -> dict[str, Any]:
        with exclusive_training_job():
            return train_als(settings, request)

    return _call(operation)


@router.get("/training/runs")
def training_runs() -> dict[str, Any]:
    return {"runs": list_training_runs(settings)}


@router.get("/training/{run_id}")
def training_run(run_id: str) -> dict[str, Any]:
    return _call(lambda: load_training_run(settings, run_id))


@router.get("/results/latest")
def latest_results() -> dict[str, Any]:
    return _call(lambda: load_training_run(settings))


@router.post("/retrieval/index")
def index_retrieval(request: RetrievalIndexRequest) -> dict[str, Any]:
    return _call(lambda: build_vector_index(settings, request.training_run_id))


@router.get("/serving/status")
def get_serving_status() -> dict[str, Any]:
    return serving_status(settings)


@router.post("/predictions")
def prediction(request: PredictionRequest) -> dict[str, Any]:
    return _call(lambda: predict_rating(settings, request))


@router.post("/recommendations")
def recommendation(request: RecommendationRequest) -> dict[str, Any]:
    return _call(lambda: recommend_movies(settings, request))
