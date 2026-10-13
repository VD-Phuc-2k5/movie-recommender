"""Request schemas that make runs explicit and reproducible."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PipelineRequest(StrictRequest):
    train_ratio: float = Field(default=0.8, gt=0, lt=1)
    validation_ratio: float = Field(default=0.1, gt=0, lt=1)
    test_ratio: float = Field(default=0.1, gt=0, lt=1)
    quantile_relative_error: float = Field(default=0.001, ge=0, le=0.1)

    @model_validator(mode="after")
    def ratios_sum_to_one(self) -> "PipelineRequest":
        total = self.train_ratio + self.validation_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-9:
            raise ValueError("train_ratio, validation_ratio, and test_ratio must sum to 1.")
        return self


class TrainingRequest(StrictRequest):
    rank: int = Field(default=80, ge=2, le=500)
    max_iter: int = Field(default=15, ge=1, le=100)
    reg_param: float = Field(default=0.08, gt=0, le=10)
    seed: int = Field(default=42, ge=0)
    nonnegative: bool = True
    positive_rating_threshold: float = Field(default=4.0, ge=0.5, le=5.0)
    ranking_k: int = Field(default=10, ge=1, le=100)
    ranking_user_limit: int = Field(default=5000, ge=100, le=100_000)
    negative_samples: int = Field(default=100, ge=10, le=1000)


class PredictionRequest(StrictRequest):
    user_id: int = Field(gt=0)
    movie_id: int = Field(gt=0)
    training_run_id: str | None = Field(default=None, min_length=1)


class RecommendationRequest(StrictRequest):
    user_id: int = Field(gt=0)
    k: int = Field(default=10, ge=1, le=100)
    strategy: Literal["auto", "exact", "ann"] = "auto"
    training_run_id: str | None = Field(default=None, min_length=1)


class RetrievalIndexRequest(StrictRequest):
    training_run_id: str | None = Field(default=None, min_length=1)
