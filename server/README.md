# Movie Recommender Server

FastAPI service and reproducible PySpark pipeline for the local MovieLens 32M
archive at `../data/raw/ml-32m.zip`.

## Setup

Requirements: Python 3.12, a Java runtime compatible with Spark, and `uv`.

```bash
cd server
uv python install 3.12
uv sync --group dev
uv run movie-recommender-api
```

PySpark 3.5's serializer is not compatible with Python 3.14. If a virtual
environment was created with a different interpreter, recreate it before
starting the API:

```bash
uv venv --clear --python 3.12
uv sync --group dev
```

The API is available at `http://localhost:8000`; interactive OpenAPI docs are at
`/docs`. The frontend assumes `http://localhost:8000/api/v1` unless
`NEXT_PUBLIC_API_BASE_URL` overrides it.

Local Spark defaults to `local[2]` to avoid exhausting memory while preprocessing
MovieLens 32M. Override `SPARK_MASTER` only when the machine has enough memory for
additional concurrent tasks. ALS training uses a `4g` driver heap by default;
override `SPARK_DRIVER_MEMORY` only when the local machine cannot accommodate it.

Optional Redis integration:

```bash
cd ..
docker compose up -d redis
cd server
cp .env.example .env
set -a && . ./.env && set +a
```

Regular response caching works with Redis. Vector candidate retrieval additionally
requires Redis Stack / RediSearch because the app creates an HNSW `FLOAT32` index
over persisted ALS item factors.

## Run order

1. `POST /api/v1/datasets/prepare` extracts only known MovieLens CSV members,
   applies declared schemas, records quality checks, performs a chronological
   train/validation/test split, and writes Parquet plus a manifest.
2. `POST /api/v1/training` trains Spark MLlib explicit-feedback ALS. It reports
   validation and held-out test metrics against global-mean and popularity
   baselines, then persists the final model trained on train + validation.
3. Optionally call `POST /api/v1/retrieval/index` after Redis Stack is healthy to
   materialize the ANN index from the saved model's item factors.
4. Use `/predictions` and `/recommendations` for live model serving. `auto`
   recommendation strategy uses ANN only when it is genuinely reachable; it
   otherwise uses exact ALS-factor scoring and reports the reason.

## Artifacts

- `data/extracted/ml-32m`: source CSVs extracted once from the provided archive.
- `data/processed/preprocess-*`: immutable partitioned Parquet outputs.
- `data/artifacts/manifests/preprocess-*.json`: schema, quality, split boundaries,
  paths, and input parameters.
- `data/artifacts/training/als-*/model`: Spark model artifact.
- `data/artifacts/training/als-*/run.json`: configuration, evaluation protocol,
  baselines, and metrics.
- `data/artifacts/training/als-*/retrieval.json`: Redis ANN index metadata after a
  successful index build.

Ranking metrics use held-out unseen ratings at or above the configured positive
threshold and deterministic unseen sampled negatives. This avoids an infeasible
full user-by-catalogue Cartesian product while keeping Recall@K/NDCG@K repeatable.
