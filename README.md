# rs-movie

MovieLens 32M recommendation application with a Spark-first data pipeline,
FastAPI serving API, optional Redis Stack ANN retrieval, and a Next.js dashboard.

## Repository layout

- `data/raw/ml-32m.zip`: the supplied MovieLens 32M source archive.
- `server`: PySpark preparation, ALS training/evaluation, persisted artifacts, and
  FastAPI endpoints.
- `client`: Next.js App Router UI with exactly five screens: Datasets, Processing,
  Training, Results, and Prediction & Recommendation.

## Start locally

Start Redis Stack for response caching and ANN candidate retrieval:

```bash
docker compose up -d redis
```

Start the backend in one terminal:

```bash
cd server
cp .env.example .env
set -a && . ./.env && set +a
uv sync --group dev
uv run movie-recommender-api
```

Start the frontend in another terminal:

```bash
cd client
printf 'NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/api/v1\n' > .env.local
npm run dev
```

Open `http://localhost:3000`. Begin with **Datasets** and wait for the preparation
manifest before training. See `server/README.md` for prerequisites, artifact
locations, API flow, evaluation protocol, and optional Redis Stack setup.
