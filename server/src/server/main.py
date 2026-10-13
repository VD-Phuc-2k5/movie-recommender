"""FastAPI entry point for the MovieLens Spark recommender."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp

from server.api.router import router, settings


def create_app() -> ASGIApp:
    api = FastAPI(
        title="Movie Recommender API",
        version="0.1.0",
        description="Reproducible MovieLens 32M Spark ALS pipeline and serving API.",
    )
    api.include_router(router)
    return CORSMiddleware(
        app=api,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
