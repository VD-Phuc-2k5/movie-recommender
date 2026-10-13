"""In-process exclusion for expensive Spark operations."""

from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock

from server.core.artifacts import JobAlreadyRunningError

_PIPELINE_LOCK = Lock()
_TRAINING_LOCK = Lock()


@contextmanager
def exclusive_pipeline_job() -> Iterator[None]:
    if not _PIPELINE_LOCK.acquire(blocking=False):
        raise JobAlreadyRunningError("A preprocessing job is already running.")
    try:
        yield
    finally:
        _PIPELINE_LOCK.release()


@contextmanager
def exclusive_training_job() -> Iterator[None]:
    if not _TRAINING_LOCK.acquire(blocking=False):
        raise JobAlreadyRunningError("A training job is already running.")
    try:
        yield
    finally:
        _TRAINING_LOCK.release()
