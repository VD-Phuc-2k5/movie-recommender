import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.version_info[:2] != (3, 12),
    reason="PySpark integration tests require the project's Python 3.12 runtime.",
)


def test_pyspark_als_dependencies_are_available() -> None:
    from pyspark.ml.recommendation import ALS

    assert ALS is not None
