from pathlib import Path

from server.core.artifacts import latest_json, read_json, write_json


def test_json_artifacts_are_published_and_loaded(tmp_path: Path) -> None:
    artifact = tmp_path / "manifests" / "preprocess.json"
    payload = {"run_id": "preprocess-1", "quality": {"rows": 42}}

    write_json(artifact, payload)

    assert read_json(artifact) == payload
    latest_path, latest_payload = latest_json(artifact.parent)
    assert latest_path == artifact
    assert latest_payload == payload
