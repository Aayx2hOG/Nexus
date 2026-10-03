"""Tests for release bundle loading and cryptographic integrity verification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nexus.bundle import BundleLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLES_DIR = PROJECT_ROOT / "artifacts" / "bundles"


def test_bundle_loader_successful_load():
    loader = BundleLoader(bundles_dir=BUNDLES_DIR, version="v1.0.0")
    bundle = loader.load()
    assert loader.is_ready is True
    assert bundle.manifest.bundle_version == "v1.0.0"
    assert bundle.decision_threshold == 0.5776925765603604
    assert len(bundle.feature_columns) == 42
    assert bundle.model is not None
    assert bundle.preprocessor is not None


def test_bundle_loader_missing_manifest(tmp_path):
    loader = BundleLoader(bundles_dir=tmp_path, version="v1.0.0")
    assert loader.is_ready is False
    with pytest.raises(FileNotFoundError):
        loader.load()
    assert loader.is_ready is False


def test_bundle_loader_corrupted_artifact_hash(tmp_path):
    # Copy v1.0.0 bundle to tmp_path and mutate model.joblib
    src = BUNDLES_DIR / "v1.0.0"
    dest = tmp_path / "v1.0.0"
    dest.mkdir(parents=True)
    for p in src.iterdir():
        dest.joinpath(p.name).write_bytes(p.read_bytes())

    # Corrupt model.joblib
    (dest / "model.joblib").write_bytes(b"corrupted model content")

    loader = BundleLoader(bundles_dir=tmp_path, version="v1.0.0")
    with pytest.raises(ValueError, match="Bundle artifact hash mismatch"):
        loader.load()
    assert loader.is_ready is False


def test_bundle_loader_corrupted_source_hash(tmp_path):
    src = BUNDLES_DIR / "v1.0.0"
    dest = tmp_path / "v1.0.0"
    dest.mkdir(parents=True)
    for p in src.iterdir():
        dest.joinpath(p.name).write_bytes(p.read_bytes())

    # Alter manifest expected source hash
    manifest_data = json.loads((dest / "manifest.json").read_text())
    manifest_data["source_hashes"]["fast_lightgbm_models.py"] = "0" * 64
    (dest / "manifest.json").write_text(json.dumps(manifest_data))

    loader = BundleLoader(bundles_dir=tmp_path, version="v1.0.0")
    with pytest.raises(ValueError, match="Source code hash mismatch"):
        loader.load()
    assert loader.is_ready is False


def test_health_ready_with_bundle(client):
    from fastapi.testclient import TestClient

    from nexus.api import create_app
    from nexus.config import Settings

    app = create_app(
        Settings(
            bundle_version="v1.0.0",
            bundles_dir=BUNDLES_DIR,
            api_token="test-token-" + "a" * 32,
        )
    )
    with TestClient(app) as test_client:
        resp = test_client.get("/health/ready")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ready", "bundle_version": "v1.0.0"}
