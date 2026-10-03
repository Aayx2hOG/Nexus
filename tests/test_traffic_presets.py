"""Tests for traffic simulation profiles, website probe, and flow presets."""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path

from nexus.api import create_app
from nexus.config import Settings
from nexus.storage import AlertStore

BUNDLES_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "bundles"
AUTH_HEADER = {"Authorization": "Bearer test-token-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}


@pytest.fixture
def app_with_bundle(tmp_path):
    db_path = tmp_path / "test_traffic.sqlite3"
    settings = Settings(
        database_path=db_path,
        bundles_dir=BUNDLES_DIR,
        bundle_version="v2.0.0",
        api_token="test-token-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        reviewer_id="tier1-analyst",
    )
    app = create_app(settings)
    store = AlertStore(db_path)
    store.initialize()
    app.state.store = store
    return app


def test_get_simulation_profiles(app_with_bundle):
    with TestClient(app_with_bundle) as client:
        resp = client.get("/api/v1/traffic/profiles")
        assert resp.status_code == 200
        profiles = resp.json()
        assert len(profiles) >= 5
        profile_ids = [p["id"] for p in profiles]
        assert "normal_web" in profile_ids
        assert "http_exploit" in profile_ids
        assert "http_dos" in profile_ids


def test_get_preset_data(app_with_bundle):
    with TestClient(app_with_bundle) as client:
        resp = client.get("/api/v1/traffic/presets/normal_50", headers=AUTH_HEADER)
        assert resp.status_code == 200
        data = resp.json()
        assert data["preset_id"] == "normal_50"
        assert data["flow_count"] == 50
        assert "dur" in data["csv_text"]

        resp_att = client.get("/api/v1/traffic/presets/attack_50", headers=AUTH_HEADER)
        assert resp_att.status_code == 200
        assert resp_att.json()["flow_count"] == 50


def test_simulate_probe_normal_web(app_with_bundle):
    with TestClient(app_with_bundle) as client:
        payload = {
            "target_url": "http://localhost:3000",
            "profile_id": "normal_web",
        }
        resp = client.post("/api/v1/traffic/simulate-probe", json=payload, headers=AUTH_HEADER)
        assert resp.status_code == 200
        result = resp.json()
        assert result["target_url"] == "http://localhost:3000"
        assert result["target_hit"] is False
        assert result["decision"] == "normal"
        assert result["threat_level"] == "CLEAN"
        assert result["score"] < result["threshold"]
        assert result["alert_id"] is None


def test_simulate_probe_http_exploit(app_with_bundle):
    with TestClient(app_with_bundle) as client:
        payload = {
            "target_url": "http://corporate-web.internal/login",
            "profile_id": "http_exploit",
        }
        resp = client.post("/api/v1/traffic/simulate-probe", json=payload, headers=AUTH_HEADER)
        assert resp.status_code == 200
        result = resp.json()
        assert result["target_url"] == "http://corporate-web.internal/login"
        assert result["target_hit"] is True
        assert result["decision"] == "alert"
        assert result["threat_level"] in ("HIGH", "CRITICAL")
        assert result["score"] >= result["threshold"]
        assert result["alert_id"] is not None
        assert len(result["top_features"]) > 0

        # Verify alert was recorded in SQLite
        alert_resp = client.get(f"/api/v1/alerts/{result['alert_id']}", headers=AUTH_HEADER)
        assert alert_resp.status_code == 200
        alert_data = alert_resp.json()
        assert alert_data["alert_id"] == result["alert_id"]
        assert alert_data["source"] == "model"
