"""Tests for Tier 1 routes: predictions, model summary, stats, sample review, and replay."""

import json
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from nexus.api import create_app
from nexus.config import Settings

TEST_TOKEN = "test-token-" + "b" * 32
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def loaded_app(tmp_path):
    """Create FastAPI application with release bundle v1.0.0 loaded and ready."""
    bundle_dir = PROJECT_ROOT / "artifacts" / "bundles"
    db_path = tmp_path / "nexus_tier1.sqlite3"
    settings = Settings(
        database_path=db_path,
        api_token=TEST_TOKEN,
        reviewer_id="tier1-analyst",
        bundles_dir=bundle_dir,
        bundle_version="v1.0.0",
    )
    app = create_app(settings)
    # Trigger startup lifespan
    with TestClient(app, headers={"Authorization": f"Bearer {TEST_TOKEN}"}) as client:
        yield client, app


@pytest.fixture
def auth_client(loaded_app):
    client, _ = loaded_app
    return client


def test_model_summary_endpoint(auth_client):
    response = auth_client.get("/api/v1/model")
    assert response.status_code == 200
    data = response.json()
    assert data["bundle_version"] == "v1.0.0"
    assert data["algorithm"] == "LightGBM"
    assert data["decision_threshold"] == pytest.approx(0.5776925765603604, abs=1e-5)
    assert data["metrics_note"] == "selection estimate, not independent confirmation"
    assert "model.joblib" in data["bundle_hashes"]
    assert "preprocessor.joblib" in data["bundle_hashes"]
    assert "accuracy" in data["selection_metrics"]


def test_batch_prediction_and_alert_generation(auth_client):
    payload_path = PROJECT_ROOT / "examples" / "flow-batch.json"
    batch_json = json.loads(payload_path.read_text())

    # Send flow batch for inference
    response = auth_client.post("/api/v1/predictions", json=batch_json)
    assert response.status_code == 200
    result = response.json()
    assert result["bundle_version"] == "v1.0.0"
    assert result["total_count"] == 1
    pred = result["predictions"][0]
    assert "flow_id" in pred
    assert "score" in pred
    assert "threshold" in pred
    assert pred["decision"] in ("alert", "normal")

    # If alert was created, verify it can be retrieved via alert API
    if pred["decision"] == "alert":
        assert pred["alert_id"] is not None
        alert_resp = auth_client.get(f"/api/v1/alerts/{pred['alert_id']}")
        assert alert_resp.status_code == 200
        alert_data = alert_resp.json()
        assert alert_data["severity"] == "alert"
        assert alert_data["source"] == "model"
        assert len(alert_data["top_features"]) <= 10
        # Assert ground-truth is NOT present in alert
        assert "label" not in alert_data
        assert "attack_cat" not in alert_data


def test_stats_and_sample_review_lifecycle(auth_client):
    # 1. Stats endpoint
    stats_resp = auth_client.get("/api/v1/stats")
    assert stats_resp.status_code == 200
    stats = stats_resp.json()
    assert "alerts_24h" in stats
    assert "awaiting_review" in stats
    assert "total_predictions_24h" in stats

    # 2. Ingest a batch to ensure we have predictions
    flow_id = str(uuid4())
    payload = {
        "schema_version": "unsw-nb15.v0",
        "flows": [
            {
                "flow_id": flow_id,
                "event_time": "2026-10-03T10:00:00Z",
                "features": {
                    "dur": 0.00001,
                    "proto": "udp",
                    "service": "-",
                    "state": "INT",
                    "spkts": 2,
                    "dpkts": 0,
                    "sbytes": 100,
                    "dbytes": 0,
                    "rate": 100000.0,
                    "sttl": 254,
                    "dttl": 0,
                    "sload": 40000000.0,
                    "dload": 0.0,
                    "sloss": 0,
                    "dloss": 0,
                    "sinpkt": 0.01,
                    "dinpkt": 0.0,
                    "sjit": 0.0,
                    "djit": 0.0,
                    "swin": 0,
                    "stcpb": 0,
                    "dtcpb": 0,
                    "dwin": 0,
                    "tcprtt": 0.0,
                    "synack": 0.0,
                    "ackdat": 0.0,
                    "smean": 50,
                    "dmean": 0,
                    "trans_depth": 0,
                    "response_body_len": 0,
                    "ct_srv_src": 1,
                    "ct_state_ttl": 2,
                    "ct_dst_ltm": 1,
                    "ct_src_dport_ltm": 1,
                    "ct_dst_sport_ltm": 1,
                    "ct_dst_src_ltm": 1,
                    "is_ftp_login": 0,
                    "ct_ftp_cmd": 0,
                    "ct_flw_http_mthd": 0,
                    "ct_src_ltm": 1,
                    "ct_srv_dst": 1,
                    "is_sm_ips_ports": 0,
                },
            }
        ],
    }
    pred_resp = auth_client.post("/api/v1/predictions", json=payload)
    assert pred_resp.status_code == 200

    # 3. Sample non-alert predictions
    sample_resp = auth_client.get("/api/v1/predictions/sample")
    assert sample_resp.status_code == 200
    sample_data = sample_resp.json()
    assert "items" in sample_data

    # If there are normal flows in sample, submit review
    if sample_data["items"]:
        sample_item = sample_data["items"][0]
        fb_id = str(uuid4())
        fb_resp = auth_client.post(
            f"/api/v1/predictions/{sample_item['flow_id']}/feedback",
            json={
                "feedback_id": fb_id,
                "expected_version": 0,
                "verdict": "false_positive",
                "notes": "Reviewed non-alerted flow.",
            },
        )
        assert fb_resp.status_code == 200
        fb_data = fb_resp.json()
        assert fb_data["flow_id"] == sample_item["flow_id"]
        assert fb_data["verdict"] == "false_positive"


def test_replay_truth_and_summary_endpoint(auth_client):
    fid = str(uuid4())
    # 1. Record replay truth
    truth_resp = auth_client.post(
        "/api/v1/replay/truth",
        json={
            "items": [
                {"flow_id": fid, "label": 1, "attack_cat": "Generic"},
            ]
        },
    )
    assert truth_resp.status_code == 200
    assert truth_resp.json()["recorded"] == 1

    # 2. Get replay summary
    summary_resp = auth_client.get("/api/v1/replay/summary")
    assert summary_resp.status_code == 200
    summary = summary_resp.json()
    assert "total_replayed" in summary
    assert "confusion_matrix" in summary
    assert "false_alerts_per_1000_normal" in summary
    assert (
        summary["summary_note"]
        == "historical replay on evaluation partition, not independent confirmation"
    )


def test_api_scores_match_frozen_winner_for_submitted_values(auth_client):
    """Real input batches must produce checkpoint scores, never canned verdicts."""
    import joblib
    import numpy as np
    import pandas as pd

    frozen = joblib.load(PROJECT_ROOT / "models/lightgbm_validated_v1/binary/model.joblib")
    rows = json.loads((PROJECT_ROOT / "data/presets/mixed_100.json").read_text())
    columns = frozen.features.native.columns
    frame = pd.DataFrame(rows).loc[:, columns]
    expected = frozen.predict_proba(frame)[:, 1]
    flows = [
        {
            "flow_id": str(uuid4()),
            "event_time": "2026-10-04T10:00:00Z",
            "features": {key: row[key] for key in columns},
        }
        for row in rows
    ]
    response = auth_client.post(
        "/api/v1/predictions", json={"schema_version": "unsw-nb15.v0", "flows": flows}
    )
    assert response.status_code == 200, response.text
    result = response.json()
    scores = [item["score"] for item in result["predictions"]]
    np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-12)
    assert len(set(scores)) > 1
    assert {p["decision"] for p in result["predictions"]} == {"normal", "alert"}
    for flow, pred in zip(flows, result["predictions"], strict=True):
        assert pred["flow_id"] == flow["flow_id"]
        assert pred["threshold"] == frozen.decision_threshold_
        assert pred["decision"] == (
            "alert" if pred["score"] >= frozen.decision_threshold_ else "normal"
        )
        if pred["alert_id"]:
            alert = auth_client.get(f"/api/v1/alerts/{pred['alert_id']}").json()
            assert alert["source"] == "model"
            assert alert["score"] == pred["score"]
            assert alert["predicted_class"] == "Attack"
