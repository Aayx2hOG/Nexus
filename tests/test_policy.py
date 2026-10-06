"""Counterfactual accounting, cohort isolation, and prediction retry regressions."""

from dataclasses import replace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from nexus.alerts import AlertData
from nexus.api import create_app
from nexus.config import Settings
from nexus.errors import APIError
from nexus.inference import FlowPrediction
from nexus.policy import PolicyQuery, policy_lab
from nexus.storage import AlertStore


def seed(store, score, label, *, bundle="test", family="Exploits"):
    pred = FlowPrediction(
        str(uuid4()),
        "2026-10-06T00:00:00Z",
        score,
        0.5,
        "alert" if score >= 0.5 else "normal",
        [],
        {"sbytes": 10},
    )
    alerts = []
    if pred.decision == "alert":
        alerts = [
            AlertData(
                alert_id=str(uuid4()),
                flow_id=pred.flow_id,
                event_time=pred.event_time,
                source="model",
                bundle_version=bundle,
                predicted_class="Attack",
                score=score,
                threshold=0.5,
                severity="alert",
                top_features=[],
            )
        ]
    result = store.record_predictions_and_alerts([pred], bundle, alerts)
    if label is not None:
        store.record_replay_truth([(pred.flow_id, label, family if label else "Normal")])
    return pred, alerts, result


@pytest.fixture
def store(tmp_path):
    result = AlertStore(tmp_path / "policy.db")
    result.initialize()
    return result


def test_counterfactual_costs_and_cohort_isolation(store):
    for score, label in [(0.8, 1), (0.4, 1), (0.4, 0), (0.1, 0), (0.9, None)]:
        seed(store, score, label)
    seed(store, 0.9, 0, bundle="other")
    result = policy_lab(store, PolicyQuery(bundle_version="test", threshold=0.4))
    assert result["total_predictions"] == 5
    assert result["labeled_predictions"] == 4
    assert result["unlabeled_predictions"] == 1
    assert result["baseline"]["recall"] == 0.5
    assert result["candidate"]["recall"] == 1
    assert result["candidate"]["false_positive_rate"] == 0.5
    assert result["recovered_attacks"] == result["added_false_positives"] == 1
    assert result["lost_attacks"] == result["removed_false_positives"] == 0
    assert result["families"] == [
        {"family": "Exploits", "total": 2, "baseline_detected": 1, "candidate_detected": 2}
    ]
    high = policy_lab(store, PolicyQuery(bundle_version="test", threshold=0.9))
    assert high["lost_attacks"] == 1
    assert high["candidate"]["precision"] is None
    with store.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM alerts").fetchone()[0] == 3


def test_no_data_and_absent_denominators(store):
    result = policy_lab(store, PolicyQuery(bundle_version="empty", threshold=0.5))
    assert result["total_predictions"] == 0
    assert result["baseline"]["recall"] is None
    assert result["candidate"]["false_positive_rate"] is None
    seed(store, 0.5, 1)
    result = policy_lab(store, PolicyQuery(bundle_version="test", threshold=0.5))
    assert result["candidate"]["recall"] == 1
    assert result["candidate"]["false_positive_rate"] is None


def test_retry_returns_original_alert_without_duplicates(store):
    pred, alerts, initial = seed(store, 0.8, 1)
    retry = alerts[0].model_copy(update={"alert_id": str(uuid4())})
    result = store.record_predictions_and_alerts([pred], "test", [retry])
    assert result[0][1].alert_id == initial[0][1].alert_id
    with store.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM alerts").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM prediction_log").fetchone()[0] == 1
    for changed in [
        replace(pred, raw_features={"sbytes": 11}),
        replace(pred, event_time="2026-10-06T01:00:00Z"),
        replace(pred, threshold=0.6),
    ]:
        with pytest.raises(APIError) as exc:
            store.record_predictions_and_alerts([changed], "test", [retry])
        assert exc.value.code == "prediction_conflict"


def test_policy_auth_and_query_validation(tmp_path):
    token = "a" * 32
    app = create_app(Settings(database_path=tmp_path / "api.db", api_token=token))
    with TestClient(app) as client:
        path = "/api/v1/replay/policy?bundle_version=test&threshold=0.5"
        assert client.get(path).status_code == 401
        client.headers["Authorization"] = f"Bearer {token}"
        assert client.get(path).status_code == 200
        for suffix in ["&threshold=0.2", "&unexpected=1"]:
            assert client.get(path + suffix).status_code in (400, 422)
        for value in ["nan", "inf", "-0.1", "1.1"]:
            assert client.get(path.replace("0.5", value)).status_code == 422
