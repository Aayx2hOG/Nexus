"""Prediction retry regressions."""

from dataclasses import replace
from uuid import uuid4

import pytest

from nexus.alerts import AlertData
from nexus.errors import APIError
from nexus.inference import FlowPrediction
from nexus.storage import AlertStore


def seed(store, score, *, bundle="test"):
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
    return pred, alerts, result


@pytest.fixture
def store(tmp_path):
    result = AlertStore(tmp_path / "retries.db")
    result.initialize()
    return result


def test_retry_returns_original_alert_without_duplicates(store):
    pred, alerts, initial = seed(store, 0.8)
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
