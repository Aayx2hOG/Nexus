"""Tests for prediction logging, storage schema migration, and replay truth."""

import sqlite3
from pathlib import Path
from uuid import uuid4

import pytest

from nexus.alerts import AlertData, FeatureContribution, FeedbackInput, PageQuery
from nexus.errors import APIError
from nexus.inference import FlowPrediction
from nexus.storage import SCHEMA_V1, Storage


def test_schema_migration_v1_to_v2(tmp_path: Path):
    db_path = tmp_path / "migrate.sqlite3"
    alert_id = str(uuid4())
    # Create a legacy v1 database
    with sqlite3.connect(db_path) as db:
        db.executescript(SCHEMA_V1)
        # Insert a sample alert into v1
        db.execute(
            """
            INSERT INTO alerts(alert_id, severity, payload, created_at)
            VALUES (?, 'high', '{}', '2026-10-01T00:00:00Z')
            """,
            (alert_id,),
        )

    # Initialize Storage, which should run migration to v2
    store = Storage(db_path)
    store.initialize()

    with store.connection() as db:
        v = db.execute("PRAGMA user_version").fetchone()[0]
        assert v == 3

        # Check prediction_log table exists
        cols = [r[1] for r in db.execute("PRAGMA table_info(prediction_log)").fetchall()]
        assert "flow_id" in cols
        assert "bundle_version" in cols
        assert "decision" in cols

        # Check legacy alert is preserved
        legacy_row = db.execute(
            "SELECT alert_id FROM alerts WHERE alert_id = ?", (alert_id,)
        ).fetchone()
        assert legacy_row is not None


def test_record_predictions_and_alerts_atomic(tmp_path: Path):
    db_path = tmp_path / "test.sqlite3"
    store = Storage(db_path)
    store.initialize()

    alert_id = str(uuid4())
    flow1_id = str(uuid4())
    flow2_id = str(uuid4())

    alert_data = AlertData(
        alert_id=alert_id,
        flow_id=flow1_id,
        event_time="2026-10-03T10:00:00Z",
        source="model",
        bundle_version="v1.0.0",
        predicted_class="Generic",
        score=0.85,
        threshold=0.577,
        severity="alert",
        top_features=[FeatureContribution(feature="sbytes", contribution=0.5, value=1200.0)],
    )

    predictions = [
        FlowPrediction(
            flow_id=flow1_id,
            event_time="2026-10-03T10:00:00Z",
            score=0.85,
            threshold=0.577,
            decision="alert",
            top_features=[FeatureContribution(feature="sbytes", contribution=0.5, value=1200.0)],
            raw_features={"sbytes": 1200},
        ),
        FlowPrediction(
            flow_id=flow2_id,
            event_time="2026-10-03T10:00:01Z",
            score=0.12,
            threshold=0.577,
            decision="normal",
            top_features=[],
            raw_features={"sbytes": 40},
        ),
    ]

    results = store.record_predictions_and_alerts(predictions, "v1.0.0", [alert_data])
    assert len(results) == 2
    assert results[0][1] is not None
    assert results[0][1].alert_id == alert_id
    assert results[1][1] is None

    # Verify rows in prediction_log
    with store.connection() as db:
        rows = db.execute(
            "SELECT flow_id, decision, alert_id FROM prediction_log ORDER BY sequence"
        ).fetchall()
        assert len(rows) == 2
        assert rows[0]["flow_id"] == flow1_id
        assert rows[0]["decision"] == "alert"
        assert rows[0]["alert_id"] == alert_id

        assert rows[1]["flow_id"] == flow2_id
        assert rows[1]["decision"] == "normal"
        assert rows[1]["alert_id"] is None


def test_prediction_idempotency_and_conflict(tmp_path: Path):
    db_path = tmp_path / "test_idempotent.sqlite3"
    store = Storage(db_path)
    store.initialize()

    flow_id = str(uuid4())
    pred = [
        FlowPrediction(
            flow_id=flow_id,
            event_time="2026-10-03T10:00:00Z",
            score=0.20,
            threshold=0.577,
            decision="normal",
            top_features=[],
            raw_features={"sbytes": 50},
        )
    ]

    # First insert succeeds
    store.record_predictions_and_alerts(pred, "v1.0.0", [])

    # Duplicate exact insert is idempotent (no-op)
    store.record_predictions_and_alerts(pred, "v1.0.0", [])

    # Conflicting insert (different decision/score for same flow_id and bundle_version)
    # raises conflict
    conflicting = [
        FlowPrediction(
            flow_id=flow_id,
            event_time="2026-10-03T10:00:00Z",
            score=0.99,
            threshold=0.577,
            decision="alert",
            top_features=[],
            raw_features={"sbytes": 50},
        )
    ]
    with pytest.raises(APIError) as exc_info:
        store.record_predictions_and_alerts(conflicting, "v1.0.0", [])
    assert exc_info.value.code == "prediction_conflict"


def test_stats_and_feedback(tmp_path: Path):
    db_path = tmp_path / "test_stats.sqlite3"
    store = Storage(db_path)
    store.initialize()

    alert_id = str(uuid4())
    flow1_id = str(uuid4())
    flow2_id = str(uuid4())

    alert_data = AlertData(
        alert_id=alert_id,
        flow_id=flow1_id,
        event_time="2026-10-03T10:00:00Z",
        source="model",
        bundle_version="v1.0.0",
        predicted_class="Exploits",
        score=0.92,
        threshold=0.577,
        severity="alert",
        top_features=[FeatureContribution(feature="sttl", contribution=0.3, value=64.0)],
    )

    preds = [
        FlowPrediction(
            flow_id=flow1_id,
            event_time="2026-10-03T10:00:00Z",
            score=0.92,
            threshold=0.577,
            decision="alert",
            top_features=[],
            raw_features={"sttl": 64},
        ),
        FlowPrediction(
            flow_id=flow2_id,
            event_time="2026-10-03T10:00:01Z",
            score=0.15,
            threshold=0.577,
            decision="normal",
            top_features=[],
            raw_features={"sttl": 64},
        ),
    ]

    store.record_predictions_and_alerts(preds, "v1.0.0", [alert_data])

    stats = store.get_stats()
    assert stats["alerts_24h"] == 1
    assert stats["awaiting_review"] == 1
    assert stats["total_predictions_24h"] == 2
    assert stats["confirmed_attacks"] == 0

    # Add feedback to the alert
    feedback = FeedbackInput(
        feedback_id=str(uuid4()),
        expected_version=0,
        verdict="confirmed_attack",
        attack_category="Exploits",
        notes="Validated exploit.",
    )
    store.add_feedback(alert_id, feedback, reviewer_id="analyst-1")

    stats = store.get_stats()
    assert stats["alerts_24h"] == 1
    assert stats["awaiting_review"] == 0
    assert stats["confirmed_attacks"] == 1


def test_sample_non_alert_and_feedback(tmp_path: Path):
    db_path = tmp_path / "test_sample.sqlite3"
    store = Storage(db_path)
    store.initialize()

    flow_ids = [str(uuid4()) for _ in range(5)]
    preds = [
        FlowPrediction(
            flow_id=flow_ids[i],
            event_time="2026-10-03T10:00:00Z",
            score=0.1 * i,
            threshold=0.577,
            decision="normal",
            top_features=[],
            raw_features={"dur": 0.05 * i},
        )
        for i in range(5)
    ]
    store.record_predictions_and_alerts(preds, "v1.0.0", [])

    # Sample non-alert predictions
    items, next_after = store.sample_non_alert_predictions(PageQuery(limit=3, after=0))
    assert len(items) == 3
    assert next_after is not None
    assert items[0]["flow_id"] == flow_ids[0]
    assert items[0]["review"] is None

    # Add sample feedback to flow_ids[0]
    fb = FeedbackInput(
        feedback_id=str(uuid4()),
        expected_version=0,
        verdict="pending",
        notes="Missed low-and-slow scan.",
    )
    fb_record = store.add_sample_feedback(flow_ids[0], fb, reviewer_id="senior-analyst")
    assert fb_record["flow_id"] == flow_ids[0]
    assert fb_record["reviewer_id"] == "senior-analyst"

    # Re-fetch sample and check review attached
    items, _ = store.sample_non_alert_predictions(PageQuery(limit=1, after=0))
    assert items[0]["review"] is not None
    assert items[0]["review"]["verdict"] == "pending"
    assert items[0]["review"]["reviewer_id"] == "senior-analyst"


def test_replay_truth_and_summary(tmp_path: Path):
    db_path = tmp_path / "test_replay.sqlite3"
    store = Storage(db_path)
    store.initialize()

    f1, f2, f3, f4 = str(uuid4()), str(uuid4()), str(uuid4()), str(uuid4())
    a1, a2 = str(uuid4()), str(uuid4())

    # 4 flows:
    # 1: pred=alert, actual=1 (TP)
    # 2: pred=alert, actual=0 (FP)
    # 3: pred=normal, actual=1 (FN)
    # 4: pred=normal, actual=0 (TN)
    preds = [
        FlowPrediction(
            flow_id=f1,
            event_time="2026-10-03T10:00:00Z",
            score=0.9,
            threshold=0.577,
            decision="alert",
            top_features=[],
            raw_features={},
        ),
        FlowPrediction(
            flow_id=f2,
            event_time="2026-10-03T10:00:00Z",
            score=0.8,
            threshold=0.577,
            decision="alert",
            top_features=[],
            raw_features={},
        ),
        FlowPrediction(
            flow_id=f3,
            event_time="2026-10-03T10:00:00Z",
            score=0.2,
            threshold=0.577,
            decision="normal",
            top_features=[],
            raw_features={},
        ),
        FlowPrediction(
            flow_id=f4,
            event_time="2026-10-03T10:00:00Z",
            score=0.1,
            threshold=0.577,
            decision="normal",
            top_features=[],
            raw_features={},
        ),
    ]
    alerts = [
        AlertData(
            alert_id=a1,
            flow_id=f1,
            event_time="2026-10-03T10:00:00Z",
            source="model",
            bundle_version="v1.0.0",
            predicted_class="Generic",
            score=0.9,
            threshold=0.577,
            severity="alert",
            top_features=[],
        ),
        AlertData(
            alert_id=a2,
            flow_id=f2,
            event_time="2026-10-03T10:00:00Z",
            source="model",
            bundle_version="v1.0.0",
            predicted_class="Generic",
            score=0.8,
            threshold=0.577,
            severity="alert",
            top_features=[],
        ),
    ]

    store.record_predictions_and_alerts(preds, "v1.0.0", alerts)

    # Record replay truth separately
    store.record_replay_truth(
        [
            (f1, 1, "Generic"),
            (f2, 0, "Normal"),
            (f3, 1, "Exploits"),
            (f4, 0, "Normal"),
        ]
    )

    summary = store.get_replay_summary()
    assert summary["total_replayed"] == 4
    assert summary["true_positives"] == 1
    assert summary["false_positives"] == 1
    assert summary["true_negatives"] == 1
    assert summary["false_negatives"] == 1
    assert summary["accuracy"] == 0.5
    assert summary["recall"] == 0.5
    assert summary["false_positive_rate"] == 0.5
    assert summary["false_alerts_per_1000_normal"] == 500.0
