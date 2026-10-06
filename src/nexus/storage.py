"""SQLite persistence with atomic reviews, prediction logging, and stable, idempotent ingestion."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from nexus.alerts import (
    AlertData,
    AlertPage,
    AlertQuery,
    AlertRecord,
    FeedbackInput,
    FeedbackPage,
    FeedbackRecord,
    PageQuery,
)
from nexus.errors import APIError
from nexus.inference import FlowPrediction

SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS alerts (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id TEXT NOT NULL UNIQUE,
    severity TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    feedback_version INTEGER NOT NULL DEFAULT 0 CHECK (feedback_version >= 0)
);
CREATE INDEX IF NOT EXISTS alerts_severity_sequence ON alerts(severity, sequence);

CREATE TABLE IF NOT EXISTS feedback (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    feedback_id TEXT NOT NULL UNIQUE,
    alert_id TEXT NOT NULL REFERENCES alerts(alert_id),
    version INTEGER NOT NULL CHECK (version > 0),
    request TEXT NOT NULL,
    reviewer_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(alert_id, version)
);
CREATE INDEX IF NOT EXISTS feedback_alert_sequence ON feedback(alert_id, sequence);

PRAGMA user_version = 1;
"""

SCHEMA_V2 = """
CREATE TABLE IF NOT EXISTS alerts (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id TEXT NOT NULL UNIQUE,
    severity TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    feedback_version INTEGER NOT NULL DEFAULT 0 CHECK (feedback_version >= 0)
);
CREATE INDEX IF NOT EXISTS alerts_severity_sequence ON alerts(severity, sequence);

CREATE TABLE IF NOT EXISTS feedback (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    feedback_id TEXT NOT NULL UNIQUE,
    alert_id TEXT NOT NULL REFERENCES alerts(alert_id),
    version INTEGER NOT NULL CHECK (version > 0),
    request TEXT NOT NULL,
    reviewer_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(alert_id, version)
);
CREATE INDEX IF NOT EXISTS feedback_alert_sequence ON feedback(alert_id, sequence);

CREATE TABLE IF NOT EXISTS prediction_log (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    flow_id TEXT NOT NULL,
    bundle_version TEXT NOT NULL,
    event_time TEXT,
    ingest_time TEXT NOT NULL,
    features TEXT NOT NULL,
    score REAL NOT NULL,
    threshold REAL NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('alert', 'normal')),
    alert_id TEXT REFERENCES alerts(alert_id),
    UNIQUE(flow_id, bundle_version)
);
CREATE INDEX IF NOT EXISTS prediction_log_decision_seq ON prediction_log(decision, sequence);
CREATE INDEX IF NOT EXISTS prediction_log_ingest ON prediction_log(ingest_time);

CREATE TABLE IF NOT EXISTS replay_truth (
    flow_id TEXT PRIMARY KEY,
    label INTEGER NOT NULL CHECK (label IN (0, 1)),
    attack_cat TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sample_feedback (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    feedback_id TEXT NOT NULL UNIQUE,
    flow_id TEXT NOT NULL,
    verdict TEXT NOT NULL CHECK (
        verdict IN ('confirmed_attack', 'false_positive', 'needs_investigation', 'pending')
    ),
    attack_category TEXT,
    notes TEXT,
    reviewer_id TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS sample_feedback_flow ON sample_feedback(flow_id);

PRAGMA user_version = 2;
"""

MIGRATION_V1_TO_V2 = """
CREATE TABLE IF NOT EXISTS prediction_log (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    flow_id TEXT NOT NULL,
    bundle_version TEXT NOT NULL,
    event_time TEXT,
    ingest_time TEXT NOT NULL,
    features TEXT NOT NULL,
    score REAL NOT NULL,
    threshold REAL NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('alert', 'normal')),
    alert_id TEXT REFERENCES alerts(alert_id),
    UNIQUE(flow_id, bundle_version)
);
CREATE INDEX IF NOT EXISTS prediction_log_decision_seq ON prediction_log(decision, sequence);
CREATE INDEX IF NOT EXISTS prediction_log_ingest ON prediction_log(ingest_time);

CREATE TABLE IF NOT EXISTS replay_truth (
    flow_id TEXT PRIMARY KEY,
    label INTEGER NOT NULL CHECK (label IN (0, 1)),
    attack_cat TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sample_feedback (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    feedback_id TEXT NOT NULL UNIQUE,
    flow_id TEXT NOT NULL,
    verdict TEXT NOT NULL CHECK (
        verdict IN ('confirmed_attack', 'false_positive', 'needs_investigation', 'pending')
    ),
    attack_category TEXT,
    notes TEXT,
    reviewer_id TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS sample_feedback_flow ON sample_feedback(flow_id);

PRAGMA user_version = 2;
"""


MIGRATION_V2_TO_V3 = """
CREATE TABLE shadow_predictions (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    flow_id TEXT NOT NULL,
    production_bundle_version TEXT NOT NULL,
    shadow_bundle_version TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('scored', 'error')),
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(flow_id, production_bundle_version, manifest_sha256),
    FOREIGN KEY(flow_id, production_bundle_version)
        REFERENCES prediction_log(flow_id, bundle_version)
);
CREATE INDEX shadow_predictions_cohort
    ON shadow_predictions(production_bundle_version, manifest_sha256, sequence);
PRAGMA user_version = 3;
"""


def now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def alert_record(row: sqlite3.Row) -> AlertRecord:
    return AlertRecord(
        **AlertData.model_validate_json(row["payload"]).model_dump(),
        sequence=row["sequence"],
        created_at=row["created_at"],
        feedback_version=row["feedback_version"],
    )


def feedback_record(row: sqlite3.Row) -> FeedbackRecord:
    data = FeedbackInput.model_validate_json(row["request"])
    return FeedbackRecord(
        **data.model_dump(exclude={"expected_version"}),
        sequence=row["sequence"],
        alert_id=row["alert_id"],
        version=row["version"],
        reviewer_id=row["reviewer_id"],
        created_at=row["created_at"],
    )


class AlertStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def connection(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys = ON")
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            if write:
                db.commit()
        except Exception:
            if write:
                db.rollback()
            raise
        finally:
            db.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1, 2, 3):
                raise RuntimeError("Unsupported Nexus database schema version")
            db.execute("PRAGMA journal_mode = WAL")
            if version == 0:
                db.executescript("BEGIN IMMEDIATE;\n" + SCHEMA_V2 + "\nCOMMIT;")
            elif version == 1:
                db.executescript("BEGIN IMMEDIATE;\n" + MIGRATION_V1_TO_V2 + "\nCOMMIT;")
            if version < 3:
                db.executescript("BEGIN IMMEDIATE;\n" + MIGRATION_V2_TO_V3 + "\nCOMMIT;")

    @staticmethod
    def _find_alert(db: sqlite3.Connection, alert_id: str) -> sqlite3.Row:
        row = db.execute("SELECT * FROM alerts WHERE alert_id = ?", (alert_id,)).fetchone()
        if row is None:
            raise APIError(404, "alert_not_found", "Alert does not exist.")
        return row

    def record_alert(self, alert: AlertData) -> AlertRecord:
        """An identical retry is safe; a conflicting reuse of an ID is rejected."""
        payload = alert.model_dump_json()
        with self.connection(write=True) as db:
            db.execute(
                "INSERT INTO alerts(alert_id, severity, payload, created_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(alert_id) DO NOTHING",
                (alert.alert_id, alert.severity, payload, now()),
            )
            row = self._find_alert(db, alert.alert_id)
            if row["payload"] != payload:
                raise APIError(409, "alert_conflict", "Alert ID already has different content.")
            return alert_record(row)

    def get_alert(self, alert_id: str) -> AlertRecord:
        with self.connection() as db:
            return alert_record(self._find_alert(db, alert_id))

    def list_alerts(self, query: AlertQuery) -> AlertPage:
        sql = "SELECT * FROM alerts WHERE sequence > ?"
        params: list[object] = [query.after]
        if query.severity is not None:
            sql += " AND severity = ?"
            params.append(query.severity)
        sql += " ORDER BY sequence LIMIT ?"
        params.append(query.limit + 1)
        with self.connection() as db:
            rows = db.execute(sql, params).fetchall()
        items = [alert_record(row) for row in rows[: query.limit]]
        return AlertPage(
            items=items,
            next_after=items[-1].sequence if len(rows) > query.limit else None,
        )

    def add_feedback(
        self,
        alert_id: str,
        feedback: FeedbackInput,
        reviewer_id: str,
    ) -> FeedbackRecord:
        request = feedback.model_dump_json()
        with self.connection(write=True) as db:
            alert = self._find_alert(db, alert_id)
            existing = db.execute(
                "SELECT * FROM feedback WHERE feedback_id = ?",
                (feedback.feedback_id,),
            ).fetchone()
            if existing is not None:
                if (existing["alert_id"], existing["request"], existing["reviewer_id"]) != (
                    alert_id,
                    request,
                    reviewer_id,
                ):
                    raise APIError(
                        409, "feedback_conflict", "Feedback ID already has different content."
                    )
                return feedback_record(existing)
            if feedback.expected_version != alert["feedback_version"]:
                raise APIError(
                    409, "stale_feedback", "Reload the alert before submitting another review."
                )
            version = alert["feedback_version"] + 1
            db.execute(
                "INSERT INTO feedback "
                "(feedback_id, alert_id, version, request, reviewer_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (feedback.feedback_id, alert_id, version, request, reviewer_id, now()),
            )
            db.execute(
                "UPDATE alerts SET feedback_version = ? WHERE alert_id = ?", (version, alert_id)
            )
            row = db.execute(
                "SELECT * FROM feedback WHERE feedback_id = ?",
                (feedback.feedback_id,),
            ).fetchone()
            return feedback_record(row)

    def list_feedback(self, alert_id: str, query: PageQuery) -> FeedbackPage:
        with self.connection() as db:
            self._find_alert(db, alert_id)
            rows = db.execute(
                "SELECT * FROM feedback WHERE alert_id = ? AND sequence > ? "
                "ORDER BY sequence LIMIT ?",
                (alert_id, query.after, query.limit + 1),
            ).fetchall()
        items = [feedback_record(row) for row in rows[: query.limit]]
        return FeedbackPage(
            items=items,
            next_after=items[-1].sequence if len(rows) > query.limit else None,
        )

    def record_predictions_and_alerts(
        self,
        predictions: list[FlowPrediction],
        bundle_version: str,
        alerts: list[AlertData],
    ) -> list[tuple[FlowPrediction, AlertRecord | None]]:
        """Atomically persist prediction logs and any created alerts in a single transaction.

        Idempotent on (flow_id, bundle_version). Returns list of (prediction, alert_record).
        """
        results: list[tuple[FlowPrediction, AlertRecord | None]] = []
        ingest_timestamp = now()

        # Map alerts by flow_id
        alert_map: dict[str, AlertData] = {a.flow_id: a for a in alerts}

        with self.connection(write=True) as db:
            # Resolve retries before creating alerts, including older random alert IDs.
            retried: dict[str, AlertRecord | None] = {}
            for pred in predictions:
                existing = db.execute(
                    "SELECT * FROM prediction_log WHERE flow_id = ? AND bundle_version = ?",
                    (pred.flow_id, bundle_version),
                ).fetchone()
                if existing is None:
                    continue
                if (
                    existing["decision"] != pred.decision
                    or abs(existing["score"] - pred.score) > 1e-6
                    or existing["threshold"] != pred.threshold
                    or existing["event_time"] != pred.event_time
                    or json.loads(existing["features"]) != pred.raw_features
                ):
                    raise APIError(409, "prediction_conflict", "Flow ID has different content.")
                retried[pred.flow_id] = (
                    alert_record(self._find_alert(db, existing["alert_id"]))
                    if existing["alert_id"]
                    else None
                )
            # 1. Insert alerts
            created_alert_records: dict[str, AlertRecord] = {}
            for flow_id, alert in alert_map.items():
                if flow_id in retried:
                    continue
                payload = alert.model_dump_json()
                db.execute(
                    "INSERT INTO alerts(alert_id, severity, payload, created_at) "
                    "VALUES (?, ?, ?, ?) ON CONFLICT(alert_id) DO NOTHING",
                    (alert.alert_id, alert.severity, payload, ingest_timestamp),
                )
                row = self._find_alert(db, alert.alert_id)
                if row["payload"] != payload:
                    raise APIError(409, "alert_conflict", "Alert ID already has different content.")
                created_alert_records[flow_id] = alert_record(row)

            # 2. Insert prediction log entries
            for pred in predictions:
                if pred.flow_id in retried:
                    results.append((pred, retried[pred.flow_id]))
                    continue
                feat_json = json.dumps(pred.raw_features)
                alert_rec = created_alert_records.get(pred.flow_id)
                alert_id = alert_rec.alert_id if alert_rec else None

                db.execute(
                    "INSERT INTO prediction_log(flow_id, bundle_version, event_time, "
                    "ingest_time, features, score, threshold, decision, alert_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(flow_id, bundle_version) DO NOTHING",
                    (
                        pred.flow_id,
                        bundle_version,
                        pred.event_time,
                        ingest_timestamp,
                        feat_json,
                        pred.score,
                        pred.threshold,
                        pred.decision,
                        alert_id,
                    ),
                )
                existing = db.execute(
                    "SELECT score, decision FROM prediction_log "
                    "WHERE flow_id = ? AND bundle_version = ?",
                    (pred.flow_id, bundle_version),
                ).fetchone()
                if existing is not None and (
                    existing["decision"] != pred.decision
                    or abs(existing["score"] - pred.score) > 1e-6
                ):
                    raise APIError(
                        409,
                        "prediction_conflict",
                        f"Flow {pred.flow_id} already has a conflicting prediction "
                        f"in bundle {bundle_version}.",
                    )
                results.append((pred, alert_rec))

        return results

    def record_replay_truth(self, truths: list[tuple[str, int, str]]) -> None:
        """Store ground-truth labels for replayed flows in a separate replay_truth table."""
        if not truths:
            return
        timestamp = now()
        with self.connection(write=True) as db:
            db.executemany(
                "INSERT INTO replay_truth(flow_id, label, attack_cat, recorded_at) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(flow_id) DO NOTHING",
                [(fid, lbl, cat, timestamp) for fid, lbl, cat in truths],
            )

    def get_stats(self) -> dict[str, Any]:
        """Aggregate alert and prediction statistics for the dashboard."""
        with self.connection() as db:
            cutoff = (datetime.now(UTC) - timedelta(days=1)).isoformat().replace("+00:00", "Z")

            alerts_24h = db.execute(
                "SELECT COUNT(*) FROM alerts WHERE created_at >= ?", (cutoff,)
            ).fetchone()[0]

            awaiting_review = db.execute(
                "SELECT COUNT(*) FROM alerts WHERE feedback_version = 0"
            ).fetchone()[0]

            # Latest feedback verdict counts across all reviewed alerts
            verdict_rows = db.execute(
                """
                SELECT json_extract(request, '$.verdict') as verdict, COUNT(*)
                FROM feedback f1
                WHERE version = (
                    SELECT MAX(version) FROM feedback f2 WHERE f2.alert_id = f1.alert_id
                )
                GROUP BY verdict
                """
            ).fetchall()

            verdict_counts: dict[str, int] = {
                "confirmed_attack": 0,
                "false_positive": 0,
                "needs_investigation": 0,
            }
            for row in verdict_rows:
                v = row[0]
                if v in verdict_counts:
                    verdict_counts[v] += row[1]
                elif v == "pending":
                    verdict_counts["needs_investigation"] += row[1]

            total_predictions_24h = db.execute(
                "SELECT COUNT(*) FROM prediction_log WHERE ingest_time >= ?", (cutoff,)
            ).fetchone()[0]

            return {
                "alerts_24h": alerts_24h,
                "awaiting_review": awaiting_review,
                "confirmed_attacks": verdict_counts["confirmed_attack"],
                "false_positives": verdict_counts["false_positive"],
                "needs_investigation": verdict_counts["needs_investigation"],
                "total_predictions_24h": total_predictions_24h,
            }

    def sample_non_alert_predictions(
        self, query: PageQuery
    ) -> tuple[list[dict[str, Any]], int | None]:
        """Fetch sequential sample of non-alerted flows for missed-attack review."""
        with self.connection() as db:
            rows = db.execute(
                """
                SELECT p.sequence, p.flow_id, p.bundle_version, p.event_time, p.ingest_time,
                       p.features, p.score, p.threshold, p.decision,
                       sf.verdict, sf.attack_category, sf.notes, sf.reviewer_id,
                       sf.created_at as reviewed_at
                FROM prediction_log p
                LEFT JOIN sample_feedback sf ON p.flow_id = sf.flow_id
                WHERE p.decision = 'normal' AND p.sequence > ?
                ORDER BY p.sequence LIMIT ?
                """,
                (query.after, query.limit + 1),
            ).fetchall()

            items = []
            for row in rows[: query.limit]:
                review = None
                if row["verdict"] is not None:
                    review = {
                        "verdict": row["verdict"],
                        "attack_category": row["attack_category"],
                        "notes": row["notes"],
                        "reviewer_id": row["reviewer_id"],
                        "created_at": row["reviewed_at"],
                    }
                items.append(
                    {
                        "sequence": row["sequence"],
                        "flow_id": row["flow_id"],
                        "bundle_version": row["bundle_version"],
                        "event_time": row["event_time"],
                        "ingest_time": row["ingest_time"],
                        "features": json.loads(row["features"]),
                        "score": row["score"],
                        "threshold": row["threshold"],
                        "decision": row["decision"],
                        "review": review,
                    }
                )
            next_after = items[-1]["sequence"] if len(rows) > query.limit else None
            return items, next_after

    def add_sample_feedback(
        self,
        flow_id: str,
        feedback: FeedbackInput,
        reviewer_id: str,
    ) -> dict[str, Any]:
        """Record analyst verdict on a non-alerted flow (missed-attack review)."""
        with self.connection(write=True) as db:
            # Check flow exists in prediction_log
            flow_row = db.execute(
                "SELECT * FROM prediction_log WHERE flow_id = ?", (flow_id,)
            ).fetchone()
            if flow_row is None:
                raise APIError(404, "flow_not_found", "Flow not found in prediction log.")

            existing = db.execute(
                "SELECT * FROM sample_feedback WHERE feedback_id = ?", (feedback.feedback_id,)
            ).fetchone()
            if existing is not None:
                return {
                    "sequence": existing["sequence"],
                    "feedback_id": existing["feedback_id"],
                    "flow_id": existing["flow_id"],
                    "verdict": existing["verdict"],
                    "attack_category": existing["attack_category"],
                    "notes": existing["notes"],
                    "reviewer_id": existing["reviewer_id"],
                    "created_at": existing["created_at"],
                }

            created_timestamp = now()
            db.execute(
                "INSERT INTO sample_feedback(feedback_id, flow_id, verdict, attack_category, "
                "notes, reviewer_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    feedback.feedback_id,
                    flow_id,
                    feedback.verdict,
                    feedback.attack_category,
                    feedback.notes,
                    reviewer_id,
                    created_timestamp,
                ),
            )
            row = db.execute(
                "SELECT * FROM sample_feedback WHERE feedback_id = ?", (feedback.feedback_id,)
            ).fetchone()
            return {
                "sequence": row["sequence"],
                "feedback_id": row["feedback_id"],
                "flow_id": row["flow_id"],
                "verdict": row["verdict"],
                "attack_category": row["attack_category"],
                "notes": row["notes"],
                "reviewer_id": row["reviewer_id"],
                "created_at": row["created_at"],
            }

    def record_shadow(self, production_version: str, report: dict) -> list[dict]:
        """Append shadow evidence only after live predictions commit; first result wins."""
        results = []
        with self.connection(write=True) as db:
            for result in report["results"]:
                db.execute(
                    "INSERT INTO shadow_predictions(flow_id, production_bundle_version, "
                    "shadow_bundle_version, manifest_sha256, status, payload, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT "
                    "(flow_id, production_bundle_version, manifest_sha256) DO NOTHING",
                    (
                        result["flow_id"],
                        production_version,
                        report["bundle_version"],
                        report["manifest_sha256"],
                        result["status"],
                        json.dumps(result, allow_nan=False),
                        now(),
                    ),
                )
                row = db.execute(
                    "SELECT payload FROM shadow_predictions WHERE flow_id = ? "
                    "AND production_bundle_version = ? AND manifest_sha256 = ?",
                    (result["flow_id"], production_version, report["manifest_sha256"]),
                ).fetchone()
                results.append(json.loads(row["payload"]))
        return results

    def list_shadow(self, production_version: str, manifest_sha256: str, query: PageQuery) -> dict:
        with self.connection() as db:
            rows = db.execute(
                "SELECT s.sequence, s.payload, s.created_at, p.decision AS live_decision "
                "FROM shadow_predictions s JOIN prediction_log p ON s.flow_id = p.flow_id "
                "AND s.production_bundle_version = p.bundle_version "
                "WHERE s.production_bundle_version = ? AND s.manifest_sha256 = ? "
                "AND s.sequence > ? ORDER BY s.sequence LIMIT ?",
                (production_version, manifest_sha256, query.after, query.limit + 1),
            ).fetchall()
        items = [
            {
                **json.loads(row["payload"]),
                "sequence": row["sequence"],
                "created_at": row["created_at"],
                "live_decision": row["live_decision"],
            }
            for row in rows[: query.limit]
        ]
        return {
            "items": items,
            "next_after": items[-1]["sequence"] if len(rows) > query.limit else None,
        }

    def shadow_counts(self, production_version: str, manifest_sha256: str) -> list[dict]:
        with self.connection() as db:
            rows = db.execute(
                """
                SELECT s.status, p.decision AS live,
                    json_extract(s.payload, '$.reference_decision') AS reference,
                    json_extract(s.payload, '$.decision') AS candidate,
                    t.label, t.attack_cat, COUNT(*) AS n
                FROM prediction_log p
                LEFT JOIN shadow_predictions s ON s.flow_id = p.flow_id
                    AND s.production_bundle_version = p.bundle_version
                    AND s.manifest_sha256 = ?
                LEFT JOIN replay_truth t ON t.flow_id = p.flow_id
                WHERE p.bundle_version = ?
                GROUP BY s.status, live, reference, candidate, t.label, t.attack_cat
                """,
                (manifest_sha256, production_version),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_policy_counts(self, bundle_version: str, threshold: float) -> list[dict[str, Any]]:
        """Aggregate a single consistent replay snapshot without loading raw features."""
        with self.connection() as db:
            rows = db.execute(
                """
                SELECT t.label, t.attack_cat, p.decision AS baseline,
                       (p.score >= ?) AS candidate, COUNT(*) AS n,
                       MAX(p.sequence) AS last_sequence
                FROM prediction_log p LEFT JOIN replay_truth t ON p.flow_id = t.flow_id
                WHERE p.bundle_version = ?
                GROUP BY t.label, t.attack_cat, p.decision, candidate
                """,
                (threshold, bundle_version),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_replay_summary(self) -> dict[str, Any]:
        """Compute empirical replay confusion matrix and FPR vs replay_truth."""
        with self.connection() as db:
            rows = db.execute(
                """
                SELECT p.decision, t.label
                FROM prediction_log p
                INNER JOIN replay_truth t ON p.flow_id = t.flow_id
                """
            ).fetchall()

            tp = fp = tn = fn = 0
            for r in rows:
                pred = 1 if r["decision"] == "alert" else 0
                actual = int(r["label"])
                if pred == 1 and actual == 1:
                    tp += 1
                elif pred == 1 and actual == 0:
                    fp += 1
                elif pred == 0 and actual == 0:
                    tn += 1
                elif pred == 0 and actual == 1:
                    fn += 1

            total = tp + fp + tn + fn
            accuracy = (tp + tn) / total if total > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            false_alerts_per_1000 = fpr * 1000.0

            return {
                "total_replayed": total,
                "confusion_matrix": [[tn, fp], [fn, tp]],
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
                "accuracy": accuracy,
                "recall": recall,
                "false_positive_rate": fpr,
                "false_alerts_per_1000_normal": false_alerts_per_1000,
            }


Storage = AlertStore
