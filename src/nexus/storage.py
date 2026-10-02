"""SQLite persistence with atomic reviews and stable, idempotent ingestion."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

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

SCHEMA = """
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
            if version not in (0, 1):
                raise RuntimeError("Unsupported Nexus database schema version")
            db.execute("PRAGMA journal_mode = WAL")
            if version == 0:
                db.executescript("BEGIN IMMEDIATE;\n" + SCHEMA + "\nCOMMIT;")

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
