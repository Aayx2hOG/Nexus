import sqlite3
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from nexus.alerts import AlertData, FeedbackInput
from nexus.demo import demo_alerts
from nexus.errors import APIError

BASE = "/api/v1/alerts"


def test_empty_alerts(client):
    assert client.get(BASE).json() == {"items": [], "next_after": None}


def test_alert_detail_and_pagination(client, store):
    expected = [store.record_alert(item) for item in demo_alerts()]
    page = client.get(BASE, params={"limit": 1}).json()
    assert page["items"][0]["alert_id"] == expected[0].alert_id
    assert page["items"][0]["source"] == "mock"
    second = client.get(BASE, params={"limit": 1, "after": page["next_after"]}).json()
    assert second["items"][0]["alert_id"] == expected[1].alert_id
    assert second["next_after"] is None
    filtered = client.get(BASE, params={"severity": "critical"}).json()
    assert len(filtered["items"]) == 1
    assert filtered["items"][0]["severity"] == "critical"
    assert client.get(f"{BASE}/{expected[0].alert_id}").json() == expected[0].model_dump()


@pytest.mark.parametrize(
    "query",
    [
        "limit=0",
        "limit=101",
        "limit=1.0",
        "limit=true",
        "limit=01",
        "limit=+1",
        "after=-1",
        "after=1e2",
        "after=9007199254740992",
        "unknown=1",
        "severity=urgent",
    ],
)
def test_query_validation(client, query):
    assert client.get(f"{BASE}?{query}").status_code == 422


def test_duplicate_queries(client):
    assert client.get(f"{BASE}?limit=1&limit=2").status_code == 400
    assert client.get(f"{BASE}?limit=1&%6cimit=2").status_code == 400


@pytest.mark.parametrize("identifier,status", [("invalid", 422), (str(uuid4()), 404)])
def test_missing_and_invalid_alert(client, identifier, status):
    assert client.get(f"{BASE}/{identifier}").status_code == status
    assert client.get(f"{BASE}/{identifier}/feedback").status_code == status


def test_review_lifecycle(client, alert, feedback):
    url = f"{BASE}/{alert.alert_id}/feedback"
    first = client.post(url, json=feedback)
    assert first.status_code == 200
    record = first.json()
    assert record["reviewer_id"] == "test-analyst"
    assert record["provenance"] == "analyst_api"
    assert record["version"] == 1
    assert record["training_eligible"] is False
    assert client.post(url, json=feedback).json() == record
    second_input = {
        "feedback_id": str(uuid4()),
        "expected_version": 1,
        "verdict": "pending",
        "notes": "Needs another review.",
    }
    second = client.post(url, json=second_input)
    assert second.status_code == 200
    assert second.json()["version"] == 2
    assert client.get(f"{BASE}/{alert.alert_id}").json()["feedback_version"] == 2
    history = client.get(url, params={"limit": 1}).json()
    assert history["items"] == [record]
    tail = client.get(url, params={"after": history["next_after"]}).json()
    assert tail["items"] == [second.json()]
    # A delayed retry must not create another version or overwrite the latest review.
    assert client.post(url, json=feedback).json() == record
    assert len(client.get(url).json()["items"]) == 2


def test_stale_and_conflicting_feedback(client, alert, feedback):
    url = f"{BASE}/{alert.alert_id}/feedback"
    assert client.post(url, json=feedback).status_code == 200
    conflicting = feedback | {"notes": "Different content."}
    assert client.post(url, json=conflicting).json()["error"]["code"] == "feedback_conflict"
    stale = feedback | {"feedback_id": str(uuid4())}
    assert client.post(url, json=stale).json()["error"]["code"] == "stale_feedback"
    assert client.get(f"{BASE}/{alert.alert_id}").json()["feedback_version"] == 1
    assert len(client.get(url).json()["items"]) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"expected_version": True},
        {"expected_version": "0"},
        {"expected_version": -1},
        {"reviewer_id": "impersonated"},
        {"created_at": "2026-10-02T10:00:00Z"},
        {"training_eligible": True},
        {"verdict": "benign"},
        {"notes": " "},
        {"notes": "x" * 2001},
        {"notes": "hello\u0000"},
        {"notes": "\ud800"},
        {"attack_category": None},
        {"verdict": "pending"},
        {"verdict": "false_positive"},
        {"feedback_id": "invalid"},
    ],
)
def test_feedback_validation(client, alert, feedback, changes):
    import json

    response = client.post(
        f"{BASE}/{alert.alert_id}/feedback",
        content=json.dumps(feedback | changes),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert client.get(f"{BASE}/{alert.alert_id}").json()["feedback_version"] == 0


def test_false_positive_and_sql_text_are_stored_as_data(client, alert, feedback):
    feedback.update(
        verdict="false_positive", attack_category=None, notes="'); DROP TABLE alerts; --"
    )
    response = client.post(f"{BASE}/{alert.alert_id}/feedback", json=feedback)
    assert response.status_code == 200
    assert response.json()["notes"] == feedback["notes"]
    assert client.get(f"{BASE}/{alert.alert_id}").status_code == 200


def test_feedback_for_missing_alert(client, feedback):
    assert client.post(f"{BASE}/{uuid4()}/feedback", json=feedback).status_code == 404


def test_authentication(client, alert, feedback, app_factory):
    for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "Basic abc"}):
        with TestClient(client.app, headers=headers) as anonymous:
            for path in (BASE, f"{BASE}/{alert.alert_id}", f"{BASE}/{alert.alert_id}/feedback"):
                response = anonymous.get(path)
                assert response.status_code == 401
                assert response.headers["www-authenticate"] == "Bearer"
            assert (
                anonymous.post(f"{BASE}/{alert.alert_id}/feedback", json=feedback).status_code
                == 401
            )
    with TestClient(app_factory(token=None)) as disabled:
        assert disabled.get(BASE).status_code == 503


def test_duplicate_authorization(client):
    token = client.headers["authorization"]
    assert (
        client.get(BASE, headers=[("Authorization", token), ("Authorization", token)]).status_code
        == 401
    )


def test_alert_ingestion_is_idempotent(store):
    data = demo_alerts()[0]
    original = store.record_alert(data)
    assert store.record_alert(data) == original
    changed = AlertData.model_validate(data.model_dump() | {"risk": 0.5})
    with pytest.raises(APIError, match="different content"):
        store.record_alert(changed)
    assert store.get_alert(data.alert_id) == original


def test_persistence_across_app_restarts(client, alert, feedback, app_factory):
    client.post(f"{BASE}/{alert.alert_id}/feedback", json=feedback)
    with TestClient(app_factory(), headers=dict(client.headers)) as restarted:
        assert restarted.get(f"{BASE}/{alert.alert_id}").json()["feedback_version"] == 1
        assert len(restarted.get(f"{BASE}/{alert.alert_id}/feedback").json()["items"]) == 1


def test_concurrent_reviews_have_one_winner(store, alert, feedback):
    def submit(_):
        data = FeedbackInput.model_validate(feedback | {"feedback_id": str(uuid4())})
        try:
            return store.add_feedback(alert.alert_id, data, "test-analyst").version
        except APIError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(submit, range(2)))
    assert results.count(1) == 1
    assert results.count("stale_feedback") == 1
    assert store.get_alert(alert.alert_id).feedback_version == 1


def test_failed_review_rolls_back_history_and_version(client, store, alert, feedback):
    with store.connection(write=True) as db:
        db.execute(
            "CREATE TRIGGER fail_update BEFORE UPDATE ON alerts "
            "BEGIN SELECT RAISE(ABORT, 'simulated storage failure'); END"
        )
    response = client.post(f"{BASE}/{alert.alert_id}/feedback", json=feedback)
    assert response.status_code == 503
    assert "simulated storage failure" not in response.text
    assert client.get(f"{BASE}/{alert.alert_id}/feedback").json()["items"] == []
    assert store.get_alert(alert.alert_id).feedback_version == 0


def test_feedback_id_cannot_be_reused_for_another_alert(client, store, alert, feedback):
    other = store.record_alert(demo_alerts()[1])
    assert client.post(f"{BASE}/{alert.alert_id}/feedback", json=feedback).status_code == 200
    response = client.post(f"{BASE}/{other.alert_id}/feedback", json=feedback)
    assert response.status_code == 409
    assert store.get_alert(other.alert_id).feedback_version == 0


def test_storage_failure_does_not_leak_details(client, store, monkeypatch):
    def unavailable(query):
        raise sqlite3.OperationalError("private database path")

    monkeypatch.setattr(store, "list_alerts", unavailable)
    response = client.get(BASE)
    assert response.status_code == 503
    assert "private database path" not in response.text
