import json
from copy import deepcopy

import pytest

from nexus.transport import RequestLimits

MAX_BODY_BYTES = RequestLimits().max_body_bytes


def test_health_and_schema(client):
    assert client.get("/health/live").json() == {"status": "alive"}
    response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}
    schema = client.get("/api/v1/schema").json()
    assert schema["additionalProperties"] is False
    assert len(schema["$defs"]["FlowFeatures"]["required"]) == 42
    assert client.get("/openapi.json").status_code == 200


def test_valid_batch_and_unavailable_model(client, payload):
    response = client.post("/api/v1/flows/validate", json=payload)
    assert response.status_code == 200
    assert response.json() == {
        "valid": True,
        "schema_version": "unsw-nb15.v0",
        "flow_count": 1,
        "model_compatible": False,
    }
    response = client.post("/api/v1/predictions", json=payload)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_unavailable"


@pytest.mark.parametrize("endpoint", ["flows/validate", "predictions"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("spkts", True),
        ("spkts", "3"),
        ("spkts", 3.0),
        ("spkts", -1),
        ("spkts", 2**53),
        ("dur", "0.1"),
        ("dur", False),
        ("dur", -0.1),
        ("dur", 1e16),
        ("dur", None),
        ("dur", []),
        ("dur", {}),
        ("sttl", 256),
        ("dttl", -1),
        ("swin", 65536),
        ("stcpb", 2**32),
        ("is_sm_ips_ports", 2),
        ("is_ftp_login", True),
        ("proto", ""),
        ("proto", " tcp"),
        ("proto", "tcp\n"),
        ("proto", "tсp"),
        ("proto", "a" * 33),
        ("proto", 6),
        ("state", "<script>"),
        ("label", 1),
        ("attack_cat", "Backdoor"),
        ("id", 7),
    ],
)
def test_strict_features(client, payload, endpoint, field, value):
    payload["flows"][0]["features"][field] = value
    response = client.post(f"/api/v1/{endpoint}", json=payload)
    assert response.status_code == 422
    assert "Backdoor" not in response.text


@pytest.mark.parametrize(
    "field,value",
    [
        ("flow_id", "not-a-uuid"),
        ("flow_id", "00000000-0000-0000-0000-000000000000"),
        ("flow_id", 123),
        ("event_time", "2026-10-02T10:00:00"),
        ("event_time", "2026-02-30T10:00:00Z"),
        ("event_time", 123456789),
        ("event_time", "2026-10-02 10:00:00Z"),
        ("event_time", "2026-10-02T10:00:00+25:00"),
        ("event_time", "2026-10-02T10:00:00+01:60"),
        ("unexpected", "secret"),
    ],
)
def test_flow_metadata(client, payload, field, value):
    payload["flows"][0][field] = value
    response = client.post("/api/v1/flows/validate", json=payload)
    assert response.status_code == 422
    assert "secret" not in response.text


@pytest.mark.parametrize(
    "mutation", ["empty", "too_many", "duplicate", "missing", "version", "extra"]
)
def test_batch_constraints(client, payload, mutation):
    if mutation == "empty":
        payload["flows"] = []
    elif mutation == "too_many":
        payload["flows"] *= 101
    elif mutation == "duplicate":
        payload["flows"].append(deepcopy(payload["flows"][0]))
    elif mutation == "missing":
        del payload["flows"][0]["features"]["dur"]
    elif mutation == "version":
        payload["schema_version"] = "unknown"
    else:
        payload["debug"] = True
    assert client.post("/api/v1/flows/validate", json=payload).status_code == 422


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"{",
        b"null",
        b"[]",
        b'"text"',
        b"\xff",
        b'{"x": NaN}',
        b'{"x": Infinity}',
        b'{"x": -Infinity}',
        b'{"x": 1e999}',
        b'{"x": 1, "x": 2}',
        b'{"features": {"dur": 1, "dur": 2}}',
        b'{"x":1,"\\u0078":2}',
        pytest.param(b'{"x":' + b"[" * 2000 + b"]" * 2000 + b"}", id="deep-nesting"),
    ],
)
def test_invalid_json(client, raw):
    response = client.post(
        "/api/v1/flows/validate", content=raw, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_json"


def test_content_and_body_limits(client, payload):
    url = "/api/v1/flows/validate"
    assert client.post(url, content=json.dumps(payload)).status_code == 415
    assert client.post(url, json=payload, headers={"Content-Encoding": "gzip"}).status_code == 415
    assert (
        client.post(
            url, content=b" " * (MAX_BODY_BYTES + 1), headers={"Content-Type": "application/json"}
        ).status_code
        == 413
    )
    assert client.post(url, json=payload, headers={"Content-Length": "1"}).status_code == 400
    assert client.post(url, json=payload, headers={"Content-Length": "-1"}).status_code == 400
    assert (
        client.post(url, json=payload, headers={"Transfer-Encoding": "chunked"}).status_code == 400
    )
    assert (
        client.post(
            url,
            json=payload,
            headers=[("Content-Type", "application/json"), ("Content-Type", "text/plain")],
        ).status_code
        == 400
    )
    assert client.post(url + "?debug=true", json=payload).status_code == 400
    assert client.request("GET", "/health/live", content=b"x").status_code == 400


def test_full_batch_and_numeric_boundaries(client, payload):
    from uuid import UUID

    feature = payload["flows"][0]["features"]
    feature.update(dur=0, sttl=255, swin=65535, stcpb=2**32 - 1, spkts=2**53 - 1)
    original = payload["flows"][0]
    payload["flows"] = []
    for number in range(1, 101):
        flow = deepcopy(original)
        flow["flow_id"] = str(UUID(int=number))
        payload["flows"].append(flow)
    response = client.post("/api/v1/flows/validate", json=payload)
    assert response.status_code == 200
    assert response.json()["flow_count"] == 100
