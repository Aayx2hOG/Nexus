"""Shadow non-interference, artifact integrity, evidence accounting and migrations."""

import json
import shutil
import sqlite3
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from nexus.api import create_app
from nexus.config import Settings
from nexus.replay import row_to_flow
from nexus.shadow import ShadowService, load_shadow
from nexus.shadow_routes import summarize
from nexus.storage import SCHEMA_V2, AlertStore

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "artifacts/shadow_bundles/selective-exploits-s42-b03-v2"
TOKEN = "shadow-tests-" + "a" * 32


@pytest.fixture(scope="module")
def frozen():
    if not BUNDLE.exists():
        pytest.skip("Export the local parity-verified shadow bundle first")
    return load_shadow(BUNDLE)


@pytest.fixture
def payload():
    rows = json.loads((ROOT / "data/presets/mixed_100.json").read_text())
    flows = [row_to_flow(row, str(uuid4()), "2026-10-06T00:00:00Z") for row in rows]
    return flows, rows


@pytest.fixture
def client(tmp_path, frozen):
    app = create_app(
        Settings(
            database_path=tmp_path / "shadow.db",
            api_token=TOKEN,
            bundles_dir=ROOT / "artifacts/bundles",
            bundle_version="v1.0.0",
        )
    )
    with TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"}) as c:
        app.state.shadow_service.bundle = frozen
        yield c, app


def submit(c, flows):
    return c.post(
        "/api/v1/predictions",
        json={
            "schema_version": "unsw-nb15.v0",
            "flows": [f.model_dump(mode="json") for f in flows],
        },
    )


def test_live_results_unchanged_and_shadow_durable(client, payload):
    c, app = client
    flows, truth = payload
    # Compute the authoritative response with shadow disabled, then retry with shadow enabled.
    shadow = app.state.shadow_service.bundle
    app.state.shadow_service.bundle = None
    before = submit(c, flows).json()
    assert before["shadow"]["status"] == "disabled"
    app.state.shadow_service.bundle = shadow
    response = submit(c, flows)
    assert response.status_code == 200
    result = response.json()
    assert result["predictions"] == before["predictions"]
    assert result["alert_count"] == before["alert_count"]
    assert result["shadow"]["status"] == "scored"
    assert result["shadow"]["persisted"] is True
    assert result["shadow"]["manifest_sha256"] == shadow.manifest_sha256
    retry = submit(c, flows).json()
    assert retry["shadow"]["results"] == result["shadow"]["results"]
    assert all(
        row["decision"] == "alert"
        for row in result["shadow"]["results"]
        if row["reference_decision"] == "alert"
    )
    with app.state.store.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM alerts").fetchone()[0] == before["alert_count"]
        assert db.execute("SELECT COUNT(*) FROM shadow_predictions").fetchone()[0] == 100
    page = c.get("/api/v1/shadow/predictions?limit=20").json()
    assert len(page["items"]) == 20
    next_page = c.get(f"/api/v1/shadow/predictions?limit=20&after={page['next_after']}").json()
    assert set(i["flow_id"] for i in page["items"]).isdisjoint(
        i["flow_id"] for i in next_page["items"]
    )
    summary = c.get("/api/v1/shadow").json()
    assert summary["scored"] == 100 and summary["labeled_scored"] == 0
    assert summary["metrics"]["candidate"]["recall"] is None
    assert (
        c.post(
            "/api/v1/replay/truth",
            json={
                "items": [
                    {"flow_id": f.flow_id, "label": t["label"], "attack_cat": t["attack_cat"]}
                    for f, t in zip(flows, truth, strict=True)
                ]
            },
        ).status_code
        == 200
    )
    summary = c.get("/api/v1/shadow").json()
    assert summary["labeled_scored"] == 100
    assert summary["vs_reference"]["lost_attacks"] == 0
    assert summary["vs_reference"]["removed_false_positives"] == 0
    assert sum(f["total"] for f in summary["families"]) == sum(t["label"] for t in truth)


def test_shadow_failure_and_busy_do_not_change_live(client, payload):
    c, app = client
    flows, _ = payload
    with patch.object(app.state.shadow_service.bundle, "predict", side_effect=RuntimeError("boom")):
        response = submit(c, flows[:2])
    assert response.status_code == 200
    assert response.json()["shadow"]["status"] == "error"
    assert response.json()["shadow"]["results"][0]["error_code"] == "shadow_inference_failed"
    app.state.shadow_service.lock.acquire()
    try:
        busy = submit(c, flows[2:4])
    finally:
        app.state.shadow_service.lock.release()
    assert busy.status_code == 200
    assert busy.json()["shadow"]["results"][0]["error_code"] == "shadow_busy"
    with patch.object(app.state.store, "record_shadow", side_effect=sqlite3.OperationalError()):
        failed = submit(c, flows[4:6])
    assert failed.status_code == 200
    assert failed.json()["shadow"]["error_code"] == "shadow_persistence_failed"
    summary = c.get("/api/v1/shadow").json()
    assert summary["errors"] == 4 and summary["not_shadowed"] == 2
    assert summary["scored"] == 0
    assert summary["total_predictions"] == 6


def test_disabled_unavailable_and_auth(tmp_path):
    settings = Settings(
        database_path=tmp_path / "db", api_token=TOKEN, shadow_bundle_dir=tmp_path / "missing"
    )
    with TestClient(create_app(settings)) as c:
        assert c.get("/api/v1/shadow").status_code == 401
        c.headers["Authorization"] = f"Bearer {TOKEN}"
        assert c.get("/api/v1/shadow").json()["status"] == "unavailable"
        assert c.get("/api/v1/shadow?unexpected=1").status_code == 400
        assert c.get("/api/v1/shadow/predictions?limit=1&limit=2").status_code == 400
    service = ShadowService(None)
    service.load()
    assert service.evaluate([])["status"] == "disabled"


def test_tampered_artifact_fails_closed(tmp_path, frozen):
    target = tmp_path / "tampered"
    shutil.copytree(BUNDLE, target)
    with (target / "models.joblib").open("ab") as file:
        file.write(b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_shadow(target)
    service = ShadowService(target)
    service.load()
    assert service.status()["status"] == "unavailable"
    assert service.bundle is None


def test_migration_preserves_v2_rows(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as db:
        db.executescript(SCHEMA_V2)
        db.execute("INSERT INTO replay_truth VALUES ('example', 0, 'Normal', 'time')")
    store = AlertStore(path)
    store.initialize()
    store.initialize()
    with store.connection() as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 3
        assert db.execute("SELECT COUNT(*) FROM replay_truth").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM shadow_predictions").fetchone()[0] == 0


def test_summary_reports_live_regression_and_reference_recovery():
    rows = [
        {
            "status": "scored",
            "live": "normal",
            "reference": "normal",
            "candidate": "alert",
            "label": 1,
            "attack_cat": "Exploits",
            "n": 3,
        },
        {
            "status": "scored",
            "live": "alert",
            "reference": "normal",
            "candidate": "normal",
            "label": 1,
            "attack_cat": "Exploits",
            "n": 2,
        },
        {
            "status": "scored",
            "live": "normal",
            "reference": "normal",
            "candidate": "alert",
            "label": 0,
            "attack_cat": "Normal",
            "n": 1,
        },
        {
            "status": "scored",
            "live": "normal",
            "reference": "normal",
            "candidate": "alert",
            "label": None,
            "attack_cat": None,
            "n": 5,
        },
    ]
    result = summarize(rows)
    assert result["vs_reference"]["recovered_attacks"] == 3
    assert result["vs_live"]["lost_attacks"] == 2
    assert result["vs_live"]["added_false_positives"] == 1
    assert result["candidate_additions_vs_live"] == 9
    assert result["labeled_scored"] == 6 and result["unlabeled_scored"] == 5


def test_startup_loads_configured_bundle(tmp_path, frozen, payload):
    flows, _ = payload
    settings = Settings(
        database_path=tmp_path / "startup.db",
        api_token=TOKEN,
        bundles_dir=ROOT / "artifacts/bundles",
        bundle_version="v1.0.0",
        shadow_bundle_dir=BUNDLE,
    )
    with TestClient(create_app(settings), headers={"Authorization": f"Bearer {TOKEN}"}) as c:
        assert c.get("/health/ready").status_code == 200
        status = c.get("/api/v1/shadow").json()
        assert status["status"] == "ready"
        assert status["parity"]["decision_mismatches"] == 0
        assert submit(c, flows[:1]).json()["shadow"]["status"] == "scored"
    broken = Settings(
        database_path=tmp_path / "missing.db",
        api_token=TOKEN,
        bundles_dir=ROOT / "artifacts/bundles",
        bundle_version="v1.0.0",
        shadow_bundle_dir=tmp_path / "missing-bundle",
    )
    with TestClient(create_app(broken)) as c:
        assert c.get("/health/ready").status_code == 200
        result = submit(c, flows[:1])
        assert result.status_code == 200
        assert result.json()["shadow"]["status"] == "unavailable"


def test_serving_batch_sizes_preserve_archived_boundary_decisions(frozen):
    import numpy as np
    import pandas as pd

    source = ROOT / "experiments/fusion_ablation_20261005T033313Z/Exploits/seed_42"
    raw = pd.read_csv(
        ROOT / "data/raw/CSV_Files/Training and Testing Sets/UNSW_NB15_training-set.csv"
    )
    with np.load(source / "evaluation_scores.npz") as scores:
        chosen = np.unique(
            np.concatenate(
                [
                    np.argsort(np.abs(scores["learned_fusion"] - frozen.policy.recovery_threshold))[
                        :8
                    ],
                    np.argsort(np.abs(scores["lightgbm"] - frozen.policy.baseline_threshold))[:8],
                    np.argsort(
                        np.abs(
                            scores["lightgbm"]
                            - frozen.policy.baseline_threshold * frozen.policy.uncertain_lower_ratio
                        )
                    )[:8],
                ]
            )
        )
        indices = scores["row_indices"][chosen]
    with np.load(source / "evaluation_decisions.npz") as decisions:
        expected = decisions["selective_fusion__0.03"][chosen].tolist()
    flows = [
        row_to_flow(row, str(uuid4()), "2026-10-06T00:00:00Z")
        for row in raw.iloc[indices].to_dict("records")
    ]
    for batch_size in (1, 7, 100):
        actual = []
        for start in range(0, len(flows), batch_size):
            actual.extend(
                item["decision"] == "alert"
                for item in frozen.predict(flows[start : start + batch_size])
            )
        assert actual == expected


def test_export_rejects_changed_dataset_and_existing_destination(tmp_path):
    from training.export_shadow_bundle import export_shadow

    source = ROOT / "experiments/fusion_ablation_20261005T033313Z/Exploits/seed_42"
    if not source.exists():
        pytest.skip("Local research artifacts absent")
    wrong_csv = tmp_path / "wrong.csv"
    wrong_csv.write_text("label\n0\n")
    output = tmp_path / "candidate"
    with pytest.raises(ValueError, match="mismatched source data"):
        export_shadow(source, wrong_csv, output, "test", 0.03)
    assert not output.exists()
    output.mkdir()
    with pytest.raises(FileExistsError):
        export_shadow(source, wrong_csv, output, "test", 0.03)


def test_runtime_source_mismatch_rejects_bundle(tmp_path, frozen):
    from nexus.bundle import sha256_file

    target = tmp_path / "tampered-source"
    shutil.copytree(BUNDLE, target)
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["serving_source_hashes"]["src/nexus/shadow.py"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="implementation changed"):
        load_shadow(target)
    # A parity report saying passed must also explicitly record zero decision changes.
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    parity_path = target / "parity.json"
    parity = json.loads(parity_path.read_text())
    parity["decision_mismatches"] = 1
    parity_path.write_text(json.dumps(parity))
    manifest["artifact_hashes"]["parity.json"] = sha256_file(parity_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="parity evidence"):
        load_shadow(target)


def test_shadow_config_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("NEXUS_SHADOW_BUNDLE_DIR", str(tmp_path))
    assert Settings.from_env().shadow_bundle_dir == tmp_path
    monkeypatch.delenv("NEXUS_SHADOW_BUNDLE_DIR")
    assert Settings.from_env().shadow_bundle_dir is None


def test_curated_demo_shows_real_recovery_and_cost(client, tmp_path):
    from training.build_shadow_demo import build_demo

    c, _ = client
    source = ROOT / "experiments/fusion_ablation_20261005T033313Z/Exploits/seed_42"
    csv = ROOT / "data/raw/CSV_Files/Training and Testing Sets/UNSW_NB15_training-set.csv"
    output = tmp_path / "curated"
    evidence = build_demo(BUNDLE, source, csv, output)
    predictions = json.loads((output / "predictions.json").read_text())
    assert all("label" not in flow["features"] for flow in predictions["flows"])
    response = c.post("/api/v1/predictions", json=predictions)
    assert response.status_code == 200
    assert response.json()["shadow"]["status"] == "scored"
    assert (
        c.post(
            "/api/v1/replay/truth", json=json.loads((output / "truth.json").read_text())
        ).status_code
        == 200
    )
    summary = c.get("/api/v1/shadow").json()
    assert summary["vs_reference"] == {
        "recovered_attacks": 3,
        "lost_attacks": 0,
        "added_false_positives": 3,
        "removed_false_positives": 0,
    }
    assert summary["labeled_scored"] == 15
    actual = {r["flow_id"]: r for r in response.json()["shadow"]["results"]}
    for case in evidence["cases"]:
        assert actual[case["flow_id"]]["decision"] == case["expected_shadow"]["decision"]
    assert "not representative" in evidence["note"]
    with pytest.raises(FileExistsError):
        build_demo(BUNDLE, source, csv, output)
    wrong_csv = tmp_path / "wrong.csv"
    wrong_csv.write_text("label\n0\n")
    with pytest.raises(ValueError, match="dataset hash"):
        build_demo(BUNDLE, source, wrong_csv, tmp_path / "bad-demo")
