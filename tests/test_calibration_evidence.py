"""Small fixed-score fixtures only: no fitting, calibration or model inference."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from calibration_evidence import (
    ARCHIVE,
    audit_directory,
    digest,
    main,
    tail_statistics,
    write_evidence,
)
from complementary_fusion import SelectiveFusion


def fixture_inputs(tmp_path, empty_route=False):
    split = {
        "calibration": np.arange(5),
        "evaluation": np.array([5, 6]),
        "fit": np.array([7]),
        "early": np.array([8]),
        "fusion": np.array([9]),
    }
    np.savez_compressed(tmp_path / "split_indices.npz", **split)
    tree = np.array([0.9, 0.3, 0.4, 0.2, 0.6])
    ranks = np.zeros(5) if empty_route else np.ones(5)
    policy = SelectiveFusion(
        0.8, np.nextafter(0.7, np.inf), 1.0 if empty_route else 0.5, 0.99, 4, 1, 0, 1
    )
    (tmp_path / "models.joblib").write_bytes(b"synthetic identifier, never loaded")
    return dict(
        split=split,
        labels=np.array([0, 0, 0, 0, 1]),
        families=np.array(["Normal"] * 4 + ["Exploits"]),
        scores={
            "lightgbm": tree,
            "learned_fusion": np.array([0.95, 0.7, 0.7, 0.4, 0.9]),
            "ae_denoising": np.arange(5, dtype=float),
        },
        anomaly_rank=ranks,
        policies={"0.25": policy.to_dict()},
        budgets=[0.25],
        seed=42,
        held_family="none",
        data_sha256="synthetic source",
        model_sha256=digest(tmp_path / "models.joblib"),
    )


def seal(tmp_path, inputs):
    (tmp_path / "calibration.json").write_text(
        json.dumps(
            {
                "partition": "calibration",
                "uses_evaluation_labels": False,
                "selective_policies": inputs["policies"],
            }
        )
    )
    manifest = {key: inputs[key] for key in ("seed", "held_family", "data_sha256")}
    manifest.update(
        status="complete",
        artifact_hashes={
            name: digest(tmp_path / name)
            for name in (ARCHIVE, "split_indices.npz", "models.joblib", "calibration.json")
        },
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))


def test_zero_headroom_ties_and_separate_fp_branches(tmp_path):
    inputs = fixture_inputs(tmp_path)
    write_evidence(tmp_path, **inputs)
    seal(tmp_path, inputs)
    report = audit_directory(tmp_path, "0.25")
    assert report["primary_calibration_fps"] == 1
    assert report["recovery_calibration_fps"] == 0
    assert report["observed_calibration_fpr"] == 0.25
    assert report["recovery_allowance"] == 0
    tail = report["routed_benign"]
    assert tail["rows"] == 3
    assert tail["above_cutoff"] == tail["equal_cutoff"] == 0
    assert tail["predecessor_float_ties"] == tail["maximum_ties"] == 2
    assert tail["maximum_to_next_distinct_gap"] == pytest.approx(0.3)
    assert report["all_benign"]["above_cutoff"] == 1  # primary FP, not recovery FP
    assert report["evaluation_fps"] is None


def test_empty_routing_and_exact_cutoff_ties(tmp_path):
    inputs = fixture_inputs(tmp_path, empty_route=True)
    write_evidence(tmp_path, **inputs)
    seal(tmp_path, inputs)
    report = audit_directory(tmp_path, "0.25")
    assert report["routed_benign"]["rows"] == 0
    assert report["routed_benign"]["quantiles"] is None
    tail = tail_statistics(np.array([0.4, 0.7, 0.7, 0.8]), 0.7, 0)
    assert tail["equal_cutoff"] == tail["within_window"] == 2
    assert tail["passing_ge_cutoff"] == 3
    assert tail["above_cutoff"] == 1


def test_primary_and_recovery_fp_counts_with_allowance(tmp_path):
    inputs = fixture_inputs(tmp_path)
    policy = SelectiveFusion(0.8, 0.7, 0.5, 0.99, 4, 1, 2, 3)
    inputs.update(policies={"0.75": policy.to_dict()}, budgets=[0.75])
    write_evidence(tmp_path, **inputs)
    seal(tmp_path, inputs)
    report = audit_directory(tmp_path, "0.75")
    assert report["primary_calibration_fps"] == 1
    assert report["recovery_calibration_fps"] == 2
    assert report["total_calibration_fps"] == 3
    assert report["unused_recovery_allowance"] == 0


def test_writer_rejects_overlap_misalignment_and_held_family(tmp_path):
    inputs = fixture_inputs(tmp_path)
    inputs["split"]["evaluation"] = np.array([0, 6])
    with pytest.raises(ValueError, match="Overlapping"):
        write_evidence(tmp_path, **inputs)
    inputs = fixture_inputs(tmp_path)
    inputs["scores"]["lightgbm"] = np.array([0.1])
    with pytest.raises(ValueError, match="alignment"):
        write_evidence(tmp_path, **inputs)
    inputs = fixture_inputs(tmp_path)
    inputs["held_family"] = "Exploits"
    with pytest.raises(ValueError, match="Held family"):
        write_evidence(tmp_path, **inputs)


def test_auditor_checks_partition_alignment_even_with_matching_hash(tmp_path):
    inputs = fixture_inputs(tmp_path)
    write_evidence(tmp_path, **inputs)
    split = inputs["split"]
    split["calibration"] = np.arange(4, -1, -1)
    np.savez_compressed(tmp_path / "split_indices.npz", **split)
    # Simulate a consistently rehashed but misaligned artifact bundle.
    with np.load(tmp_path / ARCHIVE) as a:
        arrays = {key: a[key] for key in a.files}
    metadata = json.loads(arrays["metadata_json"].item())
    metadata["partition_sha256"] = digest(tmp_path / "split_indices.npz")
    arrays["metadata_json"] = np.array(json.dumps(metadata))
    np.savez_compressed(tmp_path / ARCHIVE, **arrays)
    seal(tmp_path, inputs)
    with pytest.raises(ValueError, match="partition mismatch"):
        audit_directory(tmp_path, "0.25")


def test_hash_policy_and_seed_mismatch(tmp_path):
    inputs = fixture_inputs(tmp_path)
    write_evidence(tmp_path, **inputs)
    seal(tmp_path, inputs)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["seed"] = 123
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="mismatch: seed"):
        audit_directory(tmp_path, "0.25")
    seal(tmp_path, inputs)
    with pytest.raises(ValueError, match="No archived policy"):
        audit_directory(tmp_path, "0.01")
    with (tmp_path / ARCHIVE).open("ab") as stream:
        stream.write(b"tampering")
    with pytest.raises(ValueError, match="hash mismatch"):
        audit_directory(tmp_path, "0.25")


def test_missing_evidence_and_exclusive_writes(tmp_path, monkeypatch, capsys):
    with pytest.raises(FileNotFoundError, match="No inference or regeneration"):
        audit_directory(tmp_path, "0.01")
    inputs = fixture_inputs(tmp_path)
    write_evidence(tmp_path, **inputs)
    original = digest(tmp_path / ARCHIVE)
    with pytest.raises(FileExistsError):
        write_evidence(tmp_path, **inputs)
    assert digest(tmp_path / ARCHIVE) == original
    seal(tmp_path, inputs)
    args = ["calibration_evidence.py", "--run-dir", str(tmp_path), "--policy-key", "0.25"]
    monkeypatch.setattr(sys, "argv", args)
    main()
    assert json.loads(capsys.readouterr().out)["recovery_calibration_fps"] == 0
    target = tmp_path / "audit.json"
    target.write_text("preserve me")
    monkeypatch.setattr(sys, "argv", args + ["--output-json", str(target)])
    with pytest.raises(FileExistsError):
        main()
    assert target.read_text() == "preserve me"
