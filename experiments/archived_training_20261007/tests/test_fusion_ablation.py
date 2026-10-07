import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from anomaly_detection_models import fpr_threshold
from complementary_fusion import calibrate_selective
from fusion_ablation import fit_ablations, slice_diagnostics, validate_partitions
from run_novelty_experiment import metrics, parse_args, write_json
from summarize_anomaly_results import enrich, statistical_summary


def bundle():
    split = {
        name: np.arange(i * 4, (i + 1) * 4)
        for i, name in enumerate(("fit", "early", "fusion", "calibration", "evaluation"))
    }
    labels = np.tile([0, 1, 0, 1], 5)
    families = np.where(labels == 0, "Normal", "Exploits")
    scores = {
        part: {
            key: np.array([0.1, 0.8, 0.3, 0.7])
            for key in ("lightgbm", "ae_plain", "ae_denoising", "ae_latent")
        }
        for part in ("fusion", "calibration", "evaluation")
    }
    return split, labels, families, scores


def test_matched_controls_ignore_evaluation_labels_and_scores():
    split, labels, families, scores = bundle()
    fitted = fit_ablations(
        scores, split, labels, families, "none", 7, 1.0, ["ae_plain", "ae_denoising"]
    )
    assert len(fitted) == 6
    assert fitted["fusion_control"]["columns"] == ("lightgbm",)
    before = fitted["fusion_ae_plain"]["model"].named_steps["logisticregression"].coef_.copy()
    labels[split["evaluation"]] = 1 - labels[split["evaluation"]]
    scores["evaluation"]["ae_plain"] *= 100
    after = fit_ablations(scores, split, labels, families, "none", 7, 1.0, ["ae_plain"])
    np.testing.assert_array_equal(
        before, after["fusion_ae_plain"]["model"].named_steps["logisticregression"].coef_
    )


def test_partition_and_missing_representation_guards():
    split, labels, families, scores = bundle()
    with pytest.raises(ValueError, match="Held family"):
        validate_partitions(split, labels, families, "Exploits")
    split["calibration"] = split["evaluation"]
    with pytest.raises(ValueError, match="overlapping"):
        validate_partitions(split, labels, families, "none")
    split, labels, families, scores = bundle()
    del scores["fusion"]["ae_plain"]
    with pytest.raises(ValueError, match="Missing saved representation"):
        fit_ablations(scores, split, labels, families, "none", 7, 1.0, ["ae_plain"])


def test_allocated_policy_cap_and_honest_losses():
    tree = np.arange(100) / 100
    cutoff = fpr_threshold(tree, 0.2)
    policy = calibrate_selective(tree, tree, tree, cutoff, 0.2, primary_budget_fraction=0.5)
    assert policy.baseline_false_positives <= 10
    assert policy.allowed_additional_false_positives <= 10
    assert policy.calibration_false_positives <= 20
    assert policy.to_dict()["full_budget_baseline_threshold"] == cutoff
    values = np.array([0.85, 0.99, 0.2, 0.3])
    prediction = policy.predict(values, np.zeros(4), np.zeros(4))
    result = metrics(
        np.array([1, 1, 0, 0]),
        values,
        prediction,
        values >= cutoff,
        np.array(["Exploits", "Exploits", "Normal", "Normal"]),
        "Exploits",
    )
    assert result["lost_lightgbm_detections"] == 1
    assert result["net_recovered_attacks"] == -1
    original = calibrate_selective(tree, tree, tree, cutoff, 0.2)
    assert np.all(original.predict(tree, tree, tree)[tree >= cutoff])


def test_overlapping_diagnostics_and_empty_slices():
    frame = pd.DataFrame(dict(proto=["tcp", "tcp"], service=["http", "-"], state=["FIN", "FIN"]))
    rows = slice_diagnostics(
        frame,
        np.array([1, 0]),
        np.array(["Reconnaissance", "Normal"]),
        {"lightgbm": np.array([0.1, 0.8])},
        {"lightgbm__.1": np.array([False, True]), "fusion__.1": np.array([True, False])},
    )
    tcp = next(
        r
        for r in rows
        if r["slice"] == "proto=tcp"
        and r["cohort"] == "Reconnaissance"
        and r["method"] == "fusion__.1"
    )
    assert tcp["recovered"] == 1 and tcp["overlapping"]
    assert all(
        r["score_distributions"]["lightgbm"] is None for r in rows if r["slice"] == "proto=udp"
    )


def test_exclusive_json_and_cli_defaults(tmp_path):
    path = tmp_path / "result.json"
    write_json(path, {"original": True})
    with pytest.raises(FileExistsError):
        write_json(path, {})
    args = parse_args(["--output-dir", str(tmp_path)])
    assert not args.fusion_ablations and not args.reconnaissance_diagnostics
    assert args.primary_budget_fractions == [1.0]


def paired_rows():
    rows = []
    for seed in [7, 21]:
        for model in ["lightgbm", "fusion_control"]:
            row = metrics(
                np.array([0, 0, 1, 1]),
                np.array([0.1, 0.2, 0.7, 0.9]),
                np.array([False, False, True, True]),
                np.array([False, False, True, True]),
                np.array(["Normal", "Normal", "Exploits", "Exploits"]),
                "Exploits",
            )
            row.update(
                seed=seed,
                model=model,
                held_family="Exploits",
                budget=0.1,
                evaluation_budget_met=True,
                partition_sha256=str(seed),
                data_sha256="source",
                calibration_policy="benign",
                evaluation_policy="frozen",
            )
            rows.append(row)
    return pd.DataFrame(rows)


def test_summary_rejects_unpaired_context_and_seeds():
    table = paired_rows()
    summary = statistical_summary(table)
    assert (summary["recall_std"] == 0).all()
    assert (summary["recall_min"] == summary["recall_max"]).all()
    with pytest.raises(ValueError, match="seed sets"):
        enrich(table.iloc[:-1])
    table.loc[1, "partition_sha256"] = "different"
    with pytest.raises(ValueError, match="different partitions"):
        enrich(table)


def test_synthetic_score_bundle_reporting_and_reuse_identity(tmp_path):
    """Exercise result wiring from four-row score arrays, without detector training."""
    import json

    import joblib
    from run_novelty_experiment import finish_run

    split, labels, families, scores = bundle()
    frame = pd.DataFrame({"proto": ["tcp"] * 20, "service": ["-"] * 20, "state": ["FIN"] * 20})
    source = tmp_path / "source.csv"
    source.write_text("synthetic score fixture\n")
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--train-csv",
            str(source),
            "--models",
            "selective_fusion",
            "--fusion-ablations",
            "--fusion-representations",
            "ae_plain",
            "ae_denoising",
            "--primary-budget-fractions",
            "1",
            ".5",
            "--budget",
            ".25",
            "--reconnaissance-diagnostics",
        ]
    )
    for part in scores:
        scores[part]["learned_fusion"] = scores[part]["lightgbm"].copy()
    saved = {"rank_references": {"ae_denoising": np.array([0.1, 0.3])}}
    output = tmp_path / "first"
    output.mkdir()
    np.savez_compressed(output / "split_indices.npz", **split)
    results = finish_run(args, labels, families, "none", 7, output, split, scores, saved, frame)
    assert len(results) == 9  # baseline, six matched controls, two allocations
    assert len({r["partition_sha256"] for r in results}) == 1
    assert len({r["conditions_sha256"] for r in results}) == 1
    assert (output / "reconnaissance_diagnostics.json").exists()
    manifest = json.loads((output / "manifest.json").read_text())
    reused = joblib.load(output / "models.joblib")
    reused["detector_checkpoint_sha256"] = manifest["artifact_hashes"]["models.joblib"]
    other = tmp_path / "second"
    other.mkdir()
    np.savez_compressed(other / "split_indices.npz", **split)
    repeated = finish_run(args, labels, families, "none", 7, other, split, scores, reused, frame)
    assert results == repeated
    with pytest.raises(FileExistsError):
        finish_run(args, labels, families, "none", 7, output, split, scores, saved, frame)
