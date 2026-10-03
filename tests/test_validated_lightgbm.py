"""End-to-end checks for train-only selection and frozen overlap-free evaluation."""

import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from evaluate_frozen_lightgbm import verify_frozen  # noqa: E402
from train_validated_lightgbm import canonical_features, evaluate, search_plan  # noqa: E402
from tune_lightgbm_fast import read_data  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def invoke(script, *args):
    return subprocess.run(
        [sys.executable, str(ROOT / "training" / script), *map(str, args)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=90,
    )


def test_plan_reproducible_and_includes_weight_ablations():
    plan = search_plan("multiclass", 16, 42)
    assert plan == search_plan("multiclass", 16, 42)
    assert plan != search_plan("multiclass", 16, 43)
    assert {p["weight_alpha"] for p in plan[:5]} == {0, 0.25, 0.55, 0.7, 1}


def test_numeric_hashing_stable_across_csv_dtype_inference():
    integer = canonical_features(pd.DataFrame({"x": [1, 2], "proto": ["tcp", "udp"]}))
    floating = canonical_features(pd.DataFrame({"x": [1.0, 2.0], "proto": ["tcp", "udp"]}))
    np.testing.assert_array_equal(
        pd.util.hash_pandas_object(integer, index=False),
        pd.util.hash_pandas_object(floating, index=False),
    )


def test_train_only_paired_baseline_and_frozen_evaluation(tmp_path):
    rng = np.random.default_rng(42)
    names = np.array(["Normal", "A", "B"])

    def dataset(n):
        y = np.arange(n) % 3
        return pd.DataFrame(
            {
                "id": np.arange(n),
                "label": (y > 0).astype(int),
                "attack_cat": names[y],
                "x": y + rng.normal(0, 0.25, n),
                "proto": np.where(y % 2, "tcp", "udp"),
            }
        )

    frame = dataset(700)
    # A conflict and ordinary duplicates must stay grouped, not be silently dropped.
    conflict = frame.iloc[[0]].copy()
    conflict["label"], conflict["attack_cat"] = 1, "A"
    frame = pd.concat([frame, frame.iloc[:10], conflict], ignore_index=True)
    csv = tmp_path / "train.csv"
    frame.to_csv(csv, index=False)
    experiment = tmp_path / "experiment"
    run = invoke(
        "train_validated_lightgbm.py",
        "--train-csv",
        csv,
        "--experiment-dir",
        experiment,
        "--trials",
        2,
        "--max-rounds",
        5,
        "--patience",
        2,
        "--rf-trees",
        3,
        "--n-jobs",
        1,
        "--recall-confidence",
        0,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert not (experiment / "final_evaluation").exists()
    manifest = verify_frozen(experiment)
    assert manifest["status"] == "frozen"
    assert "test_metrics" not in (experiment / "validation_summary.json").read_text()
    audit = json.loads((experiment / "data_audit.json").read_text())
    assert audit["conflicting_label_groups"] == 1
    raw, y, class_names = read_data(csv)
    raw = canonical_features(raw)
    groups = pd.util.hash_pandas_object(raw, index=False).to_numpy()
    split = np.load(experiment / "split_indices.npz")
    for a in split.files:
        for b in split.files:
            if a != b:
                assert not set(groups[split[a]]) & set(groups[split[b]])
    refit = np.r_[split["fit"], split["early"]]
    for task in ("binary", "multiclass"):
        directory = experiment / task
        selected = json.loads((directory / "selected_validation.json").read_text())
        assert selected["refit"]["training_rows"] == len(refit)
        assert selected["refit"]["effective_rounds"] == selected["refit"]["requested_rounds"]
        for filename, record in [
            ("model.joblib", selected),
            (
                "random_forest.joblib",
                json.loads((directory / "random_forest_validation.json").read_text()),
            ),
        ]:
            bundle = joblib.load(directory / filename)
            result = evaluate(
                bundle, raw.iloc[split["selection"]], y[split["selection"]], task, class_names
            )
            assert result == record["validation_metrics"]
            if task == "binary":
                np.testing.assert_array_equal(
                    bundle.predict(raw),
                    (bundle.predict_proba(raw)[:, 1] >= record["decision_threshold"]).astype(int),
                )
    # Evaluation is a separate explicit command, excluding all development overlap.
    heldout = pd.concat([frame.iloc[:10], dataset(120)], ignore_index=True)
    test_csv = tmp_path / "evaluation.csv"
    heldout.to_csv(test_csv, index=False)
    result = invoke(
        "evaluate_frozen_lightgbm.py", "--experiment-dir", experiment, "--evaluation-csv", test_csv
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((experiment / "final_evaluation/audit.json").read_text())
    assert report["excluded_development_overlap_rows"] == 10
    assert report["evaluated_rows"] == 120
    repeated = invoke(
        "evaluate_frozen_lightgbm.py", "--experiment-dir", experiment, "--evaluation-csv", test_csv
    )
    assert repeated.returncode != 0
    (experiment / "binary/model.joblib").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Frozen artifact changed"):
        verify_frozen(experiment)
