"""Check constrained selection and the experiment artifact lifecycle on synthetic data."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from training_utils import evaluate_binary_classifier  # noqa: E402
from tune_lightgbm import (  # noqa: E402
    choose_threshold,
    class_weights,
    outperforms,
    rank_trial,
    report,
    run_task,
)


def test_threshold_matches_exhaustive_search_with_tied_scores():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.5, 0.5, 0.9, 0.5, 0.6, 0.8, 0.9])
    for minimum in (0.5, 0.75, 0.95, 1.0):
        threshold = choose_threshold(y, scores, minimum)
        candidates = []
        for value in np.unique(scores):
            pred = scores >= value
            recall = pred[y == 1].mean()
            if recall >= minimum:
                candidates.append((pred[y == 0].mean(), -recall, -value))
        assert threshold == -min(candidates)[2]
    assert choose_threshold(y, np.full(len(y), 0.5), 0.95) == 0.5


def test_constraint_and_baseline_gate():
    baseline = {"false_positive_rate": 0.18}
    low_recall = {"false_positive_rate": 0.01, "recall": 0.94, "f1": 0.95}
    feasible = {"false_positive_rate": 0.1, "recall": 0.95, "f1": 0.9}
    assert rank_trial(feasible, "binary", 0.95) < rank_trial(low_recall, "binary", 0.95)
    assert not outperforms(low_recall, baseline, "binary", 0.95)
    assert outperforms(feasible, baseline, "binary", 0.95)
    assert not outperforms(feasible, feasible, "binary", 0.95)


def test_weights_are_normalized_and_tempered():
    y = np.array([0] * 100 + [1] * 10)
    assert class_weights(y, 0) is None
    full = class_weights(y, 1)
    soft = class_weights(y, 0.5)
    assert full[1] / full[0] == pytest.approx(10)
    assert soft[1] / soft[0] == pytest.approx(np.sqrt(10))
    assert np.mean([soft[int(c)] for c in y]) == pytest.approx(1)


def test_winning_markdown_contains_arguments_and_metrics(tmp_path):
    result = {
        "task": "binary",
        "model_parameters": {"num_leaves": 63, "class_weight": None},
        "decision_threshold": 0.73,
        "test_metrics": {"false_positive_rate": 0.1, "recall": 0.96},
    }
    assert report(
        tmp_path,
        result,
        {"false_positive_rate": 0.18, "recall": 0.97},
        SimpleNamespace(minimum_recall=0.95),
    )
    content = (tmp_path / "outperforming_result.md").read_text()
    assert '"num_leaves": 63' in content
    assert '"decision_threshold": 0.73' in content
    assert "0.960000" in content


def test_evaluator_uses_saved_threshold(tmp_path):
    from lightgbm import LGBMClassifier

    X = np.arange(100).reshape(-1, 1)
    y = (X[:, 0] >= 50).astype(int)
    model = LGBMClassifier(n_estimators=3, min_child_samples=5, n_jobs=1, verbosity=-1)
    model.fit(X, y)
    model.decision_threshold_ = 1.0
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    metrics = evaluate_binary_classifier(joblib.load(path), X, y)
    assert metrics["recall"] == 0
    assert metrics["false_positive_rate"] == 0
    assert metrics["decision_threshold"] == 1.0


@pytest.mark.parametrize("task", ["binary", "multiclass"])
def test_experiment_outputs_on_synthetic_data(tmp_path, task):
    rng = np.random.default_rng(42)
    data = tmp_path / "data"
    baseline_dir = tmp_path / "baselines"
    data.mkdir()
    baseline_dir.mkdir()
    for split, rows in (("train", 300), ("test", 90)):
        y = np.arange(rows) % (2 if task == "binary" else 3)
        X = rng.normal(size=(rows, 4))
        X[:, 0] += y * 4
        sparse.save_npz(data / f"X_{split}_tree.npz", sparse.csr_matrix(X))
        suffix = "" if task == "binary" else "_multiclass"
        np.save(data / f"y_{split}{suffix}.npy", y)
    (data / "multiclass_labels.json").write_text(json.dumps({"class_names": ["A", "B", "C"]}))
    name = "lightgbm" if task == "binary" else "multiclass_lightgbm"
    # Deliberately unbeatable synthetic baseline: exercise honest no-improvement reporting.
    baseline = {"false_positive_rate": 0.0, "recall": 1.0, "f1_macro": 1.0}
    (baseline_dir / f"{name}_metrics.json").write_text(json.dumps(baseline))
    args = SimpleNamespace(
        data_dir=data,
        baseline_dir=baseline_dir,
        seed=42,
        selection_size=0.2,
        early_stopping_size=0.2,
        trials=3,
        max_rounds=5,
        patience=2,
        n_jobs=1,
        minimum_recall=0.95,
    )
    output = tmp_path / task
    result = run_task(task, output, args)
    assert not result["outperforms_baseline"]
    assert (output / "results.md").is_file()
    assert not (output / "outperforming_result.md").exists()
    assert len(list((output / "trials").glob("*.json"))) == 3
    selected = json.loads((output / "selected_result.json").read_text())
    assert "confusion_matrix" in selected["test_metrics"]
    splits = np.load(output / "split_indices.npz")
    assert len(set(splits["fit"]) & set(splits["selection"])) == 0
    assert len(set(splits["fit"]) & set(splits["early_stopping"])) == 0
    assert len(set(splits["selection"]) & set(splits["early_stopping"])) == 0
    model = joblib.load(output / "model.joblib")
    if task == "binary":
        assert selected["validation_metrics"]["recall"] >= 0.95
        assert model.decision_threshold_ == selected["decision_threshold"]
