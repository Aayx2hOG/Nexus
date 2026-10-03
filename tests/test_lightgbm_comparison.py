"""Test paired representations, refits, and the complete comparison command."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from compare_lightgbm import experiment_plan, train  # noqa: E402
from fast_lightgbm_models import NativeFeatures, OneHotFeatures  # noqa: E402
from tune_lightgbm_fast import evaluate  # noqa: E402


def test_onehot_is_fit_only_and_retains_the_same_numeric_values():
    fit = pd.DataFrame({"n": [1.0, np.nan], "proto": ["tcp", "udp"]})
    heldout = pd.DataFrame({"n": [2.0], "proto": ["sctp"]})
    onehot = OneHotFeatures().fit(fit)
    native = NativeFeatures().fit(fit)
    np.testing.assert_allclose(onehot.transform(heldout).toarray(), [[2, 0, 0]])
    assert np.isnan(onehot.transform(fit).toarray()[1, 0])
    assert native.transform(heldout).proto.isna().all()
    numeric_only = OneHotFeatures().fit(fit[["n"]])
    assert numeric_only.transform(heldout).shape == (1, 1)


def test_plan_is_small_and_controls_representation_at_alpha_half():
    alphas = [0.4, 0.45, 0.5, 0.55, 0.6]
    plan = experiment_plan("multiclass", alphas)
    assert len(plan) == 6
    assert plan[0][0] == "onehot" and plan[1][0] == "native"
    assert plan[0][1] == plan[1][1]
    assert {p["weight_alpha"] for r, p in plan if r == "onehot"} == set(alphas)
    assert len(experiment_plan("multiclass", alphas, full_grid=True)) == 10
    assert len(experiment_plan("binary", alphas)) == 2


def test_refit_time_limit_returns_a_usable_truncated_model():
    X = pd.DataFrame({"x": np.arange(200, dtype=float)})
    y = (np.arange(200) % 2).astype(int)
    args = SimpleNamespace(max_rounds=20, n_jobs=1, seed=42, fit_seconds=1e-9, patience=3)
    params = experiment_plan("binary", [0.5])[0][1]
    model, details = train(X, y, np.arange(180), np.arange(180, 200), params, args, rounds=20)
    assert details["time_limited"]
    assert details["effective_rounds"] == 1
    assert details["requested_rounds"] == 20
    assert model.predict_proba(X).shape == (200, 2)


def test_full_comparison_cli_and_saved_model_predictions(tmp_path):
    rng = np.random.default_rng(42)
    raw = tmp_path / "raw"
    raw.mkdir()
    names = ["Normal", "A", "B", "C"]
    train_y = np.arange(1400) % 4
    training = None
    for split, y in (("training", train_y), ("testing", np.arange(280) % 4)):
        frame = pd.DataFrame(
            {
                "id": np.arange(len(y)),
                "label": (y > 0).astype(int),
                "attack_cat": np.array(names)[y],
                "n": y + rng.normal(0, 0.3, len(y)),
                "proto": np.where(y % 2, "tcp", "udp"),
            }
        )
        frame.to_csv(raw / f"UNSW_NB15_{split}-set.csv", index=False)
        if split == "training":
            training = frame
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"false_positive_rate": 0, "recall": 1, "f1_macro": 1}))
    output = tmp_path / "experiment"
    root = Path(__file__).resolve().parents[1]
    run = subprocess.run(
        [
            sys.executable,
            str(root / "training/compare_lightgbm.py"),
            "--task",
            "both",
            "--fit-seconds",
            "0",
            "--alphas",
            "0.5",
            "--max-rounds",
            "6",
            "--patience",
            "2",
            "--n-jobs",
            "1",
            "--raw-dir",
            str(raw),
            "--experiment-dir",
            str(output),
            "--binary-baseline",
            str(baseline),
            "--multiclass-baseline",
            str(baseline),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=root,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    selected = np.load(output / "split_indices.npz")["selection"]
    for task, count in (("binary", 8), ("multiclass", 4)):
        directory = output / task
        trials = json.loads((directory / "all_trial_metrics.json").read_text())
        assert len(trials) == count
        assert {t["refit"] for t in trials} == {False, True}
        for trial in trials:
            if trial["refit"]:
                assert (
                    trial["model_parameters"]["effective_rounds"]
                    == trial["model_parameters"]["requested_rounds"]
                )
        model = joblib.load(directory / "model.joblib")
        result = json.loads((directory / "selected_result.json").read_text())
        metrics = evaluate(model, training.iloc[selected], train_y[selected], task, names)
        assert metrics == result["validation_metrics"]
        assert (directory / "validation_comparison.md").exists()
        assert not (directory / "outperforming_result.md").exists()
    assert (output / "summary.json").exists()
    assert (
        "refitted" in (output / "binary/results.md").read_text()
        or not json.loads((output / "binary/selected_result.json").read_text())["refit"]
    )
