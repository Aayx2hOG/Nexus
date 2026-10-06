"""Isolation and threshold invariants for the controlled anomaly experiments."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from anomaly_detection_models import AnomalyFeatures, Autoencoder, fpr_threshold  # noqa: E402
from complementary_fusion import calibrate_selective  # noqa: E402
from run_novelty_experiment import grouped_split, metrics, parse_args, run  # noqa: E402
from summarize_anomaly_results import enrich, statistical_summary  # noqa: E402


@pytest.mark.parametrize("budget", [0.01, 0.03, 0.05, 0.5])
def test_threshold_respects_budget_with_ties(budget):
    scores = np.repeat(np.arange(10, dtype=float), 20)
    cutoff = fpr_threshold(scores, budget)
    assert np.mean(scores >= cutoff) <= budget


def test_threshold_rejects_nonfinite_scores():
    with pytest.raises(ValueError):
        fpr_threshold([0.1, float("nan")], 0.05)


def test_groups_and_withheld_family_are_isolated():
    unique = 400
    frame = pd.DataFrame({"dur": np.repeat(np.arange(unique), 2)})
    labels = np.repeat(np.arange(unique) % 2, 2)
    families = np.where(labels == 0, "Normal", "Exploits")
    families[-20:] = "DoS"
    labels[-20:] = 1
    # Include a conflicting-label duplicate: the entire group must be exiled.
    families[-1], labels[-1] = "Normal", 0
    split, groups = grouped_split(frame, labels, families, "DoS", 42)
    memberships = [set(groups[rows]) for rows in split.values()]
    for i, left in enumerate(memberships):
        for right in memberships[i + 1 :]:
            assert left.isdisjoint(right)
    np.testing.assert_array_equal(
        np.sort(np.concatenate(list(split.values()))), np.arange(len(frame))
    )
    for name, rows in split.items():
        if name != "evaluation":
            assert not (families[rows] == "DoS").any()
    assert len(frame) - 1 in split["evaluation"]


def test_preprocessing_preserves_unseen_category_signal_without_refitting():
    fit = pd.DataFrame({"dur": [1.0, 2.0, 3.0], "proto": ["tcp", "udp", "tcp"]})
    model = AnomalyFeatures().fit(fit)
    medians = model.medians.copy()
    transformed = model.transform(pd.DataFrame({"dur": [4.0], "proto": ["new"]}))
    assert transformed[0, -1] > 0
    assert "new" not in model.encoder.categories_[0]
    pd.testing.assert_series_equal(model.medians, medians)
    assert np.isfinite(transformed).all()


def test_autoencoder_reconstruction_and_latent_scores_are_finite():
    rng = np.random.default_rng(7)
    values = rng.normal(size=(48, 6)).astype(np.float32)
    model = Autoencoder(seed=7, epochs=2, patience=1, batch_size=16, noise=0.1)
    model.fit(values[:32], values[32:])
    for latent in (False, True):
        scores = model.score(values[32:], latent=latent)
        assert scores.shape == (16,)
        assert np.isfinite(scores).all()
        assert (scores >= 0).all()


def test_complementarity_reports_recovered_and_lost_detections():
    labels = np.array([0, 0, 1, 1])
    baseline = np.array([True, False, True, False])
    prediction = ~baseline
    result = metrics(
        labels,
        prediction.astype(float),
        prediction,
        baseline,
        np.array(["Normal", "Normal", "DoS", "DoS"]),
        "DoS",
    )
    assert result["recovered_lightgbm_misses"] == 1
    assert result["lost_lightgbm_detections"] == 1
    assert result["additional_false_positives"] == 1
    assert result["removed_false_positives"] == 1
    assert result["net_additional_false_positives"] == 0
    assert result["recovered_per_additional_fp"] == 1
    assert result["net_recovered_attacks"] == 0


def test_selective_preserves_alerts_and_spends_only_remaining_budget():
    tree = np.linspace(0, 1, 100)
    fused = np.linspace(0, 1, 100)[::-1]
    rank = np.ones(100)
    cutoff = fpr_threshold(tree, 0.03)
    policy = calibrate_selective(tree, fused, rank, cutoff, 0.03)
    predictions = policy.predict(tree, fused, rank)
    assert (predictions[tree >= cutoff]).all()
    assert predictions.sum() <= 3
    assert policy.allowed_additional_false_positives == 0
    # A suspicious, confidently missed sample can still be rescued above the
    # maximum routed benign score, while an ordinary confident negative stays put.
    new_tree = np.array([0.01, 0.01, 1.0])
    new_fusion = np.array([1.1, 1.1, 0.0])
    new_rank = np.array([1.0, 0.1, 0.1])
    np.testing.assert_array_equal(
        policy.predict(new_tree, new_fusion, new_rank), [True, False, True]
    )


@pytest.mark.parametrize("budget", [0.01, 0.03, 0.05])
def test_selective_budget_with_tied_scores(budget):
    tree = np.repeat(np.arange(10) / 10, 20)
    fusion = np.full(200, 0.5)
    policy = calibrate_selective(tree, fusion, np.ones(200), fpr_threshold(tree, budget), budget)
    assert policy.predict(tree, fusion, np.ones(200)).mean() <= budget


def test_cli_aliases_and_extra_seeds():
    args = parse_args(
        [
            "--output-dir",
            "/tmp/cli-only",
            "--model",
            "learned_fusion",
            "--budget",
            "0.03",
            "--held-family",
            "DoS",
            "--seed",
            "7",
            "21",
            "314",
        ]
    )
    assert args.models == ["learned_fusion"]
    assert args.fpr_budgets == [0.03]
    assert args.families == ["DoS"]
    assert args.seeds == [7, 21, 314]
    with pytest.raises(SystemExit):
        parse_args(["--output-dir", "/tmp/cli-only", "--suspicious-quantile", "1.1"])


def test_small_training_reuse_and_csv_summary(tmp_path):
    # Small synthetic fixture only: never touches benchmark data or checkpoints.
    rng = np.random.default_rng(5)
    labels = np.tile([0, 1], 200)
    families = np.where(labels == 0, "Normal", "Exploits")
    families[1:21:2] = "DoS"
    frame = pd.DataFrame(
        {
            "dur": rng.uniform(0, 4, len(labels)),
            "sbytes": rng.uniform(1, 100, len(labels)),
            "proto": np.where(labels == 0, "tcp", "udp"),
        }
    )
    dataset = tmp_path / "training.csv"
    frame.assign(label=labels, attack_cat=families).to_csv(dataset, index=False)
    root = tmp_path / "trained"
    args = parse_args(
        [
            "--output-dir",
            str(root),
            "--train-csv",
            str(dataset),
            "--families",
            "DoS",
            "--seeds",
            "42",
            "--epochs",
            "1",
            "--max-rounds",
            "2",
            "--n-jobs",
            "1",
        ]
    )
    trained = root / "DoS" / "seed_42"
    results = run(args, frame, labels, families, "DoS", 42, trained)
    assert len(results) == 30
    assert all(
        r["lost_lightgbm_detections"] == 0 for r in results if r["model"] == "selective_fusion"
    )
    original = json.loads((trained / "calibration.json").read_text())
    args.reuse_dir = root
    reused = tmp_path / "reused" / "DoS" / "seed_42"
    repeated = run(args, frame, labels, families, "DoS", 42, reused)
    assert original == json.loads((reused / "calibration.json").read_text())
    assert (trained / "split_indices.npz").read_bytes() == (
        reused / "split_indices.npz"
    ).read_bytes()
    assert results == repeated
    table = enrich(pd.DataFrame(results))
    summary = statistical_summary(table)
    assert summary.seed_count.eq(1).all()
    assert summary.recall_std.isna().all()
    assert (
        table.loc[table.additional_false_positives.eq(0), "recovered_per_additional_fp"]
        .isna()
        .all()
    )
    table.to_csv(tmp_path / "comparison.csv", index=False)
    restored = pd.read_csv(tmp_path / "comparison.csv", keep_default_na=False, na_values=[""])
    assert len(enrich(restored)) == 30
    with pytest.raises(ValueError, match="Duplicate"):
        enrich(pd.concat([table, table], ignore_index=True))
    with pytest.raises(FileExistsError):
        run(args, frame, labels, families, "DoS", 42, reused)
