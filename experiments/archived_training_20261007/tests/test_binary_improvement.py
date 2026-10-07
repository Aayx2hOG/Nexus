"""Small checks for ablation isolation, weighting, and promotion criteria."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from improve_binary_detector import eligible, isolated_splits, weights  # noqa: E402


def test_reduced_feature_collisions_are_removed_from_both_partitions():
    frame = pd.DataFrame(
        {
            "sttl": [1, 2, 3, 4],
            "dttl": [1, 2, 3, 4],
            "ct_state_ttl": [1, 2, 3, 4],
            "dur": [10, 20, 10, 30],
        }
    )
    result = isolated_splits(frame, {"fit": np.array([0, 1]), "early": np.array([2, 3])})
    np.testing.assert_array_equal(result["fit"], [1])
    np.testing.assert_array_equal(result["early"], [3])


def test_fuzzers_weight_changes_only_fuzzers():
    target = np.array([0, 0, 1, 1])
    family = np.array([0, 0, 2, 3])
    np.testing.assert_array_equal(weights(target, family, 2, 2), [1, 1, 2, 1])


def test_no_promotion_when_recall_or_fuzzers_regresses():
    baseline = {"recall": 0.96, "false_positive_rate": 0.04, "fuzzers_recall": 0.8}
    improved = dict(baseline, false_positive_rate=0.03)
    assert eligible(improved, baseline, 0.95)
    assert not eligible(dict(improved, recall=0.94), baseline, 0.95)
    assert not eligible(dict(improved, fuzzers_recall=0.79), baseline, 0.95)
    assert not eligible(baseline, baseline, 0.95)


def test_hard_normal_weight_never_upweights_attacks_or_easy_normals():
    from improve_binary_detector import hard_normal_weights

    actual = hard_normal_weights(
        np.ones(4), np.array([0, 0, 1, 1]), np.array([0.8, 0.2, 0.8, 0.2]), 2.0
    )
    np.testing.assert_array_equal(actual, [2, 1, 1, 1])


def test_mining_keeps_duplicate_predictors_together_and_covers_every_row():
    from improve_binary_detector import mining_folds

    frame = pd.DataFrame({"x": np.repeat(np.arange(12), 2)})
    target = np.repeat(np.arange(12) % 2, 2)
    counts = np.zeros(len(frame), dtype=int)
    for train, held in mining_folds(frame, target):
        assert not set(frame.iloc[train].x) & set(frame.iloc[held].x)
        counts[held] += 1
    np.testing.assert_array_equal(counts, np.ones(len(frame)))


def test_duplicate_weights_reduce_repeated_rows_without_losing_alignment():
    from improve_binary_detector import robust_weights

    frame = pd.DataFrame({"x": [1, 1, 2, 3]}, index=[10, 20, 30, 40])
    target = np.array([0, 0, 1, 1])
    actual = robust_weights(frame, target, target, 1, 1, np.zeros(4), 1, {"duplicate_power": 0.5})
    assert actual[0] == actual[1]
    assert actual[0] < actual[2]
    np.testing.assert_allclose(actual.mean(), 1)


def test_control_weights_preserve_previous_experiment():
    from improve_binary_detector import robust_weights

    target = np.array([0, 0, 0, 1])
    frame = pd.DataFrame({"x": range(4)})
    actual = robust_weights(frame, target, target, 1, 1, np.zeros(4), 1, {})
    np.testing.assert_array_equal(actual, weights(target, target, 1, 1))
