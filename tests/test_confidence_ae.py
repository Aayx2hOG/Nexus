"""Behavior and aggregation checks for confidence-dependent AE fusion."""

import numpy as np
import pytest
from compare_working_v2_fusion import metrics
from experiment_confidence_ae import count_metrics, predict_rule, reliability


def test_confidence_overrides_and_boundary_equality():
    pred, modes = predict_rule(
        [0.94, 0.95, 0.6, 0.59, 0.3, 0.29, 0.1, 0.09],
        [0, 0, 1, 1, 2, 2, 3, 3],
        [1, 2, 3],
        [0.95, 0.6, 0.3, 0.1],
    )
    np.testing.assert_array_equal(modes, [0, 0, 1, 1, 2, 2, 3, 3])
    np.testing.assert_array_equal(pred, [False, True, True, False, True, False, True, False])


def test_always_normal_and_always_attack_even_at_probability_extremes():
    pred, _ = predict_rule([1, 0], [0, 3], [1, 2, 3], [1.000001, 0.5, 0, 0])
    np.testing.assert_array_equal(pred, [False, True])


def test_count_aggregation_matches_direct_predictions():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, 500)
    p = rng.random(500)
    a = rng.random(500) * 4
    pred, modes = predict_rule(p, a, [1, 2, 3], [0.9, 0.7, 0.4, 0.1])
    tp = fp = 0
    for mode, cut in enumerate([0.9, 0.7, 0.4, 0.1]):
        selected = (modes == mode) & (p >= cut)
        tp += np.count_nonzero(selected & (y == 1))
        fp += np.count_nonzero(selected & (y == 0))
    result = count_metrics(np.count_nonzero(y == 0) - fp, fp, y.sum() - tp, tp)
    assert result == metrics(y, pred)


def test_reliability_includes_score_one():
    result = reliability(np.array([0, 1]), np.array([0.0, 1.0]))
    assert result["brier"] == 0
    assert result["ece"] == 0
    assert sum(r["rows"] for r in result["bins"]) == 2


@pytest.mark.parametrize("probability", [[np.nan], [1.1], [-0.1]])
def test_invalid_probabilities(probability):
    with pytest.raises(ValueError):
        predict_rule(probability, [0], [1, 2, 3], [0.9, 0.7, 0.4, 0.1])
