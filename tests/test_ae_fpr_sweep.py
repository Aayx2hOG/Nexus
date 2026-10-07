"""Verify empirical AE budgets and the adapted frozen-rule semantics."""

import numpy as np
import pytest
from anomaly_detection_models import fpr_threshold
from experiment_ae_fpr_sweep import ae_gate, mode_rules


@pytest.mark.parametrize("budget", [0.05, 0.1])
def test_conservative_budget_handles_ties(budget):
    reference = np.repeat(np.arange(20, dtype=float), 10)
    threshold = fpr_threshold(reference, budget)
    assert np.count_nonzero(reference >= threshold) <= int(np.floor(len(reference) * budget))
    assert threshold > reference[-int(np.floor(len(reference) * budget)) - 1]


def test_four_mode_and_uncertainty_rule_boundaries():
    p = np.array([0.9, 0.5, 0.6, 0.2, 0.1, 0.8])
    error = np.array([0.0, 1.0, 1.0, 2.0, 3.0, 0.0])
    result = mode_rules(p, error, 0.6, [1.0, 2.0, 3.0], [0.65, 0.6, 0.6, 0.6], 0.1, 0.65)
    np.testing.assert_array_equal(
        result["Four-mode hard override"], [False, False, True, True, True, False]
    )
    np.testing.assert_array_equal(
        result["Uncertainty-band four-mode"], [True, False, True, True, True, True]
    )
    np.testing.assert_array_equal(
        result["Mode confidence"], [True, False, True, False, False, True]
    )


def test_gate_preserves_binary_below_threshold_and_uses_fusion_at_equality():
    result = ae_gate([True, False, True], [False, True, False], [0.0, 1.0, 2.0], 1.0)
    np.testing.assert_array_equal(result, [True, True, False])


def test_or_cannot_lose_binary_attacks_and_and_cannot_add_alerts():
    p = np.linspace(0, 1, 11)
    result = mode_rules(p, p, 0.6, [0.3, 0.6, 0.9], [0.65, 0.6, 0.6, 0.6], 0.1, 0.65)
    base = p >= 0.6
    assert np.all(result["OR"][base])
    assert not np.any(result["AND"][~base])


def test_invalid_boundaries():
    with pytest.raises(ValueError):
        mode_rules([0.5], [1.0], 0.6, [1.0, 1.0, 2.0], [0.65, 0.6, 0.6, 0.6], 0.1, 0.65)


def test_float32_scores_do_not_round_cutoff_back_onto_excluded_tie():
    values = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    threshold = fpr_threshold(values, 0.1)
    result = ae_gate(np.zeros(4, dtype=bool), np.ones(4, dtype=bool), values, threshold)
    assert not result.any()
    result = mode_rules(
        np.ones(4), values, 0.6, [threshold, 2.0, 3.0], [0.65, 0.6, 0.6, 0.6], 0.1, 0.65
    )
    assert not result["AE alone"].any()
