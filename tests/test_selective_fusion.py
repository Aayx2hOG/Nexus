import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from selective_fusion import (  # noqa: E402
    LOWER_BOUNDS,
    metric_row,
    pareto_mask,
    predictions_for_rule,
    threshold_candidates,
)


def test_selective_rule_only_escalates_inside_gray_zone():
    probabilities = np.array([0.60, 0.49, 0.30, 0.29, 0.49])
    scores = np.array([-1.0, 0.8, 0.8, 0.8, 0.2])
    result = predictions_for_rule(probabilities, scores, 0.30, 0.5)
    assert result.tolist() == [1, 1, 1, 0, 0]


def test_metric_row_reports_incremental_tradeoff():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    baseline = np.array([0, 1, 0, 1, 0, 0])
    fused = np.array([1, 1, 0, 1, 1, 0])
    result = metric_row(y_true, fused, baseline)
    assert result["lightgbm_false_negatives_recovered"] == 1
    assert result["additional_false_positives"] == 1
    assert result["attacks_recovered_per_100_additional_false_positives"] == 100.0


def test_pareto_mask_removes_dominated_rules():
    frame = pd.DataFrame(
        {
            "lightgbm_false_negatives_recovered": [2, 3, 4, 3],
            "additional_false_positives": [1, 2, 4, 3],
        }
    )
    assert pareto_mask(frame).tolist() == [True, True, True, False]


def test_required_gray_zone_bounds_are_searched():
    assert LOWER_BOUNDS == (0.10, 0.20, 0.30, 0.35, 0.40, 0.45)


def test_if_thresholds_only_name_training_or_calibration_sources():
    candidates = threshold_candidates(np.linspace(-1, 1, 50), np.linspace(-2, 0, 50))
    assert candidates
    assert all(
        row["if_threshold_source"].startswith(("calibration_all_", "model_training_normal_"))
        for row in candidates
    )
