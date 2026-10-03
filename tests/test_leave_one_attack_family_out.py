import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from leave_one_attack_family_out import (  # noqa: E402
    comparison_row,
    detection_metrics,
    normal_metrics,
    select_if_threshold,
)


def test_if_threshold_obeys_fpr_limit_and_maximizes_recall():
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.35, 0.45, 0.9])
    threshold = select_if_threshold(y, scores, maximum_fpr=0.2)
    predictions = scores >= threshold
    assert predictions[y == 0].mean() <= 0.2
    assert predictions[y == 1].mean() == 2 / 3


def test_detection_and_normal_metrics_are_json_safe():
    metrics = detection_metrics(
        np.array([0.2, 0.8, 0.7]), np.array([False, True, True]), "attack_probability"
    )
    assert metrics == {
        "samples": 3, "detected": 2, "missed": 1, "recall": 2 / 3,
        "mean_attack_probability": np.mean([0.2, 0.8, 0.7]),
        "median_attack_probability": 0.7,
    }
    assert normal_metrics(np.array([False, True]))["false_positive_rate"] == 0.5


def test_comparison_row_exposes_required_fields():
    model = {"samples": 2, "recall": 0.5}
    metrics = {
        "held_out_family": "Example", "lightgbm": model,
        "random_forest": model, "isolation_forest": model,
        "complementarity": {
            "both_supervised_miss": 1,
            "shared_supervised_misses_if_catches": 1,
            "if_recovery_rate_of_shared_supervised_misses": 1.0,
            "all_three_miss": 0,
        },
        "normal_traffic": {
            name: {"false_positive_rate": 0.1}
            for name in ("lightgbm", "random_forest", "isolation_forest")
        },
    }
    row = comparison_row(metrics)
    assert row["if_recovered"] == 1
    assert row["all_three_miss"] == 0
    assert row["lightgbm_normal_fpr"] == 0.1
