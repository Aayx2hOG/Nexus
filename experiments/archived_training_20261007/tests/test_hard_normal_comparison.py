"""Small paired-comparison checks without fitting or loading models."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from compare_hard_normal_candidate import metrics, paired_changes  # noqa: E402


def test_recoveries_and_regressions_are_reported_separately():
    y = np.array([1, 1, 0, 0])
    baseline = np.array([False, True, True, False])
    candidate = ~baseline
    assert paired_changes(y, baseline, candidate) == {
        "recovered_attacks": 1,
        "lost_attack_detections": 1,
        "removed_false_alarms": 1,
        "added_false_alarms": 1,
    }


def test_single_family_has_no_false_positive_rate():
    result = metrics(np.array([1, 1]), np.array([True, False]))
    assert result["recall"] == 0.5
    assert result["false_positive_rate"] is None
