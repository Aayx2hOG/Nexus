import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from analyze_autoencoder_complementarity import (  # noqa: E402
    MODELS,
    analyze,
    recovery,
    same_rows,
    verify_scores,
)


def test_alignment_rejects_same_label_category_permutation_by_ids_and_scores():
    raw = pd.DataFrame({"id": [1, 2], "label": [1, 1], "attack_cat": ["Fuzzers"] * 2})
    swapped = pd.DataFrame({"source_id": [2, 1], "record_index": [0, 1],
                            "true_label": [1, 1], "attack_cat": ["Fuzzers"] * 2})
    with pytest.raises(ValueError, match="source IDs/order"):
        same_rows(raw, swapped, "fixture", check_ids=True)
    with pytest.raises(ValueError, match="alignment cannot be established"):
        verify_scores(np.array([0.02, 0.01]), np.array([0.01, 0.02]), "AE")


def test_alignment_rejects_missing_duplicate_rows_and_nonfinite_scores():
    raw = pd.DataFrame({"id": [1, 2], "label": [1, 1], "attack_cat": ["Fuzzers"] * 2})
    frame = pd.DataFrame({"source_id": [1, 1], "record_index": [0, 1],
                          "true_label": [1, 1], "attack_cat": ["Fuzzers"] * 2})
    with pytest.raises(ValueError, match="duplicate"):
        same_rows(raw, frame, "fixture", check_ids=True)
    with pytest.raises(ValueError, match="row count"):
        same_rows(raw, frame.iloc[:1], "fixture")
    with pytest.raises(ValueError, match="non-finite"):
        verify_scores(np.array([np.nan]), np.array([0.1]), "AE")


def test_four_way_overlap_and_normal_only_semantics():
    # Shared supervised misses: both anomaly, IF only, AE only, neither.
    bits = np.array([[0, 0, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0],
                     [1, 1, 1, 1], [0, 0, 0, 1], [0, 0, 1, 0], [1, 1, 1, 1]])
    frame = pd.DataFrame(bits, columns=[f"{m}_prediction" for m in MODELS])
    frame["true_label"] = [1] * 5 + [0] * 3
    frame["attack_cat"] = ["Fuzzers"] * 5 + ["Normal"] * 3
    cohorts, per_attack, patterns, important, fp = analyze(frame)
    shared = cohorts["shared_supervised_misses"]
    assert shared["total_misses"] == 4
    assert shared["ae_recovered"] == shared["if_recovered"] == 2
    assert shared["either_recovered"] == 3
    assert shared["ae_recovery_percentage"] == 50
    assert all(shared[k] == 1 for k in ("both_if_and_ae", "only_if", "only_ae", "neither"))
    assert cohorts["previous_three_model_misses"]["ae_recovered"] == 1
    assert important["ae_only_detections"] == important["if_only_detections"] == 1
    assert per_attack["Fuzzers"]["missed_by_all_four"] == 1
    assert patterns.attack_count.sum() == 5
    assert patterns.normal_count.sum() == 3
    assert fp["ae_only_among_all_four"] == fp["if_only_among_all_four"] == 1
    assert fp["ae_and_if_inclusive"] == fp["all_four"] == 1
    assert fp["standalone"]["autoencoder"]["count"] == 2


def test_empty_cohort_is_not_reported_as_zero_percent_recovery():
    empty = np.zeros(3, dtype=bool)
    result = recovery(empty, ~empty, ~empty)
    assert result["total_misses"] == 0
    assert result["ae_recovery_percentage"] is None
    assert result["if_recovery_percentage"] is None
