"""Small synthetic checks; no trained model or dataset required."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from compare_normal_traffic import compare_groups  # noqa: E402


def test_decomposition_accounts_for_shared_and_unmatched_groups():
    dev = pd.DataFrame({"service": ["a", "a", "b", "b"], "alert": [0, 1, 0, 0]})
    hist = pd.DataFrame({"service": ["a", "a", "a", "c"], "alert": [1, 1, 0, 1]})
    result = compare_groups(dev, hist, ["service"])
    total = (
        result[["mix_contribution_pp", "within_group_contribution_pp", "unmatched_contribution_pp"]]
        .sum()
        .sum()
    )
    assert total == pytest.approx(100 * (hist.alert.mean() - dev.alert.mean()))
    assert result.set_index("service").loc["c", "normal_rows_development"] == 0
    assert pd.isna(result.set_index("service").loc["c", "fpr_change_pp"])


def test_identical_populations_have_zero_shift():
    frame = pd.DataFrame({"service": ["a", "a"], "sttl": [64, 64], "alert": [0, 1]})
    result = compare_groups(frame, frame, ["service", "sttl"])
    assert result.share_change_pp.iloc[0] == 0
    assert result.fpr_change_pp.iloc[0] == 0
