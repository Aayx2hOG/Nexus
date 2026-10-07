"""Decision-table and boundary checks for the four-mode experiment."""

import numpy as np
import pytest
from experiment_four_mode_ae import four_mode


def test_all_eight_decisions_and_exact_boundaries():
    scores = np.repeat([0.0, 1.0, 2.0, 3.0], 2)
    pred, modes = four_mode([False, True] * 4, scores, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(modes, [0, 0, 1, 1, 2, 2, 3, 3])
    np.testing.assert_array_equal(pred, [False, False, False, True, True, True, True, True])


def test_high_boundary_does_not_change_binary_decision():
    scores = np.array([0.0, 1.5, 2.5, 4.0, 10.0])
    first, _ = four_mode([True] * 5, scores, [1.0, 2.0, 3.0])
    second, _ = four_mode([True] * 5, scores, [1.0, 2.0, 9.0])
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("thresholds", [[1, 1, 2], [3, 2, 1], [1, 2], [1, 2, np.nan]])
def test_invalid_thresholds(thresholds):
    with pytest.raises(ValueError):
        four_mode([True], [1.0], thresholds)
