"""Boundary behavior, residual aggregation, and model-selection checks."""

import numpy as np
import pandas as pd
import pytest
from experiment_uncertainty_residuals import (
    band_predict,
    meta_features,
    recall_threshold,
    residual_features,
    select_candidate,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def test_band_preserves_confident_predictions_and_includes_endpoints():
    p = np.array([0.1, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9])
    error = np.array([100.0, 2.0, 1.0, 0.0, 2.0, 0.0, 0.0])
    pred = band_predict(p, error, 0.6, 0.2, 0.8, 0.5, 2.0)
    np.testing.assert_array_equal(pred, [False, True, False, False, True, False, True])


def test_band_error_threshold_equality():
    pred = band_predict([0.6, 0.5], [0.5, 2.0], 0.6, 0.2, 0.8, 0.5, 2.0)
    np.testing.assert_array_equal(pred, [True, True])


class ZeroReconstructor:
    def predict(self, inputs):
        return np.zeros(inputs.shape, dtype=np.float32)


def test_categorical_residuals_aggregate_into_original_feature():
    frame = pd.DataFrame({"number": [0.0, 2.0], "category": ["a", "b"]})
    bundle = {
        "numerical_columns": ["number"],
        "categorical_columns": ["category"],
        "imputer": SimpleImputer().fit(frame[["number"]]),
        "encoder": OneHotEncoder(dtype=np.float32).fit(frame[["category"]]),
        "scaler": StandardScaler().fit(frame[["number"]].to_numpy()),
        "model": ZeroReconstructor(),
    }
    mse, residuals, names = residual_features(bundle, frame, batch_size=1)
    assert names == ["number", "category"]
    np.testing.assert_allclose(residuals, [[1, 1], [1, 1]])
    np.testing.assert_allclose(mse, [2 / 3, 2 / 3])


def test_extreme_scores_have_finite_meta_features():
    for kind, width in [
        ("Score-only logistic", 1),
        ("Total-error logistic", 3),
        ("Per-feature logistic", 4),
    ]:
        x = meta_features(np.array([0.0, 1.0]), np.array([0.0, 3.0]), np.ones((2, 3)), kind)
        assert x.shape == (2, width)
        assert np.isfinite(x).all()


def test_selection_enforces_recall_before_fpr():
    candidates = [
        {"selection": {"recall": 0.94, "fpr": 0.01, "f1": 0.96}},
        {"selection": {"recall": 0.96, "fpr": 0.05, "f1": 0.95}},
        {"selection": {"recall": 0.95, "fpr": 0.04, "f1": 0.95}},
    ]
    assert select_candidate(candidates) is candidates[2]


def test_recall_threshold_handles_ties():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.5, 0.5, 0.9])
    assert recall_threshold(y, p, 0.95) == 0.5


def test_invalid_band():
    with pytest.raises(ValueError):
        band_predict([0.5], [1], 0.6, 0.7, 0.9, 0.1, 2)
