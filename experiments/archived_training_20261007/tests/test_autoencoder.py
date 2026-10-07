import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from train_autoencoder import make_autoencoder_features, reconstruction_errors  # noqa: E402
from optimize_autoencoder import (  # noqa: E402
    EXPECTED_BASELINE,
    add_baseline_differences,
    selection_key,
)


def test_make_autoencoder_features_scales_numeric_and_preserves_one_hot():
    X = sparse.csr_matrix(
        np.array([[1.0, 10.0, 1.0, 0.0], [3.0, 20.0, 0.0, 1.0]], dtype=np.float32)
    )
    scaler = StandardScaler().fit(X[:, :2].toarray())

    transformed = make_autoencoder_features(X, 2, scaler).toarray()

    np.testing.assert_allclose(transformed[:, :2].mean(axis=0), 0.0, atol=1e-7)
    np.testing.assert_array_equal(transformed[:, 2:], X[:, 2:].toarray())


def test_reconstruction_errors_returns_per_row_mse():
    X = sparse.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))
    model = MLPRegressor(hidden_layer_sizes=(2,), max_iter=1, random_state=42)
    model.fit(X, X.toarray())

    expected = np.mean(np.square(model.predict(X) - X.toarray()), axis=1)

    np.testing.assert_allclose(reconstruction_errors(model, X, batch_size=1), expected)


def _candidate(recall, pr_auc, f1=0.8, balanced_accuracy=0.8, roc_auc=0.9):
    return {
        "architecture": [128, 32, 128],
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "operating_points": {
            "fpr_10": {
                "recall": recall,
                "f1": f1,
                "balanced_accuracy": balanced_accuracy,
            }
        },
    }


def test_selection_key_prioritizes_recall_at_fpr_10():
    lower_recall = _candidate(0.80, 0.99)
    higher_recall = _candidate(0.81, 0.90)

    assert max([lower_recall, higher_recall], key=selection_key) is higher_recall


def test_candidate_differences_use_published_baseline():
    candidate = _candidate(0.80, 0.97, f1=0.87, balanced_accuracy=0.86)

    add_baseline_differences(candidate, EXPECTED_BASELINE)

    differences = candidate["improvement_over_existing_baseline"]
    assert differences["recall"] == 0.80 - EXPECTED_BASELINE["recall"]
    assert differences["pr_auc"] == 0.97 - EXPECTED_BASELINE["pr_auc"]
