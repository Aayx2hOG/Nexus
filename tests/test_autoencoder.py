import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from train_autoencoder import make_autoencoder_features, reconstruction_errors  # noqa: E402


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
