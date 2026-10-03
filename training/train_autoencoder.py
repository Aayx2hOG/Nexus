"""Train an autoencoder anomaly detector on normal UNSW-NB15 traffic.

The model follows the same evaluation protocol as ``train_isolation_forest.py``:
only normal rows from the internal training split are used for fitting, and a
labelled validation split is used to freeze an operating threshold.  The
official test split is never scored by this script.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import sparse
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from train_isolation_forest import select_operating_point
from training_utils import DEFAULT_DATA_DIR, PROJECT_ROOT, load_tree_data

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "autoencoder"
MODEL_FILENAME = "autoencoder.joblib"
CONFIG_FILENAME = "autoencoder_config.json"
VALIDATION_METRICS_FILENAME = "autoencoder_validation_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[128, 32, 128])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _json_dump(value: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_autoencoder_features(
    X: sparse.csr_matrix, numerical_feature_count: int, scaler: StandardScaler
) -> sparse.csr_matrix:
    """Scale numeric inputs and retain sparse one-hot categorical inputs."""
    numeric = scaler.transform(X[:, :numerical_feature_count].toarray()).astype(np.float32)
    categorical = X[:, numerical_feature_count:].astype(np.float32)
    return sparse.hstack([sparse.csr_matrix(numeric), categorical], format="csr")


def reconstruction_errors(
    model: MLPRegressor, X: sparse.csr_matrix, batch_size: int
) -> np.ndarray:
    """Return per-row mean squared reconstruction errors in bounded memory."""
    errors = np.empty(X.shape[0], dtype=np.float64)
    for start in range(0, X.shape[0], batch_size):
        stop = min(start + batch_size, X.shape[0])
        batch = X[start:stop]
        reconstructed = model.predict(batch)
        residual = reconstructed - batch.toarray()
        errors[start:stop] = np.mean(np.square(residual), axis=1)
    return errors


def fit_autoencoder(
    X_fit: sparse.csr_matrix,
    X_early_stop: sparse.csr_matrix,
    hidden_layers: tuple[int, ...],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    random_state: int,
    alpha: float = 1e-4,
) -> tuple[MLPRegressor, list[dict[str, float | int]], int]:
    """Train mini-batches and restore the epoch with lowest normal validation MSE."""
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch-size, and patience must all be positive")

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        batch_size=batch_size,
        learning_rate_init=learning_rate,
        alpha=alpha,
        max_iter=1,
        shuffle=False,
        random_state=random_state,
    )
    rng = np.random.default_rng(random_state)
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_weights: tuple[list[np.ndarray], list[np.ndarray]] | None = None
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        order = rng.permutation(X_fit.shape[0])
        for start in range(0, order.size, batch_size):
            batch = X_fit[order[start : start + batch_size]]
            # Avoid sklearn clipping (and warning about) the final short batch.
            model.batch_size = min(batch_size, batch.shape[0])
            model.partial_fit(batch, batch.toarray())

        validation_loss = float(
            reconstruction_errors(model, X_early_stop, batch_size).mean()
        )
        history.append({"epoch": epoch, "normal_validation_mse": validation_loss})
        print(f"Epoch {epoch:02d}: normal validation MSE={validation_loss:.8f}")
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_weights = (copy.deepcopy(model.coefs_), copy.deepcopy(model.intercepts_))
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_weights is None:
        raise RuntimeError("Autoencoder training did not produce a model")
    model.coefs_, model.intercepts_ = best_weights
    return model, history, best_epoch


def main() -> None:
    args = parse_args()
    X_train, _, y_train, _ = load_tree_data(args.data_dir)
    metadata = json.loads(
        (args.data_dir / "preprocessing_metadata.json").read_text(encoding="utf-8")
    )
    numerical_feature_count = len(metadata["numerical_columns"])

    train_indices, validation_indices = train_test_split(
        np.arange(y_train.size),
        test_size=0.2,
        stratify=y_train,
        random_state=args.random_state,
    )
    normal_train_indices = train_indices[y_train[train_indices] == 0]
    normal_validation_indices = validation_indices[y_train[validation_indices] == 0]

    scaler = StandardScaler()
    scaler.fit(X_train[normal_train_indices, :numerical_feature_count].toarray())
    X_fit = make_autoencoder_features(
        X_train[normal_train_indices], numerical_feature_count, scaler
    )
    X_early_stop = make_autoencoder_features(
        X_train[normal_validation_indices], numerical_feature_count, scaler
    )
    X_validation = make_autoencoder_features(
        X_train[validation_indices], numerical_feature_count, scaler
    )
    y_validation = y_train[validation_indices]

    started = time.perf_counter()
    model, history, best_epoch = fit_autoencoder(
        X_fit,
        X_early_stop,
        tuple(args.hidden_layers),
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.patience,
        args.random_state,
    )
    training_seconds = time.perf_counter() - started

    fit_scores = reconstruction_errors(model, X_fit, args.batch_size)
    validation_scores = reconstruction_errors(model, X_validation, args.batch_size)
    operating_points = {
        f"fpr_{int(limit * 100):02d}": select_operating_point(
            y_validation, validation_scores, limit
        )
        for limit in (0.05, 0.10, 0.15)
    }
    selected = operating_points["fpr_10"]
    normalization_min = float(min(fit_scores.min(), validation_scores.min()))
    normalization_max = float(max(fit_scores.max(), validation_scores.max()))
    if normalization_max <= normalization_min:
        raise RuntimeError("Reconstruction errors are constant; cannot normalize scores")

    validation_metrics = {
        "roc_auc": float(roc_auc_score(y_validation, validation_scores)),
        "pr_auc": float(average_precision_score(y_validation, validation_scores)),
        "selected_operating_point": selected,
        "operating_points": operating_points,
        "validation_rows": int(validation_indices.size),
        "validation_attack_rows": int(y_validation.sum()),
        "validation_normal_rows": int((y_validation == 0).sum()),
    }
    config = {
        "model": "MLPRegressorAutoencoder",
        "parameters": {
            "hidden_layers": args.hidden_layers,
            "activation": "relu",
            "optimizer": "adam",
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "maximum_epochs": args.epochs,
            "patience": args.patience,
        },
        "random_state": args.random_state,
        "split": {
            "official_training_rows": int(y_train.size),
            "internal_training_rows": int(train_indices.size),
            "fit_normal_rows": int(normal_train_indices.size),
            "validation_rows": int(validation_indices.size),
            "validation_fraction": 0.2,
            "stratified_on": "binary label",
        },
        "feature_scaling": "numeric StandardScaler fitted on normal fit rows; one-hot unchanged",
        "anomaly_score": "per-row mean squared reconstruction error",
        "threshold_selection": "maximum attack recall subject to FPR <= 0.10",
        "selected_threshold": selected["threshold"],
        "operating_points": operating_points,
        "normalization": {
            "method": "min-max with clipping",
            "fit_data": "normal internal training and complete validation scores",
            "minimum": normalization_min,
            "maximum": normalization_max,
        },
        "training": {
            "best_epoch": best_epoch,
            "epochs_completed": len(history),
            "history": history,
            "training_seconds": float(training_seconds),
        },
        "feature_count": int(X_train.shape[1]),
        "numerical_feature_count": numerical_feature_count,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "numeric_scaler": scaler}, args.output_dir / MODEL_FILENAME
    )
    _json_dump(config, args.output_dir / CONFIG_FILENAME)
    _json_dump(validation_metrics, args.output_dir / VALIDATION_METRICS_FILENAME)
    print(json.dumps(validation_metrics, indent=2, sort_keys=True))
    print(f"Training time: {training_seconds:.3f} seconds")
    print(f"Saved artifacts to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
