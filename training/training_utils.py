"""Shared utilities for the tree-model training scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "models"


def load_tree_data(
    data_dir: Path,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Load and validate the preprocessed sparse matrices and binary targets."""
    required = {
        "X_train": data_dir / "X_train_tree.npz",
        "X_test": data_dir / "X_test_tree.npz",
        "y_train": data_dir / "y_train.npy",
        "y_test": data_dir / "y_test.npy",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing preprocessed file(s): " + ", ".join(missing))

    X_train = sparse.load_npz(required["X_train"]).tocsr()
    X_test = sparse.load_npz(required["X_test"]).tocsr()
    y_train = np.load(required["y_train"], allow_pickle=False)
    y_test = np.load(required["y_test"], allow_pickle=False)

    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train and y_train contain different numbers of rows")
    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError("X_test and y_test contain different numbers of rows")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Train and test matrices contain different numbers of features")
    if not set(np.unique(y_train)).issubset({0, 1}):
        raise ValueError("y_train must contain binary labels encoded as 0 and 1")

    return X_train, X_test, y_train, y_test


def evaluate_binary_classifier(
    model: Any, X_test: sparse.csr_matrix, y_test: np.ndarray
) -> dict[str, Any]:
    """Return JSON-serializable metrics for a fitted binary classifier."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()

    false_positive_rate = fp / (fp + tn) if fp + tn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "false_positive_rate": float(false_positive_rate),
        "specificity": float(specificity),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["normal", "attack"],
            output_dict=True,
            zero_division=0,
        ),
    }


def save_results(model: Any, metrics: dict[str, Any], output_dir: Path, name: str) -> None:
    """Persist a fitted estimator and its evaluation metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{name}.joblib"
    metrics_path = output_dir / f"{name}_metrics.json"

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")
