"""Shared data loading, evaluation, and persistence for multiclass models."""

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

from training_utils import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, PROJECT_ROOT


def load_multiclass_tree_data(
    data_dir: Path,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray, np.ndarray, list[str]]:
    required = {
        "X_train": data_dir / "X_train_tree.npz",
        "X_test": data_dir / "X_test_tree.npz",
        "y_train": data_dir / "y_train_multiclass.npy",
        "y_test": data_dir / "y_test_multiclass.npy",
        "labels": data_dir / "multiclass_labels.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing multiclass input file(s): "
            + ", ".join(missing)
            + ". Run preprocessing/prepare_multiclass_targets.py first."
        )

    X_train = sparse.load_npz(required["X_train"]).tocsr()
    X_test = sparse.load_npz(required["X_test"]).tocsr()
    y_train = np.load(required["y_train"], allow_pickle=False)
    y_test = np.load(required["y_test"], allow_pickle=False)
    metadata = json.loads(required["labels"].read_text(encoding="utf-8"))
    class_names = metadata.get("class_names")

    if not isinstance(class_names, list) or not all(
        isinstance(name, str) for name in class_names
    ):
        raise ValueError("multiclass_labels.json must contain a string class_names list")
    if len(class_names) < 3:
        raise ValueError("Multiclass training requires at least three classes")
    if X_train.shape[0] != y_train.shape[0] or X_test.shape[0] != y_test.shape[0]:
        raise ValueError("Feature matrices and multiclass targets have inconsistent row counts")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Train and test matrices contain different numbers of features")

    expected = np.arange(len(class_names))
    if not np.array_equal(np.unique(y_train), expected):
        raise ValueError("Training labels must contain every class ID from 0 to n_classes - 1")
    if not set(np.unique(y_test)).issubset(set(expected)):
        raise ValueError("Test labels contain an unknown class ID")
    return X_train, X_test, y_train, y_test, class_names


def evaluate_multiclass_classifier(
    model: Any,
    X_test: sparse.csr_matrix,
    y_test: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    labels = list(range(len(class_names)))

    if list(model.classes_) != labels:
        raise ValueError("Model probability columns do not match the configured class IDs")

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "precision_macro": float(
            precision_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_test, predictions, average="weighted", zero_division=0)
        ),
        "roc_auc_ovr_macro": float(
            roc_auc_score(y_test, probabilities, labels=labels, multi_class="ovr", average="macro")
        ),
        "roc_auc_ovr_weighted": float(
            roc_auc_score(
                y_test, probabilities, labels=labels, multi_class="ovr", average="weighted"
            )
        ),
        "class_names": class_names,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels)
        .astype(int)
        .tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }


def save_multiclass_results(
    model: Any, metrics: dict[str, Any], output_dir: Path, name: str
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{name}.joblib"
    metrics_path = output_dir / f"{name}_metrics.json"
    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")


__all__ = [
    "DEFAULT_DATA_DIR",
    "DEFAULT_OUTPUT_DIR",
    "PROJECT_ROOT",
    "evaluate_multiclass_classifier",
    "load_multiclass_tree_data",
    "save_multiclass_results",
]
