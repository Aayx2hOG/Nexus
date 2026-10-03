"""Train Isolation Forest on normal UNSW-NB15 traffic and select a threshold.

This script reuses the sparse tree matrices produced by
``preprocessing/preprocess_unsw_nb15.py``.  The official test split is loaded
only to validate the processed artifacts; it is not scored during training.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import sparse
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from training_utils import DEFAULT_DATA_DIR, PROJECT_ROOT, load_tree_data

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "isolation_forest"
MODEL_FILENAME = "isolation_forest.joblib"
CONFIG_FILENAME = "isolation_forest_config.json"
VALIDATION_METRICS_FILENAME = "isolation_forest_validation_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _json_dump(value: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if fp + tn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "false_positive_rate": float(fpr),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def select_operating_point(
    y_true: np.ndarray, scores: np.ndarray, maximum_fpr: float
) -> dict[str, Any]:
    """Maximize attack recall subject to the empirical FPR constraint.

    ``roc_curve`` returns thresholds compatible with ``scores >= threshold``.
    Ties in recall are resolved by lower FPR, then by the highest threshold.
    """
    fprs, recalls, thresholds = roc_curve(y_true, scores, pos_label=1, drop_intermediate=False)
    eligible = np.flatnonzero(fprs <= maximum_fpr + np.finfo(float).eps)
    if eligible.size == 0:
        raise RuntimeError(f"No threshold satisfies FPR <= {maximum_fpr}")
    best_recall = recalls[eligible].max()
    eligible = eligible[recalls[eligible] == best_recall]
    best_fpr = fprs[eligible].min()
    eligible = eligible[fprs[eligible] == best_fpr]
    index = eligible[np.argmax(thresholds[eligible])]
    result = binary_metrics(y_true, scores, float(thresholds[index]))
    result["fpr_limit"] = float(maximum_fpr)
    return result


def feature_diagnostics(X: sparse.csr_matrix, feature_names: list[str]) -> dict[str, Any]:
    data = X.data
    nan_count = int(np.isnan(data).sum())
    infinite_count = int(np.isinf(data).sum())
    minima = np.asarray(X.min(axis=0).toarray()).ravel()
    maxima = np.asarray(X.max(axis=0).toarray()).ravel()
    constant_indices = np.flatnonzero(minima == maxima)
    return {
        "nan_values": nan_count,
        "infinite_values": infinite_count,
        "constant_feature_count": int(constant_indices.size),
        "constant_features": [feature_names[index] for index in constant_indices],
    }


def main() -> None:
    args = parse_args()
    X_train, _, y_train, _ = load_tree_data(args.data_dir)
    metadata_path = args.data_dir / "preprocessing_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_names = metadata["tree_feature_names"]
    if len(feature_names) != X_train.shape[1]:
        raise ValueError("Metadata feature names do not match the processed matrix")

    train_indices, validation_indices = train_test_split(
        np.arange(y_train.size),
        test_size=0.2,
        stratify=y_train,
        random_state=42,
    )
    normal_train_indices = train_indices[y_train[train_indices] == 0]
    X_fit = X_train[normal_train_indices]
    X_validation = X_train[validation_indices]
    y_validation = y_train[validation_indices]

    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        max_features=1.0,
        max_samples="auto",
        bootstrap=False,
        n_jobs=-1,
        random_state=42,
    )
    started = time.perf_counter()
    model.fit(X_fit)
    training_seconds = time.perf_counter() - started

    # Higher values must consistently mean more anomalous traffic.
    validation_scores = -model.decision_function(X_validation)
    fit_scores = -model.decision_function(X_fit)
    operating_points = {
        f"fpr_{int(limit * 100):02d}": select_operating_point(
            y_validation, validation_scores, limit
        )
        for limit in (0.05, 0.10, 0.15)
    }
    selected = operating_points["fpr_10"]

    # The normalization is fitted exclusively on fit/validation scores. Values
    # outside this observed range are clipped to keep final scores in [0, 1].
    normalization_min = float(min(fit_scores.min(), validation_scores.min()))
    normalization_max = float(max(fit_scores.max(), validation_scores.max()))
    if normalization_max <= normalization_min:
        raise RuntimeError("Anomaly scores are constant; cannot fit 0-1 normalization")

    diagnostics = feature_diagnostics(X_train, feature_names)
    diagnostics.update(
        {
            "feature_count": int(X_train.shape[1]),
            "numerical_feature_count": len(metadata["numerical_columns"]),
            "one_hot_feature_count": int(
                X_train.shape[1] - len(metadata["numerical_columns"])
            ),
            "categorical_source_columns": metadata["categorical_columns"],
            "feature_count_explanation": (
                f"{len(metadata['numerical_columns'])} numeric columns + "
                f"{X_train.shape[1] - len(metadata['numerical_columns'])} one-hot columns "
                f"from {metadata['categorical_columns']} = {X_train.shape[1]} features"
            ),
        }
    )
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
        "model": "IsolationForest",
        "parameters": model.get_params(),
        "random_state": 42,
        "split": {
            "official_training_rows": int(y_train.size),
            "internal_training_rows": int(train_indices.size),
            "fit_normal_rows": int(normal_train_indices.size),
            "validation_rows": int(validation_indices.size),
            "validation_fraction": 0.2,
            "stratified_on": "binary label",
        },
        "anomaly_score": "-model.decision_function(X)",
        "threshold_selection": "maximum attack recall subject to FPR <= 0.10",
        "selected_threshold": selected["threshold"],
        "operating_points": operating_points,
        "normalization": {
            "method": "min-max with clipping",
            "fit_data": "normal internal training and complete validation scores",
            "minimum": normalization_min,
            "maximum": normalization_max,
        },
        "training_seconds": float(training_seconds),
        "feature_diagnostics": diagnostics,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output_dir / MODEL_FILENAME)
    _json_dump(config, args.output_dir / CONFIG_FILENAME)
    _json_dump(validation_metrics, args.output_dir / VALIDATION_METRICS_FILENAME)

    print(json.dumps(validation_metrics, indent=2, sort_keys=True))
    print(json.dumps({"feature_diagnostics": diagnostics}, indent=2, sort_keys=True))
    print(f"Training time: {training_seconds:.3f} seconds")
    print(f"Saved artifacts to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
