"""Select RF and LightGBM probability thresholds on training-only validation data.

The saved full-training models are not modified. Same-parameter selection clones
are fitted on an internal portion of the official training split so validation
predictions are leakage-free. The resulting thresholds can then be transferred
to the full-training models for one final official-test evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import sparse
from sklearn.base import clone
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from training_utils import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR

MODEL_FILES = {
    "random_forest": "random_forest.joblib",
    "lightgbm": "lightgbm.joblib",
}
DEFAULT_THRESHOLD_DIR = DEFAULT_OUTPUT_DIR / "binary_thresholds"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_THRESHOLD_DIR)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_official_training_data(data_dir: Path) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Load only official-training artifacts; never touch the official test set."""
    matrix_path = data_dir / "X_train_tree.npz"
    target_path = data_dir / "y_train.npy"
    if not matrix_path.is_file() or not target_path.is_file():
        raise FileNotFoundError("Missing X_train_tree.npz or y_train.npy")
    X_train = sparse.load_npz(matrix_path).tocsr()
    y_train = np.load(target_path, allow_pickle=False)
    if X_train.shape[0] != y_train.size:
        raise ValueError("Official training matrix and target row counts differ")
    if not set(np.unique(y_train)).issubset({0, 1}):
        raise ValueError("Training labels must be binary 0/1")
    return X_train, y_train


def threshold_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if fp + tn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "probability_threshold": float(threshold),
        "false_positive_rate": float(fpr),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def select_operating_point(
    y_true: np.ndarray, probabilities: np.ndarray, fpr_limit: float
) -> dict[str, Any]:
    fprs, recalls, thresholds = roc_curve(
        y_true, probabilities, pos_label=1, drop_intermediate=False
    )
    eligible = np.flatnonzero(fprs <= fpr_limit + np.finfo(float).eps)
    if not eligible.size:
        raise RuntimeError(f"No probability threshold satisfies FPR <= {fpr_limit}")

    # Primary objective: maximum recall. Required tie-breaker: maximum F1.
    best_recall = recalls[eligible].max()
    recall_ties = eligible[recalls[eligible] == best_recall]
    candidates = [threshold_metrics(y_true, probabilities, thresholds[i]) for i in recall_ties]
    selected = max(
        candidates,
        key=lambda result: (result["f1"], result["probability_threshold"]),
    )
    selected["fpr_limit"] = float(fpr_limit)
    return selected


def main() -> None:
    args = parse_args()
    if not 0 < args.validation_size < 1:
        raise ValueError("--validation-size must be between 0 and 1")

    X_train, y_train = load_official_training_data(args.data_dir)
    fit_indices, validation_indices = train_test_split(
        np.arange(y_train.size),
        test_size=args.validation_size,
        stratify=y_train,
        random_state=args.random_state,
    )
    X_fit, y_fit = X_train[fit_indices], y_train[fit_indices]
    X_validation, y_validation = X_train[validation_indices], y_train[validation_indices]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, filename in MODEL_FILES.items():
        full_training_model = joblib.load(args.model_dir / filename)
        selection_model = clone(full_training_model)
        selection_model.fit(X_fit, y_fit)
        probabilities = selection_model.predict_proba(X_validation)[:, 1]
        operating_points = {
            f"fpr_{int(limit * 100):02d}": select_operating_point(
                y_validation, probabilities, limit
            )
            for limit in (0.05, 0.10, 0.15)
        }
        config = {
            "model": model_name,
            "source_model": str((args.model_dir / filename).resolve()),
            "source_model_parameters": full_training_model.get_params(),
            "selection_model_parameters": selection_model.get_params(),
            "split": {
                "official_training_rows": int(y_train.size),
                "internal_fit_rows": int(fit_indices.size),
                "validation_rows": int(validation_indices.size),
                "validation_normal_rows": int((y_validation == 0).sum()),
                "validation_attack_rows": int(y_validation.sum()),
                "validation_fraction": float(args.validation_size),
                "stratified_on": "binary label",
                "random_state": int(args.random_state),
            },
            "probability": "model.predict_proba(X)[:, 1]",
            "prediction": "(attack_probability >= selected_threshold).astype(int)",
            "selection_rule": (
                "maximum attack recall subject to FPR <= 0.10; "
                "highest F1 breaks recall ties"
            ),
            "selected_threshold": operating_points["fpr_10"]["probability_threshold"],
            "selected_operating_point": operating_points["fpr_10"],
            "operating_points": operating_points,
            "official_test_used": False,
        }
        output_path = args.output_dir / f"{model_name}_threshold_config.json"
        output_path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(config, indent=2, sort_keys=True))
        print(f"Saved {output_path.resolve()}")


if __name__ == "__main__":
    main()
