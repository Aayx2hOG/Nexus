"""Evaluate frozen RF and LightGBM thresholds once on the official test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from select_binary_thresholds import DEFAULT_THRESHOLD_DIR, MODEL_FILES, threshold_metrics
from training_utils import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, PROJECT_ROOT, load_tree_data

DEFAULT_RAW_TEST = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "CSV_Files"
    / "Training and Testing Sets"
    / "UNSW_NB15_testing-set.csv"
)
BASELINES = {
    "random_forest": {
        "probability_threshold": 0.5,
        "precision": 0.8166,
        "recall": 0.9876,
        "f1": 0.8940,
        "false_positive_rate": 0.2717,
        "roc_auc": 0.9804,
    },
    "lightgbm": {
        "probability_threshold": 0.5,
        "precision": 0.8662,
        "recall": 0.9703,
        "f1": 0.9153,
        "false_positive_rate": 0.1836,
        "roc_auc": 0.9842,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold-dir", type=Path, default=DEFAULT_THRESHOLD_DIR)
    parser.add_argument("--raw-test-csv", type=Path, default=DEFAULT_RAW_TEST)
    return parser.parse_args()


def load_attack_categories(path: Path, y_test: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(path, usecols=["attack_cat", "label"], low_memory=False)
    labels = frame["label"].to_numpy(dtype=np.int64)
    if labels.shape != y_test.shape or not np.array_equal(labels, y_test):
        raise ValueError("Raw test CSV does not align with processed test labels")
    return frame["attack_cat"].fillna("Normal").astype(str).to_numpy()


def main() -> None:
    args = parse_args()
    _, X_test, _, y_test = load_tree_data(args.data_dir)
    attack_categories = load_attack_categories(args.raw_test_csv, y_test)

    for model_name, filename in MODEL_FILES.items():
        config_path = args.threshold_dir / f"{model_name}_threshold_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        threshold = float(config["selected_threshold"])
        model = joblib.load(args.model_dir / filename)
        probabilities = model.predict_proba(X_test)[:, 1]
        metrics = threshold_metrics(y_test, probabilities, threshold)
        metrics.update(
            {
                "model": model_name,
                "roc_auc": float(roc_auc_score(y_test, probabilities)),
                "pr_auc": float(average_precision_score(y_test, probabilities)),
                "baseline_at_0_5": BASELINES[model_name],
                "threshold_source": str(config_path.resolve()),
            }
        )
        # Explicit calculations guard the shared metric contract in this final report.
        predictions = (probabilities >= threshold).astype(np.int64)
        tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
        metrics["precision"] = float(precision_score(y_test, predictions, zero_division=0))
        metrics["recall"] = float(recall_score(y_test, predictions, zero_division=0))
        metrics["f1"] = float(f1_score(y_test, predictions, zero_division=0))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_test, predictions))
        metrics["specificity"] = float(tn / (tn + fp)) if tn + fp else 0.0

        per_attack: dict[str, Any] = {}
        for category in sorted(np.unique(attack_categories)):
            mask = attack_categories == category
            category_labels = y_test[mask]
            category_predictions = predictions[mask]
            attack_rows = int(category_labels.sum())
            detected = int(((category_labels == 1) & (category_predictions == 1)).sum())
            per_attack[category] = {
                "records": int(mask.sum()),
                "attack_records": attack_rows,
                "detected_attacks": detected,
                "missed_attacks": int(attack_rows - detected),
                "attack_recall": float(detected / attack_rows) if attack_rows else None,
            }

        metrics_path = args.threshold_dir / f"{model_name}_tuned_test_metrics.json"
        per_attack_path = args.threshold_dir / f"{model_name}_tuned_per_attack_metrics.json"
        metrics_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        per_attack_path.write_text(
            json.dumps(per_attack, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(metrics, indent=2, sort_keys=True))
        print(json.dumps({"model": model_name, "per_attack": per_attack}, indent=2))
        print(f"Saved {metrics_path.resolve()} and {per_attack_path.resolve()}")


if __name__ == "__main__":
    main()
