"""Evaluate a frozen Isolation Forest threshold on official UNSW-NB15 test data."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from train_isolation_forest import (
    CONFIG_FILENAME,
    DEFAULT_OUTPUT_DIR,
    MODEL_FILENAME,
)
from training_utils import DEFAULT_DATA_DIR, PROJECT_ROOT, load_tree_data

DEFAULT_RAW_TEST = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "CSV_Files"
    / "Training and Testing Sets"
    / "UNSW_NB15_testing-set.csv"
)
METRICS_FILENAME = "isolation_forest_metrics.json"
PER_ATTACK_FILENAME = "isolation_forest_per_attack_metrics.json"
SCORES_FILENAME = "isolation_forest_test_scores.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--raw-test-csv", type=Path, default=DEFAULT_RAW_TEST)
    return parser.parse_args()


def _json_dump(value: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_attack_categories(path: Path, y_test: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(path, usecols=["attack_cat", "label"], low_memory=False)
    raw_labels = frame["label"].to_numpy(dtype=np.int64)
    if raw_labels.shape != y_test.shape or not np.array_equal(raw_labels, y_test):
        raise ValueError("Raw test CSV does not align with processed y_test")
    return frame["attack_cat"].fillna("Normal").astype(str).to_numpy()


def main() -> None:
    args = parse_args()
    _, X_test, _, y_test = load_tree_data(args.data_dir)
    attack_categories = load_attack_categories(args.raw_test_csv, y_test)

    model = joblib.load(args.artifact_dir / MODEL_FILENAME)
    config = json.loads((args.artifact_dir / CONFIG_FILENAME).read_text(encoding="utf-8"))
    threshold = float(config["selected_threshold"])
    norm_min = float(config["normalization"]["minimum"])
    norm_max = float(config["normalization"]["maximum"])

    started = time.perf_counter()
    scores = -model.decision_function(X_test)
    predictions = (scores >= threshold).astype(np.int64)
    inference_seconds = time.perf_counter() - started
    throughput = len(y_test) / inference_seconds if inference_seconds else float("inf")

    normalized_scores = np.clip((scores - norm_min) / (norm_max - norm_min), 0.0, 1.0)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if fp + tn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    metrics = {
        "model": "IsolationForest",
        "test_rows": int(y_test.size),
        "selected_threshold": threshold,
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "false_positive_rate": float(fpr),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "pr_auc": float(average_precision_score(y_test, scores)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["normal", "attack"],
            output_dict=True,
            zero_division=0,
        ),
        "timing": {
            "inference_seconds": float(inference_seconds),
            "records_per_second": float(throughput),
            "milliseconds_per_record": float(1000.0 * inference_seconds / len(y_test)),
            "includes": "decision_function and thresholding",
        },
    }

    per_attack: dict[str, Any] = {}
    for category in sorted(np.unique(attack_categories)):
        mask = attack_categories == category
        category_labels = y_test[mask]
        category_predictions = predictions[mask]
        attack_rows = int(category_labels.sum())
        detected_attacks = int(((category_labels == 1) & (category_predictions == 1)).sum())
        per_attack[category] = {
            "records": int(mask.sum()),
            "attack_records": attack_rows,
            "detected_attacks": detected_attacks,
            "missed_attacks": int(attack_rows - detected_attacks),
            "attack_recall": float(detected_attacks / attack_rows) if attack_rows else None,
            "mean_anomaly_score": float(scores[mask].mean()),
            "median_anomaly_score": float(np.median(scores[mask])),
            "anomaly_prediction_rate": float(category_predictions.mean()),
        }

    score_frame = pd.DataFrame(
        {
            "true_label": y_test,
            "attack_cat": attack_categories,
            "anomaly_score": scores,
            "normalized_anomaly_score": normalized_scores,
            "anomaly_prediction": predictions,
        }
    )
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    _json_dump(metrics, args.artifact_dir / METRICS_FILENAME)
    _json_dump(per_attack, args.artifact_dir / PER_ATTACK_FILENAME)
    score_frame.to_csv(args.artifact_dir / SCORES_FILENAME, index=False)

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps({"per_attack_metrics": per_attack}, indent=2, sort_keys=True))
    print(f"Saved evaluation artifacts to {args.artifact_dir.resolve()}")


if __name__ == "__main__":
    main()
