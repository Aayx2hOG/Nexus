"""Evaluate the frozen autoencoder on the official UNSW-NB15 test split."""

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

from evaluation_isolation_forest import DEFAULT_RAW_TEST, load_attack_categories
from train_autoencoder import (
    CONFIG_FILENAME,
    DEFAULT_OUTPUT_DIR,
    MODEL_FILENAME,
    make_autoencoder_features,
    reconstruction_errors,
)
from training_utils import DEFAULT_DATA_DIR

METRICS_FILENAME = "autoencoder_metrics.json"
PER_ATTACK_FILENAME = "autoencoder_per_attack_metrics.json"
SCORES_FILENAME = "autoencoder_test_scores.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--raw-test-csv", type=Path, default=DEFAULT_RAW_TEST)
    parser.add_argument("--batch-size", type=int, default=2048)
    return parser.parse_args()


def _json_dump(value: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    from training_utils import load_tree_data

    _, X_test, _, y_test = load_tree_data(args.data_dir)
    attack_categories = load_attack_categories(args.raw_test_csv, y_test)
    bundle = joblib.load(args.artifact_dir / MODEL_FILENAME)
    config = json.loads((args.artifact_dir / CONFIG_FILENAME).read_text(encoding="utf-8"))
    threshold = float(config["selected_threshold"])
    norm_min = float(config["normalization"]["minimum"])
    norm_max = float(config["normalization"]["maximum"])
    X_autoencoder = make_autoencoder_features(
        X_test, int(config["numerical_feature_count"]), bundle["numeric_scaler"]
    )

    started = time.perf_counter()
    scores = reconstruction_errors(bundle["model"], X_autoencoder, args.batch_size)
    predictions = (scores >= threshold).astype(np.int64)
    inference_seconds = time.perf_counter() - started
    normalized_scores = np.clip((scores - norm_min) / (norm_max - norm_min), 0.0, 1.0)

    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    metrics = {
        "model": "MLPRegressorAutoencoder",
        "test_rows": int(y_test.size),
        "selected_threshold": threshold,
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
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
            "records_per_second": float(len(y_test) / inference_seconds),
            "milliseconds_per_record": float(1000 * inference_seconds / len(y_test)),
            "includes": "reconstruction and thresholding",
        },
    }

    per_attack: dict[str, Any] = {}
    for category in sorted(np.unique(attack_categories)):
        mask = attack_categories == category
        labels = y_test[mask]
        category_predictions = predictions[mask]
        attack_rows = int(labels.sum())
        detected = int(((labels == 1) & (category_predictions == 1)).sum())
        per_attack[category] = {
            "records": int(mask.sum()),
            "attack_records": attack_rows,
            "detected_attacks": detected,
            "missed_attacks": attack_rows - detected,
            "attack_recall": float(detected / attack_rows) if attack_rows else None,
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
    _json_dump(metrics, args.artifact_dir / METRICS_FILENAME)
    _json_dump(per_attack, args.artifact_dir / PER_ATTACK_FILENAME)
    score_frame.to_csv(args.artifact_dir / SCORES_FILENAME, index=False)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Saved evaluation artifacts to {args.artifact_dir.resolve()}")


if __name__ == "__main__":
    main()
