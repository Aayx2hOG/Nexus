"""Analyze frozen RF, LightGBM, and Isolation Forest predictions on UNSW-NB15.

This script performs inference only. It never fits a model, changes preprocessing,
or selects a threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from training_utils import DEFAULT_DATA_DIR, PROJECT_ROOT, load_tree_data


DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_IF_DIR = DEFAULT_MODEL_DIR / "isolation_forest"
DEFAULT_RAW_TEST = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "CSV_Files"
    / "Training and Testing Sets"
    / "UNSW_NB15_testing-set.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "model_complementarity"
RF_THRESHOLD = 0.5
LIGHTGBM_THRESHOLD = 0.5
IF_THRESHOLD = -0.02622396704713703
EXPECTED_RF_FALSE_NEGATIVES = 561
EXPECTED_LIGHTGBM_FALSE_NEGATIVES = 1348


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--isolation-forest-dir", type=Path, default=DEFAULT_IF_DIR)
    parser.add_argument("--raw-test-csv", type=Path, default=DEFAULT_RAW_TEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def dump_json(value: Any, path: Path) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def positive_probabilities(model: Any, X_test: Any, model_name: str) -> np.ndarray:
    classes = np.asarray(model.classes_)
    positive_columns = np.flatnonzero(classes == 1)
    if positive_columns.size != 1:
        raise ValueError(f"{model_name} does not contain exactly one class labelled 1")
    probabilities = np.asarray(model.predict_proba(X_test))[:, positive_columns[0]]
    if probabilities.shape != (X_test.shape[0],):
        raise ValueError(f"Unexpected {model_name} probability shape: {probabilities.shape}")
    return probabilities


def metrics(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
    }


def recovery(mask: np.ndarray, if_predictions: np.ndarray) -> dict[str, Any]:
    total = int(mask.sum())
    detected = int((mask & (if_predictions == 1)).sum())
    return {
        "total": total,
        "isolation_forest_detected": detected,
        "isolation_forest_missed": total - detected,
        "isolation_forest_recovery_rate": float(detected / total) if total else None,
        "isolation_forest_recovery_percentage": float(100.0 * detected / total)
        if total
        else None,
    }


def category_recovery(
    attack_categories: np.ndarray, mask: np.ndarray, if_predictions: np.ndarray
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for category in sorted(np.unique(attack_categories[mask])):
        category_mask = mask & (attack_categories == category)
        result[str(category)] = recovery(category_mask, if_predictions)
    return result


def main() -> None:
    args = parse_args()
    _, X_test, _, y_test = load_tree_data(args.data_dir)

    raw = pd.read_csv(args.raw_test_csv, usecols=["id", "attack_cat", "label"])
    raw_labels = raw["label"].to_numpy(dtype=np.int64)
    if len(raw) != X_test.shape[0] or not np.array_equal(raw_labels, y_test):
        raise ValueError("Official raw test rows do not align with processed X_test/y_test")
    attack_categories = raw["attack_cat"].fillna("Normal").astype(str).to_numpy()

    rf = joblib.load(args.model_dir / "random_forest.joblib")
    lightgbm = joblib.load(args.model_dir / "lightgbm.joblib")
    isolation_forest = joblib.load(args.isolation_forest_dir / "isolation_forest.joblib")
    if_config = json.loads(
        (args.isolation_forest_dir / "isolation_forest_config.json").read_text(
            encoding="utf-8"
        )
    )
    saved_if_threshold = float(if_config["selected_threshold"])
    if not np.isclose(saved_if_threshold, IF_THRESHOLD, rtol=0.0, atol=1e-15):
        raise ValueError(
            f"Saved IF threshold {saved_if_threshold!r} differs from frozen {IF_THRESHOLD!r}"
        )

    rf_probability = positive_probabilities(rf, X_test, "Random Forest")
    lightgbm_probability = positive_probabilities(lightgbm, X_test, "LightGBM")
    # Preserve each estimator's original 0.5 predict behavior. In particular,
    # sklearn's RF resolves an exact 0.5/0.5 tie to class 0; using ``>= 0.5``
    # would silently change 93 predictions and violate the established baseline.
    rf_prediction = np.asarray(rf.predict(X_test), dtype=np.int64)
    lightgbm_prediction = np.asarray(lightgbm.predict(X_test), dtype=np.int64)
    if not np.array_equal(rf_prediction, (rf_probability > RF_THRESHOLD).astype(np.int64)):
        raise ValueError("RF predictions do not match original 0.5 threshold behavior")
    if not np.array_equal(
        lightgbm_prediction, (lightgbm_probability > LIGHTGBM_THRESHOLD).astype(np.int64)
    ):
        raise ValueError("LightGBM predictions do not match original 0.5 threshold behavior")
    if_score = -np.asarray(isolation_forest.decision_function(X_test))
    if_prediction = (if_score >= IF_THRESHOLD).astype(np.int64)
    norm_min = float(if_config["normalization"]["minimum"])
    norm_max = float(if_config["normalization"]["maximum"])
    if norm_max <= norm_min:
        raise ValueError("Invalid Isolation Forest normalization range")
    if_normalized_score = np.clip((if_score - norm_min) / (norm_max - norm_min), 0, 1)

    lgbm_fn = (y_test == 1) & (lightgbm_prediction == 0)
    rf_fn = (y_test == 1) & (rf_prediction == 0)
    both_fn = lgbm_fn & rf_fn
    if int(lgbm_fn.sum()) != EXPECTED_LIGHTGBM_FALSE_NEGATIVES:
        raise RuntimeError(
            f"LightGBM FN baseline mismatch: got {lgbm_fn.sum()}, "
            f"expected {EXPECTED_LIGHTGBM_FALSE_NEGATIVES}"
        )
    if int(rf_fn.sum()) != EXPECTED_RF_FALSE_NEGATIVES:
        raise RuntimeError(
            f"RF FN baseline mismatch: got {rf_fn.sum()}, expected {EXPECTED_RF_FALSE_NEGATIVES}"
        )

    attack_mask = y_test == 1
    agreement: dict[str, Any] = {}
    for rf_value in (1, 0):
        for lgbm_value in (1, 0):
            for if_value in (1, 0):
                mask = (
                    attack_mask
                    & (rf_prediction == rf_value)
                    & (lightgbm_prediction == lgbm_value)
                    & (if_prediction == if_value)
                )
                key = f"rf_{rf_value}_lightgbm_{lgbm_value}_if_{if_value}"
                count = int(mask.sum())
                agreement[key] = {
                    "count": count,
                    "percentage_of_attacks": float(100.0 * count / attack_mask.sum()),
                }

    normal_mask = y_test == 0
    rf_flag = rf_prediction == 1
    lgbm_flag = lightgbm_prediction == 1
    if_flag = if_prediction == 1
    false_positive_overlap = {
        "normal_records": int(normal_mask.sum()),
        "random_forest_false_positives": int((normal_mask & rf_flag).sum()),
        "lightgbm_false_positives": int((normal_mask & lgbm_flag).sum()),
        "isolation_forest_false_positives": int((normal_mask & if_flag).sum()),
        "shared_by_rf_and_lightgbm": int((normal_mask & rf_flag & lgbm_flag).sum()),
        "shared_by_all_three": int((normal_mask & rf_flag & lgbm_flag & if_flag).sum()),
        "only_isolation_forest": int((normal_mask & ~rf_flag & ~lgbm_flag & if_flag).sum()),
        "only_random_forest": int((normal_mask & rf_flag & ~lgbm_flag & ~if_flag).sum()),
        "only_lightgbm": int((normal_mask & ~rf_flag & lgbm_flag & ~if_flag).sum()),
    }

    lgbm_if_or = (lgbm_flag | if_flag).astype(np.int64)
    all_or = (rf_flag | lgbm_flag | if_flag).astype(np.int64)
    summary = {
        "dataset": {
            "test_records": int(y_test.size),
            "attack_records": int(attack_mask.sum()),
            "normal_records": int(normal_mask.sum()),
            "alignment_checks_passed": True,
        },
        "thresholds": {
            "random_forest": RF_THRESHOLD,
            "lightgbm": LIGHTGBM_THRESHOLD,
            "isolation_forest": IF_THRESHOLD,
            "supervised_tie_handling": (
                "original estimator predict behavior; exact probability 0.5 resolves to class 0"
            ),
        },
        "lightgbm_false_negatives": recovery(lgbm_fn, if_prediction),
        "random_forest_false_negatives": recovery(rf_fn, if_prediction),
        "missed_by_both_supervised": recovery(both_fn, if_prediction),
        "all_three_misses": int((both_fn & ~if_flag).sum()),
        "attack_agreement_patterns": agreement,
        "false_positive_overlap": false_positive_overlap,
        "model_metrics": {
            "lightgbm_alone": metrics(y_test, lightgbm_prediction),
            "lightgbm_or_isolation_forest": metrics(y_test, lgbm_if_or),
            "rf_or_lightgbm_or_isolation_forest": metrics(y_test, all_or),
        },
    }
    per_attack = {
        "lightgbm_false_negatives": category_recovery(
            attack_categories, lgbm_fn, if_prediction
        ),
        "random_forest_false_negatives": category_recovery(
            attack_categories, rf_fn, if_prediction
        ),
        "missed_by_both_supervised": category_recovery(
            attack_categories, both_fn, if_prediction
        ),
        "all_three_misses": {
            str(category): int(
                (both_fn & ~if_flag & (attack_categories == category)).sum()
            )
            for category in sorted(np.unique(attack_categories[both_fn & ~if_flag]))
        },
    }
    combined = pd.DataFrame(
        {
            "record_index": np.arange(y_test.size),
            "source_id": raw["id"].to_numpy(),
            "true_label": y_test,
            "attack_cat": attack_categories,
            "random_forest_attack_probability": rf_probability,
            "random_forest_prediction": rf_prediction,
            "lightgbm_attack_probability": lightgbm_probability,
            "lightgbm_prediction": lightgbm_prediction,
            "isolation_forest_raw_anomaly_score": if_score,
            "isolation_forest_normalized_anomaly_score": if_normalized_score,
            "isolation_forest_anomaly_prediction": if_prediction,
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(summary, args.output_dir / "summary.json")
    dump_json(per_attack, args.output_dir / "per_attack_overlap.json")
    combined.to_csv(args.output_dir / "combined_predictions.csv", index=False)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved analysis artifacts to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
