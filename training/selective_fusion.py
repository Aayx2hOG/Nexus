"""Run a leakage-free LightGBM/Isolation Forest selective-fusion experiment.

The official training split is divided once into model-training (80%) and
calibration (20%) rows. All learned preprocessing and all three estimators are
fit without calibration rows. Fusion-rule selection uses calibration rows
only, and official test labels are consulted only after the rule is frozen.
Production model and preprocessing artifacts are never loaded or overwritten.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from scipy import sparse
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from training_utils import PROJECT_ROOT

DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "CSV_Files"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "selective_fusion_leakage_free"
SPLIT_DIR_NAME = "Training and Testing Sets"
TRAIN_FILE_NAME = "UNSW_NB15_training-set.csv"
TEST_FILE_NAME = "UNSW_NB15_testing-set.csv"
FEATURE_CATALOG_NAME = "NUSW-NB15_features.csv"
LIGHTGBM_THRESHOLD = 0.5
LOWER_BOUNDS = (0.10, 0.20, 0.30, 0.35, 0.40, 0.45)
IF_QUANTILES = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99)
RANDOM_STATE = 42
CALIBRATION_FRACTION = 0.20
EXCLUDED_COLUMNS = ("id", "attack_cat", "label")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--maximum-additional-calibration-fpr", type=float, default=0.01)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--lightgbm-n-estimators", type=int, default=1_000)
    parser.add_argument("--lightgbm-early-stopping-rounds", type=int, default=75)
    parser.add_argument("--random-forest-n-estimators", type=int, default=300)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required input does not exist: {path}")
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        frame = pd.read_csv(path, encoding="cp1252", low_memory=False)
    frame.columns = frame.columns.str.strip()
    return frame


def categorical_columns(raw_dir: Path, feature_columns: list[str]) -> list[str]:
    catalog = read_csv(raw_dir / FEATURE_CATALOG_NAME)
    if not {"Name", "Type"}.issubset(catalog.columns):
        raise ValueError("Feature catalog must contain Name and Type columns")
    nominal = set(
        catalog.loc[
            catalog["Type"].astype(str).str.strip().str.casefold() == "nominal", "Name"
        ].astype(str).str.strip().str.casefold()
    )
    result = [column for column in feature_columns if column.casefold() in nominal]
    if not result:
        raise ValueError("No categorical predictors were identified")
    return result


def validate_frame(frame: pd.DataFrame, name: str) -> None:
    missing = set(EXCLUDED_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    labels = set(frame["label"].dropna().unique())
    if frame["label"].isna().any() or not labels.issubset({0, 1}):
        raise ValueError(f"Official {name} labels must be binary 0/1 without missing values")


def fit_preprocessor(
    model_train: pd.DataFrame,
    categorical: list[str],
    numerical: list[str],
) -> dict[str, Any]:
    """Fit every learned transformation on model-training rows only."""
    imputer = SimpleImputer(strategy="median")
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
    imputer.fit(model_train[numerical])
    encoder.fit(model_train[categorical].fillna("__MISSING__").astype(str))
    return {
        "categorical_columns": categorical,
        "numerical_columns": numerical,
        "numeric_imputer": imputer,
        "tree_encoder": encoder,
    }


def transform(frame: pd.DataFrame, preprocessor: dict[str, Any]) -> sparse.csr_matrix:
    numerical = preprocessor["numerical_columns"]
    categorical = preprocessor["categorical_columns"]
    numeric = preprocessor["numeric_imputer"].transform(frame[numerical]).astype(np.float32)
    encoded = preprocessor["tree_encoder"].transform(
        frame[categorical].fillna("__MISSING__").astype(str)
    )
    matrix = sparse.hstack([sparse.csr_matrix(numeric), encoded], format="csr")
    if not np.isfinite(matrix.data).all():
        raise ValueError("Transformed predictors contain non-finite values")
    return matrix


def make_lightgbm(n_estimators: int, n_jobs: int) -> lgb.LGBMClassifier:
    """Existing binary LightGBM configuration from train_lightgbm.py."""
    return lgb.LGBMClassifier(
        objective="binary", n_estimators=n_estimators, learning_rate=0.05,
        num_leaves=31, class_weight="balanced", subsample=0.8,
        subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        n_jobs=n_jobs, random_state=RANDOM_STATE, verbosity=-1,
    )


def fit_lightgbm(
    X: sparse.csr_matrix,
    y: np.ndarray,
    n_estimators: int,
    early_stopping_rounds: int,
    n_jobs: int,
) -> tuple[lgb.LGBMClassifier, int]:
    """Choose rounds on an inner split, then refit on all model-training rows."""
    fit_indices, stopping_indices = train_test_split(
        np.arange(y.size), test_size=0.1, stratify=y, random_state=RANDOM_STATE
    )
    selector = make_lightgbm(n_estimators, n_jobs)
    selector.fit(
        X[fit_indices], y[fit_indices],
        eval_set=[(X[stopping_indices], y[stopping_indices])],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    best_iteration = int(selector.best_iteration_ or n_estimators)
    model = make_lightgbm(best_iteration, n_jobs)
    model.fit(X, y)
    return model, best_iteration


def positive_probabilities(model: Any, matrix: Any) -> np.ndarray:
    classes = np.asarray(model.classes_)
    columns = np.flatnonzero(classes == 1)
    if columns.size != 1:
        raise ValueError("Classifier must contain exactly one class labelled 1")
    return np.asarray(model.predict_proba(matrix))[:, columns[0]]


def predictions_for_rule(
    probabilities: np.ndarray,
    anomaly_scores: np.ndarray,
    lower_bound: float,
    anomaly_threshold: float,
) -> np.ndarray:
    baseline = probabilities >= LIGHTGBM_THRESHOLD
    escalation = (
        (probabilities >= lower_bound)
        & (probabilities < LIGHTGBM_THRESHOLD)
        & (anomaly_scores >= anomaly_threshold)
    )
    return (baseline | escalation).astype(np.int64)


def metric_row(
    y_true: np.ndarray,
    predictions: np.ndarray,
    baseline_predictions: np.ndarray,
) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    base_tn, base_fp, base_fn, base_tp = confusion_matrix(
        y_true, baseline_predictions, labels=[0, 1]
    ).ravel()
    recovered = int(base_fn - fn)
    additional_fp = int(fp - base_fp)
    efficiency = float(100.0 * recovered / additional_fp) if additional_fp else None
    return {
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "lightgbm_false_negatives_recovered": recovered,
        "additional_false_positives": additional_fp,
        "attacks_recovered_per_100_additional_false_positives": efficiency,
        "baseline_tp": int(base_tp), "baseline_fp": int(base_fp),
        "baseline_tn": int(base_tn), "baseline_fn": int(base_fn),
    }


def classifier_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = (probabilities >= 0.5).astype(np.int64)
    return metric_row(y_true, predictions, predictions)


def pareto_mask(frame: pd.DataFrame) -> np.ndarray:
    """Keep rules not dominated on recovery (higher) and added FP (lower)."""
    recovered = frame["lightgbm_false_negatives_recovered"].to_numpy()
    added_fp = frame["additional_false_positives"].to_numpy()
    keep = np.ones(len(frame), dtype=bool)
    for index in range(len(frame)):
        dominates = (
            (recovered >= recovered[index])
            & (added_fp <= added_fp[index])
            & ((recovered > recovered[index]) | (added_fp < added_fp[index]))
        )
        keep[index] = not dominates.any()
    return keep


def threshold_candidates(
    calibration_scores: np.ndarray, training_normal_scores: np.ndarray
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for source, values in (
        ("calibration_all", calibration_scores),
        ("model_training_normal", training_normal_scores),
    ):
        for quantile, threshold in zip(IF_QUANTILES, np.quantile(values, IF_QUANTILES)):
            candidates.append({
                "if_threshold_source": f"{source}_q{quantile:g}",
                "if_quantile": quantile,
                "if_anomaly_threshold": float(threshold),
            })
    unique: dict[float, dict[str, Any]] = {}
    for candidate in candidates:
        unique.setdefault(candidate["if_anomaly_threshold"], candidate)
    return list(unique.values())


def select_rule(
    y: np.ndarray,
    probabilities: np.ndarray,
    anomaly_scores: np.ndarray,
    training_normal_scores: np.ndarray,
    maximum_additional_fpr: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    baseline = (probabilities >= LIGHTGBM_THRESHOLD).astype(np.int64)
    baseline_metrics = metric_row(y, baseline, baseline)
    rows: list[dict[str, Any]] = []
    for lower_bound in LOWER_BOUNDS:
        for candidate in threshold_candidates(anomaly_scores, training_normal_scores):
            predictions = predictions_for_rule(
                probabilities, anomaly_scores, lower_bound, candidate["if_anomaly_threshold"]
            )
            row = {"lightgbm_gray_zone_lower_bound": lower_bound, **candidate,
                   **metric_row(y, predictions, baseline)}
            row["additional_fpr"] = row["fpr"] - baseline_metrics["fpr"]
            rows.append(row)
    search = pd.DataFrame(rows).sort_values(
        ["additional_false_positives", "lightgbm_false_negatives_recovered"],
        ascending=[True, False],
    )
    useful = search[(search["lightgbm_false_negatives_recovered"] > 0) & pareto_mask(search)]
    eligible = useful[useful["additional_fpr"] <= maximum_additional_fpr + 1e-15]
    if eligible.empty:
        raise RuntimeError("No useful rule satisfies the calibration FPR budget")
    selected = eligible.sort_values(
        ["lightgbm_false_negatives_recovered", "additional_false_positives", "f1",
         "lightgbm_gray_zone_lower_bound", "if_anomaly_threshold"],
        ascending=[False, True, False, False, False],
    ).iloc[0].to_dict()
    return search, useful, selected, baseline_metrics


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def dump_json(value: Any, path: Path) -> None:
    path.write_text(
        json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if not 0 <= args.maximum_additional_calibration_fpr <= 1:
        raise ValueError("--maximum-additional-calibration-fpr must be in [0, 1]")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty experiment directory: {args.output_dir}"
        )

    train = read_csv(args.raw_dir / SPLIT_DIR_NAME / TRAIN_FILE_NAME)
    validate_frame(train, "train")
    all_indices = np.arange(len(train))
    model_indices, calibration_indices = train_test_split(
        all_indices, test_size=CALIBRATION_FRACTION,
        stratify=train["label"].to_numpy(dtype=np.int64), random_state=RANDOM_STATE,
    )
    model_frame = train.iloc[model_indices]
    calibration_frame = train.iloc[calibration_indices]
    y_model = model_frame["label"].to_numpy(dtype=np.int64)
    y_calibration = calibration_frame["label"].to_numpy(dtype=np.int64)

    feature_columns = [column for column in train.columns if column not in EXCLUDED_COLUMNS]
    categorical = categorical_columns(args.raw_dir, feature_columns)
    numerical = [column for column in feature_columns if column not in categorical]
    preprocessor = fit_preprocessor(model_frame, categorical, numerical)
    X_model = transform(model_frame, preprocessor)
    X_calibration = transform(calibration_frame, preprocessor)

    started = time.perf_counter()
    lightgbm, best_iteration = fit_lightgbm(
        X_model, y_model, args.lightgbm_n_estimators,
        args.lightgbm_early_stopping_rounds, args.n_jobs,
    )
    random_forest = RandomForestClassifier(
        n_estimators=args.random_forest_n_estimators, max_depth=None,
        min_samples_leaf=1, class_weight="balanced_subsample", n_jobs=args.n_jobs,
        random_state=RANDOM_STATE, verbose=0,
    ).fit(X_model, y_model)
    normal_mask = y_model == 0
    isolation_forest = IsolationForest(
        n_estimators=300, contamination="auto", max_features=1.0,
        max_samples="auto", bootstrap=False, n_jobs=args.n_jobs,
        random_state=RANDOM_STATE,
    ).fit(X_model[normal_mask])
    training_seconds = time.perf_counter() - started

    calibration_lgb_probability = positive_probabilities(lightgbm, X_calibration)
    calibration_rf_probability = positive_probabilities(random_forest, X_calibration)
    calibration_if_score = -np.asarray(isolation_forest.decision_function(X_calibration))
    training_normal_if_score = -np.asarray(
        isolation_forest.decision_function(X_model[normal_mask])
    )
    search, pareto, selected_rule, calibration_baseline = select_rule(
        y_calibration, calibration_lgb_probability, calibration_if_score,
        training_normal_if_score, args.maximum_additional_calibration_fpr,
    )
    # The selected rule is immutable before any official test metric is computed.
    selected_rule = json_ready(selected_rule)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dump_json(selected_rule, args.output_dir / "selected_rule_before_test.json")

    # The official test CSV is not even read until selection is complete.
    test = read_csv(args.raw_dir / SPLIT_DIR_NAME / TEST_FILE_NAME)
    validate_frame(test, "test")
    if train.columns.tolist() != test.columns.tolist():
        raise ValueError("Official train and test columns differ")
    X_test = transform(test, preprocessor)
    y_test = test["label"].to_numpy(dtype=np.int64)
    test_lgb_probability = positive_probabilities(lightgbm, X_test)
    test_rf_probability = positive_probabilities(random_forest, X_test)
    test_if_score = -np.asarray(isolation_forest.decision_function(X_test))
    test_baseline_predictions = (test_lgb_probability >= LIGHTGBM_THRESHOLD).astype(np.int64)
    test_fused_predictions = predictions_for_rule(
        test_lgb_probability, test_if_score,
        float(selected_rule["lightgbm_gray_zone_lower_bound"]),
        float(selected_rule["if_anomaly_threshold"]),
    )
    test_baseline = metric_row(y_test, test_baseline_predictions, test_baseline_predictions)
    test_fusion = metric_row(y_test, test_fused_predictions, test_baseline_predictions)
    test_fusion["additional_fpr"] = test_fusion["fpr"] - test_baseline["fpr"]

    split = {
        "official_training_rows": len(train), "model_training_rows": len(model_indices),
        "calibration_rows": len(calibration_indices),
        "calibration_fraction": CALIBRATION_FRACTION,
        "stratified_on": "binary label", "random_state": RANDOM_STATE,
        "model_training_class_counts": {
            str(k): int(v) for k, v in zip(*np.unique(y_model, return_counts=True))
        },
        "calibration_class_counts": {
            str(k): int(v) for k, v in zip(*np.unique(y_calibration, return_counts=True))
        },
    }
    summary = {
        "methodology": {
            "selection_data": "held-out calibration subset of official training set only",
            "test_usage": "one final evaluation after freezing the selected rule",
            "preprocessing_fit_data": "model-training subset only",
            "model_fit_data": "model-training subset only",
            "isolation_forest_fit_data": "normal model-training rows only",
            "production_artifacts_loaded_or_overwritten": False,
        },
        "split": split, "feature_count": X_model.shape[1],
        "lightgbm_best_iteration": best_iteration, "training_seconds": training_seconds,
        "selection_policy": {
            "objective": (
                "maximize recovered LightGBM false negatives within calibration FPR budget"
            ),
            "maximum_absolute_additional_fpr": args.maximum_additional_calibration_fpr,
            "lightgbm_attack_threshold": LIGHTGBM_THRESHOLD,
            "candidate_lower_bounds": list(LOWER_BOUNDS),
            "candidate_if_quantiles": list(IF_QUANTILES),
            "if_threshold_sources": ["calibration_all", "model_training_normal"],
        },
        "calibration_lightgbm_baseline": calibration_baseline,
        "calibration_random_forest": classifier_metrics(y_calibration, calibration_rf_probability),
        "selected_rule": selected_rule,
        "official_test_lightgbm_baseline": test_baseline,
        "official_test_selected_fusion": test_fusion,
        "official_test_random_forest": classifier_metrics(y_test, test_rf_probability),
    }
    manifest = {
        "command": sys.argv, "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__, "lightgbm": lgb.__version__,
        "raw_training_csv": str(args.raw_dir / SPLIT_DIR_NAME / TRAIN_FILE_NAME),
        "raw_test_csv": str(args.raw_dir / SPLIT_DIR_NAME / TEST_FILE_NAME),
    }

    np.savez_compressed(
        args.output_dir / "split_indices.npz",
        model_training_indices=model_indices, calibration_indices=calibration_indices,
    )
    joblib.dump(preprocessor, args.output_dir / "preprocessor.joblib")
    joblib.dump(lightgbm, args.output_dir / "lightgbm.joblib")
    joblib.dump(random_forest, args.output_dir / "random_forest.joblib")
    joblib.dump(isolation_forest, args.output_dir / "isolation_forest.joblib")
    search.to_csv(args.output_dir / "calibration_candidate_search.csv", index=False)
    pareto.to_csv(args.output_dir / "calibration_pareto_candidates.csv", index=False)
    dump_json(summary, args.output_dir / "summary.json")
    dump_json(manifest, args.output_dir / "manifest.json")
    print(json.dumps(json_ready(summary), indent=2, sort_keys=True))
    print(f"Saved leakage-free experiment to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
