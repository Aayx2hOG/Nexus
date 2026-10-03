"""Run simulated unseen-attack-family generalization on UNSW-NB15.

For every requested attack family, all of its rows are removed before fitting
the preprocessor and models.  The official test split is used exactly once for
final family and normal-traffic evaluation.  Artifacts are written only below
the experiment output directory; production artifacts are never read or
modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from scipy import sparse
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from training_utils import PROJECT_ROOT

DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "CSV_Files"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "leave_one_attack_family_out"
SPLIT_DIR = "Training and Testing Sets"
TRAIN_FILE = "UNSW_NB15_training-set.csv"
TEST_FILE = "UNSW_NB15_testing-set.csv"
FEATURE_CATALOG = "NUSW-NB15_features.csv"
DEFAULT_FAMILIES = ("Fuzzers", "Exploits", "DoS", "Reconnaissance", "Analysis", "Backdoor", "Shellcode")
OPTIONAL_FAMILY = "Generic"
EXCLUDED_COLUMNS = ("id", "attack_cat", "label")
RANDOM_STATE = 42
SUPERVISED_THRESHOLD = 0.5
VALIDATION_FRACTION = 0.20
IF_MAXIMUM_FPR = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--families", nargs="+", default=None)
    parser.add_argument("--include-generic", action="store_true")
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


def canonical_attack_categories(values: pd.Series) -> pd.Series:
    return values.fillna("Normal").astype(str).str.strip()


def validate_data(train: pd.DataFrame, test: pd.DataFrame) -> None:
    if train.columns.tolist() != test.columns.tolist():
        raise ValueError("Official train and test columns differ")
    missing = set(EXCLUDED_COLUMNS).difference(train.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    for name, frame in (("train", train), ("test", test)):
        labels = set(frame["label"].dropna().unique())
        if frame["label"].isna().any() or not labels.issubset({0, 1}):
            raise ValueError(f"Official {name} labels must be binary 0/1")
        normal = canonical_attack_categories(frame["attack_cat"]).str.casefold() == "normal"
        if not (frame.loc[normal, "label"] == 0).all():
            raise ValueError(f"Official {name} has Normal rows labelled as attacks")


def get_feature_columns(raw_dir: Path, train: pd.DataFrame) -> tuple[list[str], list[str]]:
    features = [column for column in train.columns if column not in EXCLUDED_COLUMNS]
    catalog = read_csv(raw_dir / FEATURE_CATALOG)
    if not {"Name", "Type"}.issubset(catalog.columns):
        raise ValueError("Feature catalog must contain Name and Type")
    nominal = set(
        catalog.loc[
            catalog["Type"].astype(str).str.strip().str.casefold() == "nominal", "Name"
        ].astype(str).str.strip().str.casefold()
    )
    categorical = [column for column in features if column.casefold() in nominal]
    numerical = [column for column in features if column not in categorical]
    if not categorical:
        raise ValueError("No categorical predictors were identified")
    return categorical, numerical


def fit_preprocessor(frame: pd.DataFrame, categorical: list[str], numerical: list[str]) -> dict[str, Any]:
    imputer = SimpleImputer(strategy="median")
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
    imputer.fit(frame[numerical])
    encoder.fit(frame[categorical].fillna("__MISSING__").astype(str))
    return {"categorical": categorical, "numerical": numerical, "imputer": imputer, "encoder": encoder}


def transform(frame: pd.DataFrame, preprocessor: dict[str, Any]) -> sparse.csr_matrix:
    numeric = preprocessor["imputer"].transform(frame[preprocessor["numerical"]]).astype(np.float32)
    categories = preprocessor["encoder"].transform(
        frame[preprocessor["categorical"]].fillna("__MISSING__").astype(str)
    )
    matrix = sparse.hstack([sparse.csr_matrix(numeric), categories], format="csr")
    if not np.isfinite(matrix.data).all():
        raise ValueError("Transformed data contains non-finite values")
    return matrix


def make_lightgbm(n_estimators: int, n_jobs: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary", n_estimators=n_estimators, learning_rate=0.05,
        num_leaves=31, class_weight="balanced", subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0, n_jobs=n_jobs,
        random_state=RANDOM_STATE, verbosity=-1,
    )


def fit_lightgbm(X: sparse.csr_matrix, y: np.ndarray, maximum_rounds: int,
                 patience: int, n_jobs: int) -> tuple[lgb.LGBMClassifier, int]:
    fit, stopping = train_test_split(
        np.arange(y.size), test_size=0.1, stratify=y, random_state=RANDOM_STATE
    )
    selector = make_lightgbm(maximum_rounds, n_jobs)
    selector.fit(
        X[fit], y[fit], eval_set=[(X[stopping], y[stopping])],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(patience, verbose=False)],
    )
    rounds = int(selector.best_iteration_ or maximum_rounds)
    model = make_lightgbm(rounds, n_jobs)
    model.fit(X, y)
    return model, rounds


def select_if_threshold(y: np.ndarray, scores: np.ndarray,
                        maximum_fpr: float = IF_MAXIMUM_FPR) -> float:
    """Maximize validation attack recall under an empirical normal FPR cap."""
    fprs, recalls, thresholds = roc_curve(y, scores, pos_label=1, drop_intermediate=False)
    eligible = np.flatnonzero(fprs <= maximum_fpr + np.finfo(float).eps)
    best_recall = recalls[eligible].max()
    eligible = eligible[recalls[eligible] == best_recall]
    best_fpr = fprs[eligible].min()
    eligible = eligible[fprs[eligible] == best_fpr]
    return float(thresholds[eligible[np.argmax(thresholds[eligible])]])


def detection_metrics(values: np.ndarray, predictions: np.ndarray,
                      score_name: str) -> dict[str, Any]:
    count = int(values.size)
    detected = int(predictions.sum())
    return {
        "samples": count,
        "detected": detected,
        "missed": count - detected,
        "recall": float(detected / count) if count else None,
        f"mean_{score_name}": float(values.mean()) if count else None,
        f"median_{score_name}": float(np.median(values)) if count else None,
    }


def normal_metrics(predictions: np.ndarray) -> dict[str, Any]:
    return {
        "samples": int(predictions.size), "false_positives": int(predictions.sum()),
        "false_positive_rate": float(predictions.mean()) if predictions.size else None,
    }


def run_family(family: str, train: pd.DataFrame, test: pd.DataFrame,
               categorical: list[str], numerical: list[str], args: argparse.Namespace,
               output_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    train_family = canonical_attack_categories(train["attack_cat"])
    test_family = canonical_attack_categories(test["attack_cat"])
    held_train = train_family.str.casefold() == family.casefold()
    held_test = test_family.str.casefold() == family.casefold()
    normal_test = test["label"].to_numpy(dtype=np.int64) == 0
    if not held_train.any():
        raise ValueError(f"No training rows found for family {family!r}")
    if not held_test.any():
        raise ValueError(f"No official-test rows found for family {family!r}")
    if not (test.loc[held_test, "label"] == 1).all():
        raise ValueError(f"Held-out family {family!r} contains non-attack test labels")

    retained = train.loc[~held_train].reset_index(drop=True)
    y = retained["label"].to_numpy(dtype=np.int64)
    preprocessor = fit_preprocessor(retained, categorical, numerical)
    X = transform(retained, preprocessor)
    evaluation = test.loc[held_test | normal_test].copy()
    X_evaluation = transform(evaluation, preprocessor)
    is_attack = evaluation["label"].to_numpy(dtype=np.int64) == 1

    started = time.perf_counter()
    lightgbm, rounds = fit_lightgbm(
        X, y, args.lightgbm_n_estimators, args.lightgbm_early_stopping_rounds, args.n_jobs
    )
    rf = RandomForestClassifier(
        n_estimators=args.random_forest_n_estimators, class_weight="balanced_subsample",
        n_jobs=args.n_jobs, random_state=RANDOM_STATE,
    ).fit(X, y)

    # Match the established IF method: stratified internal validation, fit only
    # normal rows, score as -decision_function, and select the <=10% FPR point.
    if_fit, if_validation = train_test_split(
        np.arange(y.size), test_size=VALIDATION_FRACTION, stratify=y,
        random_state=RANDOM_STATE,
    )
    normal_fit = if_fit[y[if_fit] == 0]
    isolation_forest = IsolationForest(
        n_estimators=300, contamination="auto", max_features=1.0,
        max_samples="auto", bootstrap=False, n_jobs=args.n_jobs,
        random_state=RANDOM_STATE,
    ).fit(X[normal_fit])
    validation_scores = -isolation_forest.decision_function(X[if_validation])
    if_threshold = select_if_threshold(y[if_validation], validation_scores)

    lgb_probability = lightgbm.predict_proba(X_evaluation)[:, 1]
    rf_probability = rf.predict_proba(X_evaluation)[:, 1]
    if_score = -isolation_forest.decision_function(X_evaluation)
    lgb_prediction = lgb_probability >= SUPERVISED_THRESHOLD
    rf_prediction = rf_probability >= SUPERVISED_THRESHOLD
    if_prediction = if_score >= if_threshold

    attack_lgb, attack_rf, attack_if = (
        lgb_prediction[is_attack], rf_prediction[is_attack], if_prediction[is_attack]
    )
    shared_miss = ~attack_lgb & ~attack_rf
    if_recovered = shared_miss & attack_if
    shared_count = int(shared_miss.sum())
    complementarity = {
        "lightgbm_misses": int((~attack_lgb).sum()),
        "random_forest_misses": int((~attack_rf).sum()),
        "both_supervised_miss": shared_count,
        "lightgbm_misses_if_catches": int((~attack_lgb & attack_if).sum()),
        "random_forest_misses_if_catches": int((~attack_rf & attack_if).sum()),
        "shared_supervised_misses_if_catches": int(if_recovered.sum()),
        "if_recovery_rate_of_shared_supervised_misses": (
            float(if_recovered.sum() / shared_count) if shared_count else None
        ),
        "all_three_miss": int((shared_miss & ~attack_if).sum()),
    }
    normal = ~is_attack
    metrics = {
        "held_out_family": family,
        "training": {
            "official_rows": int(len(train)), "removed_family_rows": int(held_train.sum()),
            "supervised_rows": int(len(retained)), "supervised_attack_rows": int(y.sum()),
            "supervised_normal_rows": int((y == 0).sum()), "feature_count": int(X.shape[1]),
            "lightgbm_best_iteration": rounds, "if_fit_normal_rows": int(normal_fit.size),
            "if_validation_rows": int(if_validation.size), "if_threshold": if_threshold,
        },
        "lightgbm": detection_metrics(lgb_probability[is_attack], attack_lgb, "attack_probability"),
        "random_forest": detection_metrics(rf_probability[is_attack], attack_rf, "attack_probability"),
        "isolation_forest": detection_metrics(if_score[is_attack], attack_if, "anomaly_score"),
        "complementarity": complementarity,
        "normal_traffic": {
            "lightgbm": normal_metrics(lgb_prediction[normal]),
            "random_forest": normal_metrics(rf_prediction[normal]),
            "isolation_forest": normal_metrics(if_prediction[normal]),
        },
        "training_seconds": float(time.perf_counter() - started),
    }
    predictions = pd.DataFrame({
        "official_test_index": evaluation.index.to_numpy(),
        "attack_cat": canonical_attack_categories(evaluation["attack_cat"]).to_numpy(),
        "label": evaluation["label"].to_numpy(dtype=np.int64),
        "lightgbm_probability": lgb_probability,
        "lightgbm_prediction": lgb_prediction.astype(np.int64),
        "random_forest_probability": rf_probability,
        "random_forest_prediction": rf_prediction.astype(np.int64),
        "isolation_forest_anomaly_score": if_score,
        "isolation_forest_prediction": if_prediction.astype(np.int64),
    })
    family_dir = output_dir / family.lower().replace(" ", "_")
    family_dir.mkdir(parents=True, exist_ok=True)
    write_json(family_dir / "metrics.json", metrics)
    predictions.to_csv(family_dir / "predictions.csv", index=False)
    return metrics, predictions


def comparison_row(metrics: dict[str, Any]) -> dict[str, Any]:
    comp = metrics["complementarity"]
    normal = metrics["normal_traffic"]
    return {
        "held_out_family": metrics["held_out_family"],
        "samples": metrics["lightgbm"]["samples"],
        "lightgbm_recall": metrics["lightgbm"]["recall"],
        "random_forest_recall": metrics["random_forest"]["recall"],
        "isolation_forest_recall": metrics["isolation_forest"]["recall"],
        "both_supervised_miss": comp["both_supervised_miss"],
        "if_recovered": comp["shared_supervised_misses_if_catches"],
        "if_recovery_rate_of_shared_supervised_misses": comp["if_recovery_rate_of_shared_supervised_misses"],
        "all_three_miss": comp["all_three_miss"],
        "lightgbm_normal_fpr": normal["lightgbm"]["false_positive_rate"],
        "random_forest_normal_fpr": normal["random_forest"]["false_positive_rate"],
        "isolation_forest_normal_fpr": normal["isolation_forest"]["false_positive_rate"],
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_metrics_csv(path: Path, metrics: Iterable[dict[str, Any]]) -> None:
    rows = []
    for item in metrics:
        flat = comparison_row(item)
        flat.update({
            "lightgbm_mean_attack_probability": item["lightgbm"]["mean_attack_probability"],
            "lightgbm_median_attack_probability": item["lightgbm"]["median_attack_probability"],
            "random_forest_mean_attack_probability": item["random_forest"]["mean_attack_probability"],
            "random_forest_median_attack_probability": item["random_forest"]["median_attack_probability"],
            "isolation_forest_mean_anomaly_score": item["isolation_forest"]["mean_anomaly_score"],
            "isolation_forest_median_anomaly_score": item["isolation_forest"]["median_anomaly_score"],
            "lightgbm_misses": item["complementarity"]["lightgbm_misses"],
            "random_forest_misses": item["complementarity"]["random_forest_misses"],
            "lightgbm_misses_if_catches": item["complementarity"]["lightgbm_misses_if_catches"],
            "random_forest_misses_if_catches": item["complementarity"]["random_forest_misses_if_catches"],
        })
        rows.append(flat)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.lightgbm_early_stopping_rounds < 1:
        raise ValueError("--lightgbm-early-stopping-rounds must be positive")
    families = list(args.families or DEFAULT_FAMILIES)
    if args.include_generic and OPTIONAL_FAMILY not in families:
        families.append(OPTIONAL_FAMILY)
    if len({family.casefold() for family in families}) != len(families):
        raise ValueError("Duplicate held-out families were requested")

    train = read_csv(args.raw_dir / SPLIT_DIR / TRAIN_FILE)
    test = read_csv(args.raw_dir / SPLIT_DIR / TEST_FILE)
    validate_data(train, test)
    categorical, numerical = get_feature_columns(args.raw_dir, train)
    available = {value.casefold(): value for value in canonical_attack_categories(train["attack_cat"]).unique()}
    unknown = [family for family in families if family.casefold() not in available]
    if unknown:
        raise ValueError(f"Unknown training attack families: {unknown}; available: {sorted(available.values())}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment": "simulated unseen-attack-family generalization",
        "families": families, "generic_is_optional": True, "worms_skipped": True,
        "random_state": RANDOM_STATE, "supervised_threshold": SUPERVISED_THRESHOLD,
        "isolation_forest_score": "-model.decision_function(X)",
        "isolation_forest_threshold_selection": "validation maximum recall subject to FPR <= 0.10",
        "validation_fraction": VALIDATION_FRACTION,
        "official_test_usage": "final evaluation only",
        "preprocessing": "refit after held-out family removal; unknown categories ignored",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__,
                     "lightgbm": lgb.__version__},
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    write_json(args.output_dir / "config.json", config)
    all_metrics = []
    for requested in families:
        family = available[requested.casefold()]
        print(f"Running LOAFO experiment for {family}", flush=True)
        metrics, _ = run_family(
            family, train, test, categorical, numerical, args, args.output_dir
        )
        all_metrics.append(metrics)
        print(json.dumps(comparison_row(metrics), indent=2), flush=True)

    comparison = [comparison_row(item) for item in all_metrics]
    summary = {"experiment": config["experiment"], "families": all_metrics, "comparison": comparison}
    write_json(args.output_dir / "summary.json", summary)
    pd.DataFrame(comparison).to_csv(args.output_dir / "comparison.csv", index=False)
    write_metrics_csv(args.output_dir / "per_family_metrics.csv", all_metrics)
    print(f"Saved experiment artifacts to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
