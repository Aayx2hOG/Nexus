"""Validation-only LightGBM search with a paired Random Forest baseline."""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from compare_lightgbm import rows, train
from fast_lightgbm_models import NativeModel, OneHotFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from training_utils import PROJECT_ROOT
from tune_lightgbm import write_json
from tune_lightgbm_fast import (
    audit_training,
    candidates,
    conservative_threshold,
    grouped_splits,
    read_data,
    selection_rank,
)
from tune_lightgbm_fast import (
    evaluate as evaluate_base,
)


def evaluate(model, X, y, task, names):
    metrics = evaluate_base(model, X, y, task, names)
    probabilities = model.predict_proba(X)
    if task == "binary":
        metrics["average_precision"] = float(average_precision_score(y > 0, probabilities[:, 1]))
        metrics["false_alerts_per_1000_normal"] = 1000 * metrics["false_positive_rate"]
    else:
        matrix = np.asarray(metrics["confusion_matrix"])
        normal_total = int(matrix[0].sum())
        metrics["normal_false_positive_rate"] = float((normal_total - matrix[0, 0]) / normal_total)
        metrics["false_alerts_per_1000_normal"] = 1000 * metrics["normal_false_positive_rate"]
    return metrics


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_features(frame):
    """Stable numeric dtypes for train/evaluation feature-identity auditing."""
    frame = frame.copy()
    for column in frame:
        if pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = frame[column].astype(np.float64)
        else:
            frame[column] = frame[column].astype("string")
    return frame


def load_split_indices(path, y, groups):
    """Load saved row partitions, including an extracted NPZ directory."""
    parts = ("fit", "early", "calibration", "selection")
    if path.is_dir():
        split = {part: np.load(path / f"{part}.npy", allow_pickle=False) for part in parts}
    else:
        with np.load(path, allow_pickle=False) as archive:
            split = {part: archive[part] for part in parts}
    seen_groups = set()
    for part, indices in split.items():
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"{part}: indices must be a one-dimensional integer array")
        if not len(indices) or np.any(indices < 0) or np.any(indices >= len(y)):
            raise ValueError(f"{part}: empty partition or out-of-range indices")
        partition_groups = set(groups[indices])
        if seen_groups.intersection(partition_groups):
            raise ValueError(f"{part}: identical predictors cross partition boundaries")
        seen_groups.update(partition_groups)
        if set(np.unique(y[indices])) != set(np.unique(y)):
            raise ValueError(f"{part}: saved partition is missing a class")
    combined = np.concatenate(list(split.values()))
    if not np.array_equal(np.sort(combined), np.arange(len(y))):
        raise ValueError("Saved partitions must cover every training row exactly once")
    return split


def search_plan(task, count, seed):
    base = candidates(task, 1)[0]
    if task == "multiclass":
        base["weight_alpha"] = 0.55
    plan = [base]
    # Explicit weighting ablations precede the reproducible bounded random search.
    for alpha in (0.0, 0.25, 0.7, 1.0):
        if alpha != base["weight_alpha"]:
            plan.append(dict(base, weight_alpha=alpha))
    rng = np.random.default_rng(seed)
    while len(plan) < count:
        settings = dict(base)
        for key, values in {
            "num_leaves": [15, 31, 63],
            "min_child_samples": [20, 50, 100, 200],
            "learning_rate": [0.02, 0.03, 0.05, 0.08],
            "reg_alpha": [0.0, 0.1, 1.0, 2.0],
            "reg_lambda": [1.0, 3.0, 5.0, 10.0],
            "weight_alpha": [0.0, 0.25, 0.4, 0.55, 0.7, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "subsample": [0.7, 0.85, 1.0],
        }.items():
            settings[key] = rng.choice(values).item()
        if settings not in plan:
            plan.append(settings)
    return plan[:count]


def score_candidate(model, X, y, split, task, names, args):
    target = (y > 0).astype(int) if task == "binary" else y
    cal, select = split["calibration"], split["selection"]
    if task == "binary":
        model.decision_threshold_, _ = conservative_threshold(
            target[cal],
            model.predict_proba(rows(X, cal))[:, 1],
            args.minimum_recall,
            args.recall_confidence,
        )
    metrics = evaluate(model, rows(X, select), y[select], task, names)
    return {
        "validation_metrics": metrics,
        "decision_threshold": getattr(model, "decision_threshold_", None),
        "selection_constraint_met": task != "binary" or metrics["recall"] >= args.minimum_recall,
    }


def train_task(task, output, raw, y, names, split, args):
    output.mkdir()
    (output / "trials").mkdir()
    target = (y > 0).astype(int) if task == "binary" else y
    fit = split["fit"]
    refit = np.sort(np.r_[fit, split["early"]])
    # Early-stopping vocabulary excludes early rows; the refit gets its own encoder.
    early_features = OneHotFeatures().fit(raw.iloc[fit])
    X_early = early_features.transform(raw)
    final_features = OneHotFeatures().fit(raw.iloc[refit])
    X = final_features.transform(raw)
    print(f"{task}: training paired Random Forest on {len(refit):,} rows", flush=True)
    forest = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        RandomForestClassifier(
            n_estimators=args.rf_trees,
            class_weight="balanced_subsample",
            max_features="sqrt",
            n_jobs=args.n_jobs,
            random_state=args.seed,
        ),
    )
    forest.fit(rows(X, refit), target[refit])
    baseline = score_candidate(forest, X, y, split, task, names, args)
    baseline["parameters"] = forest[-1].get_params()
    joblib.dump(NativeModel(final_features, forest), output / "random_forest.joblib")
    write_json(output / "random_forest_validation.json", baseline)
    del forest
    best_rank = None
    winner = None
    for trial_id, params in enumerate(search_plan(task, args.trials, args.seed), 1):
        print(f"{task}: LightGBM trial {trial_id}/{args.trials}", flush=True)
        initial, stopping = train(X_early, target, fit, split["early"], params, args)
        rounds = stopping["effective_rounds"]
        del initial
        model, details = train(X, target, refit, split["early"], params, args, rounds=rounds)
        result = score_candidate(model, X, y, split, task, names, args)
        result.update(
            trial_id=trial_id, search_arguments=params, early_stopping=stopping, refit=details
        )
        write_json(output / "trials" / f"trial_{trial_id:04d}.json", result)
        rank = selection_rank(result["validation_metrics"], task, args.minimum_recall)
        if best_rank is None or rank < best_rank:
            best_rank, winner = rank, result
            pending = output / "model.pending.joblib"
            joblib.dump(NativeModel(final_features, model), pending)
            pending.replace(output / "model.joblib")
            write_json(output / "selected_validation.json", winner)
        del model
    rf_rank = selection_rank(baseline["validation_metrics"], task, args.minimum_recall)
    return {
        "selected_trial": winner["trial_id"],
        "lightgbm": winner["validation_metrics"],
        "random_forest": baseline["validation_metrics"],
        "selection_constraint_met": winner["selection_constraint_met"],
        "beats_paired_rf_on_validation": bool(
            winner["selection_constraint_met"] and best_rank < rf_rank
        ),
        "interpretation": "Selection-set comparison, not an independent performance estimate. "
        "RF is a fixed baseline; LightGBM receives a larger search budget.",
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=PROJECT_ROOT
        / "data/raw/CSV_Files/Training and Testing Sets/UNSW_NB15_training-set.csv",
    )
    parser.add_argument("--task", choices=["binary", "multiclass", "both"], default="both")
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument(
        "--split-indices", type=Path,
        help="Saved split NPZ file or directory containing fit/early/calibration/selection.npy",
    )
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--max-rounds", type=int, default=2500)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--rf-trees", type=int, default=300)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-recall", type=float, default=0.95)
    parser.add_argument("--recall-confidence", type=float, default=0.95)
    args = parser.parse_args()
    if min(args.trials, args.max_rounds, args.patience, args.rf_trees) < 1 or args.n_jobs == 0:
        parser.error("trials/rounds/patience/trees must be positive; n-jobs cannot be zero")
    if not 0 < args.minimum_recall < 1:
        parser.error("minimum-recall must be in (0, 1)")
    if args.recall_confidence != 0 and not 0.5 < args.recall_confidence < 1:
        parser.error("recall-confidence must be 0 or in (0.5, 1)")
    args.fit_seconds = 0  # Converged early stopping and complete fixed-round refits.
    return args


def main():
    args = parse_args()
    raw, y, names = read_data(args.train_csv)
    raw = canonical_features(raw)
    groups, audit = audit_training(raw, y)
    split = (
        load_split_indices(args.split_indices, y, groups)
        if args.split_indices is not None else grouped_splits(y, groups, args.seed)
    )
    output = args.experiment_dir or PROJECT_ROOT / "experiments" / (
        "validated_lightgbm_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    )
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / "split_indices.npz", **split)
    np.save(output / "development_feature_hashes.npy", np.unique(groups))
    audit["split_class_counts"] = {
        part: {names[c]: int(np.sum(y[idx] == c)) for c in range(len(names))}
        for part, idx in split.items()
    }
    audit["conflict_policy"] = "Keep all labels; identical predictors stay in the same partition."
    write_json(output / "data_audit.json", audit)
    manifest = {
        "status": "training",
        "command": sys.argv,
        "class_names": names,
        "feature_columns": list(raw.columns),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "train_sha256": digest(args.train_csv),
        "source_hashes": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
        "versions": {
            "lightgbm": lgb.__version__,
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "limitations": [
            "Exact-row grouping does not identify sessions or temporal dependence.",
            "Single grouped split; selection metrics are optimistic after search.",
            "Existing official test results informed development; use fresh data "
            "for independent confirmation.",
            "Wilson recall bound is pointwise, not a guarantee after selection or drift.",
        ],
    }
    write_json(output / "manifest.json", manifest)
    tasks = ["binary", "multiclass"] if args.task == "both" else [args.task]
    summary = {}
    for task in tasks:
        summary[task] = train_task(task, output / task, raw, y, names, split, args)
        write_json(output / "validation_summary.json", summary)
    manifest["status"] = "frozen"
    manifest["artifact_hashes"] = {
        str(p.relative_to(output)): digest(p)
        for p in [
            output / "development_feature_hashes.npy",
            output / "split_indices.npz",
            *output.glob("*/*.joblib"),
            *output.glob("*/*validation.json"),
        ]
    }
    write_json(output / "manifest.json", manifest)
    print(f"Frozen validation-only experiment: {output}\nNo test data was read.", flush=True)


if __name__ == "__main__":
    main()
