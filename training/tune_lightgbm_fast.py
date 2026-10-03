"""Time-bounded, duplicate-grouped native-categorical LightGBM experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from fast_lightgbm_models import AttackCascade, NativeFeatures, NativeModel
from multiclass_training_utils import evaluate_multiclass_classifier
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from training_utils import PROJECT_ROOT, evaluate_binary_classifier
from tune_lightgbm import choose_threshold, class_weights, outperforms, report, write_json


def grouped_splits(y, groups, seed):
    """Four disjoint subsets; exact duplicates cannot cross boundaries."""
    # Stratify groups by their majority label, then map group IDs back to rows.
    # Avoid the expensive per-group fold-assignment loop in StratifiedGroupKFold.
    unique, inverse = np.unique(groups, return_inverse=True)
    counts = np.zeros((len(unique), int(np.max(y)) + 1), dtype=np.int64)
    np.add.at(counts, (inverse, y), 1)
    group_labels = counts.argmax(axis=1)
    remaining = np.arange(len(unique))
    result = {}
    for offset, (name, folds) in enumerate((("selection", 7), ("calibration", 6), ("early", 5))):
        remaining, hold = train_test_split(
            remaining,
            test_size=1 / folds,
            stratify=group_labels[remaining],
            random_state=seed + offset,
        )
        result[name] = np.flatnonzero(np.isin(inverse, hold))
    result["fit"] = np.flatnonzero(np.isin(inverse, remaining))
    expected = set(np.unique(y))
    for name, indices in result.items():
        if set(np.unique(y[indices])) != expected:
            raise ValueError(f"Grouped {name} split is missing a class; try another --seed")
    return result


def audit_training(frame, y):
    # Hash predictors only: duplicate rows with conflicting labels stay together too.
    groups = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    table = pd.DataFrame({"group": groups, "label": y})
    counts = table.groupby("group").agg(rows=("label", "size"), labels=("label", "nunique"))
    conflicting = counts.index[counts["labels"] > 1]
    audit = {
        "rows": len(frame),
        "unique_feature_rows": len(counts),
        "duplicate_rows_beyond_first": int((counts["rows"] - 1).sum()),
        "conflicting_label_groups": len(conflicting),
        "rows_in_conflicting_groups": int(counts.loc[conflicting, "rows"].sum()),
        "conflicting_rows_by_class": {
            str(k): int(v)
            for k, v in table[table.group.isin(conflicting)].label.value_counts().items()
        },
        "grouping": "Exact predictor-row hashes; no inferred session or temporal groups.",
    }
    return groups, audit


def conservative_threshold(y, scores, minimum_recall, confidence):
    """Use a one-sided Wilson lower bound on calibration recall when requested."""
    target = minimum_recall
    if confidence:
        n = int(np.sum(y == 1))
        p = np.arange(n + 1) / n
        z = norm.ppf(confidence)
        lower = (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (
            1 + z * z / n
        )
        feasible = np.flatnonzero(lower >= minimum_recall)
        if not len(feasible):
            raise ValueError("Too few calibration attacks for the recall confidence requirement")
        target = float(p[feasible[0]])
    return choose_threshold(y, scores, target), target


def macro_f1(y, probabilities):
    predicted = probabilities.argmax(axis=1)
    count = probabilities.shape[1]
    matrix = np.bincount(y.astype(int) * count + predicted, minlength=count * count).reshape(
        count, count
    )
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    scores = np.divide(
        2 * np.diag(matrix), denominator, out=np.zeros(count, dtype=float), where=denominator != 0
    )
    return "macro_f1", float(scores.mean()), True


class Deadline:
    """Stop at a boosting-round boundary and return the best metric seen so far."""

    order = 25
    before_iteration = False

    def __init__(self, end):
        self.end = end
        self.best = None
        self.triggered = False

    def __call__(self, env):
        metric = env.evaluation_result_list[0]
        score = metric[2] if metric[3] else -metric[2]
        if self.best is None or score > self.best[0]:
            self.best = (score, env.iteration, list(env.evaluation_result_list))
        if time.monotonic() >= self.end:
            self.triggered = True
            raise lgb.callback.EarlyStopException(self.best[1], self.best[2])


def candidates(task, count):
    base = dict(
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        reg_lambda=1.0,
        reg_alpha=0.0,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        weight_alpha=0.5,
        stopping="logloss",
    )
    if task == "binary":
        base.update(
            learning_rate=0.03,
            num_leaves=63,
            min_child_samples=100,
            subsample=0.85,
            colsample_bytree=0.85,
            weight_alpha=0.0,
        )
        changes = [
            {},
            {
                "num_leaves": 31,
                "min_child_samples": 20,
                "weight_alpha": 1.0,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
            {"min_child_samples": 150, "reg_lambda": 5.0},
            {"weight_alpha": 0.5},
            {"num_leaves": 95, "min_child_samples": 200},
            {"reg_lambda": 10.0},
            {"num_leaves": 31, "min_child_samples": 50},
            {"num_leaves": 95, "reg_alpha": 0.1},
        ]
    else:
        changes = [
            {},
            {"stopping": "macro_f1"},
            {"weight_alpha": 0.4},
            {"weight_alpha": 0.6},
            {"weight_alpha": 0.45},
            {"weight_alpha": 0.55},
            {"num_leaves": 63, "min_child_samples": 100, "reg_alpha": 1.0},
            {"reg_lambda": 10.0, "max_depth": 8},
        ]
    return [dict(base, **change) for change in changes[:count]]


def fit_model(X, y, split, params, args, deadline, attack_only=False):
    fit, early = split["fit"], split["early"]
    if attack_only:
        fit, early = fit[y[fit] > 0], early[y[early] > 0]
        y = y - 1
    settings = dict(params)
    alpha = settings.pop("weight_alpha")
    stopping = settings.pop("stopping")
    binary = len(np.unique(y[fit])) == 2
    model = lgb.LGBMClassifier(
        **settings,
        objective="binary" if binary else "multiclass",
        class_weight=class_weights(y[fit], alpha),
        n_estimators=args.max_rounds,
        n_jobs=args.n_jobs,
        random_state=args.seed,
        verbosity=-1,
        subsample_freq=1,
        metric="None",
        deterministic=True,
        force_col_wise=True,
    )
    metric = (
        macro_f1
        if stopping == "macro_f1" and not binary
        else ("binary_logloss" if binary else "multi_logloss")
    )
    model.fit(
        X.iloc[fit],
        y[fit],
        eval_set=[(X.iloc[early], y[early])],
        eval_metric=metric,
        callbacks=[Deadline(deadline), lgb.early_stopping(args.patience, verbose=False)],
    )
    return model


def evaluate(model, X, y, task, names):
    if task == "binary":
        return evaluate_binary_classifier(model, X, (y > 0).astype(int))
    return evaluate_multiclass_classifier(model, X, y, names)


def selection_rank(metrics, task, minimum):
    if task == "binary":
        if metrics["recall"] < minimum:
            return (1, -metrics["recall"], metrics["false_positive_rate"])
        return (0, metrics["false_positive_rate"], -metrics["recall"])
    return (-metrics["f1_macro"], -metrics["balanced_accuracy"], -metrics["accuracy"])


def parameters_of(model):
    if isinstance(model, AttackCascade):
        return {
            "gate": parameters_of(model.gate),
            "attack_classifier": parameters_of(model.attack_model),
            "gate_threshold": model.decision_threshold,
        }
    return {"parameters": model.get_params(), "best_iteration": int(model.best_iteration_)}


def load_baseline(task, args):
    previous = args.binary_baseline if task == "binary" else args.multiclass_baseline
    if previous.is_file():
        content = json.loads(previous.read_text())
        return content.get("test_metrics", content), str(previous)
    fallback = (
        PROJECT_ROOT
        / "models"
        / ("lightgbm_metrics.json" if task == "binary" else "multiclass_lightgbm_metrics.json")
    )
    return json.loads(fallback.read_text()), str(fallback)


def run_search(task, output, X, y, split, features, names, args, gate=None):
    output.mkdir()
    (output / "trials").mkdir()
    start = time.monotonic()
    end = start + args.minutes * 60
    winner, best, best_rank = None, None, None
    trials = []
    cal, selection = split["calibration"], split["selection"]
    target_y = (y > 0).astype(int) if task == "binary" else y
    plan = candidates(task, args.trials)
    # A standalone multiclass run trains its own gate without reading prior fitted models.
    cascade_enabled = task == "multiclass" and not args.no_cascade
    if cascade_enabled and gate is None:
        gate = fit_model(
            X,
            (y > 0).astype(int),
            split,
            candidates("binary", 1)[0],
            args,
            min(end, start + args.minutes * 60 * 0.25),
        )
        gate.decision_threshold_, _ = conservative_threshold(
            (y[cal] > 0).astype(int),
            gate.predict_proba(X.iloc[cal])[:, 1],
            args.minimum_recall,
            args.recall_confidence,
        )
    jobs = [("direct", p) for p in plan]
    if cascade_enabled:
        jobs.insert(1, ("cascade", candidates("multiclass", 1)[0]))
    for kind, params in jobs:
        if winner is not None and time.monotonic() >= end:
            break
        trial_id = len(trials) + 1
        # Give every pending candidate a fair slice; no first trial can use the whole budget.
        remaining = len(jobs) - len(trials)
        allowance = max(0.1, (end - time.monotonic()) / remaining)
        deadline = min(end, time.monotonic() + allowance)
        print(f"{task} trial {trial_id}/{len(jobs)} ({kind}), budget {allowance:.0f}s", flush=True)
        model = fit_model(X, target_y, split, params, args, deadline, attack_only=kind == "cascade")
        calibration_target = None
        if task == "binary":
            model.decision_threshold_, calibration_target = conservative_threshold(
                target_y[cal],
                model.predict_proba(X.iloc[cal])[:, 1],
                args.minimum_recall,
                args.recall_confidence,
            )
        elif kind == "cascade":
            model = AttackCascade(gate, model, gate.decision_threshold_, len(names))
        metrics = evaluate(model, X.iloc[selection], y[selection], task, names)
        calibration_metrics = evaluate(model, X.iloc[cal], y[cal], task, names)
        trial = dict(
            trial_id=trial_id,
            task=task,
            kind=kind,
            search_arguments=params,
            model_parameters=parameters_of(model),
            validation_metrics=metrics,
            calibration_metrics=calibration_metrics,
            calibration_recall_target=calibration_target,
            decision_threshold=getattr(model, "decision_threshold_", None),
            elapsed_seconds=time.monotonic() - start,
        )
        trials.append(trial)
        write_json(output / "trials" / f"trial_{trial_id:04d}.json", trial)
        write_json(output / "all_trial_metrics.json", trials)
        rank = selection_rank(metrics, task, args.minimum_recall)
        if best_rank is None or rank < best_rank:
            winner, best, best_rank = trial, model, rank
            # Checkpoint every new leader before the next expensive fit.
            temporary = output / "model.pending.joblib"
            joblib.dump(NativeModel(features, best), temporary)
            temporary.replace(output / "model.joblib")
            write_json(output / "selected_validation.json", winner)
        print(f"  selection rank={rank}; checkpoint saved", flush=True)
    winner["selection_constraint_met"] = (
        task != "binary" or winner["validation_metrics"]["recall"] >= args.minimum_recall
    )
    winner["elapsed_seconds"] = time.monotonic() - start
    write_json(output / "selected_validation.json", winner)
    return NativeModel(features, best), winner


def read_data(path, names=None):
    frame = pd.read_csv(path, low_memory=False)
    categories = frame["attack_cat"].astype(str).str.strip()
    if names is None:
        names = ["Normal"] + sorted(set(categories) - {"Normal"})
    ids = categories.map({name: i for i, name in enumerate(names)})
    if ids.isna().any() or not np.array_equal((ids.to_numpy() > 0).astype(int), frame["label"]):
        raise ValueError("Unknown attack category or inconsistent binary labels")
    X = frame.drop(columns=["id", "label", "attack_cat"])
    return X, ids.to_numpy(dtype=int), names


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["binary", "multiclass", "both"], default="both")
    parser.add_argument("--minutes", type=float, default=10, help="Training minutes per task")
    parser.add_argument("--trials", type=int, default=6, help="Direct candidates per task (1-8)")
    parser.add_argument("--max-rounds", type=int, default=2500)
    parser.add_argument("--patience", type=int, default=75)
    parser.add_argument("--minimum-recall", type=float, default=0.95)
    parser.add_argument(
        "--recall-confidence",
        type=float,
        default=0.95,
        help="One-sided calibration Wilson confidence; 0 disables",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--no-cascade", action="store_true")
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data/raw/CSV_Files/Training and Testing Sets",
    )
    parser.add_argument(
        "--binary-baseline",
        type=Path,
        default=PROJECT_ROOT / "experiments/lightgbm_binary_recall95/binary/selected_result.json",
    )
    parser.add_argument(
        "--multiclass-baseline",
        type=Path,
        default=PROJECT_ROOT
        / "experiments/lightgbm_multiclass_macrof1/multiclass/selected_result.json",
    )
    args = parser.parse_args()
    if args.minutes <= 0 or not 1 <= args.trials <= 8 or min(args.max_rounds, args.patience) < 1:
        parser.error("minutes/rounds/patience must be positive; trials must be 1-8")
    if not 0 < args.minimum_recall <= 1:
        parser.error("minimum-recall must be in (0, 1]")
    if args.recall_confidence != 0 and not 0.5 < args.recall_confidence < 1:
        parser.error("recall-confidence must be 0 or in (0.5, 1)")
    return args


def main():
    args = parse_args()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    output = args.experiment_dir or PROJECT_ROOT / "experiments" / f"lightgbm_fast_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    train_path = args.raw_dir / "UNSW_NB15_training-set.csv"
    test_path = args.raw_dir / "UNSW_NB15_testing-set.csv"
    raw, y, names = read_data(train_path)
    groups, audit = audit_training(raw, y)
    split = grouped_splits(y, groups, args.seed)
    audit["split_class_counts"] = {
        name: {
            names[int(c)]: int(n)
            for c, n in zip(*np.unique(y[idx], return_counts=True), strict=True)
        }
        for name, idx in split.items()
    }
    write_json(output / "data_audit.json", audit)
    np.savez(output / "split_indices.npz", **split)
    features = NativeFeatures().fit(raw.iloc[split["fit"]])
    X = features.transform(raw)
    hashes = {}
    for path in (
        train_path,
        test_path,
        Path(__file__),
        Path(__file__).with_name("fast_lightgbm_models.py"),
    ):
        with path.open("rb") as stream:
            hashes[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
    write_json(
        output / "manifest.json",
        dict(
            command=sys.argv,
            arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            created_utc=stamp,
            hashes=hashes,
            versions={
                "lightgbm": lgb.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            class_names=names,
            preprocessing="Native categoricals fitted on fitting rows only; numeric NaNs retained.",
            limitations=[
                "Exact duplicate grouping does not identify all related flows.",
                "One grouped holdout; no repeated CV. Rare-class scores remain uncertain.",
                "Recall confidence bound is pointwise on calibration, not a deployment guarantee.",
                "Budget excludes loading, scoring and saving; stops at round boundaries.",
                "Prior test results informed this work; test is not an untouched holdout.",
            ],
        ),
    )
    tasks = ["binary", "multiclass"] if args.task == "both" else [args.task]
    selected = {}
    gate = None
    for task in tasks:
        model, result = run_search(task, output / task, X, y, split, features, names, args, gate)
        selected[task] = (model, result)
        if task == "binary":
            gate = model.estimator
    # All model and decision-rule choices are frozen before any test prediction.
    X_test, y_test, _ = read_data(test_path, names)
    test_groups = pd.util.hash_pandas_object(X_test, index=False).to_numpy()
    audit["test_rows_with_training_feature_match"] = int(np.isin(test_groups, groups).sum())
    write_json(output / "data_audit.json", audit)
    summary = {}
    for task, (model, result) in selected.items():
        directory = output / task
        baseline, source = load_baseline(task, args)
        write_json(directory / "baseline_metrics.json", baseline)
        result["baseline_source"] = source
        result["test_metrics"] = evaluate(model, X_test, y_test, task, names)
        result["original_baseline_comparison"] = None
        original = (
            PROJECT_ROOT
            / "models"
            / ("lightgbm_metrics.json" if task == "binary" else "multiclass_lightgbm_metrics.json")
        )
        if original.is_file():
            original_metrics = json.loads(original.read_text())
            write_json(directory / "original_baseline_metrics.json", original_metrics)
            result["original_baseline_comparison"] = outperforms(
                result["test_metrics"], original_metrics, task, args.minimum_recall
            )
        write_json(directory / "selected_result.json", result)
        improved = report(directory, result, baseline, args)
        summary[task] = dict(
            outperforms_baseline=improved,
            baseline_source=source,
            selection_constraint_met=result["selection_constraint_met"],
            selected_trial=result["trial_id"],
            test_metrics=result["test_metrics"],
        )
        write_json(output / "summary.json", summary)
    print(f"Completed: {output}", flush=True)


if __name__ == "__main__":
    main()
