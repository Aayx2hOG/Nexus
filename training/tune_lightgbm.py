"""Validation-only LightGBM search; evaluate each frozen winner once on the test set."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import sklearn
from multiclass_training_utils import evaluate_multiclass_classifier, load_multiclass_tree_data
from sklearn.metrics import roc_curve
from sklearn.model_selection import ParameterSampler, train_test_split
from training_utils import (
    DEFAULT_DATA_DIR,
    PROJECT_ROOT,
    evaluate_binary_classifier,
    load_tree_data,
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def choose_threshold(y, probabilities, minimum_recall):
    """Minimize empirical FPR, then maximize recall, then threshold, including tied scores."""
    if not 0 < minimum_recall <= 1:
        raise ValueError("minimum_recall must be in (0, 1]")
    if set(np.unique(y)) != {0, 1} or not np.isfinite(probabilities).all():
        raise ValueError("Threshold selection requires both classes and finite scores")
    fpr, recall, thresholds = roc_curve(y, probabilities, drop_intermediate=False)
    eligible = np.flatnonzero((recall >= minimum_recall) & np.isfinite(thresholds))
    best = min(eligible, key=lambda i: (fpr[i], -recall[i], -thresholds[i]))
    return float(thresholds[best])


def class_weights(y, alpha):
    if alpha == 0:
        return None
    labels, counts = np.unique(y, return_counts=True)
    weights = (len(y) / (len(labels) * counts)) ** alpha
    weights /= np.dot(counts, weights) / len(y)
    return {int(c): float(w) for c, w in zip(labels, weights, strict=True)}


def rank_trial(metrics, task, minimum_recall):
    if task == "binary":
        if metrics["recall"] < minimum_recall:
            return (float("inf"), 0, 0)
        return (metrics["false_positive_rate"], -metrics["recall"], -metrics["f1"])
    return (-metrics["f1_macro"], -metrics["balanced_accuracy"], -metrics["accuracy"])


def outperforms(metrics, baseline, task, minimum_recall):
    if task == "binary":
        return (
            metrics["recall"] >= minimum_recall
            and metrics["false_positive_rate"] < baseline["false_positive_rate"]
        )
    return metrics["f1_macro"] > baseline["f1_macro"]


def configurations(args):
    # First three trials isolate weighting before searching other parameters.
    base = dict(
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        reg_alpha=0.0,
        reg_lambda=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    candidates = [dict(base, weight_alpha=a) for a in (1.0, 0.0, 0.5)]
    space = dict(
        learning_rate=[0.02, 0.03, 0.05],
        num_leaves=[15, 31, 63, 127],
        max_depth=[-1, 6, 8, 10],
        min_child_samples=[10, 20, 50, 100],
        reg_alpha=[0.0, 0.1, 1.0, 5.0],
        reg_lambda=[0.0, 1.0, 5.0, 10.0],
        subsample=[0.7, 0.85, 1.0],
        colsample_bytree=[0.7, 0.85, 1.0],
        weight_alpha=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    for params in ParameterSampler(space, n_iter=args.trials * 2, random_state=args.seed):
        if params["max_depth"] > 0 and params["num_leaves"] > 2 ** params["max_depth"]:
            continue
        candidates.append(params)
        if len(candidates) >= args.trials:
            break
    return candidates[: args.trials]


def report(directory, result, baseline, args):
    task = result["task"]
    metrics = result["test_metrics"]
    improved = outperforms(metrics, baseline, task, args.minimum_recall)
    objective = (
        f"Lower false-positive rate with recall >= {args.minimum_recall:.1%}"
        if task == "binary"
        else "Higher macro F1"
    )
    lines = [
        f"# {task.title()} LightGBM experiment",
        "",
        f"Objective: {objective}.",
        "",
        f"**Outperforms current baseline: {'YES' if improved else 'NO'}.**",
        "",
        "Selection used validation only. Test results below did not select the model.",
        "The recall constraint is empirical on validation; test/deployment recall may differ.",
        "Models are saved without refitting, preserving the validated probability scale.",
        "",
        "| Metric | Current baseline | Selected test result | Change |",
        "|---|---:|---:|---:|",
    ]
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and isinstance(baseline.get(key), (int, float)):
            lines.append(
                f"| {key} | {baseline[key]:.6f} | {value:.6f} | {value - baseline[key]:+.6f} |"
            )
    lines += [
        "",
        "## Selected model and exact arguments",
        "",
        "```json",
        json.dumps({k: v for k, v in result.items() if k != "test_metrics"}, indent=2),
        "```",
        "",
        "## Complete test metrics",
        "",
        "```json",
        json.dumps(metrics, indent=2),
        "```",
        "",
        "## Run arguments",
        "",
        "```json",
        json.dumps(vars(args), indent=2, default=str),
        "```",
        "",
        "See manifest.json for data hashes, versions, command and preprocessing limitations.",
        "See split_indices.npz and trials/ for split membership and every validation result.",
    ]
    content = "\n".join(lines) + "\n"
    (directory / "results.md").write_text(content, encoding="utf-8")
    if improved:
        (directory / "outperforming_result.md").write_text(content, encoding="utf-8")
    return improved


def run_task(task, directory, args):
    directory.mkdir()
    (directory / "trials").mkdir()
    if task == "binary":
        X, X_test, y, y_test = load_tree_data(args.data_dir)
        names = None
        baseline_path = args.baseline_dir / "lightgbm_metrics.json"
    else:
        X, X_test, y, y_test, names = load_multiclass_tree_data(args.data_dir)
        baseline_path = args.baseline_dir / "multiclass_lightgbm_metrics.json"
    baseline = json.loads(baseline_path.read_text())
    write_json(directory / "baseline_metrics.json", baseline)
    indices = np.arange(len(y))
    rest, selection = train_test_split(
        indices,
        test_size=args.selection_size,
        stratify=y,
        random_state=args.seed,
    )
    fit, early = train_test_split(
        rest,
        test_size=args.early_stopping_size / (1 - args.selection_size),
        stratify=y[rest],
        random_state=args.seed,
    )
    np.savez(directory / "split_indices.npz", fit=fit, early_stopping=early, selection=selection)
    split_counts = {
        name: {str(k): int(v) for k, v in zip(*np.unique(y[idx], return_counts=True), strict=True)}
        for name, idx in (("fit", fit), ("early_stopping", early), ("selection", selection))
    }
    write_json(directory / "split_class_counts.json", split_counts)
    best_rank = None
    best_model = None
    winner = None
    all_trials = []
    for trial_id, candidate in enumerate(configurations(args), start=1):
        params = dict(candidate)
        alpha = params.pop("weight_alpha")
        model = lgb.LGBMClassifier(
            **params,
            objective="binary" if task == "binary" else "multiclass",
            class_weight=class_weights(y[fit], alpha),
            n_estimators=args.max_rounds,
            subsample_freq=1,
            random_state=args.seed,
            n_jobs=args.n_jobs,
            verbosity=-1,
        )
        print(f"{task}: trial {trial_id}/{args.trials}: {candidate}", flush=True)
        model.fit(
            X[fit],
            y[fit],
            eval_set=[(X[early], y[early])],
            eval_metric="binary_logloss" if task == "binary" else "multi_logloss",
            callbacks=[lgb.early_stopping(args.patience, verbose=False)],
        )
        if task == "binary":
            model.decision_threshold_ = choose_threshold(
                y[selection],
                model.predict_proba(X[selection])[:, 1],
                args.minimum_recall,
            )
            metrics = evaluate_binary_classifier(model, X[selection], y[selection])
        else:
            metrics = evaluate_multiclass_classifier(model, X[selection], y[selection], names)
        trial = dict(
            trial_id=trial_id,
            task=task,
            search_arguments=candidate,
            model_parameters=model.get_params(),
            best_iteration=int(model.best_iteration_),
            decision_threshold=getattr(model, "decision_threshold_", None),
            validation_metrics=metrics,
            early_stopping_scores=model.best_score_,
            fit_rows=len(fit),
            early_stopping_rows=len(early),
            selection_rows=len(selection),
        )
        write_json(directory / "trials" / f"trial_{trial_id:04d}.json", trial)
        all_trials.append(trial)
        rank = rank_trial(metrics, task, args.minimum_recall)
        if best_rank is None or rank < best_rank:
            best_rank, best_model, winner = rank, model, trial
        print(f"  validation objective: {rank}", flush=True)
    write_json(directory / "all_trial_metrics.json", all_trials)
    # The winner and threshold are frozen BEFORE accessing test predictions.
    write_json(directory / "selected_validation.json", winner)
    joblib.dump(best_model, directory / "model.joblib")
    test_metrics = (
        evaluate_binary_classifier(best_model, X_test, y_test)
        if task == "binary"
        else evaluate_multiclass_classifier(best_model, X_test, y_test, names)
    )
    result = dict(winner, test_metrics=test_metrics)
    write_json(directory / "selected_result.json", result)
    improved = report(directory, result, baseline, args)
    print(
        f"{task}: baseline outperformed = {improved}; report: {directory / 'results.md'}",
        flush=True,
    )
    return {
        "outperforms_baseline": improved,
        "selected_trial": winner["trial_id"],
        "test_metrics": test_metrics,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["binary", "multiclass", "both"], default="both")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--experiment-dir", type=Path, default=None)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--max-rounds", type=int, default=3000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--minimum-recall", type=float, default=0.95)
    parser.add_argument("--selection-size", type=float, default=0.15)
    parser.add_argument("--early-stopping-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    if min(args.trials, args.max_rounds, args.patience) < 1:
        parser.error("trials, max-rounds and patience must be positive")
    if not 0 < args.minimum_recall <= 1:
        parser.error("minimum-recall must be in (0, 1]")
    if not (
        0 < args.selection_size < 1
        and 0 < args.early_stopping_size < 1
        and args.selection_size + args.early_stopping_size < 1
    ):
        parser.error("split fractions must be positive and sum to less than 1")
    return args


def main():
    args = parse_args()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = args.experiment_dir or PROJECT_ROOT / "experiments" / f"lightgbm_{stamp}"
    directory.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for path in sorted(args.data_dir.iterdir()):
        if path.suffix in {".npz", ".npy", ".json", ".joblib"}:
            with path.open("rb") as stream:
                hashes[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = dict(
        command=sys.argv,
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        created_utc=stamp,
        data_sha256=hashes,
        versions=dict(
            python=platform.python_version(),
            lightgbm=lgb.__version__,
            sklearn=sklearn.__version__,
            numpy=np.__version__,
        ),
        limitation="Uses existing preprocessing fitted on the official training split, including "
        "internal validation rows. Internal splits are stratified, not group/dedup audited. "
        "Rare-class validation metrics may be unstable. No test-driven selection or refit.",
    )
    write_json(directory / "manifest.json", manifest)
    tasks = ["binary", "multiclass"] if args.task == "both" else [args.task]
    summary = {task: run_task(task, directory / task, args) for task in tasks}
    write_json(directory / "summary.json", summary)
    print(f"Experiment complete: {directory}", flush=True)


if __name__ == "__main__":
    main()
