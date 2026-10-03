"""Controlled feature/weight/refit comparisons on identical grouped training splits."""

from __future__ import annotations

import argparse
import copy
import hashlib
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from fast_lightgbm_models import NativeFeatures, NativeModel, OneHotFeatures
from training_utils import PROJECT_ROOT
from tune_lightgbm import class_weights, report, write_json
from tune_lightgbm_fast import (
    Deadline,
    audit_training,
    candidates,
    conservative_threshold,
    evaluate,
    grouped_splits,
    load_baseline,
    read_data,
    selection_rank,
)


def rows(X, indices):
    return X.iloc[indices] if isinstance(X, pd.DataFrame) else X[indices]


class RefitDeadline:
    """A fixed-round refit has no validation metrics; keep its last completed round."""

    order = 25
    before_iteration = False

    def __init__(self, end):
        self.end = end
        self.triggered = False

    def __call__(self, env):
        if time.monotonic() >= self.end:
            self.triggered = True
            raise lgb.callback.EarlyStopException(env.iteration, [])


def experiment_plan(task, alphas, full_grid=False):
    base = candidates(task, 1)[0]
    if task == "binary":
        return [(representation, dict(base)) for representation in ("onehot", "native")]
    # Isolate representation at the winning weight, then scan weights on one-hot.
    ordered = [0.5] + [a for a in alphas if a != 0.5]
    plan = [(r, dict(base)) for r in ("onehot", "native")]
    plan += [("onehot", dict(base, weight_alpha=a)) for a in ordered[1:]]
    if full_grid:
        plan += [("native", dict(base, weight_alpha=a)) for a in ordered[1:]]
    return plan


def train(X, y, fit, early, params, args, rounds=None):
    settings = dict(params)
    alpha = settings.pop("weight_alpha")
    settings.pop("stopping")
    requested = rounds if rounds is not None else args.max_rounds
    model = lgb.LGBMClassifier(
        **settings,
        objective="binary" if len(np.unique(y)) == 2 else "multiclass",
        class_weight=class_weights(y[fit], alpha),
        n_estimators=requested,
        n_jobs=args.n_jobs,
        random_state=args.seed,
        verbosity=-1,
        subsample_freq=1,
        deterministic=True,
        force_col_wise=True,
        metric="None",
    )
    end = time.monotonic() + args.fit_seconds if args.fit_seconds else float("inf")
    started = time.monotonic()
    if rounds is None:
        deadline = Deadline(end)
        model.fit(
            rows(X, fit),
            y[fit],
            eval_set=[(rows(X, early), y[early])],
            eval_metric="binary_logloss"
            if len(model.class_weight or {}) == 2 or len(np.unique(y)) == 2
            else "multi_logloss",
            callbacks=[deadline, lgb.early_stopping(args.patience, verbose=False)],
        )
        limited = deadline.triggered
    else:
        deadline = RefitDeadline(end)
        model.fit(rows(X, fit), y[fit], callbacks=[deadline])
        limited = deadline.triggered
    return model, {
        "parameters": model.get_params(),
        "effective_rounds": int(model.n_estimators_),
        "requested_rounds": int(requested),
        "time_limited": bool(limited),
        "training_seconds": time.monotonic() - started,
        "training_rows": len(fit),
    }


def compare_task(task, output, raw, y, split, cache, names, args):
    output.mkdir()
    (output / "trials").mkdir()
    target = (y > 0).astype(int) if task == "binary" else y
    plan = experiment_plan(task, args.alphas, args.full_grid)
    winner = best_bundle = best_rank = None
    trials = []
    cal, select = split["calibration"], split["selection"]
    for config_id, (representation, params) in enumerate(plan, 1):
        rounds = None
        for refit in (False, True):
            features, X = cache[(representation, refit)]
            fit = np.sort(np.r_[split["fit"], split["early"]]) if refit else split["fit"]
            print(
                f"{task} config {config_id}/{len(plan)}: {representation}, "
                f"alpha={params['weight_alpha']}, refit={refit}",
                flush=True,
            )
            model, details = train(X, target, fit, split["early"], params, args, rounds)
            if not refit:
                rounds = details["effective_rounds"]
            confidences = (0.0, 0.95) if task == "binary" else (None,)
            for confidence in confidences:
                calibration_target = None
                if task == "binary":
                    model.decision_threshold_, calibration_target = conservative_threshold(
                        target[cal],
                        model.predict_proba(rows(X, cal))[:, 1],
                        args.minimum_recall,
                        confidence,
                    )
                metrics = evaluate(model, rows(X, select), y[select], task, names)
                trial = dict(
                    trial_id=len(trials) + 1,
                    task=task,
                    representation=representation,
                    refit=refit,
                    search_arguments=params,
                    model_parameters=details,
                    class_names=names,
                    decision_threshold=getattr(model, "decision_threshold_", None),
                    recall_confidence=confidence,
                    calibration_recall_target=calibration_target,
                    validation_metrics=metrics,
                    calibration_metrics=evaluate(model, rows(X, cal), y[cal], task, names),
                    selection_constraint_met=task != "binary"
                    or metrics["recall"] >= args.minimum_recall,
                )
                trials.append(trial)
                write_json(output / "trials" / f"trial_{trial['trial_id']:04d}.json", trial)
                write_json(output / "all_trial_metrics.json", trials)
                rank = selection_rank(metrics, task, args.minimum_recall)
                if best_rank is None or rank < best_rank:
                    # Snapshot before the next threshold variant mutates this fit.
                    best_rank, winner = rank, trial
                    best_bundle = NativeModel(features, copy.deepcopy(model))
                    pending = output / "model.pending.joblib"
                    joblib.dump(best_bundle, pending)
                    pending.replace(output / "model.joblib")
                    write_json(output / "selected_validation.json", winner)
                print(f"  variant {trial['trial_id']}: {rank}, confidence={confidence}", flush=True)
    lines = [
        f"# {task.title()} controlled validation comparison",
        "",
        "All variants use the same grouped splits; no test metrics select the winner.",
        "",
        "| Trial | Features | Weight alpha | Refit | Recall confidence | Rounds | "
        "Time limited | Selection FPR | Selection recall | Selection macro F1 |",
        "|---|---|---:|---|---|---:|---|---:|---:|---:|",
    ]
    for t in trials:
        m = t["validation_metrics"]
        lines.append(
            f"| {t['trial_id']} | {t['representation']} | "
            f"{t['search_arguments']['weight_alpha']} | {t['refit']} | "
            f"{t['recall_confidence']} | {t['model_parameters']['effective_rounds']} | "
            f"{t['model_parameters']['time_limited']} | "
            f"{m.get('false_positive_rate', '—')} | {m.get('recall', '—')} | "
            f"{m.get('f1_macro', '—')} |"
        )
    lines += [
        "",
        f"Selected trial: **{winner['trial_id']}**.",
        "",
        "Time-limited refits may have fewer rounds than requested; "
        "they do not isolate the effect of extra training rows alone.",
    ]
    (output / "validation_comparison.md").write_text("\n".join(lines) + "\n")
    return best_bundle, winner


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["binary", "multiclass", "both"], default="both")
    parser.add_argument(
        "--fit-seconds", type=float, default=45, help="Approximate cap per fit; 0 disables the cap"
    )
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.4, 0.45, 0.5, 0.55, 0.6])
    parser.add_argument("--full-grid", action="store_true", help="Test every alpha with native too")
    parser.add_argument("--max-rounds", type=int, default=2500)
    parser.add_argument("--patience", type=int, default=75)
    parser.add_argument("--minimum-recall", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=4)
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
    if not np.isfinite(args.fit_seconds) or args.fit_seconds < 0:
        parser.error("fit-seconds must be finite and >=0")
    if not 0 < args.minimum_recall < 1:
        parser.error("minimum-recall must be in (0, 1)")
    if min(args.max_rounds, args.patience) < 1 or args.n_jobs == 0:
        parser.error("rounds/patience must be positive and n-jobs cannot be 0")
    if any(not 0 <= a <= 1 for a in args.alphas):
        parser.error("alphas must be in [0, 1]")
    args.alphas = list(dict.fromkeys(args.alphas))
    return args


def main():
    args = parse_args()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    output = args.experiment_dir or PROJECT_ROOT / "experiments" / f"lightgbm_comparison_{stamp}"
    tasks = ["binary", "multiclass"] if args.task == "both" else [args.task]
    baselines = {task: load_baseline(task, args) for task in tasks}
    output.mkdir(parents=True, exist_ok=False)
    train_path = args.raw_dir / "UNSW_NB15_training-set.csv"
    test_path = args.raw_dir / "UNSW_NB15_testing-set.csv"
    raw, y, names = read_data(train_path)
    groups, audit = audit_training(raw, y)
    split = grouped_splits(y, groups, args.seed)
    np.savez(output / "split_indices.npz", **split)
    audit["split_class_counts"] = {
        name: {
            names[int(c)]: int(n)
            for c, n in zip(*np.unique(y[idx], return_counts=True), strict=True)
        }
        for name, idx in split.items()
    }
    write_json(output / "data_audit.json", audit)
    cache = {}
    for representation, cls in (("native", NativeFeatures), ("onehot", OneHotFeatures)):
        for refit in (False, True):
            fit = np.r_[split["fit"], split["early"]] if refit else split["fit"]
            features = cls().fit(raw.iloc[fit])
            cache[(representation, refit)] = features, features.transform(raw)
    hashes = {}
    for path in [train_path, test_path, *Path(__file__).parent.glob("*.py")]:
        with path.open("rb") as stream:
            hashes[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
    write_json(
        output / "manifest.json",
        dict(
            command=sys.argv,
            arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            hashes=hashes,
            class_names=names,
            created_utc=stamp,
            versions={
                "lightgbm": lgb.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            limitations=[
                "One grouped split; rare-class estimates remain uncertain.",
                "Prior test outcomes informed development; use a fresh holdout for confirmation.",
                "Confidence bounds are pointwise, not deployment or model-selection guarantees.",
                "A time-limited refit can have fewer rounds; check validation_comparison.md.",
                "Numeric NaNs are retained in both representations; not exact legacy imputation.",
            ],
        ),
    )
    selected = {
        task: compare_task(task, output / task, raw, y, split, cache, names, args) for task in tasks
    }
    # Freeze both tasks before reading final test labels or predicting on test rows.
    raw_test, y_test, _ = read_data(test_path, names)
    test_groups = pd.util.hash_pandas_object(raw_test, index=False).to_numpy()
    audit["test_rows_with_training_feature_match"] = int(np.isin(test_groups, groups).sum())
    write_json(output / "data_audit.json", audit)
    summary = {}
    for task, (bundle, result) in selected.items():
        baseline, source = baselines[task]
        result["baseline_source"] = source
        result["test_metrics"] = evaluate(bundle, raw_test, y_test, task, names)
        write_json(output / task / "baseline_metrics.json", baseline)
        write_json(output / task / "selected_result.json", result)
        improved = report(output / task, result, baseline, args)
        summary[task] = dict(
            outperforms_baseline=improved,
            selected_trial=result["trial_id"],
            test_metrics=result["test_metrics"],
        )
        write_json(output / "summary.json", summary)
    print(f"Completed: {output}", flush=True)


if __name__ == "__main__":
    main()
