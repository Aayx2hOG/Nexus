"""Development-only TTL/weighting ablations; never replace the serving bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from evaluate_frozen_lightgbm import verify_frozen
from fast_lightgbm_models import NativeModel, OneHotFeatures
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from train_validated_lightgbm import canonical_features, digest
from tune_lightgbm_fast import conservative_threshold, read_data

ROOT = Path(__file__).resolve().parents[1]
TTL = ["sttl", "dttl", "ct_state_ttl"]
PLAN = [
    ("control", False, 1.0),
    ("no_ttl", True, 1.0),
    ("fuzzers_2x", False, 2.0),
    ("no_ttl_fuzzers_2x", True, 2.0),
]


# Predeclared small search, never driven by historical evaluation scores.
REGULARIZED = {"num_leaves": 31, "min_child_samples": 200, "reg_alpha": 0.5, "reg_lambda": 10.0}
ROBUST_CONFIGS = {
    "control": {},
    "regularized": {"parameters": REGULARIZED},
    "strong_regularization": {"parameters": dict(REGULARIZED, num_leaves=15, reg_lambda=20.0)},
    "softer_balance": {"balance_power": 0.5},
    "duplicate_weighting": {"duplicate_power": 0.5},
    "regularized_duplicates": {"parameters": REGULARIZED, "duplicate_power": 0.5},
    "regularized_fuzzers": {"parameters": REGULARIZED, "fuzzers": 1.5},
    "regularized_combined": {
        "parameters": REGULARIZED,
        "duplicate_power": 0.5,
        "hard_normal": 1.25,
        "fuzzers": 1.5,
    },
}


def robust_weights(frame, target, family, fuzzers_id, multiplier, scores, hard, config):
    base = weights(target, family, fuzzers_id, 1.0) ** config.get("balance_power", 1.0)
    base *= np.where(family == fuzzers_id, multiplier, 1.0)
    base = hard_normal_weights(base, target, scores, hard)
    power = config.get("duplicate_power", 0.0)
    if power:
        hashes = pd.util.hash_pandas_object(frame, index=False)
        counts = hashes.map(hashes.value_counts()).to_numpy()
        base /= counts**power
    # Keep the average weight fixed so regularization strengths remain comparable.
    return base / base.mean() if config else base


def isolated_splits(frame, splits):
    """Remove reduced-feature identities crossing ANY original partition."""
    hashes = pd.util.hash_pandas_object(frame.drop(columns=TTL), index=False).to_numpy()
    part = np.full(len(frame), -1, dtype=int)
    for number, indices in enumerate(splits.values()):
        if np.any(part[indices] != -1):
            raise ValueError("Overlapping original split indices")
        part[indices] = number
    if np.any(part == -1):
        raise ValueError("Original splits do not cover all rows")
    groups = pd.DataFrame({"hash": hashes, "partition": part})
    counts = groups.groupby("hash").partition.nunique()
    shared = counts.index[counts > 1]
    keep = ~np.isin(hashes, shared)
    return {name: indices[keep[indices]] for name, indices in splits.items()}


def weights(target, family, fuzzers_id, multiplier):
    """Balanced binary weights computed only from the current fitting partition."""
    count = np.bincount(target, minlength=2)
    if np.any(count == 0):
        raise ValueError("Fitting partition needs both classes")
    result = (len(target) / (2 * count))[target]
    return result * np.where(family == fuzzers_id, multiplier, 1.0)


def hard_normal_weights(base, target, mining_scores, multiplier):
    """Only normal examples with held-out mining score >= 0.5 get extra weight."""
    return base * np.where((target == 0) & (mining_scores >= 0.5), multiplier, 1.0)


def mining_folds(frame, target):
    groups = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    return StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42).split(
        frame, target, groups
    )


def mine_normal_scores(raw, target, family, split, parameters, fuzzers_id, rounds):
    """OOF scores for fitting rows; fold-ensemble scores for early rows.

    Mining never uses early, calibration, or selection labels for model fitting,
    vocabulary fitting, stopping, or difficulty selection. Mining rounds and
    cutoff are fixed before the experiment.
    """
    fit, early = split["fit"], split["early"]
    frame = raw.iloc[fit]
    scores = np.full(len(raw), np.nan)
    early_scores = np.zeros(len(early))
    visits = np.zeros(len(fit), dtype=int)
    for fold, (train, held) in enumerate(mining_folds(frame, target[fit]), 1):
        print(f"Mining fold {fold}/3 ({rounds} fixed rounds)...", flush=True)
        train_indices = fit[train]
        prep = OneHotFeatures().fit(raw.iloc[train_indices])
        miner = lgb.LGBMClassifier(**dict(parameters, n_estimators=rounds))
        miner.fit(
            prep.transform(raw.iloc[train_indices]),
            target[train_indices],
            sample_weight=weights(target[train_indices], family[train_indices], fuzzers_id, 1.0),
        )
        scores[fit[held]] = miner.predict_proba(prep.transform(raw.iloc[fit[held]]))[:, 1]
        early_scores += miner.predict_proba(prep.transform(raw.iloc[early]))[:, 1] / 3
        visits[held] += 1
    if not np.all(visits == 1) or not np.isfinite(scores[fit]).all():
        raise ValueError("Mining must score every fitting row exactly once out-of-fold")
    scores[early] = early_scores
    return scores


def measure(target, family, scores, threshold, fuzzers_id):
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(target, predicted, labels=[0, 1]).ravel()
    fuzzers = family == fuzzers_id
    return {
        "accuracy": float((tp + tn) / len(target)),
        "recall": float(tp / (tp + fn)),
        "false_positive_rate": float(fp / (fp + tn)),
        "fuzzers_recall": float(predicted[fuzzers].mean()),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def eligible(candidate, baseline, minimum_recall):
    return (
        candidate["recall"] >= minimum_recall
        and candidate["false_positive_rate"] < baseline["false_positive_rate"]
        and candidate["fuzzers_recall"] >= baseline["fuzzers_recall"]
    )


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "models/lightgbm_validated_v1")
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=ROOT / "data/raw/CSV_Files/Training and Testing Sets/UNSW_NB15_training-set.csv",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/binary_improvement_v1")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-rounds", type=int, default=2500)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--experiment", choices=["ttl", "hard-normal", "robust"], default="ttl")
    parser.add_argument("--mining-rounds", type=int, default=400)
    args = parser.parse_args()
    if args.mining_rounds < 1:
        parser.error("mining-rounds must be positive")
    plan = (
        PLAN
        if args.experiment == "ttl"
        else [
            ("control", False, 1.0),
            ("hard_normal_1p5x", False, 1.0),
            ("hard_normal_2x", False, 1.0),
            ("hard_normal_2x_fuzzers_2x", False, 2.0),
        ]
    )
    hard_multipliers = {
        "hard_normal_1p5x": 1.5,
        "hard_normal_2x": 2.0,
        "hard_normal_2x_fuzzers_2x": 2.0,
    }
    if args.experiment == "robust":
        plan = [
            (name, False, config.get("fuzzers", 1.0)) for name, config in ROBUST_CONFIGS.items()
        ]
        hard_multipliers = {
            name: config.get("hard_normal", 1.0) for name, config in ROBUST_CONFIGS.items()
        }
    if min(args.threads, args.max_rounds, args.patience) < 1:
        parser.error("threads, rounds, and patience must be positive")
    if args.output.exists():
        parser.error("Output exists; use a new directory")
    manifest = verify_frozen(args.baseline)
    if digest(args.train_csv) != manifest["train_sha256"]:
        raise ValueError("Must use the original development CSV, not historical test data")
    raw, family, names = read_data(args.train_csv, manifest["class_names"])
    raw = canonical_features(raw)
    if list(raw.columns) != manifest["feature_columns"]:
        raise ValueError("Development schema mismatch")
    target = (family > 0).astype(int)
    fuzzers_id = names.index("Fuzzers")
    with np.load(args.baseline / "split_indices.npz") as saved:
        original = {name: saved[name] for name in ("fit", "early", "calibration", "selection")}
    split = isolated_splits(raw, original)
    for name, indices in split.items():
        if len(np.unique(target[indices])) != 2 or not np.any(family[indices] == fuzzers_id):
            raise ValueError(f"{name} lacks both binary classes or Fuzzers after filtering")
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez(args.output / "split_indices.npz", **split)
    protocol = {
        "status": "running",
        "train_sha256": manifest["train_sha256"],
        "baseline_sha256": manifest["artifact_hashes"]["binary/model.joblib"],
        "source_sha256": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
        "plan": plan,
        "experiment": args.experiment,
        "robust_configs": ROBUST_CONFIGS if args.experiment == "robust" else {},
        "unique_predictor_gate": args.experiment == "robust",
        "hard_normal_multipliers": hard_multipliers
        if args.experiment in ("hard-normal", "robust")
        else {},
        "mining": {
            "folds": 3,
            "rounds": args.mining_rounds,
            "cutoff": 0.5,
            "scope": "fit-only grouped OOF; early scores from fit-only fold ensemble",
        }
        if args.experiment in ("hard-normal", "robust")
        else None,
        "seed": 42,
        "max_rounds": args.max_rounds,
        "patience": args.patience,
        "minimum_recall": 0.95,
        "calibration_recall_confidence": 0.95,
        "retained_rows": {k: len(v) for k, v in split.items()},
        "removed_rows": {k: len(original[k]) - len(v) for k, v in split.items()},
        "rule": "Selection recall >=95%, lower FPR than frozen baseline, "
        "Fuzzers recall no worse than baseline. Rank eligible candidates by FPR.",
        "scope": "Reused development selection set; not independent confirmation. "
        "All candidates use identical filtered partitions. Frozen baseline retains "
        "its original training; retrained control isolates this difference. "
        "Historical test data is never read. No automatic deployment.",
    }
    write(args.output / "protocol.json", protocol)
    print("Retained partition sizes:", protocol["retained_rows"], flush=True)
    baseline = joblib.load(args.baseline / "binary/model.joblib")
    baseline.estimator.set_params(n_jobs=args.threads)
    select = split["selection"]
    cal = split["calibration"]
    baseline_metrics = measure(
        target[select],
        family[select],
        baseline.predict_proba(raw.iloc[select])[:, 1],
        baseline.decision_threshold_,
        fuzzers_id,
    )
    parameters = baseline.estimator.get_params().copy()
    parameters.update(class_weight=None, n_jobs=args.threads, random_state=42)
    mining_scores = np.zeros(len(raw))
    if args.experiment in ("hard-normal", "robust"):
        mining_scores = mine_normal_scores(
            raw, target, family, split, parameters, fuzzers_id, args.mining_rounds
        )
        np.save(args.output / "mining_scores.npy", mining_scores)
        protocol["mining"]["hard_normal_counts"] = {
            key: int(((target[idx] == 0) & (mining_scores[idx] >= 0.5)).sum())
            for key, idx in split.items()
            if key in ("fit", "early")
        }
        write(args.output / "protocol.json", protocol)
    unique_select = select[~raw.iloc[select].duplicated().to_numpy()]
    baseline_unique = measure(
        target[unique_select],
        family[unique_select],
        baseline.predict_proba(raw.iloc[unique_select])[:, 1],
        baseline.decision_threshold_,
        fuzzers_id,
    )
    summary = {
        "baseline_unique": baseline_unique,
        "baseline": baseline_metrics,
        "trials": {},
        "recommended_candidate": None,
    }
    best = None
    for name, drop_ttl, multiplier in plan:
        print(f"Training {name}...", flush=True)
        frame = raw.drop(columns=TTL) if drop_ttl else raw
        fit, early = split["fit"], split["early"]
        config = ROBUST_CONFIGS.get(name, {}) if args.experiment == "robust" else {}
        trial_parameters = dict(parameters, **config.get("parameters", {}))
        prep = OneHotFeatures().fit(frame.iloc[fit])
        model = lgb.LGBMClassifier(**dict(trial_parameters, n_estimators=args.max_rounds))
        model.fit(
            prep.transform(frame.iloc[fit]),
            target[fit],
            sample_weight=robust_weights(
                frame.iloc[fit],
                target[fit],
                family[fit],
                fuzzers_id,
                multiplier,
                mining_scores[fit],
                hard_multipliers.get(name, 1.0),
                config,
            ),
            eval_set=[(prep.transform(frame.iloc[early]), target[early])],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(args.patience, verbose=False)],
        )
        rounds = int(model.best_iteration_)
        refit = np.sort(np.r_[fit, early])
        prep = OneHotFeatures().fit(frame.iloc[refit])
        model = lgb.LGBMClassifier(**dict(trial_parameters, n_estimators=rounds))
        model.fit(
            prep.transform(frame.iloc[refit]),
            target[refit],
            sample_weight=robust_weights(
                frame.iloc[refit],
                target[refit],
                family[refit],
                fuzzers_id,
                multiplier,
                mining_scores[refit],
                hard_multipliers.get(name, 1.0),
                config,
            ),
        )
        threshold, _ = conservative_threshold(
            target[cal], model.predict_proba(prep.transform(frame.iloc[cal]))[:, 1], 0.95, 0.95
        )
        model.decision_threshold_ = float(threshold)
        candidate = NativeModel(prep, model)
        result = measure(
            target[select],
            family[select],
            candidate.predict_proba(frame.iloc[select])[:, 1],
            threshold,
            fuzzers_id,
        )
        result.update(
            threshold=float(threshold),
            rounds=rounds,
            dropped_features=TTL if drop_ttl else [],
            fuzzers_weight=multiplier,
            hard_normal_weight=hard_multipliers.get(name, 1.0),
        )
        result["unique_predictors"] = measure(
            target[unique_select],
            family[unique_select],
            candidate.predict_proba(frame.iloc[unique_select])[:, 1],
            threshold,
            fuzzers_id,
        )
        result["configuration"] = config
        result["eligible"] = eligible(result, baseline_metrics, 0.95)
        if args.experiment == "robust":
            unique = result["unique_predictors"]
            result["unique_gate_met"] = (
                unique["recall"] >= baseline_unique["recall"]
                and unique["fuzzers_recall"] >= baseline_unique["fuzzers_recall"]
                and unique["false_positive_rate"] <= baseline_unique["false_positive_rate"]
            )
            result["eligible"] = result["eligible"] and result["unique_gate_met"]
        folder = args.output / name
        folder.mkdir()
        joblib.dump(candidate, folder / "model.joblib")
        result["checkpoint_sha256"] = digest(folder / "model.joblib")
        write(folder / "metrics.json", result)
        summary["trials"][name] = result
        if result["eligible"] and (best is None or result["false_positive_rate"] < best):
            best = result["false_positive_rate"]
            summary["recommended_candidate"] = name
        write(args.output / "summary.json", summary)
        print(json.dumps(result, indent=2), flush=True)
    protocol["status"] = "complete"
    write(args.output / "protocol.json", protocol)
    print(f"Done: {args.output / 'summary.json'}. Serving bundle unchanged.", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
