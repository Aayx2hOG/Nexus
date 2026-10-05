"""Controlled binary/unseen-family experiments using the development CSV only.

Run from the repository root. Never reads the official testing CSV or modifies
existing checkpoints. Every output directory must be new. Results are internal
development holdouts, not independent confirmation on a fresh external dataset.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import platform
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from anomaly_detection_models import AnomalyFeatures, Autoencoder, benign_rank, fpr_threshold
from calibration_evidence import write_evidence
from complementary_fusion import calibrate_selective
from fast_lightgbm_models import OneHotFeatures
from fusion_ablation import fit_ablations, slice_diagnostics, validate_partitions
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "data/raw/CSV_Files/Training and Testing Sets/UNSW_NB15_training-set.csv"
PARTITIONS = ("fit", "early", "fusion", "calibration", "evaluation")
MODELS = (
    "lightgbm",
    "ae_plain",
    "ae_scaled",
    "isolation_forest",
    "ae_denoising",
    "ae_latent",
    "learned_fusion",
    "rank_or",
    "naive_or",
    "selective_fusion",
)


class Tee:
    """Copy Python progress/warnings to a log without detaching execution."""

    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, message):
        self.console.write(message)
        self.log.write(message)
        self.flush()
        return len(message)

    def flush(self):
        self.console.flush()
        self.log.flush()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def grouped_split(frame, labels, families, held_family, seed):
    """Exile whole duplicate groups containing the withheld family before splitting."""
    hashes = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    _, groups = np.unique(hashes, return_inverse=True)
    held = np.zeros(len(frame), dtype=bool)
    if held_family != "none":
        target = families == held_family
        if not target.any():
            raise ValueError(f"Unknown/empty held-out family: {held_family}")
        held = np.isin(groups, np.unique(groups[target]))
    unique = np.unique(groups[~held])
    counts = np.bincount(groups)
    attacks = np.bincount(groups, weights=labels)
    strata = (attacks / counts >= 0.5).astype(int)
    remaining = unique
    partitions = {}
    # Sequential fractions yield 55/10/10/10/15 percent of retained groups.
    for name, fraction in (
        ("evaluation", 0.15),
        ("calibration", 0.10 / 0.85),
        ("fusion", 0.10 / 0.75),
        ("early", 0.10 / 0.65),
    ):
        remaining, selected = train_test_split(
            remaining,
            test_size=fraction,
            stratify=strata[remaining],
            random_state=seed,
        )
        partitions[name] = np.flatnonzero(np.isin(groups, selected))
    partitions["fit"] = np.flatnonzero(np.isin(groups, remaining))
    partitions["evaluation"] = np.sort(np.r_[partitions["evaluation"], np.flatnonzero(held)])
    for name in PARTITIONS:
        rows = partitions[name]
        if set(np.unique(labels[rows])) != {0, 1}:
            raise ValueError(f"Partition {name} needs both normal and attack rows")
    return partitions, groups


def metrics(labels, scores, predictions, baseline, families, held_family):
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    normal, attack = labels == 0, labels == 1
    recovered = attack & ~baseline & predictions
    added_fp = normal & ~baseline & predictions
    removed_fp = normal & baseline & ~predictions
    lost = attack & baseline & ~predictions
    return {
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions)),
        "fpr": float(fp / (tn + fp)),
        "false_alerts_per_1000_benign": float(1000 * fp / (tn + fp)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "recovered_lightgbm_misses": int(recovered.sum()),
        "lost_lightgbm_detections": int(lost.sum()),
        "additional_false_positives": int(added_fp.sum()),
        "removed_false_positives": int(removed_fp.sum()),
        "net_additional_false_positives": int(added_fp.sum() - removed_fp.sum()),
        "net_recovered_attacks": int(recovered.sum() - lost.sum()),
        "recovered_per_additional_fp": (
            float(recovered.sum() / added_fp.sum()) if added_fp.any() else None
        ),
        "recovery_ratio_status": (
            "finite"
            if added_fp.any()
            else "recovery_without_additional_fp"
            if recovered.any()
            else "no_recovery_no_added_fp"
        ),
        "withheld_family_recall": (
            float(predictions[families == held_family].mean()) if held_family != "none" else None
        ),
        "per_family_recall": {
            str(family): {
                "rows": int((families == family).sum()),
                "recall": float(predictions[families == family].mean()),
            }
            for family in np.unique(families[attack])
        },
    }


def evaluate_attack_types(args, frame, labels, families, split, features, seed, output):
    """Diagnostic family classifier evaluated on known attack families only.

    This is conditional on an oracle attack gate, not an end-to-end detection
    score or an unknown-family detector. Report that limitation with its metrics.
    """
    fit = split["fit"][labels[split["fit"]] == 1]
    encoder = LabelEncoder().fit(families[fit])
    early = split["early"]
    early = early[(labels[early] == 1) & np.isin(families[early], encoder.classes_)]
    evaluation = split["evaluation"]
    known = (labels[evaluation] == 1) & np.isin(families[evaluation], encoder.classes_)
    if len(encoder.classes_) < 2 or not len(early) or not known.any():
        raise ValueError(
            "Attack-type evaluation needs two fitting families and known evaluation attacks"
        )
    model = lgb.LGBMClassifier(
        n_estimators=args.max_rounds,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=50,
        class_weight="balanced",
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=args.n_jobs,
        verbosity=-1,
    )
    model.fit(
        features.transform(frame.iloc[fit]),
        encoder.transform(families[fit]),
        eval_set=[(features.transform(frame.iloc[early]), encoder.transform(families[early]))],
        callbacks=[lgb.early_stopping(75, verbose=False)],
    )
    rows = evaluation[known]
    truth = encoder.transform(families[rows])
    predictions = model.predict(features.transform(frame.iloc[rows]))
    write_json(
        output / "attack_type_metrics.json",
        {
            "scope": "Known-family classification conditional on true attack; not end-to-end",
            "known_attack_rows": len(rows),
            "excluded_unknown_attack_rows": int(((labels[evaluation] == 1) & ~known).sum()),
            "class_names": encoder.classes_.tolist(),
            "classification_report": classification_report(
                truth,
                predictions,
                labels=np.arange(len(encoder.classes_)),
                target_names=encoder.classes_.tolist(),
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                truth,
                predictions,
                labels=np.arange(len(encoder.classes_)),
            ).tolist(),
            "unknown_rejection": "Not implemented; novelty-only alerts have no known-family label",
        },
    )
    return model, encoder


def run(args, frame, labels, families, family, seed, output):
    if args.reuse_dir:
        return reuse_run(args, frame, labels, families, family, seed, output)
    output.mkdir(parents=True, exist_ok=False)
    split, groups = grouped_split(frame, labels, families, family, seed)
    np.savez_compressed(output / "split_indices.npz", **split)
    write_json(
        output / "split_audit.json",
        {
            name: {
                "rows": len(rows),
                "groups": len(np.unique(groups[rows])),
                "benign": int((labels[rows] == 0).sum()),
                "families": pd.Series(families[rows]).value_counts().to_dict(),
            }
            for name, rows in split.items()
        },
    )
    normal = {name: rows[labels[rows] == 0] for name, rows in split.items()}
    features = OneHotFeatures().fit(frame.iloc[split["fit"]])
    print("Training LightGBM (early-stopping partition only)...", flush=True)
    tree = lgb.LGBMClassifier(
        n_estimators=args.max_rounds,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=100,
        reg_lambda=1.0,
        class_weight="balanced",
        colsample_bytree=0.85,
        subsample=0.85,
        subsample_freq=1,
        random_state=seed,
        n_jobs=args.n_jobs,
        verbosity=-1,
    )
    tree.fit(
        features.transform(frame.iloc[split["fit"]]),
        labels[split["fit"]],
        eval_set=[(features.transform(frame.iloc[split["early"]]), labels[split["early"]])],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(75, verbose=True), lgb.log_evaluation(100)],
    )
    scoring_parts = ("fusion", "calibration", "evaluation")
    scores = {
        name: {"lightgbm": tree.predict_proba(features.transform(frame.iloc[split[name]]))[:, 1]}
        for name in scoring_parts
    }
    saved = {"tree_features": features, "lightgbm": tree, "anomaly_models": {}}
    if args.attack_types:
        saved["attack_types"] = evaluate_attack_types(
            args,
            frame,
            labels,
            families,
            split,
            features,
            seed,
            output,
        )
    # Fixed ablations: no winner is selected using evaluation outcomes.
    for variant, improved, noise in (
        ("plain", False, 0.0),
        ("scaled", True, 0.0),
        ("denoising", True, 0.10),
    ):
        needed = set(args.models) | (
            set(args.fusion_representations) if args.fusion_ablations else set()
        )
        if variant == "plain" and "ae_plain" not in needed:
            continue
        if variant == "scaled" and not needed.intersection({"ae_scaled", "isolation_forest"}):
            continue
        print(f"{family} seed={seed}: {variant} AE", flush=True)
        preprocessing = AnomalyFeatures(improved=improved).fit(frame.iloc[normal["fit"]])
        fit = preprocessing.transform(frame.iloc[normal["fit"]])
        early = preprocessing.transform(frame.iloc[normal["early"]])
        model = Autoencoder(seed, args.epochs, args.patience, args.batch_size, noise).fit(
            fit, early
        )
        saved["anomaly_models"][variant] = (preprocessing, model)
        write_json(output / f"{variant}_history.json", model.history)
        for name in scoring_parts:
            transformed = preprocessing.transform(frame.iloc[split[name]])
            scores[name][f"ae_{variant}"] = model.score(transformed)
            if variant == "denoising":
                scores[name]["ae_latent"] = model.score(transformed, latent=True)
        if variant == "scaled" and "isolation_forest" in needed:
            isolation = IsolationForest(
                n_estimators=200,
                max_samples=min(2048, len(fit)),
                random_state=seed,
                n_jobs=args.n_jobs,
            ).fit(fit)
            saved["isolation_forest"] = (preprocessing, isolation)
            for name in scoring_parts:
                scores[name]["isolation_forest"] = -isolation.score_samples(
                    preprocessing.transform(frame.iloc[split[name]])
                )

    # Fusion rows were not used to fit either detector or choose AE stopping.
    fusion_columns = ("lightgbm", "ae_denoising", "ae_latent")

    def fusion_matrix(name):
        return np.column_stack(
            [
                scores[name]["lightgbm"],
                np.log1p(scores[name]["ae_denoising"]),
                np.log1p(scores[name]["ae_latent"]),
            ]
        )

    fusion = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=args.fusion_c,
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
        ),
    )
    print("Training learned fusion on the disjoint fusion partition...", flush=True)
    fusion.fit(fusion_matrix("fusion"), labels[split["fusion"]])
    references = {
        key: scores["fusion"][key][labels[split["fusion"]] == 0]
        for key in ("lightgbm", "ae_denoising")
    }
    for name in scoring_parts:
        scores[name]["learned_fusion"] = fusion.predict_proba(fusion_matrix(name))[:, 1]
        scores[name]["rank_or"] = np.maximum(
            *[benign_rank(references[key], scores[name][key]) for key in references]
        )
    saved.update(fusion=fusion, fusion_columns=fusion_columns, rank_references=references)
    return finish_run(args, labels, families, family, seed, output, split, scores, saved, frame)


def reuse_run(args, frame, labels, families, family, seed, output):
    """Reuse trusted local artifacts, verifying data and split membership first."""
    source = args.reuse_dir / family / f"seed_{seed}"
    manifest = json.loads((source / "manifest.json").read_text())
    if (
        manifest["status"] != "complete"
        or manifest["held_family"] != family
        or manifest["seed"] != seed
    ):
        raise ValueError("Reuse manifest does not match requested scenario")
    if digest(args.train_csv) != manifest["data_sha256"]:
        raise ValueError("Reuse dataset hash mismatch")
    for name in ("models.joblib", "split_indices.npz"):
        if digest(source / name) != manifest["artifact_hashes"][name]:
            raise ValueError(f"Reuse artifact hash mismatch: {name}")
    with np.load(source / "split_indices.npz") as archive:
        split = {key: archive[key] for key in PARTITIONS}
    expected, _ = grouped_split(frame, labels, families, family, seed)
    if any(not np.array_equal(split[key], expected[key]) for key in PARTITIONS):
        raise ValueError("Reuse splits differ from the frozen grouped protocol")
    saved = joblib.load(source / "models.joblib")
    saved.setdefault("detector_checkpoint_sha256", manifest["artifact_hashes"]["models.joblib"])
    if args.fusion_c != saved["fusion"].named_steps["logisticregression"].C:
        raise ValueError("--fusion-c cannot change when reusing a fitted fusion model")
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "split_indices.npz", **split)
    write_json(
        output / "reused_from.json",
        {
            "directory": str(source.resolve()),
            "manifest_sha256": digest(source / "manifest.json"),
            "training": (
                "Detector checkpoints and exact partitions reused; "
                "opt-in logistic ablations fit on fusion rows only"
            ),
        },
    )
    scores = {}
    needed = set(args.models) | {"lightgbm", "ae_denoising", "ae_latent"}
    if args.fusion_ablations:
        needed.update(args.fusion_representations)
    missing = {
        key.removeprefix("ae_")
        for key in needed
        if key in {"ae_plain", "ae_scaled", "ae_denoising"}
    } - set(saved["anomaly_models"])
    if missing:
        raise ValueError(
            f"Required saved AE checkpoints missing: {sorted(missing)}; no implicit retraining"
        )
    for name in ("fusion", "calibration", "evaluation"):
        print(f"Scoring {name} with frozen local models; no fitting...", flush=True)
        rows = frame.iloc[split[name]]
        current = {
            "lightgbm": saved["lightgbm"].predict_proba(saved["tree_features"].transform(rows))[
                :, 1
            ]
        }
        for variant in ("plain", "scaled", "denoising"):
            if f"ae_{variant}" not in needed:
                continue
            features, model = saved["anomaly_models"][variant]
            values = features.transform(rows)
            current[f"ae_{variant}"] = model.score(values)
            if variant == "denoising":
                current["ae_latent"] = model.score(values, latent=True)
        if "isolation_forest" in needed:
            features, model = saved["isolation_forest"]
            current["isolation_forest"] = -model.score_samples(features.transform(rows))
        matrix = np.column_stack(
            [current["lightgbm"], np.log1p(current["ae_denoising"]), np.log1p(current["ae_latent"])]
        )
        current["learned_fusion"] = saved["fusion"].predict_proba(matrix)[:, 1]
        current["rank_or"] = np.maximum(
            *[
                benign_rank(reference, current[key])
                for key, reference in saved["rank_references"].items()
            ]
        )
        scores[name] = current
    return finish_run(args, labels, families, family, seed, output, split, scores, saved, frame)


def finish_run(args, labels, families, family, seed, output, split, scores, saved, frame):
    validate_partitions(split, labels, families, family)
    ablations = {}
    if args.fusion_ablations:
        ablations = fit_ablations(
            scores,
            split,
            labels,
            families,
            family,
            seed,
            args.fusion_c,
            args.fusion_representations,
        )
        saved["fusion_ablations"] = ablations
    print("Calibrating on benign calibration rows, then evaluating frozen decisions...", flush=True)
    thresholds, results = {}, []
    decisions, policies = {}, {}
    diagnostic_scores = dict(scores["evaluation"])
    y_cal, y_eval = labels[split["calibration"]], labels[split["evaluation"]]
    for budget in args.fpr_budgets:
        cutoffs = {
            key: fpr_threshold(values[y_cal == 0], budget)
            for key, values in scores["calibration"].items()
        }
        previous = saved.get("thresholds", {}).get(str(budget), {}).get("lightgbm")
        if previous is not None and previous != cutoffs["lightgbm"]:
            raise ValueError("Recalibration changed the saved full-budget LightGBM baseline")
        thresholds[str(budget)] = cutoffs
        baseline = scores["evaluation"]["lightgbm"] >= cutoffs["lightgbm"]
        for key in (*MODELS, *ablations):
            if key not in scores["evaluation"] or key not in set(args.models) | {"lightgbm"} | set(
                ablations
            ):
                continue
            values = scores["evaluation"][key]
            prediction = values >= cutoffs[key]
            decisions[f"{key}__{budget}"] = prediction
            result = metrics(
                y_eval, values, prediction, baseline, families[split["evaluation"]], family
            )
            result.update(
                model=key,
                held_family=family,
                seed=seed,
                budget=budget,
                threshold=cutoffs[key],
                calibration_fpr=float(
                    (scores["calibration"][key][y_cal == 0] >= cutoffs[key]).mean()
                ),
                evaluation_budget_met=result["fpr"] <= budget,
            )
            results.append(result)
        if "selective_fusion" in args.models:
            ranks = {
                name: benign_rank(
                    saved["rank_references"]["ae_denoising"], scores[name]["ae_denoising"]
                )
                for name in ("calibration", "evaluation")
            }
            for allocation in args.primary_budget_fractions:
                method = (
                    "selective_fusion"
                    if allocation == 1.0
                    else f"selective_fusion_primary_{allocation}"
                )
                policy = calibrate_selective(
                    scores["calibration"]["lightgbm"][y_cal == 0],
                    scores["calibration"]["learned_fusion"][y_cal == 0],
                    ranks["calibration"][y_cal == 0],
                    cutoffs["lightgbm"],
                    budget,
                    args.uncertain_lower_ratio,
                    args.suspicious_quantile,
                    args.min_fusion_score,
                    primary_budget_fraction=allocation,
                )
                policies[str(budget) if allocation == 1.0 else f"{method}__{budget}"] = (
                    policy.to_dict()
                )
                tree, fusion = (scores["evaluation"][key] for key in ("lightgbm", "learned_fusion"))
                prediction = policy.predict(tree, fusion, ranks["evaluation"])
                if args.reconnaissance_diagnostics:
                    diagnostic_scores[f"{method}__{budget}"] = policy.score(
                        tree, fusion, ranks["evaluation"]
                    )
                decisions[f"{method}__{budget}"] = prediction
                result = metrics(
                    y_eval,
                    policy.score(tree, fusion, ranks["evaluation"]),
                    prediction,
                    baseline,
                    families[split["evaluation"]],
                    family,
                )
                result.update(
                    model=method,
                    primary_budget_fraction=allocation,
                    preserves_full_budget_baseline=allocation == 1.0,
                    held_family=family,
                    seed=seed,
                    budget=budget,
                    threshold=policy.recovery_threshold,
                    calibration_fpr=policy.calibration_false_positives / policy.calibration_rows,
                    evaluation_budget_met=result["fpr"] <= budget,
                    routed_fraction=float(policy.route(tree, ranks["evaluation"]).mean()),
                    auc_score="budget-dependent conditional ranking",
                )
                results.append(result)
    # Conventional OR uses each detector's full budget: report total FPR honestly.
    for budget in args.fpr_budgets:
        if "naive_or" not in args.models:
            continue
        cutoffs = thresholds[str(budget)]
        a = scores["evaluation"]["lightgbm"] >= cutoffs["lightgbm"]
        b = scores["evaluation"]["ae_denoising"] >= cutoffs["ae_denoising"]
        decisions[f"naive_or__{budget}"] = a | b
        result = metrics(
            y_eval, scores["evaluation"]["rank_or"], a | b, a, families[split["evaluation"]], family
        )
        result.update(
            model="naive_or",
            held_family=family,
            seed=seed,
            budget=budget,
            evaluation_budget_met=result["fpr"] <= budget,
            auc_score="rank_or (naive OR has two cutoffs)",
        )
        results.append(result)
    if args.reconnaissance_diagnostics:
        write_json(
            output / "reconnaissance_diagnostics.json",
            slice_diagnostics(
                frame.iloc[split["evaluation"]],
                y_eval,
                families[split["evaluation"]],
                diagnostic_scores,
                decisions,
            ),
        )
    comparison_context = {
        "partition_sha256": digest(output / "split_indices.npz"),
        "data_sha256": digest(args.train_csv),
        "calibration_policy": "benign calibration only; conservative >= cutoff with ties",
        "evaluation_policy": "frozen decisions on grouped internal development holdout",
        "scenario": "binary" if family == "none" else "withheld_family",
        "fusion_c": args.fusion_c,
        "gate_configuration": [
            args.uncertain_lower_ratio,
            args.suspicious_quantile,
            args.min_fusion_score,
        ],
        "fusion_representations": args.fusion_representations if args.fusion_ablations else [],
        "primary_budget_fractions": args.primary_budget_fractions,
    }
    conditions = {
        key: value
        for key, value in comparison_context.items()
        if key not in {"partition_sha256", "scenario"}
    }
    comparison_context["conditions_sha256"] = hashlib.sha256(
        json.dumps(conditions, sort_keys=True).encode()
    ).hexdigest()
    for result in results:
        result["conditions_sha256"] = comparison_context["conditions_sha256"]
        result.update(
            {
                k: comparison_context[k]
                for k in (
                    "partition_sha256",
                    "data_sha256",
                    "calibration_policy",
                    "evaluation_policy",
                    "scenario",
                )
            }
        )
        result.setdefault("primary_budget_fraction", 1.0)
        result["method_configuration"] = (
            list(ablations[result["model"]]["columns"])
            if result["model"] in ablations
            else result["model"]
        )
    baseline_metrics = {r["budget"]: r for r in results if r["model"] == "lightgbm"}
    for result in results:
        base = baseline_metrics[result["budget"]]
        result["recall_delta"] = result["recall"] - base["recall"]
        result["fpr_delta"] = result["fpr"] - base["fpr"]
        print(
            f"{family} seed={seed} budget={result['budget']:.0%} {result['model']}: "
            f"recall={result['recall']:.4f} FPR={result['fpr']:.4f} "
            f"recovered={result['recovered_lightgbm_misses']} "
            f"lost={result['lost_lightgbm_detections']} "
            f"net_added_FP={result['net_additional_false_positives']}",
            flush=True,
        )
    saved["thresholds"] = thresholds
    saved["selective_policies"] = policies
    write_json(
        output / "calibration.json",
        {
            "partition": "calibration",
            "benign_rows": int((y_cal == 0).sum()),
            "thresholds": thresholds,
            "selective_policies": policies,
            "uses_evaluation_labels": False,
        },
    )
    joblib.dump(saved, output / "models.joblib", compress=3)
    np.savez_compressed(
        output / "evaluation_scores.npz",
        row_indices=split["evaluation"],
        labels=y_eval,
        families=families[split["evaluation"]],
        **scores["evaluation"],
    )
    np.savez_compressed(
        output / "evaluation_decisions.npz", row_indices=split["evaluation"], **decisions
    )
    model_hash = digest(output / "models.joblib")
    write_evidence(
        output,
        split=split,
        labels=y_cal,
        families=families[split["calibration"]],
        scores=scores["calibration"],
        anomaly_rank=benign_rank(
            saved["rank_references"]["ae_denoising"], scores["calibration"]["ae_denoising"]
        ),
        policies=policies,
        budgets=args.fpr_budgets,
        seed=seed,
        held_family=family,
        data_sha256=comparison_context["data_sha256"],
        model_sha256=model_hash,
    )
    for result in results:
        result["detector_checkpoint_sha256"] = saved.get("detector_checkpoint_sha256", model_hash)
    write_json(output / "metrics.json", results)
    write_json(
        output / "manifest.json",
        {
            "status": "complete",
            "comparison_context": comparison_context,
            "assessment_sha256": "a4456766c442ba2e6f41f03c05daf681c67da23ffe666a0a8b3df80bb251f1ad",
            "held_family": family,
            "seed": seed,
            "data_sha256": digest(args.train_csv),
            "source_hashes": {
                name: digest(Path(__file__).with_name(name))
                for name in (
                    "run_novelty_experiment.py",
                    "anomaly_detection_models.py",
                    "complementary_fusion.py",
                    "calibration_evidence.py",
                    "fusion_ablation.py",
                    "fast_lightgbm_models.py",
                )
            },
            "artifact_hashes": {
                name: digest(output / name)
                for name in (
                    "models.joblib",
                    "split_indices.npz",
                    "evaluation_scores.npz",
                    "evaluation_decisions.npz",
                    "calibration.json",
                    "calibration_evidence.npz",
                    "metrics.json",
                )
            },
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "sklearn": sklearn.__version__,
                "lightgbm": lgb.__version__,
            },
            "limitations": [
                "Internal development holdout; official testing CSV was not opened.",
                "Grouped identical rows do not establish session or temporal independence.",
                "Calibration FPR caps are empirical, not guarantees under sampling or drift.",
                "LightGBM is a controlled retrained baseline, not the frozen v1 checkpoint.",
                "Evaluation informs development; independent confirmation remains necessary.",
                "Withheld families simulate unseen attacks, not proven zero-day/signature bypass.",
                "No model is automatically selected or promoted.",
            ],
        },
    )
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=TRAIN_CSV)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--families",
        "--held-family",
        nargs="+",
        default=["none", "Reconnaissance", "Exploits", "DoS"],
    )
    parser.add_argument("--seeds", "--seed", type=int, nargs="+", default=[42, 123, 2026])
    parser.add_argument(
        "--fpr-budgets", "--budget", type=float, nargs="+", default=[0.01, 0.03, 0.05]
    )
    parser.add_argument(
        "--models",
        "--model",
        nargs="+",
        choices=MODELS,
        default=list(MODELS),
        help="Methods to report; LightGBM baseline is always included",
    )
    parser.add_argument(
        "--reuse-dir",
        type=Path,
        help="Reuse frozen detectors/splits; --fusion-ablations fits logistic controls only",
    )
    parser.add_argument(
        "--fusion-ablations",
        action="store_true",
        help="Fit matched logistic controls on the disjoint fusion partition",
    )
    parser.add_argument(
        "--fusion-representations",
        nargs="+",
        choices=["ae_plain", "ae_denoising", "ae_scaled"],
        default=["ae_denoising"],
        help="Reconstruction representations for matched ablations",
    )
    parser.add_argument(
        "--primary-budget-fractions",
        type=float,
        nargs="+",
        default=[1.0],
        help="Predeclared primary shares; remaining share funds recovery; no winner selection",
    )
    parser.add_argument("--reconnaissance-diagnostics", action="store_true")
    parser.add_argument("--fusion-c", type=float, default=1.0)
    parser.add_argument(
        "--uncertain-lower-ratio",
        type=float,
        default=0.5,
        help="Route negatives scoring at least this fraction of the LightGBM cutoff",
    )
    parser.add_argument(
        "--suspicious-quantile",
        type=float,
        default=0.99,
        help="Also route negatives above this benign AE percentile",
    )
    parser.add_argument("--min-fusion-score", type=float, default=0.0)
    parser.add_argument(
        "--check-data", action="store_true", help="Validate data/splits only; no training"
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-rounds", type=int, default=1500)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument(
        "--attack-types",
        action="store_true",
        help="Also evaluate known attack-family classification",
    )
    args = parser.parse_args(argv)
    if any(not 0 < v <= 1 for v in args.primary_budget_fractions) or len(
        set(args.primary_budget_fractions)
    ) != len(args.primary_budget_fractions):
        parser.error("Primary budget fractions must be unique and in (0,1]")
    if len(set(args.fusion_representations)) != len(args.fusion_representations):
        parser.error("Fusion representations must not repeat")
    if not np.isfinite(args.fusion_c) or args.fusion_c <= 0:
        parser.error("fusion-c must be finite and positive")
    if any(
        not 0 <= v <= 1
        for v in (
            args.uncertain_lower_ratio,
            args.suspicious_quantile,
            args.min_fusion_score,
        )
    ):
        parser.error("Gate settings must be in [0,1]")
    if any(
        v <= 0 for v in (args.epochs, args.patience, args.batch_size, args.max_rounds, args.n_jobs)
    ):
        parser.error("Training limits and n-jobs must be positive")
    if any(not 0 < b < 1 for b in args.fpr_budgets):
        parser.error("FPR budgets must be in (0, 1)")
    if len(set(args.families)) != len(args.families) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Families and seeds must not repeat")
    if len(set(args.fpr_budgets)) != len(args.fpr_budgets):
        parser.error("Budgets must not repeat")
    if any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error("Seeds must be in [0, 2**32)")
    if "testing" in args.train_csv.name.lower():
        parser.error("Use the development training CSV, not the official testing CSV")
    return args


def execute(args):
    raw = pd.read_csv(args.train_csv)
    raw.columns = raw.columns.str.strip()
    families = raw["attack_cat"].astype("string").str.strip().to_numpy(dtype=str)
    labels = pd.to_numeric(raw["label"], errors="raise").to_numpy()
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be 0 or 1")
    labels = labels.astype(int)
    if not np.array_equal(families == "Normal", labels == 0):
        raise ValueError("Attack categories and binary labels disagree")
    available = set(families[labels == 1]) | {"none"}
    if not set(args.families) <= available:
        raise ValueError(f"Families must be chosen from {sorted(available)}")
    frame = raw.drop(columns=["id", "label", "attack_cat"])
    for column in frame.select_dtypes(include=["object", "string"]).columns:
        frame[column] = frame[column].astype("string").str.strip()
    write_json(
        args.output_dir / "protocol.json",
        {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "data_sha256": digest(args.train_csv),
            "status": "planned",
            "partition_group_fractions": dict(
                zip(PARTITIONS, [0.55, 0.10, 0.10, 0.10, 0.15], strict=True)
            ),
        },
    )
    results = []
    for family in args.families:
        for seed in args.seeds:
            print(f"Starting family={family} seed={seed}", flush=True)
            if args.check_data:
                split, _ = grouped_split(frame, labels, families, family, seed)
                print({name: len(rows) for name, rows in split.items()}, flush=True)
                continue
            results.extend(
                run(
                    args,
                    frame,
                    labels,
                    families,
                    family,
                    seed,
                    args.output_dir / family / f"seed_{seed}",
                )
            )
            # Completed runs remain inspectable if a later run fails.
            pd.DataFrame(results).drop(columns=["confusion_matrix", "per_family_recall"]).to_csv(
                args.output_dir / "comparison.csv", index=False
            )
    if args.check_data:
        write_json(args.output_dir / "data_check.json", {"status": "passed", "training": False})
        return
    from summarize_anomaly_results import statistical_summary

    statistical_summary(pd.DataFrame(results)).to_csv(args.output_dir / "summary.csv", index=False)
    write_json(args.output_dir / "completed.json", {"runs": len(args.families) * len(args.seeds)})
    print(f"Complete. Review {args.output_dir / 'comparison.csv'} and summary.csv", flush=True)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "run.log").open("x", buffering=1) as log:
        with (
            contextlib.redirect_stdout(Tee(sys.stdout, log)),
            contextlib.redirect_stderr(Tee(sys.stderr, log)),
        ):
            try:
                execute(args)
            except Exception:
                import traceback

                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
