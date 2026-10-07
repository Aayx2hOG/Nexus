"""Audit frozen official-test predictions and report four-model complementarity.

No fitting, threshold selection, or fusion policy is performed. Saved predictions
drive every statistic. Frozen AE inference is used only to authenticate the order
of its legacy score CSV, which lacks IDs; labels/categories alone cannot do that.
An existing output directory is refused to preserve earlier analysis artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from threadpoolctl import threadpool_limits

from evaluation_isolation_forest import DEFAULT_RAW_TEST
from train_autoencoder import make_autoencoder_features, reconstruction_errors
from training_utils import PROJECT_ROOT

EXPECTED_CM = {
    "random_forest": [[26948, 10052], [561, 44771]],
    "lightgbm": [[30205, 6795], [1348, 43984]],
    "isolation_forest": [[31289, 5711], [21286, 24046]],
    "autoencoder": [[31336, 5664], [7898, 37434]],
}
MODELS = tuple(EXPECTED_CM)
AE_THRESHOLD = 0.0008786016260273755
IF_THRESHOLD = -0.02622396704713703


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def same_rows(raw, frame, name, check_ids=False):
    require(len(frame) == len(raw), f"{name}: row count mismatch")
    require(np.array_equal(raw.label, frame.true_label), f"{name}: labels/order mismatch")
    require(np.array_equal(raw.attack_cat, frame.attack_cat),
            f"{name}: categories/order mismatch")
    if check_ids:
        require(frame.source_id.is_unique, f"{name}: duplicate source IDs")
        require(np.array_equal(raw.id, frame.source_id), f"{name}: source IDs/order mismatch")
        require(np.array_equal(frame.record_index, np.arange(len(raw))),
                f"{name}: record index mismatch")


def verify_scores(saved, reproduced, name, rtol=1e-6, atol=1e-10):
    require(np.isfinite(saved).all() and np.isfinite(reproduced).all(),
            f"{name}: non-finite scores")
    require(np.shape(saved) == np.shape(reproduced), f"{name}: score shape mismatch")
    require(np.allclose(saved, reproduced, rtol=rtol, atol=atol),
            f"{name}: per-row scores differ; row alignment cannot be established")


def binary_metrics(y, prediction, score):
    require(np.isin(prediction, [0, 1]).all(), "Non-binary predictions")
    require(np.isfinite(score).all(), "Non-finite scores")
    cm = confusion_matrix(y, prediction, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "confusion_matrix": cm.tolist(), "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp),
        "recall": float(recall_score(y, prediction)),
        "precision": float(precision_score(y, prediction)),
        "f1": float(f1_score(y, prediction)),
        "false_positive_rate": float(fp / (fp + tn)),
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
    }


def recovery(mask, if_flag, ae_flag):
    n = int(mask.sum())
    def count(flags):
        return int((mask & flags).sum())

    if_count, ae_count = count(if_flag), count(ae_flag)
    return {
        "total_misses": n,
        "if_recovered": if_count, "ae_recovered": ae_count,
        "if_missed": n - if_count, "ae_missed": n - ae_count,
        "if_recovery_percentage": 100 * if_count / n if n else None,
        "ae_recovery_percentage": 100 * ae_count / n if n else None,
        "either_recovered": count(if_flag | ae_flag),
        "still_missed": count(~if_flag & ~ae_flag),
        "both_if_and_ae": count(if_flag & ae_flag),
        "only_if": count(if_flag & ~ae_flag),
        "only_ae": count(~if_flag & ae_flag),
        "neither": count(~if_flag & ~ae_flag),
    }


def analyze(frame):
    attack = frame.true_label.to_numpy() == 1
    normal = ~attack
    rf, lgb, iff, ae = [frame[f"{m}_prediction"].to_numpy() == 1 for m in MODELS]
    shared = attack & ~rf & ~lgb
    cohorts = {
        "lightgbm_misses": attack & ~lgb,
        "random_forest_misses": attack & ~rf,
        "shared_supervised_misses": shared,
        "previous_three_model_misses": shared & ~iff,
        "either_supervised_misses": attack & (~rf | ~lgb),
    }
    recoveries = {name: recovery(mask, iff, ae) for name, mask in cohorts.items()}
    per_attack = {}
    for category in sorted(frame.loc[attack, "attack_cat"].unique()):
        mask = attack & (frame.attack_cat.to_numpy() == category)
        n = int(mask.sum())
        per_attack[category] = {
            "total_records": n,
            "recall": {m: float((mask & flag).sum() / n)
                       for m, flag in zip(MODELS, (rf, lgb, iff, ae), strict=True)},
            "cohorts": {name: recovery(mask & cohort, iff, ae)
                        for name, cohort in cohorts.items()},
            "missed_by_all_four": int((mask & ~rf & ~lgb & ~iff & ~ae).sum()),
        }
    patterns = []
    for bits in itertools.product((0, 1), repeat=4):
        mask = np.ones(len(frame), dtype=bool)
        for flag, bit in zip((rf, lgb, iff, ae), bits, strict=True):
            mask &= flag == bit
        n = int((mask & attack).sum())
        patterns.append({**dict(zip(MODELS, bits, strict=True)), "attack_count": n,
                         "percentage_of_attacks": 100 * n / int(attack.sum()),
                         "normal_count": int((mask & normal).sum())})
    def count(mask):
        return int((mask & attack).sum())

    important = {
        "detected_by_all_four": count(rf & lgb & iff & ae),
        "missed_by_all_four": count(~rf & ~lgb & ~iff & ~ae),
        "ae_only_detections": count(~rf & ~lgb & ~iff & ae),
        "if_only_detections": count(~rf & ~lgb & iff & ~ae),
        "detected_only_by_anomaly_models": count(~rf & ~lgb & (iff | ae)),
        "detected_by_both_anomaly_models_only": count(~rf & ~lgb & iff & ae),
        "shared_supervised_misses_detected_by_ae": count(~rf & ~lgb & ae),
        "shared_supervised_misses_detected_by_if": count(~rf & ~lgb & iff),
        "shared_supervised_misses_detected_by_either": count(~rf & ~lgb & (iff | ae)),
        "if_detected_ae_missed_regardless_of_supervised": count(iff & ~ae),
        "ae_detected_if_missed_regardless_of_supervised": count(ae & ~iff),
    }
    def fp_count(mask):
        return int((normal & mask).sum())

    fp = {
        "normal_records": int(normal.sum()),
        "standalone": {m: {"count": fp_count(flag), "fpr": fp_count(flag) / normal.sum()}
                       for m, flag in zip(MODELS, (rf, lgb, iff, ae), strict=True)},
        "ae_only_among_all_four": fp_count(ae & ~iff & ~rf & ~lgb),
        "if_only_among_all_four": fp_count(iff & ~ae & ~rf & ~lgb),
        "ae_only_relative_to_if": fp_count(ae & ~iff),
        "if_only_relative_to_ae": fp_count(iff & ~ae),
        "ae_and_if_inclusive": fp_count(ae & iff),
        "lightgbm_and_ae_inclusive": fp_count(lgb & ae),
        "random_forest_and_ae_inclusive": fp_count(rf & ae),
        "all_four": fp_count(rf & lgb & iff & ae),
        "semantics": "Shared overlaps are inclusive; only_among_all_four excludes all others.",
    }
    return recoveries, per_attack, pd.DataFrame(patterns), important, fp


def load_and_audit(root, raw_path):
    paths = {
        "raw_test": raw_path,
        "prior_predictions": root / "reports/model_complementarity/combined_predictions.csv",
        "prior_summary": root / "reports/model_complementarity/summary.json",
        "ae_scores": root / "models/autoencoder/autoencoder_test_scores.csv",
        "if_scores": root / "models/isolation_forest/isolation_forest_test_scores.csv",
        "ae_config": root / "models/autoencoder/autoencoder_config.json",
        "if_config": root / "models/isolation_forest/isolation_forest_config.json",
        "ae_checkpoint": root / "models/autoencoder/autoencoder.joblib",
        "preprocessor": root / "data/processed/preprocessor.joblib",
        "x_test": root / "data/processed/X_test_tree.npz",
        "y_test": root / "data/processed/y_test.npy",
    }
    for m in MODELS:
        subdir = f"{m}/" if m in ("autoencoder", "isolation_forest") else ""
        paths[f"{m}_metrics"] = root / f"models/{subdir}{m}_metrics.json"
    before = {key: sha256(path) for key, path in paths.items()}
    raw = pd.read_csv(raw_path)
    raw["attack_cat"] = raw.attack_cat.fillna("Normal").astype(str)
    require(len(raw) == 82332 and raw.id.is_unique and raw.id.notna().all(),
            "Official test count/IDs invalid")
    require(np.isin(raw.label, [0, 1]).all() and int(raw.label.sum()) == 45332,
            "Official test label baseline mismatch")
    prior = pd.read_csv(paths["prior_predictions"], float_precision="round_trip")
    ae = pd.read_csv(paths["ae_scores"], float_precision="round_trip")
    iff = pd.read_csv(paths["if_scores"], float_precision="round_trip")
    same_rows(raw, prior, "Prior three-model predictions", check_ids=True)
    same_rows(raw, ae, "AE scores")
    same_rows(raw, iff, "IF scores")
    # Verify feature-level row identity, including rows sharing the same label/family.
    prep = joblib.load(paths["preprocessor"])
    numeric = prep["numeric_imputer"].transform(raw[prep["numerical_columns"]]).astype(
        np.float32)
    categorical = prep["tree_encoder"].transform(
        raw[prep["categorical_columns"]].fillna("__MISSING__").astype(str))
    reconstructed = sparse.hstack([sparse.csr_matrix(numeric), categorical], format="csr")
    saved_x = sparse.load_npz(paths["x_test"])
    require(reconstructed.shape == saved_x.shape and (reconstructed != saved_x).nnz == 0,
            "Processed features do not match official raw rows in order")
    require(np.array_equal(np.load(paths["y_test"], allow_pickle=False), raw.label),
            "Processed test labels mismatch")
    ae_config, if_config = read_json(paths["ae_config"]), read_json(paths["if_config"])
    require(ae_config["selected_threshold"] == AE_THRESHOLD, "Frozen AE threshold changed")
    require(if_config["selected_threshold"] == IF_THRESHOLD, "Frozen IF threshold changed")
    bundle = joblib.load(paths["ae_checkpoint"])
    ae_x = make_autoencoder_features(saved_x, ae_config["numerical_feature_count"],
                                    bundle["numeric_scaler"])
    with threadpool_limits(limits=1):
        reproduced = reconstruction_errors(bundle["model"], ae_x, 2048)
    verify_scores(ae.anomaly_score.to_numpy(), reproduced, "AE")
    require(np.array_equal(reproduced >= AE_THRESHOLD, ae.anomaly_prediction),
            "AE replay decisions do not match saved predictions")
    verify_scores(iff.anomaly_score.to_numpy(),
                  prior.isolation_forest_raw_anomaly_score.to_numpy(), "IF", rtol=1e-12)
    require(np.array_equal(iff.anomaly_prediction, prior.isolation_forest_anomaly_prediction),
            "IF saved predictions disagree")
    frame = prior.rename(columns={
        "isolation_forest_anomaly_prediction": "isolation_forest_prediction",
        "isolation_forest_raw_anomaly_score": "isolation_forest_score",
        "random_forest_attack_probability": "random_forest_score",
        "lightgbm_attack_probability": "lightgbm_score",
    }).copy()
    frame["autoencoder_score"] = ae.anomaly_score
    frame["autoencoder_prediction"] = ae.anomaly_prediction
    all_metrics = {}
    for m in MODELS:
        prediction, score = frame[f"{m}_prediction"], frame[f"{m}_score"]
        cutoff = AE_THRESHOLD if m == "autoencoder" else IF_THRESHOLD
        expected = score > 0.5 if m in ("random_forest", "lightgbm") else score >= cutoff
        require(np.array_equal(prediction, expected), f"{m}: frozen decision rule mismatch")
        computed = binary_metrics(raw.label, prediction, score)
        require(computed["confusion_matrix"] == EXPECTED_CM[m], f"{m}: baseline CM mismatch")
        saved = read_json(paths[f"{m}_metrics"])
        require(saved["test_rows"] == 82332, f"{m}: saved test count mismatch")
        for key, value in computed.items():
            if key in saved:
                require(np.allclose(value, saved[key], rtol=1e-10, atol=1e-12),
                        f"{m}: baseline metric {key} mismatch")
        all_metrics[m] = computed
    prior_summary = read_json(paths["prior_summary"])
    for key, total, recovered in (("lightgbm_false_negatives", 1348, 271),
                                  ("random_forest_false_negatives", 561, 145),
                                  ("missed_by_both_supervised", 485, 120)):
        require(prior_summary[key]["total"] == total and
                prior_summary[key]["isolation_forest_detected"] == recovered,
                f"Prior summary baseline mismatch: {key}")
    require(prior_summary["all_three_misses"] == 365, "Prior three-model baseline mismatch")
    audit = {
        "alignment_checks_passed": True,
        "row_identity": "Official IDs and record indices match prior three-model CSV exactly.",
        "processed_features": "Every transformed feature equals saved X_test in raw ID order.",
        "legacy_ae_alignment": (
            "AE CSV lacks IDs. Frozen checkpoint replay on verified features matches all saved "
            "scores (rtol=1e-6, atol=1e-10) and decisions exactly; labels/categories also match. "
            "Rows with identical features/scores are indistinguishable but have identical decisions."
        ),
        "legacy_if_alignment": "IF scores/decisions match the ID-bearing three-model CSV.",
        "saved_predictions_used_for_analysis": True,
        "ae_replay_max_absolute_score_difference": float(
            np.max(np.abs(ae.anomaly_score.to_numpy() - reproduced))),
        "inputs": {key: {"path": str(path.resolve()), "sha256": before[key]}
                   for key, path in paths.items()},
    }
    return frame, all_metrics, audit, paths, before


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--raw-test-csv", type=Path, default=DEFAULT_RAW_TEST)
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "reports/autoencoder_complementarity")
    args = parser.parse_args()
    require(not args.output_dir.exists(), "Output directory already exists; use a new directory")
    frame, model_metrics, audit, paths, before = load_and_audit(args.root, args.raw_test_csv)
    cohorts, per_attack, agreement, important, fp = analyze(frame)
    for name, total, recovered in (("lightgbm_misses", 1348, 271),
                                  ("random_forest_misses", 561, 145),
                                  ("shared_supervised_misses", 485, 120),
                                  ("previous_three_model_misses", 365, 0)):
        require(cohorts[name]["total_misses"] == total and
                cohorts[name]["if_recovered"] == recovered, f"Cohort baseline mismatch: {name}")
    require(int(agreement.attack_count.sum()) == 45332 and
            int(agreement.normal_count.sum()) == 37000, "Agreement totals mismatch")
    require(all(sha256(path) == before[key] for key, path in paths.items()),
            "An input artifact changed during the analysis")
    audit["input_hashes_unchanged_after_analysis"] = True
    summary = {
        "dataset": {"test_records": 82332, "attack_records": 45332, "normal_records": 37000},
        "integrity": audit,
        "thresholds": {"random_forest": 0.5, "lightgbm": 0.5,
                       "isolation_forest": IF_THRESHOLD, "autoencoder": AE_THRESHOLD,
                       "comparison": "Supervised > threshold; anomaly >= threshold."},
        "model_metrics": model_metrics, "cohorts": cohorts,
        "important_attack_patterns": important,
        "definitions": {
            "still_missed": "Within each cohort, detected by neither anomaly detector.",
            "detected_only_by_anomaly_models": "Both supervised miss; at least one anomaly flags.",
            "ae_only_detections": "AE detects; RF, LightGBM and IF all miss.",
            "if_only_detections": "IF detects; RF, LightGBM and AE all miss.",
        },
        "scope": "Historical frozen-test analysis; no optimization, fusion rule, or zero-day claim.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for filename, data in (("summary.json", summary), ("per_attack_comparison.json", per_attack),
                           ("normal_fp_overlap.json", fp)):
        (args.output_dir / filename).write_text(
            json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    agreement.to_csv(args.output_dir / "agreement_patterns.csv", index=False)
    frame.to_csv(args.output_dir / "combined_predictions.csv", index=False)
    print(json.dumps({"cohorts": cohorts, "important_attack_patterns": important,
                      "normal_fp_overlap": fp}, indent=2))
    print(f"Saved analysis to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
