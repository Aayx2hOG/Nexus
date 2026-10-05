"""Compare two frozen binary models on identical historical rows; no tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from evaluate_frozen_lightgbm import verify_frozen
from sklearn.metrics import confusion_matrix
from train_validated_lightgbm import canonical_features, digest
from tune_lightgbm_fast import read_data

ROOT = Path(__file__).resolve().parents[1]


def metrics(y, predicted):
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()

    def divide(a, b):
        return float(a / b) if b else None

    return {
        "rows": len(y),
        "accuracy": divide(tp + tn, len(y)),
        "precision": divide(tp, tp + fp),
        "recall": divide(tp, tp + fn),
        "false_positive_rate": divide(fp, fp + tn),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def paired_changes(y, baseline, candidate):
    attack = y == 1
    return {
        "recovered_attacks": int((attack & ~baseline & candidate).sum()),
        "lost_attack_detections": int((attack & baseline & ~candidate).sum()),
        "removed_false_alarms": int((~attack & baseline & ~candidate).sum()),
        "added_false_alarms": int((~attack & ~baseline & candidate).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "models/lightgbm_validated_v1")
    parser.add_argument("--experiment", type=Path, default=ROOT / "artifacts/hard_normal_v1")
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "data/raw/CSV_Files/Training and Testing Sets/UNSW_NB15_testing-set.csv",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/hard_normal_comparison_v1")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1 or args.output.exists():
        parser.error("Use positive threads and a new output directory")
    manifest = verify_frozen(args.baseline)
    protocol = json.loads((args.experiment / "protocol.json").read_text())
    selection = json.loads((args.experiment / "summary.json").read_text())
    name = selection["recommended_candidate"]
    if protocol["status"] != "complete" or not name:
        raise ValueError("Completed experiment with a selection-qualified candidate required")
    if (
        protocol["train_sha256"] != manifest["train_sha256"]
        or protocol["baseline_sha256"] != manifest["artifact_hashes"]["binary/model.joblib"]
    ):
        raise ValueError("Candidate development provenance differs from baseline")
    candidate_path = args.experiment / name / "model.joblib"
    expected = selection["trials"][name]
    if digest(candidate_path) != expected["checkpoint_sha256"]:
        raise ValueError("Candidate checkpoint hash mismatch")
    for source in (
        "fast_lightgbm_models.py",
        "train_validated_lightgbm.py",
        "tune_lightgbm_fast.py",
        "improve_binary_detector.py",
    ):
        if digest(Path(__file__).parent / source) != protocol["source_sha256"][source]:
            raise ValueError(f"Candidate source changed: {source}")
    raw, family, names = read_data(args.csv, manifest["class_names"])
    raw = canonical_features(raw)
    if list(raw.columns) != manifest["feature_columns"]:
        raise ValueError("Feature schema mismatch")
    hashes = pd.util.hash_pandas_object(raw, index=False).to_numpy()
    keep = ~np.isin(hashes, np.load(args.baseline / "development_feature_hashes.npy"))
    if not keep.any():
        raise ValueError("No historical rows remain after development overlap removal")
    frame = raw.loc[keep]
    y = (family[keep] > 0).astype(int)
    unique = ~pd.Series(hashes[keep]).duplicated().to_numpy()
    summary = {
        "scope": "Historical diagnostic, not independent confirmation. Frozen thresholds; "
        "identical rows; no training, threshold tuning, or automatic deployment.",
        "candidate": name,
        "evaluation_sha256": digest(args.csv),
        "excluded_development_overlap": int((~keep).sum()),
        "unique_note": "First predictor occurrence retained; duplicate labels can conflict.",
        "models": {},
    }
    predictions = pd.DataFrame(
        {
            "source_row_index": np.flatnonzero(keep),
            "label": y,
            "attack_cat": [names[i] for i in family[keep]],
        }
    )
    decisions = {}
    for label, path in (
        ("baseline", args.baseline / "binary/model.joblib"),
        ("candidate", candidate_path),
    ):
        print(f"Scoring {label}: {len(frame):,} identical historical rows...", flush=True)
        model = joblib.load(path)
        np.testing.assert_array_equal(model.classes_, [0, 1])
        model.estimator.set_params(n_jobs=args.threads)
        threshold = float(model.decision_threshold_)
        if label == "candidate" and threshold != expected["threshold"]:
            raise ValueError("Candidate threshold mismatch")
        scores = model.predict_proba(frame)[:, 1]
        if not np.isfinite(scores).all():
            raise ValueError("Non-finite scores")
        predicted = scores >= threshold
        decisions[label] = predicted
        predictions[f"{label}_score"] = scores
        predictions[f"{label}_alert"] = predicted
        summary["models"][label] = {
            "checkpoint_sha256": digest(path),
            "threshold": threshold,
            "row_weighted": metrics(y, predicted),
            "unique_predictors": metrics(y[unique], predicted[unique]),
            "by_family": {
                name: metrics(y[family[keep] == i], predicted[family[keep] == i])
                for i, name in enumerate(names)
                if np.any(family[keep] == i)
            },
        }
    summary["paired_changes"] = paired_changes(y, decisions["baseline"], decisions["candidate"])
    args.output.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(args.output / "predictions.csv", index=False)
    predictions[decisions["baseline"] != decisions["candidate"]].to_csv(
        args.output / "changed_decisions.csv", index=False
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v["row_weighted"] for k, v in summary["models"].items()}, indent=2))
    print(json.dumps(summary["paired_changes"], indent=2))
    print(f"Done: {args.output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
