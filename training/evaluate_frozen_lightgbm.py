"""Evaluate frozen candidates without retraining or tuning on evaluation outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from train_validated_lightgbm import canonical_features, digest, evaluate
from tune_lightgbm import write_json
from tune_lightgbm_fast import read_data


def verify_frozen(experiment):
    manifest = json.loads((experiment / "manifest.json").read_text())
    if manifest["status"] != "frozen":
        raise ValueError("Training must finish and freeze all artifacts before evaluation")
    for name, expected in manifest["artifact_hashes"].items():
        if digest(experiment / name) != expected:
            raise ValueError(f"Frozen artifact changed: {name}")
    for name, expected in manifest["source_hashes"].items():
        path = Path(__file__).parent / name
        if not path.exists() or digest(path) != expected:
            raise ValueError(f"Training/evaluation source changed since freezing: {name}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--evaluation-csv", type=Path, required=True)
    parser.add_argument(
        "--data-role",
        choices=["historical", "fresh-holdout"],
        default="historical",
        help="Fresh holdout is a user assertion, not something code can verify",
    )
    args = parser.parse_args()
    root = args.experiment_dir
    manifest = verify_frozen(root)
    raw, y, _ = read_data(args.evaluation_csv, manifest["class_names"])
    if list(raw.columns) != manifest["feature_columns"]:
        raise ValueError("Evaluation feature schema/order differs from frozen training schema")
    raw = canonical_features(raw)
    groups = pd.util.hash_pandas_object(raw, index=False).to_numpy()
    development = np.load(root / "development_feature_hashes.npy", allow_pickle=False)
    overlap = np.isin(groups, development)
    keep = ~overlap
    if not keep.any():
        raise ValueError("No independent feature rows remain after removing development overlap")
    output = root / "final_evaluation"
    # Refuse repeated evaluation/overwriting for this frozen experiment.
    output.mkdir(exist_ok=False)
    audit = {
        "evaluation_sha256": digest(args.evaluation_csv),
        "data_role": args.data_role,
        "total_rows": len(raw),
        "excluded_development_overlap_rows": int(overlap.sum()),
        "evaluated_rows": int(keep.sum()),
        "within_evaluation_duplicate_rows": int(keep.sum() - len(np.unique(groups[keep]))),
        "class_counts_after_overlap_removal": {
            name: int(np.sum(y[keep] == i)) for i, name in enumerate(manifest["class_names"])
        },
        "scope": "Rows with predictor hashes absent from all development partitions. "
        "Duplicates inside evaluation are retained; metrics are row-weighted. "
        "This is not the full official benchmark and cannot undo prior test exposure.",
    }
    write_json(output / "audit.json", audit)
    summary = {}
    for task in ("binary", "multiclass"):
        folder = root / task
        if not folder.exists():
            continue
        target = (y[keep] > 0).astype(int) if task == "binary" else y[keep]
        expected = 2 if task == "binary" else len(manifest["class_names"])
        if len(np.unique(target)) != expected:
            summary[task] = {
                "not_scored": "Overlap-free subset lacks a required class; "
                "full ROC-AUC comparison is undefined."
            }
            continue
        summary[task] = {}
        for name, filename in (
            ("lightgbm", "model.joblib"),
            ("random_forest", "random_forest.joblib"),
        ):
            model = joblib.load(folder / filename)
            summary[task][name] = evaluate(
                model, raw.loc[keep], y[keep], task, manifest["class_names"]
            )
        if task == "binary":
            minimum = manifest["arguments"]["minimum_recall"]
            for name in ("lightgbm", "random_forest"):
                summary[task][name]["recall_constraint_met"] = (
                    summary[task][name]["recall"] >= minimum
                )
        write_json(output / "metrics.json", summary)
    write_json(output / "metrics.json", summary)
    print(f"Evaluation saved to {output}; models and thresholds were not changed.")


if __name__ == "__main__":
    main()
