"""Diagnose the frozen binary winner on historical data; never fit or tune it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from evaluate_frozen_lightgbm import verify_frozen
from scipy import sparse
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits
from train_validated_lightgbm import canonical_features, digest
from tune_lightgbm_fast import read_data

ROOT = Path(__file__).resolve().parents[1]


def metrics(y, prediction):
    if len(y):
        tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    else:
        tn = fp = fn = tp = 0

    def ratio(a, b):
        return float(a / b) if b else None

    return {
        "rows": int(len(y)),
        "accuracy": ratio(tp + tn, len(y)),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "false_positive_rate": ratio(fp, fp + tn),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir", type=Path, default=ROOT / "models/lightgbm_validated_v1"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "data/raw/CSV_Files/Training and Testing Sets/UNSW_NB15_testing-set.csv",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/error_analysis_v1")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--shap-per-group", type=int, default=25)
    args = parser.parse_args()
    if args.threads < 1 or args.shap_per_group < 0:
        parser.error("threads must be positive; shap-per-group must be nonnegative")
    if args.output.exists():
        parser.error("Output already exists; choose a new --output directory")

    print("Verifying frozen model and reading historical data...", flush=True)
    manifest = verify_frozen(args.experiment_dir)
    raw, family, names = read_data(args.csv, manifest["class_names"])
    raw = canonical_features(raw)
    if list(raw.columns) != manifest["feature_columns"]:
        raise ValueError("Feature schema differs from frozen model")
    hashes = pd.util.hash_pandas_object(raw, index=False).to_numpy()
    development = np.load(args.experiment_dir / "development_feature_hashes.npy")
    keep = ~np.isin(hashes, development)
    if not keep.any():
        raise ValueError("No rows remain after removing development overlap")
    frame = raw.loc[keep].reset_index(drop=True)
    y = (family[keep] > 0).astype(int)
    unique = ~pd.Series(hashes[keep]).duplicated().to_numpy()
    model = joblib.load(args.experiment_dir / "binary/model.joblib")
    np.testing.assert_array_equal(model.classes_, [0, 1])
    threshold = float(model.decision_threshold_)
    model.estimator.set_params(n_jobs=args.threads)
    print(f"Scoring {len(frame):,} rows at frozen threshold {threshold:.9f}...", flush=True)
    with threadpool_limits(limits=args.threads):
        scores = model.predict_proba(frame)[:, 1]
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite model scores")
    prediction = (scores >= threshold).astype(int)
    outcomes = np.select(
        [(y == 1) & (prediction == 1), (y == 0) & (prediction == 1), (y == 1) & (prediction == 0)],
        ["TP", "FP", "FN"],
        default="TN",
    )
    rows = frame.copy()
    rows.insert(0, "source_row_index", np.flatnonzero(keep))
    rows["label"] = y
    rows["attack_cat"] = [names[i] for i in family[keep]]
    rows["score"] = scores
    rows["outcome"] = outcomes
    summary = {
        "scope": "Historical diagnostic, not fresh independent validation. No tuning or fitting.",
        "evaluation_sha256": digest(args.csv),
        "checkpoint_sha256": manifest["artifact_hashes"]["binary/model.joblib"],
        "threshold": threshold,
        "original_rows": len(raw),
        "excluded_development_overlap": int((~keep).sum()),
        "row_weighted": metrics(y, prediction),
        "unique_predictors_first_occurrence": metrics(y[unique], prediction[unique]),
        "duplicate_label_conflict_groups": int(
            pd.DataFrame({"hash": hashes[keep], "label": y})
            .groupby("hash")
            .label.nunique()
            .gt(1)
            .sum()
        ),
        "unique_metric_note": "First occurrence retained; conflicting-label groups reported above.",
        "by_family": {},
        "unseen_categories": {},
    }
    for name in sorted(rows.attack_cat.unique()):
        selected = rows.attack_cat.to_numpy() == name
        summary["by_family"][name] = metrics(y[selected], prediction[selected])
    for column, known in model.features.native.categories.items():
        unseen = ~frame[column].isin(known).to_numpy()
        summary["unseen_categories"][column] = {
            "rows": int(unseen.sum()),
            "values": frame.loc[unseen, column].value_counts().to_dict(),
            "metrics": metrics(y[unseen], prediction[unseen]),
        }

    args.output.mkdir(parents=True, exist_ok=False)
    rows.to_csv(args.output / "predictions.csv", index=False)
    rows[rows.outcome.isin(["FP", "FN"])].to_csv(args.output / "errors.csv", index=False)
    numeric = frame.select_dtypes(include="number").columns.tolist()
    rows.groupby("outcome")[numeric + ["score"]].median().T.to_csv(
        args.output / "feature_medians_by_outcome.csv"
    )
    for column in model.features.categorical:
        rows.groupby([column, "outcome"]).size().unstack(fill_value=0).to_csv(
            args.output / f"outcomes_by_{column}.csv"
        )

    explanations = []
    feature_names = model.features.numeric + list(
        model.features.encoder.get_feature_names_out(model.features.categorical)
    )
    for outcome in ("FP", "TN", "FN", "TP"):
        selected = rows[rows.outcome == outcome].sample(
            n=min(args.shap_per_group, int((outcomes == outcome).sum())), random_state=42
        )
        if selected.empty:
            continue
        print(f"Explaining {len(selected)} {outcome} rows...", flush=True)
        transformed = model.features.transform(frame.loc[selected.index])
        with threadpool_limits(limits=args.threads):
            contribution = model.estimator.booster_.predict(
                transformed, pred_contrib=True, num_threads=args.threads
            )
        if sparse.issparse(contribution):
            contribution = contribution.toarray()
        for index, values in zip(selected.index, contribution[:, :-1], strict=True):
            for rank, j in enumerate(np.argsort(-np.abs(values))[:10], 1):
                explanations.append(
                    {
                        "source_row_index": int(rows.loc[index, "source_row_index"]),
                        "outcome": outcome,
                        "rank": rank,
                        "feature": feature_names[j],
                        "signed_log_odds_contribution": float(values[j]),
                    }
                )
    pd.DataFrame(explanations).to_csv(args.output / "sampled_shap.csv", index=False)
    summary["shap_note"] = (
        "Seed 42 sample per outcome; signed raw-margin contributions, not causal effects. "
        "See predictions.csv for original feature values."
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["row_weighted"], indent=2), flush=True)
    print(f"Done. Share {args.output / 'summary.json'} for review.", flush=True)


if __name__ == "__main__":
    main()
