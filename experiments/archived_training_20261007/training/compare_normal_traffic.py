"""Compare development and historical benign traffic; no training or threshold tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from evaluate_frozen_lightgbm import verify_frozen
from scipy.stats import ks_2samp
from train_validated_lightgbm import canonical_features, digest
from tune_lightgbm_fast import read_data

ROOT = Path(__file__).resolve().parents[1]


def group_rates(frame, columns):
    result = (
        frame.groupby(columns, dropna=False)
        .agg(normal_rows=("alert", "size"), false_alerts=("alert", "sum"))
        .reset_index()
    )
    result["false_positive_rate"] = result.false_alerts / result.normal_rows
    result["population_share"] = result.normal_rows / len(frame)
    return result


def compare_groups(development, historical, columns):
    left = group_rates(development, columns)
    right = group_rates(historical, columns)
    result = left.merge(right, on=columns, how="outer", suffixes=("_development", "_historical"))
    for population in ("development", "historical"):
        for metric in ("normal_rows", "false_alerts", "population_share"):
            key = f"{metric}_{population}"
            result[key] = result[key].fillna(0)
    result["share_change_pp"] = 100 * (
        result.population_share_historical - result.population_share_development
    )
    result["fpr_change_pp"] = 100 * (
        result.false_positive_rate_historical - result.false_positive_rate_development
    )
    # Exact decomposition on shared groups using development rates as reference.
    shared = (result.normal_rows_development > 0) & (result.normal_rows_historical > 0)
    result["mix_contribution_pp"] = np.where(
        shared, result.share_change_pp * result.false_positive_rate_development, np.nan
    )
    result["within_group_contribution_pp"] = np.where(
        shared, result.population_share_historical * result.fpr_change_pp, np.nan
    )
    # Groups absent from either population have no comparable within-group rate.
    result["unmatched_contribution_pp"] = np.where(
        shared,
        0,
        100
        * (
            result.false_alerts_historical / len(historical)
            - result.false_alerts_development / len(development)
        ),
    )
    return result.sort_values("false_alerts_historical", ascending=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "models/lightgbm_validated_v1")
    parser.add_argument(
        "--development-csv",
        type=Path,
        default=ROOT / "data/raw/CSV_Files/Training and Testing Sets/UNSW_NB15_training-set.csv",
    )
    parser.add_argument(
        "--historical-csv",
        type=Path,
        default=ROOT / "data/raw/CSV_Files/Training and Testing Sets/UNSW_NB15_testing-set.csv",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/normal_shift_v1")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1 or args.output.exists():
        parser.error("Use positive threads and a new output directory")
    manifest = verify_frozen(args.baseline)
    if digest(args.development_csv) != manifest["train_sha256"]:
        raise ValueError("Development CSV does not match frozen split provenance")
    model = joblib.load(args.baseline / "binary/model.joblib")
    model.estimator.set_params(n_jobs=args.threads)
    threshold = float(model.decision_threshold_)
    development, y_dev, _ = read_data(args.development_csv, manifest["class_names"])
    historical, y_hist, _ = read_data(args.historical_csv, manifest["class_names"])
    development, historical = map(canonical_features, (development, historical))
    for frame in (development, historical):
        if list(frame.columns) != manifest["feature_columns"]:
            raise ValueError("Feature schema differs from frozen model")
    with np.load(args.baseline / "split_indices.npz") as splits:
        selection = splits["selection"]
    hist_hash = pd.util.hash_pandas_object(historical, index=False).to_numpy()
    known = np.load(args.baseline / "development_feature_hashes.npy")
    overlap = np.isin(hist_hash, known)
    dev = development.iloc[selection[y_dev[selection] == 0]].copy()
    hist = historical.loc[(y_hist == 0) & ~overlap].copy()
    if dev.empty or hist.empty:
        raise ValueError("Both populations must contain normal rows")
    summary = {
        "scope": "Historical descriptive diagnostic, not independent validation or causal proof. "
        "Development = original selection partition; historical = no development overlap. "
        "Rows retain duplicates. KS distances describe marginal shift; no p-value claims.",
        "checkpoint_sha256": manifest["artifact_hashes"]["binary/model.joblib"],
        "development_sha256": digest(args.development_csv),
        "historical_sha256": digest(args.historical_csv),
        "threshold": threshold,
        "historical_overlap_excluded_all_labels": int(overlap.sum()),
        "populations": {},
        "decompositions": {},
    }
    for name, frame in (("development", dev), ("historical", hist)):
        print(f"Scoring {len(frame):,} normal {name} rows...", flush=True)
        scores = model.predict_proba(frame)[:, 1]
        if not np.isfinite(scores).all():
            raise ValueError("Non-finite scores")
        frame["alert"] = scores >= threshold
        summary["populations"][name] = {
            "rows": len(frame),
            "false_alerts": int(frame.alert.sum()),
            "false_positive_rate": float(frame.alert.mean()),
            "duplicate_predictor_rows": int(frame.duplicated(manifest["feature_columns"]).sum()),
        }
    args.output.mkdir(parents=True, exist_ok=False)
    for columns in (
        ["service"],
        ["proto"],
        ["state"],
        ["sttl"],
        ["dttl"],
        ["ct_state_ttl"],
        ["service", "sttl"],
    ):
        result = compare_groups(dev, hist, columns)
        name = "_".join(columns)
        result.to_csv(args.output / f"groups_{name}.csv", index=False)
        summary["decompositions"][name] = {
            key: float(result[key].sum())
            for key in (
                "mix_contribution_pp",
                "within_group_contribution_pp",
                "unmatched_contribution_pp",
            )
        }
    numeric = []
    for column in development.select_dtypes(include="number").columns:
        a, b = dev[column].to_numpy(), hist[column].to_numpy()
        aa, bb = a[np.isfinite(a)], b[np.isfinite(b)]
        row = {
            "feature": column,
            "development_nonfinite": int(len(a) - len(aa)),
            "historical_nonfinite": int(len(b) - len(bb)),
            "ks_distance": float(ks_2samp(aa, bb, method="asymp").statistic)
            if len(aa) and len(bb)
            else None,
        }
        for name, values in (("development", aa), ("historical", bb)):
            for quantile in (0.1, 0.5, 0.9):
                row[f"{name}_p{int(quantile * 100)}"] = (
                    float(np.quantile(values, quantile)) if len(values) else None
                )
        numeric.append(row)
    pd.DataFrame(numeric).sort_values("ks_distance", ascending=False).to_csv(
        args.output / "numeric_shift.csv", index=False
    )
    summary["decomposition_note"] = (
        "Each grouping independently decomposes the same FPR gap into composition changes, "
        "within-group rate changes, and unmatched groups. Do not add across groupings. "
        "Small groups are unstable; inspect normal_rows before interpreting rates."
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["populations"], indent=2), flush=True)
    print(f"Done: {args.output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
