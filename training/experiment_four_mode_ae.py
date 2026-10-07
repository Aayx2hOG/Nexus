"""Retrain and evaluate four-mode AE overrides without test-set threshold selection."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from compare_working_v2_fusion import metrics
from fast_lightgbm_models import NativeModel, OneHotFeatures
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits
from train_autoencoder import fit_autoencoder, make_autoencoder_features, reconstruction_errors
from train_validated_lightgbm import canonical_features, load_split_indices
from tune_lightgbm_fast import conservative_threshold

ROOT = Path(__file__).resolve().parents[1]


def four_mode(binary, scores, thresholds):
    """Boundary equality enters the higher mode; 0/1/2/3 = none/low/mid/high."""
    thresholds = np.asarray(thresholds, dtype=float)
    if (
        thresholds.shape != (3,)
        or not np.all(np.isfinite(thresholds))
        or not np.all(np.diff(thresholds) > 0)
    ):
        raise ValueError("Three finite, strictly increasing thresholds required")
    scores = np.asarray(scores)
    binary = np.asarray(binary, dtype=bool)
    if scores.shape != binary.shape or not np.all(np.isfinite(scores)):
        raise ValueError("Finite scores must match binary prediction shape")
    modes = np.searchsorted(thresholds, scores, side="right")
    return (modes >= 2) | ((modes == 1) & binary), modes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "experiments/four_mode_ae_20261007"
    )
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)

    def dump(name, value):
        (out / name).write_text(json.dumps(value, indent=2) + "\n")

    frozen = ROOT / "experiments/lightgbm_validated_v2"
    rawdir = ROOT / "data/raw/CSV_Files/Training and Testing Sets"
    trainpath = rawdir / "UNSW_NB15_training-set.csv"
    manifest = json.loads((frozen / "manifest.json").read_text())
    assert hashlib.sha256(trainpath.read_bytes()).hexdigest() == manifest["train_sha256"]
    raw = pd.read_csv(trainpath)
    x = canonical_features(raw.drop(columns=["id", "label", "attack_cat"]))
    y = raw.label.to_numpy(int)
    groups = pd.util.hash_pandas_object(x, index=False).to_numpy()
    split = load_split_indices(frozen / "split_indices.npz", y, groups)
    fit, early, cal, sel = [split[k] for k in ("fit", "early", "calibration", "selection")]
    protocol = dict(
        seed=42,
        splits={k: len(v) for k, v in split.items()},
        fixed_quantiles=[0.9, 0.95, 0.99],
        low_boundary_grid=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        mid_boundary_grid=[0.90, 0.95, 0.97, 0.99, 0.995, 0.999],
        high_quantile=0.9999,
        objective=(
            "Minimum selection FPR with recall >= 95%; otherwise highest recall; ties higher F1"
        ),
        rule="none -> normal; low -> binary; mid/high -> attack",
    )
    dump("protocol.json", protocol)
    print("Retraining binary LightGBM with previously selected configuration", flush=True)
    refit = np.sort(np.r_[fit, early])
    features = OneHotFeatures().fit(x.iloc[refit])
    params = json.loads((frozen / "binary/selected_validation.json").read_text())["refit"][
        "parameters"
    ]
    params["class_weight"] = {int(k): v for k, v in params["class_weight"].items()}
    model = lgb.LGBMClassifier(**params).fit(features.transform(x.iloc[refit]), y[refit])
    binary = NativeModel(features, model)
    p = binary.predict_proba(x)[:, 1]
    cutoff, _ = conservative_threshold(y[cal], p[cal], 0.95, 0.95)
    binary.decision_threshold_ = float(cutoff)
    joblib.dump(binary, out / "binary.joblib")
    normal = fit[y[fit] == 0]
    cats = ["proto", "service", "state"]
    nums = [c for c in x if c not in cats]
    imputer = SimpleImputer().fit(x.iloc[normal][nums])
    encoder = OneHotEncoder(handle_unknown="ignore", dtype=np.float32).fit(
        x.iloc[normal][cats].fillna("__MISSING__").astype(str)
    )

    def transform(frame):
        return sparse.hstack(
            [
                sparse.csr_matrix(imputer.transform(frame[nums]).astype(np.float32)),
                encoder.transform(frame[cats].fillna("__MISSING__").astype(str)),
            ],
            format="csr",
        )

    tree = transform(x)
    scaler = StandardScaler().fit(tree[normal, : len(nums)].toarray())
    ax = make_autoencoder_features(tree, len(nums), scaler)
    print("Training normal-only autoencoder", flush=True)
    ae, history, epoch = fit_autoencoder(
        ax[normal], ax[early[y[early] == 0]], (128, 32, 128), 20, 512, 0.001, 4, 42
    )
    a = reconstruction_errors(ae, ax, 2048)
    reference = a[cal[y[cal] == 0]]

    def candidate(name, quantiles):
        thresholds = np.quantile(reference, quantiles).tolist()
        pred, _ = four_mode(p[sel] >= cutoff, a[sel], thresholds)
        return dict(
            name=name, quantiles=quantiles, thresholds=thresholds, selection=metrics(y[sel], pred)
        )

    fixed = candidate("Fixed four-mode", [0.9, 0.95, 0.99])
    candidates = [
        candidate("Selected four-mode", [lo, mid, 0.9999])
        for lo in protocol["low_boundary_grid"]
        for mid in protocol["mid_boundary_grid"]
        if lo < mid
    ]
    eligible = [c for c in candidates if c["selection"]["recall"] >= 0.95]
    winner = (
        min(
            eligible,
            key=lambda c: (c["selection"]["fpr"], -c["selection"]["recall"], -c["selection"]["f1"]),
        )
        if eligible
        else max(
            candidates,
            key=lambda c: (c["selection"]["recall"], -c["selection"]["fpr"], c["selection"]["f1"]),
        )
    )
    dump(
        "frozen_selection.json",
        dict(
            binary_threshold=float(cutoff),
            fixed=fixed,
            winner=winner,
            constraint_met=bool(eligible),
            candidates=candidates,
        ),
    )
    dump(
        "ae_training.json",
        dict(
            history=history,
            best_epoch=epoch,
            normal_fit_rows=len(normal),
            architecture=[128, 32, 128],
        ),
    )
    joblib.dump(
        dict(
            model=ae,
            imputer=imputer,
            encoder=encoder,
            scaler=scaler,
            numerical_columns=nums,
            categorical_columns=cats,
        ),
        out / "autoencoder.joblib",
    )
    print("Thresholds frozen; evaluating official test", flush=True)
    testpath = rawdir / "UNSW_NB15_testing-set.csv"
    test = pd.read_csv(testpath)
    xt = canonical_features(test.drop(columns=["id", "label", "attack_cat"]))
    yt = test.label.to_numpy(int)
    pt = binary.predict_proba(xt)[:, 1]
    at = reconstruction_errors(
        ae, make_autoencoder_features(transform(xt), len(nums), scaler), 2048
    )
    base = pt >= cutoff
    results = [dict(name="Binary only", **metrics(yt, base))]
    cohorts = []
    scores = dict(y=yt, binary_probability=pt, ae_error=at)
    for c in (fixed, winner):
        pred, modes = four_mode(base, at, c["thresholds"])
        results.append(
            dict(
                name=c["name"],
                **metrics(yt, pred),
                recovered_attacks=int(((yt == 1) & ~base & pred).sum()),
                lost_attacks=int(((yt == 1) & base & ~pred).sum()),
                removed_false_alarms=int(((yt == 0) & base & ~pred).sum()),
                added_false_alarms=int(((yt == 0) & ~base & pred).sum()),
            )
        )
        scores[c["name"] + "_prediction"] = pred
        scores[c["name"] + "_mode"] = modes
        for mode, name in enumerate(["No anomaly", "Low", "Mid", "High"]):
            mask = modes == mode
            cohorts.append(
                dict(
                    variant=c["name"],
                    mode=name,
                    rows=int(mask.sum()),
                    normal=int(((yt == 0) & mask).sum()),
                    attack=int(((yt == 1) & mask).sum()),
                    predicted_attack=int(pred[mask].sum()),
                    tn=int(((yt == 0) & ~pred & mask).sum()),
                    fp=int(((yt == 0) & pred & mask).sum()),
                    fn=int(((yt == 1) & ~pred & mask).sum()),
                    tp=int(((yt == 1) & pred & mask).sum()),
                )
            )
    dump("test_results.json", results)
    pd.DataFrame(cohorts).to_csv(out / "mode_breakdown.csv", index=False)
    np.savez_compressed(out / "test_scores.npz", **scores)
    overlap = int(np.isin(pd.util.hash_pandas_object(xt, index=False).to_numpy(), groups).sum())
    dump(
        "audit.json",
        dict(
            train_sha256=hashlib.sha256(trainpath.read_bytes()).hexdigest(),
            test_sha256=hashlib.sha256(testpath.read_bytes()).hexdigest(),
            test_overlap_rows=overlap,
            grouped_splits_verified=True,
        ),
    )
    lines = [
        "# Four-mode AE fusion experiment — 7 October 2026",
        "",
        (
            "Both LightGBM and the autoencoder were freshly trained. "
            "The binary configuration comes from the previously "
            "selected v2 model; AE uses normal-only fitting, architecture "
            "128–32–128, seed 42, up to 20 epochs and patience "
            "4."
        ),
        "",
        "## Decision rule",
        "",
        "| AE mode | Binary normal | Binary attack |",
        "|---|---|---|",
        "| No anomaly | Normal | Normal |",
        "| Low | Normal | Attack |",
        "| Mid | Attack | Attack |",
        "| High | Attack | Attack |",
        "",
        (
            "Low + binary normal is interpreted as normal. Mid "
            "and high have identical binary decisions; their boundary "
            "only changes the severity label. These are score bands, "
            "not ground-truth anomaly severity classes."
        ),
        "",
        "## Training and threshold selection",
        "",
        (
            f"Grouped train partitions: {protocol['splits']}. AE "
            f"fit uses {len(normal):,} normal rows; best epoch {epoch}. "
            f"AE preprocessing is fitted only on those normal rows. "
            f"LightGBM refits on fit + early using the previously "
            f"selected 1,318 rounds. Binary cutoff {cutoff:.12g} "
            f"is calibrated for 95% recall with the existing 95% "
            f"confidence procedure."
        ),
        "",
        (
            "AE boundaries are quantiles of normal calibration "
            "errors. Fixed bands use q90/q95/q99. The validation "
            "search varies the no/low and low/mid boundaries over "
            "the grids in protocol.json, with mid/high fixed at "
            "q99.99. It minimizes selection FPR subject to recall "
            "≥95%, falling back to maximum recall if no candidate "
            "qualifies. All thresholds and models are saved before "
            "loading test data."
        ),
        "",
        f"Selection recall constraint met: {bool(eligible)}. Candidates: {len(candidates)}.",
        "",
        "| Variant | Quantiles | Error boundaries | Selection recall | Selection FPR |",
        "|---|---|---|---:|---:|",
    ]
    for c in (fixed, winner):
        lines.append(
            f"| {c['name']} | {c['quantiles']} | {c['thresholds']} | "
            f"{c['selection']['recall']:.4%} | {c['selection']['fpr']:.4%} |"
        )
    lines += [
        "",
        "For error e and boundaries t0 < t1 < t2: none e<t0; low t0≤e<t1; mid t1≤e<t2; high e≥t2.",
        "",
        "## Official test results",
        "",
        (
            f"Test rows: {len(yt):,}; normal {int((yt == 0).sum()):,}; "
            f"attack {int((yt == 1).sum()):,}. Attack is the positive "
            f"class."
        ),
        "",
        "| Model | Accuracy | Precision | Recall | F1 | FPR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            "| "
            + r["name"]
            + " | "
            + " | ".join(f"{r[k]:.4%}" for k in ["accuracy", "precision", "recall", "f1", "fpr"])
            + " |"
        )
    for r in results:
        lines += [
            "",
            f"### {r['name']}: confusion matrix",
            "",
            "Rows = actual; columns = predicted.",
            "",
            "| | Normal | Attack |",
            "|---|---:|---:|",
            f"| Normal | {r['tn']:,} | {r['fp']:,} |",
            f"| Attack | {r['fn']:,} | {r['tp']:,} |",
        ]
        if "recovered_attacks" in r:
            lines += [
                "",
                (
                    f"Compared with binary only: recovered {r['recovered_attacks']:,} "
                    f"attacks; lost {r['lost_attacks']:,} previously detected "
                    f"attacks; removed {r['removed_false_alarms']:,} false "
                    f"alarms; added {r['added_false_alarms']:,} false alarms."
                ),
            ]
    lines += [
        "",
        "## Test breakdown by mode",
        "",
        "| Variant | Mode | Rows | Actual normal | Actual attack | TN | FP | FN | TP |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in cohorts:
        lines.append(
            "| "
            + " | ".join(
                str(c[k])
                for k in ["variant", "mode", "rows", "normal", "attack", "tn", "fp", "fn", "tp"]
            )
            + " |"
        )
    lines += [
        "",
        "## Limits and reproduction",
        "",
        (
            f"This is a single-seed experiment on an official test "
            f"set already used in prior project development, not "
            f"a fresh holdout. {overlap:,} test rows have feature-identical "
            f"matches in the training CSV. Internal partitions are "
            f"group-disjoint. Previously selected binary hyperparameters "
            f"also reflect earlier development. Test data did not "
            f"select this experiment’s AE boundaries. Low reconstruction "
            f"error does not guarantee normal traffic; the lost-attack "
            f"counts measure the cost of that override."
        ),
        "",
        "Run from repository root:",
        "",
        "```sh",
        (
            ".venv/bin/python training/experiment_four_mode_ae.py "
            "--output-dir experiments/four_mode_ae_new"
        ),
        "```",
        "",
        (
            "Artifacts: protocol.json, frozen_selection.json, ae_training.json, "
            "audit.json, test_results.json, mode_breakdown.csv, "
            "test_scores.npz, binary.joblib, autoencoder.joblib."
        ),
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
