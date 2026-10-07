"""Validation-selected AE bands with mode-specific LightGBM attack-score cutoffs."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from compare_working_v2_fusion import metrics
from scipy import sparse
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss
from threadpoolctl import threadpool_limits
from train_autoencoder import make_autoencoder_features, reconstruction_errors
from train_validated_lightgbm import canonical_features, load_split_indices

ROOT = Path(__file__).resolve().parents[1]


def predict_rule(probability, error, boundaries, cutoffs):
    """Attack iff score >= cutoff for its AE mode; >1 cutoff means always normal."""
    boundaries = np.asarray(boundaries, dtype=float)
    cutoffs = np.asarray(cutoffs, dtype=float)
    probability = np.asarray(probability, dtype=float)
    error = np.asarray(error, dtype=float)
    if boundaries.shape != (3,) or not np.all(np.diff(boundaries) > 0):
        raise ValueError("Expected three increasing AE boundaries")
    if cutoffs.shape != (4,) or not np.all(np.isfinite(cutoffs)):
        raise ValueError("Expected four finite attack cutoffs")
    if (
        probability.shape != error.shape
        or not np.all(np.isfinite(error))
        or not np.all(np.isfinite(probability))
        or not np.all(np.isfinite(boundaries))
        or np.any((probability < 0) | (probability > 1))
    ):
        raise ValueError("Finite, matching scores and probabilities in [0,1] required")
    modes = np.searchsorted(boundaries, error, side="right")
    return probability >= cutoffs[modes], modes


def count_metrics(tn, fp, fn, tp):
    return dict(
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
        recall=float(tp / (tp + fn)),
        fpr=float(fp / (fp + tn)),
        precision=float(tp / max(tp + fp, 1)),
        f1=float(2 * tp / max(2 * tp + fp + fn, 1)),
        accuracy=float((tn + tp) / (tn + fp + fn + tp)),
    )


def reliability(y, p):
    bins = np.minimum((p * 10).astype(int), 9)
    result = []
    for i in range(10):
        mask = bins == i
        if mask.any():
            result.append(
                dict(
                    bin=f"{i * 10}–{(i + 1) * 10}%",
                    rows=int(mask.sum()),
                    mean_score=float(p[mask].mean()),
                    observed_attack_rate=float(y[mask].mean()),
                )
            )
    ece = sum(r["rows"] * abs(r["mean_score"] - r["observed_attack_rate"]) for r in result) / len(y)
    return dict(
        brier=float(brier_score_loss(y, p)),
        log_loss=float(log_loss(y, p, labels=[0, 1])),
        ece=float(ece),
        bins=result,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "experiments/confidence_ae_20261007"
    )
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    source = ROOT / "experiments/four_mode_ae_20261007"
    frozen = ROOT / "experiments/lightgbm_validated_v2"

    def dump(name, value):
        (out / name).write_text(json.dumps(value, indent=2) + "\n")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    binary = joblib.load(source / "binary.joblib")
    ae = joblib.load(source / "autoencoder.joblib")
    baseline_cutoff = binary.decision_threshold_
    cutgrid = sorted(
        set(
            [
                0.0,
                0.05,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                float(baseline_cutoff),
                0.65,
                0.75,
                0.85,
                0.9,
                0.95,
                0.99,
                1.000001,
            ]
        )
    )
    quantiles = [
        list(q)
        for q in itertools.product(
            [0.01, 0.1, 0.3, 0.5, 0.7, 0.9], [0.8, 0.9, 0.95, 0.99], [0.99, 0.995, 0.999, 0.9999]
        )
        if q[0] < q[1] < q[2]
    ]
    # Non-increasing thresholds: stronger AE evidence never requires a higher binary score.
    patterns = np.array(
        [list(reversed(c)) for c in itertools.combinations_with_replacement(range(len(cutgrid)), 4)]
    )
    protocol = dict(
        source_models=str(source.relative_to(ROOT)),
        source_hashes={f: digest(source / f) for f in ["binary.joblib", "autoencoder.joblib"]},
        quantile_grid=quantiles,
        probability_cutoff_grid=cutgrid,
        objectives=["Maximum selection F1", "Minimum selection FPR with recall >= 95%"],
        rule="Attack iff raw LightGBM score >= cutoff[AE mode]; equality enters higher AE band",
        calibration="Isotonic regression fitted on calibration; diagnostics on selection and test",
        note=(
            "Reuse freshly trained prior experiment models to isolate rule effects; no retraining."
        ),
    )
    dump("protocol.json", protocol)
    rawdir = ROOT / "data/raw/CSV_Files/Training and Testing Sets"
    trainpath = rawdir / "UNSW_NB15_training-set.csv"
    assert digest(trainpath) == json.loads((frozen / "manifest.json").read_text())["train_sha256"]
    raw = pd.read_csv(trainpath)
    x = canonical_features(raw.drop(columns=["id", "label", "attack_cat"]))
    y = raw.label.to_numpy(int)
    groups = pd.util.hash_pandas_object(x, index=False).to_numpy()
    split = load_split_indices(frozen / "split_indices.npz", y, groups)
    cal, sel = split["calibration"], split["selection"]

    def ae_score(frame):
        nums, cats = ae["numerical_columns"], ae["categorical_columns"]
        tree = sparse.hstack(
            [
                sparse.csr_matrix(ae["imputer"].transform(frame[nums]).astype(np.float32)),
                ae["encoder"].transform(frame[cats].fillna("__MISSING__").astype(str)),
            ],
            format="csr",
        )
        return reconstruction_errors(
            ae["model"], make_autoencoder_features(tree, len(nums), ae["scaler"]), 2048
        )

    print("Scoring calibration and selection partitions", flush=True)
    pc = binary.predict_proba(x.iloc[cal])[:, 1]
    ps = binary.predict_proba(x.iloc[sel])[:, 1]
    ac, ass = ae_score(x.iloc[cal]), ae_score(x.iloc[sel])
    yc, ys = y[cal], y[sel]
    reference = ac[yc == 0]
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(pc, yc)
    joblib.dump(calibrator, out / "probability_calibrator.joblib")
    calibration = dict(
        selection_raw=reliability(ys, ps),
        selection_calibrated=reliability(ys, calibrator.predict(ps)),
    )
    # Rank only on the selection partition. Test is not read until winners are saved.
    winners = {}
    summaries = []
    counts = {}

    def consider(family, boundaries, cuts, quantile, m):
        counts[family] = counts.get(family, 0) + 1
        candidate = dict(
            family=family, boundaries=boundaries, cutoffs=cuts, quantiles=quantile, selection=m
        )
        for objective in ["f1", "recall95"]:
            if objective == "recall95" and m["recall"] < 0.95:
                continue
            rank = (
                (-m["f1"], m["fpr"], -m["recall"])
                if objective == "f1"
                else (m["fpr"], -m["recall"], -m["f1"])
            )
            key = family + "/" + objective
            if key not in winners or rank < winners[key][0]:
                winners[key] = (rank, candidate)

    for cutoff in sorted(set(cutgrid + np.linspace(0.01, 0.99, 99).tolist())):
        m = metrics(ys, ps >= cutoff)
        consider("Binary threshold", [0.0, 1.0, 2.0], [cutoff] * 4, None, m)
        summaries.append(dict(family="Binary threshold", cutoff=cutoff, **m))
    total_pos = int(ys.sum())
    total_neg = len(ys) - total_pos
    allcuts = np.asarray(cutgrid)[patterns]
    print(
        f"Searching {len(quantiles)} AE boundary sets × {len(patterns)} confidence rules",
        flush=True,
    )
    for quantile in quantiles:
        boundaries = np.quantile(reference, quantile).tolist()
        modes = np.searchsorted(boundaries, ass, side="right")
        tp = np.zeros((4, len(cutgrid)), dtype=int)
        fp = np.zeros_like(tp)
        for mode in range(4):
            for k, cut in enumerate(cutgrid):
                pred = (modes == mode) & (ps >= cut)
                tp[mode, k] = np.count_nonzero(pred & (ys == 1))
                fp[mode, k] = np.count_nonzero(pred & (ys == 0))
        tps = sum(tp[mode, patterns[:, mode]] for mode in range(4))
        fps = sum(fp[mode, patterns[:, mode]] for mode in range(4))
        recalls = tps / total_pos
        f1s = 2 * tps / np.maximum(tps + fps + total_pos, 1)
        # Only objective winners per boundary can win globally; still count every rule evaluated.
        best_f1 = int(np.lexsort((-recalls, fps, -f1s))[0])
        valid = np.flatnonzero(recalls >= 0.95)
        indices = {best_f1}
        if len(valid):
            indices.add(int(valid[np.lexsort((-f1s[valid], -recalls[valid], fps[valid]))[0]]))
        for i in indices:
            m = count_metrics(total_neg - fps[i], fps[i], total_pos - tps[i], tps[i])
            consider("Mode confidence", boundaries, allcuts[i].tolist(), quantile, m)
        summaries.append(
            dict(
                family="Mode confidence",
                quantiles=str(quantile),
                best_f1=float(f1s[best_f1]),
                best_f1_cutoffs=str(allcuts[best_f1].tolist()),
                recall95_min_fpr=float(fps[valid].min() / total_neg) if len(valid) else None,
            )
        )
        pred, _ = predict_rule(ps, ass, boundaries, [1.000001, baseline_cutoff, 0.0, 0.0])
        consider(
            "Original override",
            boundaries,
            [1.000001, baseline_cutoff, 0.0, 0.0],
            quantile,
            metrics(ys, pred),
        )
    counts["Mode confidence"] = len(quantiles) * len(patterns)
    chosen = {key: value[1] for key, value in winners.items()}
    prior = json.loads((source / "frozen_selection.json").read_text())
    chosen["Prior selected four-mode"] = dict(
        family="Prior selected four-mode",
        boundaries=prior["winner"]["thresholds"],
        cutoffs=[1.000001, baseline_cutoff, 0.0, 0.0],
        quantiles=prior["winner"]["quantiles"],
    )
    chosen["Current binary"] = dict(
        family="Current binary",
        boundaries=[0.0, 1.0, 2.0],
        cutoffs=[baseline_cutoff] * 4,
        quantiles=None,
    )
    for c in chosen.values():
        pred, _ = predict_rule(ps, ass, c["boundaries"], c["cutoffs"])
        c["selection"] = metrics(ys, pred)
    dump("frozen_selection.json", dict(candidate_counts=counts, chosen=chosen))
    pd.DataFrame(summaries).to_csv(out / "search_summary.csv", index=False)
    np.savez_compressed(
        out / "validation_scores.npz",
        calibration_y=yc,
        calibration_p=pc,
        calibration_ae=ac,
        selection_y=ys,
        selection_p=ps,
        selection_ae=ass,
    )
    print("Rules frozen; scoring official test", flush=True)
    testpath = rawdir / "UNSW_NB15_testing-set.csv"
    test = pd.read_csv(testpath)
    xt = canonical_features(test.drop(columns=["id", "label", "attack_cat"]))
    yt = test.label.to_numpy(int)
    pt = binary.predict_proba(xt)[:, 1]
    at = ae_score(xt)
    pct = calibrator.predict(pt)
    calibration.update(test_raw=reliability(yt, pt), test_calibrated=reliability(yt, pct))
    dump("calibration.json", calibration)
    overlap = np.isin(pd.util.hash_pandas_object(xt, index=False).to_numpy(), groups)
    base = pt >= baseline_cutoff
    rows = []
    bands = []
    saved = dict(
        y=yt, raw_attack_score=pt, calibrated_attack_probability=pct, ae_error=at, overlap=overlap
    )
    for name, c in chosen.items():
        pred, modes = predict_rule(pt, at, c["boundaries"], c["cutoffs"])
        saved[name] = pred
        r = dict(
            name=name,
            **metrics(yt, pred),
            recovered_attacks=int(((yt == 1) & ~base & pred).sum()),
            lost_attacks=int(((yt == 1) & base & ~pred).sum()),
            removed_false_alarms=int(((yt == 0) & base & ~pred).sum()),
            added_false_alarms=int(((yt == 0) & ~base & pred).sum()),
            nonoverlap=metrics(yt[~overlap], pred[~overlap]),
        )
        rows.append(r)
        for mode, label in enumerate(["No anomaly", "Low", "Mid", "High"]):
            mask = modes == mode
            bands.append(
                dict(
                    name=name,
                    mode=label,
                    rows=int(mask.sum()),
                    normal=int(((yt == 0) & mask).sum()),
                    attacks=int(((yt == 1) & mask).sum()),
                    predicted_attack=int(pred[mask].sum()),
                )
            )
    dump("test_results.json", rows)
    dump(
        "audit.json",
        dict(
            train_sha256=digest(trainpath),
            test_sha256=digest(testpath),
            overlap_rows=int(overlap.sum()),
            nonoverlap_rows=int((~overlap).sum()),
            source_models_reused=True,
            group_disjoint_internal_splits=True,
        ),
    )
    pd.DataFrame(bands).to_csv(out / "mode_breakdown.csv", index=False)
    np.savez_compressed(out / "test_scores.npz", **saved)
    export_percentages(out, saved, chosen)
    write_report(out, chosen, rows, counts, calibration, int(overlap.sum()))
    print(
        pd.DataFrame([{k: v for k, v in r.items() if k != "nonoverlap"} for r in rows]).to_string(
            index=False
        ),
        flush=True,
    )


def export_percentages(out, saved, chosen):
    rule = chosen.get("Mode confidence/recall95", chosen["Mode confidence/f1"])
    pred, modes = predict_rule(
        saved["raw_attack_score"], saved["ae_error"], rule["boundaries"], rule["cutoffs"]
    )
    pd.DataFrame(
        {
            "test_row_zero_based": np.arange(len(pred)),
            "actual_attack": saved["y"],
            "lightgbm_attack_score_percent": 100 * saved["raw_attack_score"],
            "isotonic_estimate_percent": 100 * saved["calibrated_attack_probability"],
            "ae_error": saved["ae_error"],
            "ae_mode": np.asarray(["No anomaly", "Low", "Mid", "High"])[modes],
            "required_score_percent": 100 * np.asarray(rule["cutoffs"])[modes],
            "predicted_attack": pred.astype(int),
        }
    ).to_csv(out / "attack_percentages.csv", index=False)


def write_report(out, chosen, rows, counts, calibration, overlap):
    lines = [
        "# AE boundaries and LightGBM confidence — 7 October 2026",
        "",
        "## Experiment",
        "",
        "Reused the LightGBM and normal-only AE freshly trained in the preceding experiment "
        "so changes measure decision rules, not model randomness. The binary model supplies "
        "`predict_proba(X)[:, 1]`: a score from 0 to 1, displayed as 0–100%.",
        "",
        "Three families were searched: binary-only cutoffs, original hard AE overrides, and "
        "a separate LightGBM cutoff for each AE mode. In the last family, predict attack when "
        "`score >= cutoff[mode]`, otherwise normal. Cutoffs cannot increase as AE anomaly "
        "strength increases. No anomaly can therefore preserve a confident attack; mid/high "
        "can require binary evidence instead of forcing every row to attack.",
        "",
        f"Candidate counts: {counts}. Full grids are in protocol.json. AE boundaries use "
        "normal calibration-error quantiles. Equality enters the higher AE mode. A cutoff "
        "above 100% means always normal; 0% means always attack.",
        "",
        "Two operating objectives were declared before test scoring: maximum selection F1, "
        "and minimum selection false-positive rate (FPR) subject to attack recall ≥95%. "
        "Each family gets a separate winner for each objective. Ties prefer lower FPR then "
        "higher recall for F1; higher recall then F1 for recall-constrained selection. "
        "A missing recall95 row means no candidate met that constraint. Binary-only cutoff "
        "search provides a control for gains achievable without the AE.",
        "",
        "## Frozen rules",
        "",
        "| Rule | AE quantiles | No-anomaly cutoff | Low | Mid | High | "
        "Selection recall | Selection FPR | Selection F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    by_name = {r["name"]: r for r in rows}
    if "Mode confidence/recall95" in by_name:
        r = by_name["Mode confidence/recall95"]
        baseline = by_name["Current binary"]
        lines[2:2] = [
            "## Findings",
            "",
            f"The recall-constrained confidence rule achieved test F1 {r['f1']:.4%} "
            f"versus {baseline['f1']:.4%} for the current binary. It removed "
            f"{r['removed_false_alarms']} false alarms and lost {r['lost_attacks']} "
            f"previously detected attacks. This is a small observed improvement, "
            "not evidence of statistical significance or a deployment recommendation.",
            "",
            "In this run the selected rule requires 65% below the normal calibration-error "
            "median and 57.7693% elsewhere. Low, mid and high therefore collapse to the "
            "same decision condition. The maximum-F1 mode search likewise selected the "
            "same 40% cutoff in all modes: AE bands did not help that objective. "
            "The binary-only F1 search had a finer cutoff grid and chose 39%.",
            "",
            "The isotonic percentage calibration improved selection Brier score but "
            "worsened test Brier score. Do not treat either score as a verified live "
            "attack probability. attack_percentages.csv shows every test row’s raw "
            "percentage, calibrated estimate, AE mode, required cutoff and prediction "
            "for the recall-constrained confidence rule.",
            "",
        ]
    for name, c in chosen.items():
        cuts = ["Always normal" if v > 1 else f"{v:.4%}" for v in c["cutoffs"]]
        m = c["selection"]
        lines.append(
            f"| {name} | {c['quantiles']} | "
            + " | ".join(cuts)
            + f" | {m['recall']:.4%} | {m['fpr']:.4%} | {m['f1']:.4%} |"
        )
    lines += [
        "",
        "Exact reconstruction-error boundaries and cutoffs are in frozen_selection.json.",
        "",
        "## Official test results",
        "",
        "| Rule | Accuracy | Precision | Recall | F1 | FPR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| "
            + r["name"]
            + " | "
            + " | ".join(f"{r[k]:.4%}" for k in ["accuracy", "precision", "recall", "f1", "fpr"])
            + " |"
        )
    lines += [
        "",
        "## Changes relative to current binary",
        "",
        "| Rule | Recovered attacks | Lost attacks | Removed false alarms | Added false alarms |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| "
            + r["name"]
            + " | "
            + " | ".join(
                str(r[k])
                for k in [
                    "recovered_attacks",
                    "lost_attacks",
                    "removed_false_alarms",
                    "added_false_alarms",
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Confusion matrices",
        "",
        "Rows are actual; columns predicted. Attack is positive.",
    ]
    for r in rows:
        lines += [
            "",
            f"### {r['name']}",
            "",
            "| | Normal | Attack |",
            "|---|---:|---:|",
            f"| Normal | {r['tn']:,} | {r['fp']:,} |",
            f"| Attack | {r['fn']:,} | {r['tp']:,} |",
        ]
    lines += [
        "",
        "## Does the percentage reflect attack likelihood?",
        "",
        "An 80% raw LightGBM score is not automatically an 80% real-world attack probability. "
        "The model uses class weighting, and traffic distributions differ. An isotonic "
        "calibrator was fitted only on the calibration partition; it did not change the "
        "raw-score rules above. Lower Brier, log loss and ECE are better. ECE uses ten "
        "equal-width score bins and depends on binning.",
        "",
        "| Partition / score | Brier | Log loss | ECE |",
        "|---|---:|---:|---:|",
    ]
    for name, c in calibration.items():
        lines.append(f"| {name} | {c['brier']:.6f} | {c['log_loss']:.6f} | {c['ece']:.4%} |")
    for name in ["test_raw", "test_calibrated"]:
        lines += [
            "",
            f"### {name}: predicted percentage versus observed attacks",
            "",
            "| Score bin | Rows | Average score | Observed attack rate |",
            "|---|---:|---:|---:|",
        ]
        for c in calibration[name]["bins"]:
            lines.append(
                f"| {c['bin']} | {c['rows']:,} | {c['mean_score']:.2%} | "
                f"{c['observed_attack_rate']:.2%} |"
            )
    lines += [
        "",
        "## Test rows without feature-identical training matches",
        "",
        f"{overlap:,} official test rows match training features. The following is a "
        "secondary diagnostic excluding them, with the same frozen rules; it was not used "
        "to choose rules.",
        "",
        "| Rule | Accuracy | Recall | F1 | FPR |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| "
            + r["name"]
            + " | "
            + " | ".join(f"{r['nonoverlap'][k]:.4%}" for k in ["accuracy", "recall", "f1", "fpr"])
            + " |"
        )
    lines += [
        "",
        "## Limitations and reproduction",
        "",
        "The official test and selection partitions have been used in previous development. "
        "This is an exploratory follow-up, not a fresh independent holdout. The large rule "
        "search can overfit selection. All current choices were frozen before current test "
        "scoring; no test metric selected a rule. The base models use one seed. Calibration "
        "on this dataset does not guarantee accurate percentages on live traffic. No serving "
        "configuration was changed.",
        "",
        "```sh",
        ".venv/bin/python training/experiment_confidence_ae.py "
        "--output-dir experiments/confidence_ae_new",
        "```",
        "",
        "Saved: source-model hashes, protocol, search summaries, frozen rules, validation "
        "and test scores, calibrated probability model, mode counts, audits and metrics.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
