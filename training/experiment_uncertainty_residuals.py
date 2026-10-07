"""Compare uncertainty-band rules and grouped-CV learned AE residual fusion."""

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from train_autoencoder import make_autoencoder_features
from train_validated_lightgbm import canonical_features, load_split_indices

ROOT = Path(__file__).resolve().parents[1]


def band_predict(p, error, cutoff, lower, upper, veto, promote):
    """Outside [lower, upper], preserve binary; within it use AE veto/promotion."""
    if not 0 <= lower <= cutoff <= upper <= 1 or not 0 <= veto < promote:
        raise ValueError("Invalid confidence band or AE thresholds")
    p, error = np.asarray(p), np.asarray(error)
    if p.shape != error.shape or not np.isfinite(p).all() or not np.isfinite(error).all():
        raise ValueError("Matching finite scores required")
    base = p >= cutoff
    inside = (p >= lower) & (p <= upper)
    adjusted = (error >= promote) | ((error >= veto) & base)
    return np.where(inside, adjusted, base)


def residual_features(bundle, frame, batch_size=2048):
    """Squared numeric residuals and SUM over each original categorical one-hot block."""
    nums, cats = bundle["numerical_columns"], bundle["categorical_columns"]
    tree = sparse.hstack(
        [
            sparse.csr_matrix(bundle["imputer"].transform(frame[nums]).astype(np.float32)),
            bundle["encoder"].transform(frame[cats].fillna("__MISSING__").astype(str)),
        ],
        format="csr",
    )
    inputs = make_autoencoder_features(tree, len(nums), bundle["scaler"])
    widths = [1] * len(nums) + [len(c) for c in bundle["encoder"].categories_]
    if sum(widths) != inputs.shape[1]:
        raise ValueError("Unexpected one-hot layout")
    starts = np.r_[0, np.cumsum(widths)[:-1]]
    residuals = np.empty((len(frame), len(widths)), dtype=np.float64)
    mean_error = np.empty(len(frame), dtype=np.float64)
    for start in range(0, len(frame), batch_size):
        stop = min(start + batch_size, len(frame))
        batch = inputs[start:stop]
        squared = np.square(bundle["model"].predict(batch) - batch.toarray())
        mean_error[start:stop] = squared.mean(axis=1)
        residuals[start:stop] = np.add.reduceat(squared, starts, axis=1)
    if not np.isfinite(residuals).all():
        raise ValueError("Nonfinite reconstruction residuals")
    np.testing.assert_allclose(residuals.sum(axis=1) / inputs.shape[1], mean_error, rtol=2e-6)
    return mean_error, residuals, nums + cats


def meta_features(p, error, residuals, kind):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    if kind == "Score-only logistic":
        return logit[:, None]
    total = np.log1p(error)
    if kind == "Total-error logistic":
        return np.column_stack([logit, total, logit * total])
    if kind == "Per-feature logistic":
        return np.column_stack([logit, np.log1p(residuals)])
    raise ValueError(kind)


def recall_threshold(y, score, target):
    fpr, recall, thresholds = roc_curve(y, score, drop_intermediate=False)
    indices = np.flatnonzero(recall >= target)
    best = min(indices, key=lambda i: (fpr[i], -recall[i], -thresholds[i]))
    return float(thresholds[best])


def select_candidate(candidates):
    eligible = [c for c in candidates if c["selection"]["recall"] >= 0.95]
    if eligible:
        return min(
            eligible,
            key=lambda c: (c["selection"]["fpr"], -c["selection"]["recall"], -c["selection"]["f1"]),
        )
    return max(candidates, key=lambda c: (c["selection"]["recall"], -c["selection"]["fpr"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "experiments/uncertainty_residuals_20261007"
    )
    out = parser.parse_args().output_dir
    out.mkdir(parents=True, exist_ok=False)

    def dump(name, value):
        (out / name).write_text(json.dumps(value, indent=2) + "\n")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    source = ROOT / "experiments/four_mode_ae_20261007"
    frozen = ROOT / "experiments/lightgbm_validated_v2"
    rawdir = ROOT / "data/raw/CSV_Files/Training and Testing Sets"
    model = joblib.load(source / "binary.joblib")
    ae = joblib.load(source / "autoencoder.joblib")
    cutoff = float(model.decision_threshold_)
    protocol = dict(
        source_models=str(source.relative_to(ROOT)),
        source_hashes={f: digest(source / f) for f in ["binary.joblib", "autoencoder.joblib"]},
        lower_band=[0.1, 0.25, 0.4, 0.5],
        upper_band=[0.65, 0.75, 0.85, 0.95],
        veto_quantiles=[0.01, 0.1, 0.3, 0.5, 0.7, 0.9],
        promote_quantiles=[0.8, 0.9, 0.95, 0.99, 0.995, 0.999],
        C=[0.01, 0.1, 1.0, 10.0],
        calibration_recall_targets=[0.95, 0.96, 0.97],
        meta_folds=5,
        seed=2026,
        objective="Minimum selection FPR with recall >=95%; ties higher recall then F1",
        fallback="If no candidate qualifies, highest recall, then lowest FPR; marked infeasible",
        learned_families=["Score-only logistic", "Total-error logistic", "Per-feature logistic"],
        base_models_retrained=False,
        cross_validation="StratifiedGroupKFold on calibration, which neither base model fitted; "
        "out-of-fold scores calibrate thresholds, then refit meta-model on calibration; "
        "selection chooses C and recall target; official test scored after freezing.",
    )
    dump("protocol.json", protocol)
    trainpath = rawdir / "UNSW_NB15_training-set.csv"
    assert digest(trainpath) == json.loads((frozen / "manifest.json").read_text())["train_sha256"]
    raw = pd.read_csv(trainpath)
    x = canonical_features(raw.drop(columns=["id", "label", "attack_cat"]))
    y = raw.label.to_numpy(int)
    groups = pd.util.hash_pandas_object(x, index=False).to_numpy()
    split = load_split_indices(frozen / "split_indices.npz", y, groups)
    cal, sel = split["calibration"], split["selection"]
    print(
        "Extracting original-feature reconstruction errors for calibration and selection",
        flush=True,
    )
    pc, ps = model.predict_proba(x.iloc[cal])[:, 1], model.predict_proba(x.iloc[sel])[:, 1]
    ac, rc, names = residual_features(ae, x.iloc[cal])
    ass, rs, names_s = residual_features(ae, x.iloc[sel])
    assert names_s == names
    yc, ys = y[cal], y[sel]
    folds = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026).split(pc, yc, groups[cal])
    )
    fold_id = np.full(len(cal), -1)
    for i, (fit, valid) in enumerate(folds):
        assert not set(groups[cal[fit]]) & set(groups[cal[valid]])
        fold_id[valid] = i
    assert np.all(fold_id >= 0)
    np.savez_compressed(
        out / "meta_folds.npz", calibration_rows=cal, fold_id=fold_id, selection_rows=sel
    )
    # Artifact parity validates that residual extraction uses exactly the preceding AE scoring.
    prior = np.load(ROOT / "experiments/confidence_ae_20261007/validation_scores.npz")
    np.testing.assert_allclose(ac, prior["calibration_ae"], rtol=1e-6)
    np.testing.assert_allclose(ass, prior["selection_ae"], rtol=1e-6)
    all_candidates = []
    chosen = {}
    models = {}
    oof_saved = {}
    for t in np.linspace(0.01, 0.99, 99):
        all_candidates.append(
            dict(family="Binary threshold", threshold=float(t), selection=metrics(ys, ps >= t))
        )
    chosen["Binary threshold"] = select_candidate(all_candidates)
    band_candidates = []
    normal_errors = ac[yc == 0]
    print("Searching uncertainty-band rules", flush=True)
    for lower, upper, qv, qp in itertools.product(
        protocol["lower_band"],
        protocol["upper_band"],
        protocol["veto_quantiles"],
        protocol["promote_quantiles"],
    ):
        if qv >= qp:
            continue
        veto, promote = np.quantile(normal_errors, [qv, qp])
        pred = band_predict(ps, ass, cutoff, lower, upper, veto, promote)
        band_candidates.append(
            dict(
                family="Uncertainty band",
                lower=lower,
                upper=upper,
                veto=float(veto),
                promote=float(promote),
                veto_quantile=qv,
                promote_quantile=qp,
                selection=metrics(ys, pred),
            )
        )
    chosen["Uncertainty band"] = select_candidate(band_candidates)
    all_candidates.extend(band_candidates)
    cv_rows = []
    for kind in protocol["learned_families"]:
        xc = meta_features(pc, ac, rc, kind)
        xs = meta_features(ps, ass, rs, kind)
        candidates = []
        for c in protocol["C"]:
            print(f"Grouped 5-fold meta training: {kind}, C={c}", flush=True)
            oof = np.empty(len(yc))

            def build(regularization=c):
                return make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=regularization, max_iter=4000, random_state=2026),
                )

            for fit, valid in folds:
                estimator = build().fit(xc[fit], yc[fit])
                oof[valid] = estimator.predict_proba(xc[valid])[:, 1]
            estimator = build().fit(xc, yc)
            key = f"{kind}_C{c}"
            models[key] = estimator
            oof_saved[key] = oof
            score = estimator.predict_proba(xs)[:, 1]
            for target in protocol["calibration_recall_targets"]:
                threshold = recall_threshold(yc, oof, target)
                record = dict(
                    family=kind,
                    model_key=key,
                    C=c,
                    target=target,
                    threshold=threshold,
                    oof=metrics(yc, oof >= threshold),
                    selection=metrics(ys, score >= threshold),
                )
                candidates.append(record)
                for i, (_, valid) in enumerate(folds):
                    cv_rows.append(
                        dict(
                            model_key=key,
                            target=target,
                            fold=i,
                            **metrics(yc[valid], oof[valid] >= threshold),
                        )
                    )
        chosen[kind] = select_candidate(candidates)
        all_candidates.extend(candidates)
        winner = chosen[kind]
        joblib.dump(models[winner["model_key"]], out / (kind.replace(" ", "_") + ".joblib"))
    prior_rule = json.loads(
        (ROOT / "experiments/confidence_ae_20261007/frozen_selection.json").read_text()
    )["chosen"]["Mode confidence/recall95"]
    chosen["Previous confidence rule"] = dict(
        family="Previous confidence rule", **{k: v for k, v in prior_rule.items() if k != "family"}
    )
    chosen["Current binary"] = dict(
        family="Current binary", threshold=cutoff, selection=metrics(ys, ps >= cutoff)
    )
    for c in chosen.values():
        c["selection_constraint_met"] = bool(c["selection"]["recall"] >= 0.95)
    dump("frozen_selection.json", chosen)
    dump("all_candidates.json", all_candidates)
    pd.DataFrame(cv_rows).to_csv(out / "grouped_cv_diagnostics.csv", index=False)
    np.savez_compressed(out / "oof_scores.npz", y=yc, **oof_saved)
    np.savez_compressed(
        out / "validation_features.npz",
        calibration_y=yc,
        calibration_p=pc,
        calibration_ae=ac,
        calibration_residuals=rc,
        selection_y=ys,
        selection_p=ps,
        selection_ae=ass,
        selection_residuals=rs,
    )
    dump("residual_feature_names.json", names)

    def predict(c, p, a, r):
        family = c["family"]
        if family in ["Current binary", "Binary threshold"]:
            return p >= c["threshold"]
        if family == "Uncertainty band":
            return band_predict(p, a, cutoff, c["lower"], c["upper"], c["veto"], c["promote"])
        if family == "Previous confidence rule":
            mode = np.searchsorted(c["boundaries"], a, side="right")
            return p >= np.asarray(c["cutoffs"])[mode]
        return (
            models[c["model_key"]].predict_proba(meta_features(p, a, r, family))[:, 1]
            >= c["threshold"]
        )

    for c in chosen.values():
        assert metrics(ys, predict(c, ps, ass, rs)) == c["selection"]
    print("All choices frozen; evaluating official test", flush=True)
    testpath = rawdir / "UNSW_NB15_testing-set.csv"
    test = pd.read_csv(testpath)
    xt = canonical_features(test.drop(columns=["id", "label", "attack_cat"]))
    yt = test.label.to_numpy(int)
    pt = model.predict_proba(xt)[:, 1]
    at, rt, test_names = residual_features(ae, xt)
    assert names == test_names
    base = pt >= cutoff
    overlap = np.isin(pd.util.hash_pandas_object(xt, index=False).to_numpy(), groups)
    results = []
    family_rows = []
    predictions = {}
    for name, c in chosen.items():
        pred = predict(c, pt, at, rt)
        predictions[name] = pred
        results.append(
            dict(
                name=name,
                **metrics(yt, pred),
                recovered_attacks=int(((yt == 1) & ~base & pred).sum()),
                lost_attacks=int(((yt == 1) & base & ~pred).sum()),
                removed_false_alarms=int(((yt == 0) & base & ~pred).sum()),
                added_false_alarms=int(((yt == 0) & ~base & pred).sum()),
                nonoverlap=metrics(yt[~overlap], pred[~overlap]),
            )
        )
        for attack in sorted(test.loc[yt == 1, "attack_cat"].unique()):
            mask = (test.attack_cat == attack).to_numpy() & (yt == 1)
            family_rows.append(
                dict(
                    model=name,
                    attack_family=attack,
                    rows=int(mask.sum()),
                    detected=int(pred[mask].sum()),
                    recall=float(pred[mask].mean()),
                )
            )
    dump("test_results.json", results)
    pd.DataFrame(family_rows).to_csv(out / "attack_family_recall.csv", index=False)
    np.savez_compressed(
        out / "test_predictions.npz",
        y=yt,
        p=pt,
        ae=at,
        residuals=rt,
        overlap=overlap,
        **predictions,
    )
    # Standardized logistic coefficients are associations, not causal explanations.
    residual_model = models[chosen["Per-feature logistic"]["model_key"]]
    coefficients = pd.DataFrame(
        dict(
            feature=["LightGBM logit"] + ["AE error: " + n for n in names],
            coefficient=residual_model[-1].coef_[0],
        )
    )
    coefficients["absolute_coefficient"] = coefficients.coefficient.abs()
    coefficients.sort_values("absolute_coefficient", ascending=False).to_csv(
        out / "residual_coefficients.csv", index=False
    )
    dump(
        "audit.json",
        dict(
            train_sha256=digest(trainpath),
            test_sha256=digest(testpath),
            grouped_splits_verified=True,
            grouped_meta_folds_verified=True,
            previous_ae_score_parity=True,
            overlap_rows=int(overlap.sum()),
            calibration_rows=len(cal),
            selection_rows=len(sel),
            test_rows=len(yt),
            candidates=len(all_candidates),
            source_hashes=protocol["source_hashes"],
        ),
    )
    write_report(
        out, chosen, results, family_rows, protocol, int(overlap.sum()), len(all_candidates)
    )
    print(
        pd.DataFrame(
            [{k: v for k, v in r.items() if k != "nonoverlap"} for r in results]
        ).to_string(index=False),
        flush=True,
    )


def write_report(out, chosen, results, family_rows, protocol, overlap, count):
    lines = [
        "# Uncertainty-band and per-feature AE fusion — 7 October 2026",
        "",
        "## Design",
        "",
        "This experiment tests the two proposed next steps: restrict AE overrides to an "
        "uncertain LightGBM score band, and learn fusion from individual reconstruction errors. "
        "The previous binary and normal-only AE models are reused, isolating fusion changes. "
        "New logistic fusion models are trained in this run.",
        "",
        "Uncertainty band: outside [lower, upper], preserve the existing binary prediction. "
        "Inside the band, low AE error below veto forces normal; error at or above promote "
        "forces attack; otherwise follow binary. Both endpoints are included. AE thresholds "
        "are normal calibration-error quantiles, not probabilities.",
        "",
        "Learned models: score-only logistic is a control; total-error logistic adds log(1+MSE) "
        "and its interaction with the LightGBM logit; per-feature logistic adds log(1+squared "
        "residual) for each original numeric feature and log(1+summed squared residuals) for "
        "each categorical one-hot block. Categories remain grouped by original input feature. "
        "All inputs are standardized within each meta-training fold.",
        "",
        "The calibration partition is excluded from both base detectors’ fitting. Five "
        "stratified, group-disjoint folds generate out-of-fold meta-model scores. Those scores "
        "calibrate thresholds at 95%, 96% or 97% recall; final meta-models refit on calibration. "
        "A separate selection partition chooses regularization and threshold target. The "
        "OOF fold table is diagnostic: its thresholds use pooled OOF labels, so it is not "
        "nested-CV performance of the whole selection pipeline.",
        "",
        "Every family selects minimum selection FPR subject to recall ≥95%, ties higher "
        "recall then F1. If infeasible, highest-recall fallback is explicitly flagged. "
        "Models, thresholds and choices are saved before test scoring. This is a fixed-base "
        "grouped-CV fusion experiment, not five independent retrainings of both base models.",
        "",
        f"Total candidates: {count}. Full grids: protocol.json. "
        "All candidates: all_candidates.json.",
        "",
        "## Selection results",
        "",
        "| Method | Recall ≥95%? | Selection recall | Selection FPR | Selection F1 |",
        "|---|---|---:|---:|---:|",
    ]
    by_name = {r["name"]: r for r in results}
    previous = by_name["Previous confidence rule"]
    band_result = by_name["Uncertainty band"]
    per_feature = by_name["Per-feature logistic"]
    lines[2:2] = [
        "## Findings",
        "",
        f"Previous confidence-rule test F1: {previous['f1']:.4%}; uncertainty-band "
        f"F1: {band_result['f1']:.4%}; per-feature fusion F1: {per_feature['f1']:.4%}. "
        "These are descriptive comparisons of frozen validation-selected rules, "
        "not test-driven threshold choices.",
        "",
        f"Per-feature fusion recovered {per_feature['recovered_attacks']} attacks and "
        f"lost {per_feature['lost_attacks']} compared with the current binary, while "
        f"removing {per_feature['removed_false_alarms']} false alarms and adding "
        f"{per_feature['added_false_alarms']}. Inspect family recall below to locate "
        "the change; the overall recall gain does not by itself justify the false alarms.",
        "",
        "All methods target at least 95% selection recall, but their achieved test recalls "
        "differ. These are not comparisons at identical test recall. The learned-model "
        "threshold grid is deliberately limited to three OOF recall targets. No extra "
        "threshold tuning was performed after viewing test results.",
        "",
    ]
    for name, c in chosen.items():
        m = c["selection"]
        lines.append(
            f"| {name} | {c['selection_constraint_met']} | {m['recall']:.4%} | "
            f"{m['fpr']:.4%} | {m['f1']:.4%} |"
        )
    band = chosen["Uncertainty band"]
    lines += [
        "",
        f"Selected uncertainty band: **{band['lower']:.0%}–{band['upper']:.0%}**. "
        f"Within it, veto below normal-error q{100 * band['veto_quantile']:g} "
        f"({band['veto']:.9g}); promote at/above q{100 * band['promote_quantile']:g} "
        f"({band['promote']:.9g}); otherwise use binary cutoff 57.7693%.",
        "",
        "| Learned method | C | OOF recall target | Final score cutoff |",
        "|---|---:|---:|---:|",
    ]
    for kind in protocol["learned_families"]:
        c = chosen[kind]
        lines.append(f"| {kind} | {c['C']} | {c['target']:.0%} | {c['threshold']:.9g} |")
    lines += [
        "",
        "## Official test results",
        "",
        "| Method | Accuracy | Precision | Recall | F1 | FPR |",
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
    lines += [
        "",
        "## Errors changed versus current binary",
        "",
        "| Method | Recovered attacks | Lost attacks | Removed false alarms | Added false alarms |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in results:
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
        "Actual classes are rows; predicted classes are columns.",
    ]
    for r in results:
        lines += [
            "",
            f"### {r['name']}",
            "",
            "| | Normal | Attack |",
            "|---|---:|---:|",
            f"| Normal | {r['tn']:,} | {r['fp']:,} |",
            f"| Attack | {r['fn']:,} | {r['tp']:,} |",
        ]
    family_frame = pd.DataFrame(family_rows)
    pivot = family_frame.pivot(index="attack_family", columns="model", values="recall")
    support = family_frame.groupby("attack_family")["rows"].first()
    lines += [
        "",
        "## Recall by attack family",
        "",
        "| Family | Rows | " + " | ".join(chosen) + " |",
        "|---|---:|" + "---:|" * len(chosen),
    ]
    for family in pivot.index:
        lines.append(
            f"| {family} | {support[family]} | "
            + " | ".join(f"{pivot.loc[family, name]:.2%}" for name in chosen)
            + " |"
        )
    lines += [
        "",
        "## Excluding feature-identical training matches",
        "",
        f"{overlap:,} test rows match training features. This secondary diagnostic uses "
        "the same frozen predictions and does not select models.",
        "",
        "| Method | Accuracy | Recall | F1 | FPR |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            "| "
            + r["name"]
            + " | "
            + " | ".join(f"{r['nonoverlap'][k]:.4%}" for k in ["accuracy", "recall", "f1", "fpr"])
            + " |"
        )
    lines += [
        "",
        "## Limits and artifacts",
        "",
        "Selection and official test have been used in prior development; this remains "
        "exploratory, not an untouched holdout. Prior binary hyperparameters were selected "
        "on this selection set. Grouped CV protects meta-training from duplicate leakage "
        "but cannot undo that development history. No test score selected a current "
        "candidate, and no serving model was changed. Small attack families have unstable "
        "recall estimates. Scores are not guaranteed live attack probabilities.",
        "",
        "Saved: exact rules, meta-models, all validation candidates, fold assignments, OOF "
        "scores, validation residuals, test predictions, attack-family recall, source hashes "
        "and standardized residual coefficients. Coefficients describe fitted associations, "
        "not causal effects or reliable feature importance under correlated inputs.",
        "",
        "```sh",
        ".venv/bin/python training/experiment_uncertainty_residuals.py "
        "--output-dir experiments/uncertainty_residuals_new",
        "```",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
