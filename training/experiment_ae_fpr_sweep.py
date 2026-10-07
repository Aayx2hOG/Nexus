"""Frozen-model comparison at 5% and 10% AE calibration FPR, with explicit gates."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from anomaly_detection_models import fpr_threshold
from compare_working_v2_fusion import metrics
from evaluate_imported_seed42_test import imported_scores
from experiment_uncertainty_residuals import meta_features
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]


def mode_rules(p, error, cutoff, boundaries, confidence_cuts, lower, upper):
    """Apply fixed rules; boundaries correspond to AE tail rates b, b/2, b/10."""
    boundaries = np.asarray(boundaries, dtype=np.float64)
    if boundaries.shape != (3,) or not np.all(np.diff(boundaries) > 0):
        raise ValueError("Expected three increasing AE boundaries")
    p, error = np.asarray(p, dtype=np.float64), np.asarray(error, dtype=np.float64)
    if p.shape != error.shape or not np.isfinite(p).all() or not np.isfinite(error).all():
        raise ValueError("Matching finite scores required")
    base = p >= cutoff
    modes = np.searchsorted(boundaries, error, side="right")
    anomaly = modes >= 1
    hard = (modes >= 2) | ((modes == 1) & base)
    inside = (p >= lower) & (p <= upper)
    return {
        "AE alone": anomaly,
        "OR": base | anomaly,
        "AND": base & anomaly,
        "Four-mode hard override": hard,
        "Mode confidence": p >= np.asarray(confidence_cuts)[modes],
        "Uncertainty-band four-mode": np.where(inside, hard, base),
        "Selective recovery (p>=40%)": base | (anomaly & (p >= 0.4)),
    }


def ae_gate(base, learned, error, threshold):
    """Below AE threshold preserve binary; above it follow frozen learned fusion."""
    return np.where(np.asarray(error, dtype=np.float64) >= threshold, learned, base)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "experiments/ae_fpr_5_10_20261007"
    )
    out = parser.parse_args().output_dir
    out.mkdir(parents=True, exist_ok=False)

    def dump(name, value):
        (out / name).write_text(json.dumps(value, indent=2) + "\n")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    exp = ROOT / "experiments"
    source = exp / "four_mode_ae_20261007"
    learned_dir = exp / "uncertainty_residuals_20261007"
    imported_dir = exp / "seed_42/seed_42"
    confidence = json.loads((exp / "confidence_ae_20261007/frozen_selection.json").read_text())
    selected = json.loads((learned_dir / "frozen_selection.json").read_text())
    budget_list = [0.05, 0.10]
    cutoff = confidence["chosen"]["Current binary"]["cutoffs"][0]
    confidence_cuts = confidence["chosen"]["Mode confidence/recall95"]["cutoffs"]
    band = selected["Uncertainty band"]
    dump(
        "protocol.json",
        dict(
            ae_fpr_targets=budget_list,
            cutoff_method=(
                "Conservative empirical >= threshold on normal calibration scores; ties handled"
            ),
            four_modes="No anomaly < T(b); low [T(b),T(b/2)); mid [T(b/2),T(b/10)); high >=T(b/10)",
            original_rule="No anomaly normal; low follows binary; mid/high attack",
            confidence_cutoffs=confidence_cuts,
            binary_cutoff=cutoff,
            uncertainty_band=[band["lower"], band["upper"]],
            learned_adaptation=(
                "Keep original continuous fusion as reference; AE-gated variant follows "
                "frozen fusion when error >=T(b), otherwise original binary"
            ),
            training="All base models and learned models frozen; only AE thresholds recalibrated",
            selection="No model selected on test; report every predeclared rule at both budgets",
            scope="All previously compared rule families plus imported seed42 plain-AE+latent",
            target_interpretation=(
                "AE-only calibration FPR; combined system FPR is measured, not constrained"
            ),
        ),
    )
    trainpath = ROOT / "data/raw/CSV_Files/Training and Testing Sets/UNSW_NB15_training-set.csv"
    train = pd.read_csv(trainpath)
    train_x = train.drop(columns=["id", "label", "attack_cat"])
    manifest = json.loads((imported_dir / "manifest.json").read_text())
    assert digest(trainpath) == manifest["data_sha256"]
    assert digest(imported_dir / "models.joblib") == manifest["artifact_hashes"]["models.joblib"]
    imported = joblib.load(imported_dir / "models.joblib")
    imported_split = np.load(imported_dir / "split_indices.npz")
    ic = imported_split["calibration"]
    print("Scoring imported normal calibration partition; base models remain frozen", flush=True)
    imported_cal = imported_scores(imported, train_x.iloc[ic])
    iy = train.label.to_numpy(int)[ic]
    v = np.load(learned_dir / "validation_features.npz")
    cy = v["calibration_y"]
    reference = v["calibration_ae"][cy == 0]
    imported_reference = imported_cal["ae_plain"][iy == 0].astype(np.float64)
    frozen = {}
    calibration_rows = []
    for b in budget_list:
        bounds = [fpr_threshold(reference, rate) for rate in [b, b / 2, b / 10]]
        threshold = fpr_threshold(imported_reference, b)
        frozen[str(b)] = dict(recent_boundaries=bounds, imported_plain_threshold=threshold)
        for label, ref, t in [
            ("Recent AE", reference, bounds[0]),
            ("Imported plain AE", imported_reference, threshold),
        ]:
            observed = float((ref >= t).mean())
            assert observed <= b
            calibration_rows.append(
                dict(
                    model=label,
                    target=b,
                    threshold=t,
                    normal_rows=len(ref),
                    false_positives=int((ref >= t).sum()),
                    observed_fpr=observed,
                )
            )
    dump("frozen_thresholds.json", frozen)
    dump("ae_calibration.json", calibration_rows)
    models = {
        name: joblib.load(learned_dir / (name.replace(" ", "_") + ".joblib"))
        for name in ["Score-only logistic", "Total-error logistic", "Per-feature logistic"]
    }

    def recent_rules(p, a, r, b):
        bounds = frozen[str(b)]["recent_boundaries"]
        base = p >= cutoff
        pred = mode_rules(p, a, cutoff, bounds, confidence_cuts, band["lower"], band["upper"])
        pred["Binary only"] = base
        pred["Tuned binary (59%)"] = p >= selected["Binary threshold"]["threshold"]
        for kind, model in models.items():
            fusion = (
                model.predict_proba(meta_features(p, a, r, kind))[:, 1]
                >= selected[kind]["threshold"]
            )
            pred[kind + " (continuous reference)"] = fusion
            if kind != "Score-only logistic":
                pred[kind + " (AE-gated)"] = ae_gate(base, fusion, a, bounds[0])
        pred["Prior confidence (reference)"] = (
            p
            >= np.asarray(confidence_cuts)[
                np.searchsorted(
                    confidence["chosen"]["Mode confidence/recall95"]["boundaries"], a, side="right"
                )
            ]
        )
        return pred

    selection = []
    for b in budget_list:
        for name, pred in recent_rules(
            v["selection_p"], v["selection_ae"], v["selection_residuals"], b
        ).items():
            selection.append(dict(ae_target=b, name=name, **metrics(v["selection_y"], pred)))
    dump("selection_results.json", selection)
    print("Thresholds frozen; evaluating all policies on official test", flush=True)
    testpath = ROOT / "data/raw/CSV_Files/Training and Testing Sets/UNSW_NB15_testing-set.csv"
    expected = json.loads((learned_dir / "audit.json").read_text())
    assert digest(testpath) == expected["test_sha256"]
    test = pd.read_csv(testpath)
    y = test.label.to_numpy(int)
    current = np.load(learned_dir / "test_predictions.npz")
    imported_test = np.load(exp / "seed42_official_test_20261007/test_predictions.npz")
    np.testing.assert_array_equal(y, current["y"])
    np.testing.assert_array_equal(y, imported_test["y"])
    # Existing imported test bundle omits plain reconstruction errors, so score frozen AE here.
    features, ae = imported["anomaly_models"]["plain"]
    ia = ae.score(features.transform(test.drop(columns=["id", "label", "attack_cat"]))).astype(
        np.float64
    )
    ip = imported_test["imported_lightgbm"]
    import_cut = imported["thresholds"]["0.05"]["lightgbm"]
    import_fusion_cut = imported["thresholds"]["0.05"]["fusion_ae_plain_latent"]
    import_base = ip >= import_cut
    import_fusion = imported_test["imported_fusion"] >= import_fusion_cut
    np.testing.assert_array_equal(import_base, imported_test["Imported seed42 binary"])
    np.testing.assert_array_equal(import_fusion, imported_test["Imported seed42 plain-AE + latent"])
    base = current["p"] >= cutoff
    rows = []
    families = []
    ae_test = []
    stored = dict(y=y, recent_ae_error=current["ae"], imported_plain_error=ia)
    for b in budget_list:
        preds = recent_rules(current["p"], current["ae"], current["residuals"], b)
        threshold = frozen[str(b)]["imported_plain_threshold"]
        anomaly = ia >= threshold
        preds.update(
            {
                "Imported binary (reference)": import_base,
                "Imported plain AE alone": anomaly,
                "Imported OR": import_base | anomaly,
                "Imported AND": import_base & anomaly,
                "Imported plain+latent (continuous reference)": import_fusion,
                "Imported plain+latent (AE-gated)": ae_gate(
                    import_base, import_fusion, ia, threshold
                ),
            }
        )
        for label, a, t in [
            ("Recent AE", current["ae"], frozen[str(b)]["recent_boundaries"][0]),
            ("Imported plain AE", ia, threshold),
        ]:
            ae_test.append(dict(model=label, target=b, **metrics(y, a >= t)))
        for name, pred in preds.items():
            own_base = import_base if name.startswith("Imported") else base
            row = dict(
                ae_target=b,
                name=name,
                **metrics(y, pred),
                confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist(),
                recovered_attacks=int(((y == 1) & ~own_base & pred).sum()),
                lost_attacks=int(((y == 1) & own_base & ~pred).sum()),
                removed_false_alarms=int(((y == 0) & own_base & ~pred).sum()),
                added_false_alarms=int(((y == 0) & ~own_base & pred).sum()),
            )
            rows.append(row)
            stored[f"{b}/{name}"] = pred
            for family in sorted(test.loc[y == 1, "attack_cat"].unique()):
                mask = (test.attack_cat == family).to_numpy() & (y == 1)
                families.append(
                    dict(
                        ae_target=b,
                        model=name,
                        family=family,
                        rows=int(mask.sum()),
                        detected=int(pred[mask].sum()),
                        recall=float(pred[mask].mean()),
                    )
                )
    dump("test_results.json", rows)
    dump("ae_test_results.json", ae_test)
    pd.DataFrame(
        [{k: val for k, val in r.items() if k != "confusion_matrix"} for r in rows]
    ).to_csv(out / "test_results.csv", index=False)
    pd.DataFrame(families).to_csv(out / "attack_family_recall.csv", index=False)
    np.savez_compressed(out / "test_predictions.npz", **stored)
    audit = dict(
        train_sha256=digest(trainpath),
        test_sha256=digest(testpath),
        same_test_labels_verified=True,
        thresholds_frozen_before_test=True,
        no_retraining=True,
        no_test_selection=True,
        rows=len(y),
        source_hashes={
            str(p.relative_to(ROOT)): digest(p)
            for p in [
                source / "binary.joblib",
                source / "autoencoder.joblib",
                imported_dir / "models.joblib",
                learned_dir / "validation_features.npz",
                learned_dir / "test_predictions.npz",
                exp / "seed42_official_test_20261007/test_predictions.npz",
            ]
        },
    )
    dump("audit.json", audit)
    write_report(out, rows, calibration_rows, ae_test, selection, frozen)
    print(
        pd.DataFrame(
            [
                {k: r[k] for k in ["ae_target", "name", "recall", "fpr", "f1", "fn", "fp"]}
                for r in rows
            ]
        ).to_string(index=False),
        flush=True,
    )


def write_report(out, rows, calibration_rows, ae_test, selection, frozen):
    lines = [
        "# Fusion experiments with AE FPR targets of 5% and 10%",
        "",
        "## Scope and interpretation",
        "",
        "The 5% and 10% settings apply to the **AE alone on normal calibration traffic**. "
        "They are not constraints on the full system’s FPR, and do not guarantee AE test FPR. "
        "Every policy below uses the same official test: 82,332 rows, 37,000 normal and "
        "45,332 attack. This isolates AE operating thresholds; model weights, LightGBM "
        "cutoffs and learned-fusion cutoffs remain frozen. No winner is selected on test.",
        "",
        "The sweep covers the previously compared policy families: AE alone, OR, AND, "
        "four-mode hard overrides, mode-specific confidence, uncertainty-band overrides, "
        "selective recovery, score-only/total-error/per-feature logistic fusion, and the "
        "imported seed-42 plain-AE + latent fusion. Previous individual threshold-search "
        "trials are not all retrained or reselected.",
        "",
        "## How the budgets apply",
        "",
        "- Threshold T(b) is calibrated with a conservative empirical cutoff: no more than "
        "floor(b × normal calibration count) normal rows can meet error ≥ T(b).",
        "- Four AE modes use tail budgets b, b/2 and b/10. At b=5%, these are 5%, 2.5%, "
        "0.5%; at b=10%, they are 10%, 5%, 1%. No anomaly is below T(b).",
        "- Hard four-mode rule: no anomaly → normal; low → binary; mid/high → attack.",
        "- Confidence rule: no anomaly needs 65% LightGBM score; all other modes need "
        "57.7693%. Unlike the prior median boundary, no-anomaly now ends at T(b).",
        "- Uncertainty-band rule applies the hard four-mode rule only within the previously "
        "selected 10%–65% LightGBM score interval, preserving binary outside.",
        "- Selective recovery preserves binary positives and adds attacks when score ≥40% "
        "and AE error ≥T(b). OR/AND use the same AE anomaly decision.",
        "- Continuous learned fusion has no intrinsic AE FPR parameter. Its original output "
        "is retained as a budget-independent reference. A separately labeled AE-gated "
        "variant follows learned fusion only when AE error ≥T(b), otherwise binary.",
        "- Imported seed42 uses its own normal calibration partition and plain-AE error "
        "threshold. Its binary and fusion cutoffs remain at the saved original 5% system "
        "calibration-budget setting for both AE sweep settings. Its latent input still "
        "comes from the denoising AE; that component is not recalibrated.",
        "",
        "## AE calibration achieved",
        "",
        "| AE | Target | Threshold | Normal rows | False alarms | Calibration FPR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    findings = ["## Observed tradeoffs", ""]
    for b in [0.05, 0.10]:
        baseline = next(r for r in rows if r["ae_target"] == b and r["name"] == "Binary only")
        band = next(
            r for r in rows if r["ae_target"] == b and r["name"] == "Uncertainty-band four-mode"
        )
        either = next(r for r in rows if r["ae_target"] == b and r["name"] == "OR")
        findings += [
            f"At the {b:.0%} AE calibration target, uncertainty-band fusion has "
            f"**{band['recall']:.4%} attack recall**, **{band['fpr']:.4%} system FPR**, "
            f"and **{band['f1']:.4%} F1**. Relative to binary only it detects "
            f"{band['tp'] - baseline['tp']:+,} net attacks and changes false alarms by "
            f"{band['fp'] - baseline['fp']:+,}. OR fusion reaches {either['recall']:.4%} "
            f"recall at {either['fpr']:.4%} system FPR.",
            "",
        ]
    findings += [
        "These are observations of all predeclared test results, not a new test-selected "
        "deployment decision. The uncertainty-band configurations improve recall and "
        "false-alarm counts over binary-only on this test, while OR prioritizes higher "
        "recall at a much larger false-alarm cost. Confirmation on new data is still needed.",
        "",
    ]
    lines[2:2] = findings
    for r in calibration_rows:
        lines.append(
            f"| {r['model']} | {r['target']:.0%} | {r['threshold']:.10g} | "
            f"{r['normal_rows']} | {r['false_positives']} | {r['observed_fpr']:.4%} |"
        )
    lines += [
        "",
        "## AE alone on official test",
        "",
        "| AE | Calibration target | Actual test FPR | Attack recall |",
        "|---|---:|---:|---:|",
    ]
    for r in ae_test:
        lines.append(f"| {r['model']} | {r['target']:.0%} | {r['fpr']:.4%} | {r['recall']:.4%} |")
    for b in [0.05, 0.10]:
        lines += [
            "",
            f"## All system results — AE target {b:.0%}",
            "",
            "Attack recall and missed attacks are shown first because missing attacks is "
            "the priority. Actual system FPR remains necessary to assess false-alarm workload.",
            "",
            "| Method | Recall | Missed attacks | System FPR | False alarms | "
            "Precision | F1 | Accuracy |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in rows:
            if r["ae_target"] != b:
                continue
            lines.append(
                f"| {r['name']} | {r['recall']:.4%} | {r['fn']:,} | "
                f"{r['fpr']:.4%} | {r['fp']:,} | {r['precision']:.4%} | "
                f"{r['f1']:.4%} | {r['accuracy']:.4%} |"
            )
    lines += [
        "",
        "## Confusion matrices",
        "",
        "Rows = actual; columns = predicted. "
        "Reference methods repeat unchanged across budgets by design.",
    ]
    for r in rows:
        lines += [
            "",
            f"### {r['name']} — AE target {r['ae_target']:.0%}",
            "",
            "| | Normal | Attack |",
            "|---|---:|---:|",
            f"| Normal | {r['tn']:,} | {r['fp']:,} |",
            f"| Attack | {r['fn']:,} | {r['tp']:,} |",
        ]
    lines += [
        "",
        "## Limits and reproduction",
        "",
        "All AE cutoffs use calibration normals only and were saved before current test "
        "scoring. Selection results are saved separately for recent models; imported models "
        "have a different split protocol and are not scored on the recent selection set, "
        "which may overlap their fitting data. This is an operating-rule sweep, not a "
        "new model-training search. AE-gated learned variants are explicitly new conditions, "
        "not claims that the continuous model itself has a 5%/10% AE FPR.",
        "",
        "The official test has known training-feature overlap and repeated development "
        "use. Imported seed42 withheld Exploits during training; recent models did not. "
        "Calibration FPR is empirical and need not transfer to test/live traffic. No "
        "serving settings changed.",
        "",
        "Exact boundaries: frozen_thresholds.json. All metrics and per-baseline error "
        "changes: test_results.csv/json. Per-family recall: attack_family_recall.csv. "
        "All frozen decisions: test_predictions.npz. Sources: audit.json.",
        "",
        "```sh",
        ".venv/bin/python training/experiment_ae_fpr_sweep.py "
        "--output-dir experiments/ae_fpr_5_10_new",
        "```",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
