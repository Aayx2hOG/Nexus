"""Evaluate imported seed-42 frozen fusion on the official test, with archive parity."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from compare_working_v2_fusion import metrics
from experiment_uncertainty_residuals import residual_features
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]


def imported_scores(saved, frame):
    tree = saved["lightgbm"].predict_proba(saved["tree_features"].transform(frame))[:, 1]
    features, ae = saved["anomaly_models"]["plain"]
    plain = ae.score(features.transform(frame))
    # Archived ae_latent is the denoising AE's latent Mahalanobis distance.
    features, ae = saved["anomaly_models"]["denoising"]
    latent = ae.score(features.transform(frame), latent=True)
    scores = dict(lightgbm=tree, ae_plain=plain, ae_latent=latent)
    fusion = saved["fusion_ablations"]["fusion_ae_plain_latent"]
    matrix = np.column_stack(
        [
            scores[name] if name == "lightgbm" else np.log1p(scores[name])
            for name in fusion["columns"]
        ]
    )
    scores["fusion_ae_plain_latent"] = fusion["model"].predict_proba(matrix)[:, 1]
    assert all(np.isfinite(values).all() for values in scores.values())
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "experiments/seed42_official_test_20261007"
    )
    out = parser.parse_args().output_dir
    out.mkdir(parents=True, exist_ok=False)

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def dump(name, value):
        (out / name).write_text(json.dumps(value, indent=2) + "\n")

    source = ROOT / "experiments/seed_42/seed_42"
    recent = ROOT / "experiments/confidence_ae_20261007"
    models = ROOT / "experiments/four_mode_ae_20261007"
    rawdir = ROOT / "data/raw/CSV_Files/Training and Testing Sets"
    manifest = json.loads((source / "manifest.json").read_text())
    for name, expected in manifest["artifact_hashes"].items():
        assert digest(source / name) == expected, name
    saved = joblib.load(source / "models.joblib")
    calibration = json.loads((source / "calibration.json").read_text())
    cuts = calibration["thresholds"]["0.05"]
    for name in ["lightgbm", "fusion_ae_plain_latent"]:
        assert saved["thresholds"]["0.05"][name] == cuts[name]
    rule = json.loads((recent / "frozen_selection.json").read_text())["chosen"][
        "Mode confidence/recall95"
    ]
    dump(
        "protocol.json",
        dict(
            imported_model="fusion_ae_plain_latent",
            seed=42,
            imported_calibration_budget=0.05,
            imported_threshold=cuts["fusion_ae_plain_latent"],
            imported_binary_threshold=cuts["lightgbm"],
            recent_rule=rule,
            retrained=False,
            retuned=False,
            latent_source="Denoising AE latent Mahalanobis distance",
            source_hashes=manifest["artifact_hashes"],
        ),
    )
    trainpath = rawdir / "UNSW_NB15_training-set.csv"
    assert digest(trainpath) == manifest["data_sha256"]
    train = pd.read_csv(trainpath)
    train_x = train.drop(columns=["id", "label", "attack_cat"])
    archive = np.load(source / "evaluation_scores.npz", allow_pickle=False)
    archive_decisions = np.load(source / "evaluation_decisions.npz", allow_pickle=False)
    print("Verifying full archived holdout score and prediction parity", flush=True)
    original = imported_scores(saved, train_x.iloc[archive["row_indices"]])
    parity = {}
    for name, values in original.items():
        np.testing.assert_allclose(values, archive[name], rtol=2e-5, atol=1e-7)
        parity[name] = dict(max_absolute_error=float(np.max(np.abs(values - archive[name]))))
        if name in ["lightgbm", "fusion_ae_plain_latent"]:
            for budget, thresholds in calibration["thresholds"].items():
                actual = values >= thresholds[name]
                np.testing.assert_array_equal(actual, archive_decisions[f"{name}__{budget}"])
            parity[name]["exact_decision_parity_all_three_budgets"] = True
    dump("archive_parity.json", parity)
    print(
        "Archive reproduced. Scoring both frozen models on the same official test rows", flush=True
    )
    testpath = rawdir / "UNSW_NB15_testing-set.csv"
    recent_audit = json.loads((recent / "audit.json").read_text())
    assert digest(testpath) == recent_audit["test_sha256"]
    test = pd.read_csv(testpath)
    xt = test.drop(columns=["id", "label", "attack_cat"])
    y = test.label.to_numpy(int)
    imported = imported_scores(saved, xt)
    recent_binary = joblib.load(models / "binary.joblib")
    recent_ae = joblib.load(models / "autoencoder.joblib")
    p = recent_binary.predict_proba(xt)[:, 1]
    a, _, _ = residual_features(recent_ae, xt)
    modes = np.searchsorted(rule["boundaries"], a, side="right")
    predictions = {
        "Imported seed42 binary": imported["lightgbm"] >= cuts["lightgbm"],
        "Imported seed42 plain-AE + latent": imported["fusion_ae_plain_latent"]
        >= cuts["fusion_ae_plain_latent"],
        "Recent binary": p >= recent_binary.decision_threshold_,
        "Recent confidence rule": p >= np.asarray(rule["cutoffs"])[modes],
    }
    previous = np.load(recent / "test_scores.npz", allow_pickle=False)
    np.testing.assert_array_equal(y, previous["y"])
    np.testing.assert_array_equal(
        predictions["Recent confidence rule"], previous["Mode confidence/recall95"]
    )
    results = []
    for name, pred in predictions.items():
        results.append(
            dict(
                name=name,
                **metrics(y, pred),
                confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist(),
            )
        )
    old = predictions["Imported seed42 plain-AE + latent"]
    new = predictions["Recent confidence rule"]
    paired = dict(
        imported_extra_detected_attacks=int(((y == 1) & old & ~new).sum()),
        imported_missed_recent_detected_attacks=int(((y == 1) & ~old & new).sum()),
        imported_extra_false_alarms=int(((y == 0) & old & ~new).sum()),
        imported_removed_recent_false_alarms=int(((y == 0) & ~old & new).sum()),
    )
    family_rows = []
    for family in sorted(test.loc[y == 1, "attack_cat"].unique()):
        mask = (test.attack_cat == family).to_numpy() & (y == 1)
        for name, pred in predictions.items():
            family_rows.append(
                dict(
                    family=family,
                    model=name,
                    rows=int(mask.sum()),
                    detected=int(pred[mask].sum()),
                    recall=float(pred[mask].mean()),
                )
            )
    pd.DataFrame(family_rows).to_csv(out / "attack_family_recall.csv", index=False)
    dump("test_results.json", results)
    dump("paired_comparison.json", paired)
    np.savez_compressed(
        out / "test_predictions.npz",
        y=y,
        **predictions,
        imported_lightgbm=imported["lightgbm"],
        imported_fusion=imported["fusion_ae_plain_latent"],
    )
    dump(
        "audit.json",
        dict(
            test_sha256=digest(testpath),
            train_sha256=digest(trainpath),
            test_rows=len(y),
            normal_rows=int((y == 0).sum()),
            attack_rows=int((y == 1).sum()),
            source_hashes_verified=True,
            archive_score_parity=True,
            recent_predictions_reproduced=True,
            no_retraining=True,
            no_threshold_retuning=True,
            different_training_protocols=True,
        ),
    )
    lines = [
        "# Seed 42 versus recent confidence fusion — same official testing set",
        "",
        "**Both models below are now evaluated on the exact same 82,332 test rows: "
        "37,000 normal and 45,332 attacks.** No thresholds or model weights were changed.",
        "",
        "Imported model: `experiments/seed_42/seed_42/models.joblib`, "
        "`fusion_ae_plain_latent`, seed 42, Exploits withheld during training. Its saved "
        f"5%-budget fusion threshold is **{cuts['fusion_ae_plain_latent']:.15g}**; "
        f"its binary baseline uses **{cuts['lightgbm']:.15g}**. The 5% is the original "
        "calibration budget, not a promised FPR on this test.",
        "",
        "The imported fusion combines LightGBM score, log(1+plain-AE reconstruction error), "
        "and log(1+denoising-AE latent Mahalanobis distance). The latent component is from "
        "the denoising AE; it is not a plain-AE-only model.",
        "",
        "Recent rule: 65% LightGBM cutoff below the normal-calibration AE-error median; "
        "57.7693% elsewhere. Its saved predictions were independently reproduced on these "
        "test rows.",
        "",
        "## Test metrics",
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
    lines += ["", "## Confusion matrices", "", "Rows = actual; columns = predicted."]
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
    r = results[1]
    n = results[3]
    lines += [
        "",
        "## Direct paired comparison",
        "",
        f"Imported fusion versus recent confidence rule: "
        f"F1 difference **{100 * (r['f1'] - n['f1']):+.4f} percentage points**, "
        f"recall difference **{100 * (r['recall'] - n['recall']):+.4f} points**, "
        f"and FPR difference **{100 * (r['fpr'] - n['fpr']):+.4f} points**.",
        "",
        f"The imported fusion detects {paired['imported_extra_detected_attacks']:,} "
        f"attacks missed by the recent rule but misses "
        f"{paired['imported_missed_recent_detected_attacks']:,} attacks the recent rule "
        f"detects. It introduces {paired['imported_extra_false_alarms']:,} false alarms "
        f"and removes {paired['imported_removed_recent_false_alarms']:,} recent false alarms.",
        "",
        "## Verification and limits",
        "",
        "All six imported artifact hashes verified. Before test scoring, the full archived "
        "68,643-row holdout was rescored: saved component scores matched within numerical "
        "tolerance, and binary/fusion predictions matched exactly at all three archived "
        "budgets. See archive_parity.json for maximum score differences.",
        "",
        "This fixes the previous different-evaluation-set comparison. Training still differs: "
        "the imported run withheld Exploits, whereas the recent run included all attack "
        "families. Therefore this compares the actual frozen systems, not an isolated "
        "architecture effect. The official test has known training-feature overlap and "
        "has already been used in project development; it is not a fresh holdout.",
        "",
        "Per-attack-family recall is in attack_family_recall.csv. Frozen predictions and "
        "scores are in test_predictions.npz. No serving configuration was changed.",
        "",
        "```sh",
        ".venv/bin/python training/evaluate_imported_seed42_test.py "
        "--output-dir experiments/seed42_official_test_new",
        "```",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(pd.DataFrame(results).to_string(index=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
