"""Audit the imported seed-42 run and compare recorded results, not different holdouts."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
OLD = ROOT / "experiments/seed_42/seed_42"


def metrics(matrix):
    tn, fp, fn, tp = np.asarray(matrix).ravel().tolist()
    return dict(
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        precision=tp / (tp + fp),
        recall=tp / (tp + fn),
        f1=2 * tp / (2 * tp + fp + fn),
        fpr=fp / (tn + fp),
        accuracy=(tp + tn) / (tp + tn + fp + fn),
    )


def main():
    manifest = json.loads((OLD / "manifest.json").read_text())
    hashes = {
        name: hashlib.sha256((OLD / name).read_bytes()).hexdigest() == expected
        for name, expected in manifest["artifact_hashes"].items()
    }
    assert all(hashes.values())
    old = json.loads((OLD / "metrics.json").read_text())
    scores = np.load(OLD / "evaluation_scores.npz", allow_pickle=False)
    decisions = np.load(OLD / "evaluation_decisions.npz", allow_pickle=False)
    split = np.load(OLD / "split_indices.npz", allow_pickle=False)
    np.testing.assert_array_equal(scores["row_indices"], decisions["row_indices"])
    np.testing.assert_array_equal(scores["row_indices"], split["evaluation"])
    for row in old:
        pred = decisions[f"{row['model']}__{row['budget']}"]
        matrix = confusion_matrix(scores["labels"], pred, labels=[0, 1]).tolist()
        assert matrix == row["confusion_matrix"]
        actual = metrics(matrix)
        for key in ["precision", "recall", "f1", "fpr"]:
            assert np.isclose(actual[key], row[key])
    old_best = max((r for r in old if r["evaluation_budget_met"]), key=lambda r: r["f1"])
    old_raw = max(old, key=lambda r: r["f1"])
    old_base = next(
        r for r in old if r["model"] == "lightgbm" and r["budget"] == old_best["budget"]
    )
    recent = []
    for folder in [
        "four_mode_ae_20261007",
        "confidence_ae_20261007",
        "uncertainty_residuals_20261007",
    ]:
        for r in json.loads((ROOT / "experiments" / folder / "test_results.json").read_text()):
            recent.append(dict(experiment=folder, **r))
    for folder in ["working_v2_baseline_fusion_20261005", "stacked_baseline_ae_20261005"]:
        for r in pd.read_csv(ROOT / "experiments" / folder / "official_test.csv").to_dict(
            "records"
        ):
            recent.append(dict(experiment=folder, **r))
    best = max(recent, key=lambda r: r["f1"])
    base = next(
        r
        for r in recent
        if r["experiment"] == "confidence_ae_20261007" and r["name"] == "Current binary"
    )
    new_saved = np.load(ROOT / "experiments/confidence_ae_20261007/test_scores.npz")
    assert best["name"] == "Mode confidence/recall95"
    new_matrix = confusion_matrix(new_saved["y"], new_saved[best["name"]], labels=[0, 1]).tolist()
    assert metrics(new_matrix) == {k: best[k] for k in metrics(new_matrix)}
    old_m, old_b = metrics(old_best["confusion_matrix"]), metrics(old_base["confusion_matrix"])
    n_old = len(scores["labels"])
    n_new = len(new_saved["y"])
    rows = [
        dict(dataset="Internal Exploits-withheld holdout", model=old_base["model"], **old_b),
        dict(dataset="Internal Exploits-withheld holdout", model=old_best["model"], **old_m),
        dict(dataset="Official test", model=base["name"], **{k: base[k] for k in old_m}),
        dict(dataset="Official test", model=best["name"], **{k: best[k] for k in old_m}),
    ]
    pd.DataFrame(rows).to_csv(OUT / "comparison.csv", index=False)
    pd.DataFrame(
        [
            {
                k: r[k]
                for k in ["experiment", "name", "accuracy", "precision", "recall", "f1", "fpr"]
            }
            for r in recent
        ]
    ).sort_values("f1", ascending=False).to_csv(OUT / "recent_results_inventory.csv", index=False)
    audit = dict(
        source_hash_checks=hashes,
        verified_archived_confusion_matrices=len(old),
        verified_recent_best_matrix=True,
        seed=manifest["seed"],
        held_family=manifest["held_family"],
        old_best_model=old_best["model"],
        old_best_budget=old_best["budget"],
        old_best_selection="Retrospective highest recorded holdout F1 among rows meeting their FPR budget",
        recent_best=dict(experiment=best["experiment"], name=best["name"]),
        recent_ranking_scope="Five recent experiments on the same official test, descriptive F1 ranking",
        same_evaluation_set=False,
        models_retrained=False,
    )
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    lines = [
        "# Imported seed 42 versus the best recent experiment",
        "",
        "## Finding",
        "",
        f"The imported run’s best recorded budget-compliant result is **{old_best['model']}** "
        f"at a {old_best['budget']:.0%} FPR budget: **{old_m['f1']:.4%} F1**. "
        f"The best recorded official-test F1 across the five recent experiment folders is "
        f"**{best['name']}**, **{best['f1']:.4%} F1**. These evaluate different rows under "
        "different training and threshold protocols, so the larger archived F1 does not "
        "establish a better model.",
        "",
        "“Best” here is a retrospective descriptive ranking, not a newly validated model "
        "selection. The archived winner was identified by its recorded holdout performance; "
        "the recent rule was originally chosen on selection data before test scoring.",
        "",
        "## What seed 42 contains",
        "",
        "Source: `experiments/seed_42/seed_42`. Its manifest identifies **Exploits withheld**: "
        "that family is absent from fitting, early stopping, fusion training and calibration. "
        "The official testing CSV was not used in that run. Both old and recent models use "
        "seed 42, so this is a comparison of protocols and models, not a seed-effect study.",
        "",
        f"All six manifest artifact hashes verified. All {len(old)} saved confusion matrices "
        "and associated precision/recall/F1/FPR values reproduce from the stored decisions. "
        "The recent best matrix also reproduces.",
        "",
        "| Property | Imported seed 42 | Recent best |",
        "|---|---|---|",
        f"| Evaluation rows | {n_old:,} internal holdout | {n_new:,} official test |",
        f"| Normal / attack | {old_m['tn'] + old_m['fp']:,} / {old_m['fn'] + old_m['tp']:,} | "
        f"{best['tn'] + best['fp']:,} / {best['fn'] + best['tp']:,} |",
        f"| Attack prevalence | {scores['labels'].mean():.2%} | {new_saved['y'].mean():.2%} |",
        "| Training | Exploits withheld; separate fusion partition | All attack families available |",
        "| Threshold objective | Benign calibration FPR budget | Selection recall ≥95%, minimize FPR |",
        "| Fusion | Plain AE reconstruction + latent representation | Mode-specific LightGBM cutoffs |",
        "",
        "The difference in attack prevalence particularly affects precision and F1. "
        "FPR and recall also depend on which normal and attack traffic the holdout contains.",
        "",
        "## Recorded results — not a matched head-to-head comparison",
        "",
        "| Dataset | Model | Accuracy | Precision | Recall | F1 | FPR |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r['model']} | "
            + " | ".join(f"{r[k]:.4%}" for k in ["accuracy", "precision", "recall", "f1", "fpr"])
            + " |"
        )
    lines += [
        "",
        "## Improvement over each experiment’s own binary baseline",
        "",
        "Changes below are percentage points; they help describe the fusion contribution "
        "within each experiment but do not remove differences between evaluation sets.",
        "",
        "| Change | Imported plain-AE + latent fusion | Recent confidence rule |",
        "|---|---:|---:|",
    ]
    for label, key in [
        ("F1", "f1"),
        ("Attack recall", "recall"),
        ("False-positive rate", "fpr"),
        ("Accuracy", "accuracy"),
    ]:
        lines.append(
            f"| {label} | {100 * (old_m[key] - old_b[key]):+.4f} pp | {100 * (best[key] - base[key]):+.4f} pp |"
        )
    lines += [
        "",
        f"The archived fusion recovered {old_best['recovered_lightgbm_misses']} attacks "
        f"and lost {old_best['lost_lightgbm_detections']}: net **{old_m['tp'] - old_b['tp']:+,}** "
        f"detected attacks, with **{old_m['fp'] - old_b['fp']:+,}** false positives. "
        f"Exploits recall rose from {old_base['withheld_family_recall']:.4%} to "
        f"{old_best['withheld_family_recall']:.4%}.",
        "",
        f"The recent rule recovered {best['recovered_attacks']} and lost {best['lost_attacks']} "
        f"attacks: net **{best['tp'] - base['tp']:+,}** detected attacks, with "
        f"**{best['fp'] - base['fp']:+,}** false positives. The archived fusion is promising "
        "for recovering attacks; the recent rule primarily suppresses false alarms.",
        "",
        "## Confusion matrices",
        "",
        "Rows = actual; columns = predicted.",
    ]
    for label, r in [
        ("Imported plain-AE + latent fusion", old_m),
        ("Recent confidence rule", best),
    ]:
        lines += [
            "",
            f"### {label}",
            "",
            "| | Normal | Attack |",
            "|---|---:|---:|",
            f"| Normal | {r['tn']:,} | {r['fp']:,} |",
            f"| Attack | {r['fn']:,} | {r['tp']:,} |",
        ]
    lines += [
        "",
        "## Highest raw archived F1",
        "",
        f"`{old_raw['model']}` at the {old_raw['budget']:.0%} budget recorded "
        f"{old_raw['f1']:.4%} F1, but its observed FPR was {old_raw['fpr']:.4%}, "
        "exceeding its budget. It should not be described as the best budget-compliant run.",
        "",
        "## Conclusion and next comparison",
        "",
        "The archived plain-AE + latent method deserves a matched evaluation: it improved "
        "both recall and FPR relative to its own baseline. We cannot conclude that it beats "
        "the recent method from these saved metrics alone. Freeze the archived rule and "
        "score it on the same official test as a historical check; for a stronger comparison, "
        "train both approaches using identical grouped partitions, data access and selection "
        "objectives, then evaluate on a fresh holdout. In particular, do not evaluate the "
        "recent model on this archived internal holdout without checking training overlap.",
        "",
        "No retraining, retuning or promotion was performed. Source artifacts remain unchanged. "
        "The recent official test has known training-feature overlap and repeated development "
        "use, documented in its original reports.",
        "",
        "Reproduce: `.venv/bin/python experiments/seed42_comparison_20261007/compare.py`.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
