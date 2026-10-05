"""Aggregate existing comparison CSVs, paired seed summaries, and six plots.

No training or threshold selection. Refuses duplicate scenario/model/budget/seed
rows and existing output directories. Seed summaries describe repeated splits;
overlapping evaluation populations do not establish independent replications.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["held_family", "model", "budget", "seed"]
FAMILIES = ["none", "Reconnaissance", "Exploits", "DoS"]
MEASURES = [
    "precision",
    "recall",
    "f1",
    "fpr",
    "false_alerts_per_1000_benign",
    "roc_auc",
    "pr_auc",
    "recovered_lightgbm_misses",
    "lost_lightgbm_detections",
    "additional_false_positives",
    "removed_false_positives",
    "net_additional_false_positives",
    "net_recovered_attacks",
    "recovered_per_additional_fp",
    "withheld_family_recall",
    "evaluation_budget_met",
    "recall_delta",
    "fpr_delta",
]
COLORS = {
    "lightgbm": "#1f77b4",
    "learned_fusion": "#009E73",
    "selective_fusion": "#CC79A7",
    "naive_or": "#D55E00",
    "rank_or": "#E69F00",
    "ae_plain": "#999999",
    "ae_scaled": "#8c564b",
    "ae_denoising": "#9467bd",
    "ae_latent": "#17becf",
    "isolation_forest": "#bcbd22",
}


def enrich(table):
    table = table.copy()
    if table.duplicated(KEYS).any():
        raise ValueError("Duplicate scenario/model/budget/seed rows: do not mix old and new suites")
    context = ["partition_sha256", "data_sha256", "calibration_policy", "evaluation_policy"]
    if "conditions_sha256" in table:
        context.append("conditions_sha256")
    if any(column in table for column in context):
        if any(column not in table or table[column].isna().any() for column in context):
            raise ValueError("Cannot mix provenance-bearing results with unaudited historical rows")
        paired = table.groupby(["held_family", "budget", "seed"])
        if (paired[context].nunique() != 1).any().any():
            raise ValueError("Paired methods have different partitions or calibration conditions")
        for _, rows in table.groupby(["held_family", "budget"]):
            if "conditions_sha256" in rows and rows.conditions_sha256.nunique() != 1:
                raise ValueError("Cannot aggregate seeds with different method conditions")
            seeds = rows.groupby("model").seed.apply(lambda values: frozenset(values))
            if len(set(seeds)) != 1:
                raise ValueError("Compared methods must have identical seed sets")
    added = table.additional_false_positives
    recovered = table.recovered_lightgbm_misses
    table["net_recovered_attacks"] = recovered - table.lost_lightgbm_detections
    table["recovered_per_additional_fp"] = recovered / added.where(added > 0)
    table["recovery_ratio_status"] = np.where(
        added > 0,
        "finite",
        np.where(recovered > 0, "recovery_without_additional_fp", "no_recovery_no_added_fp"),
    )
    if table.evaluation_budget_met.dtype == object:
        mapped = table.evaluation_budget_met.map({"True": True, "False": False})
        if mapped.isna().any():
            raise ValueError("Invalid evaluation_budget_met column")
        table["evaluation_budget_met"] = mapped
    join = ["held_family", "budget", "seed"]
    baseline = table.loc[table.model.eq("lightgbm"), join + ["recall", "fpr"]]
    table = table.drop(
        columns=["recall_delta", "fpr_delta", "baseline_recall", "baseline_fpr"], errors="ignore"
    ).merge(
        baseline.rename(columns={"recall": "baseline_recall", "fpr": "baseline_fpr"}),
        on=join,
        how="left",
        validate="many_to_one",
    )
    if table.baseline_recall.isna().any():
        raise ValueError("Every result requires its paired LightGBM baseline")
    table["recall_delta"] = table.recall - table.baseline_recall
    table["fpr_delta"] = table.fpr - table.baseline_fpr
    return table


def statistical_summary(table):
    table = enrich(table)
    group = table.groupby(["held_family", "model", "budget"], sort=False)
    columns = [name for name in MEASURES if name in table]
    summary = group[columns].agg(["count", "mean", "std", "min", "max"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    summary["seed_count"] = group.seed.nunique()
    summary["zero_additional_fp_seed_count"] = group.additional_false_positives.apply(
        lambda values: int((values == 0).sum())
    )
    summary["pooled_recovered_per_additional_fp"] = (
        group.recovered_lightgbm_misses.sum()
        / group.additional_false_positives.sum().replace(0, np.nan)
    )
    return summary.reset_index()


def generate_plots(table, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(exist_ok=False)
    families = [name for name in FAMILIES if name in set(table.held_family)]
    methods = [name for name in COLORS if name in set(table.model)]

    def finish(fig, name):
        fig.tight_layout()
        fig.savefig(output / f"{name}.png", dpi=160, bbox_inches="tight")
        fig.savefig(output / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    for number, x, y, title in (
        (1, "fpr", "recall", "Recall vs observed FPR"),
        (
            2,
            "net_additional_false_positives",
            "recovered_lightgbm_misses",
            "Recovered misses vs net additional false positives",
        ),
        (3, "budget", "recall", "Recall vs calibration FPR budget"),
    ):
        fig, axes = plt.subplots(1, len(families), figsize=(5 * len(families), 4.8), squeeze=False)
        for ax, family in zip(axes[0], families, strict=True):
            data = table[table.held_family.eq(family)]
            for method in methods:
                if number == 2 and method not in {
                    "lightgbm",
                    "learned_fusion",
                    "selective_fusion",
                    "naive_or",
                    "rank_or",
                }:
                    continue
                rows = data[data.model.eq(method)]
                if rows.empty:
                    continue
                grouped = rows.groupby("budget")
                xm, ym = grouped[x].mean(), grouped[y].mean()
                xerr = None if x == "budget" else grouped[x].std().fillna(0)
                ax.errorbar(
                    xm,
                    ym,
                    xerr=xerr,
                    yerr=grouped[y].std().fillna(0),
                    marker="o",
                    capsize=2,
                    label=method,
                    color=COLORS[method],
                )
                if number == 2 and method in {"learned_fusion", "selective_fusion", "naive_or"}:
                    for budget, xv, yv in zip(xm.index, xm, ym, strict=True):
                        ax.annotate(f"{budget:.0%}", (xv, yv), fontsize=7)
            ax.set(title=family, xlabel=x.replace("_", " "), ylabel=y.replace("_", " "))
            ax.grid(alpha=0.2)
            if number == 2:
                ax.axvline(0, color="black", linewidth=0.6)
        axes[0, -1].legend(fontsize=7)
        fig.suptitle(title + " (mean ± seed SD)", y=1.03)
        finish(fig, f"{number:02d}_{x}_vs_{y}")

    selected = [
        m
        for m in ("lightgbm", "learned_fusion", "selective_fusion", "naive_or", "rank_or")
        if m in methods
    ]
    budgets = sorted(table.budget.unique())
    fig, axes = plt.subplots(1, len(budgets), figsize=(5 * len(budgets), 4.8), squeeze=False)
    held = [family for family in families if family != "none"]
    for ax, budget in zip(axes[0], budgets, strict=True):
        for i, method in enumerate(selected):
            data = table[table.budget.eq(budget) & table.model.eq(method)]
            grouped = data.groupby("held_family").withheld_family_recall
            means = grouped.mean().reindex(held)
            errors = grouped.std().reindex(held).fillna(0)
            position = np.arange(len(held)) + (i - (len(selected) - 1) / 2) * 0.15
            if held:
                ax.bar(
                    position,
                    means,
                    yerr=errors,
                    width=0.15,
                    label=method,
                    color=COLORS[method],
                    capsize=2,
                )
        ax.set(
            xticks=np.arange(len(held)),
            xticklabels=held,
            title=f"Budget {budget:.0%}",
            ylabel="Withheld-family recall",
            ylim=(0, 1.05),
        )
        ax.tick_params(axis="x", rotation=15)
    if held:
        axes[0, -1].legend(fontsize=7)
    fig.suptitle("Held-out-family detection (mean ± seed SD)", y=1.02)
    finish(fig, "04_held_family_performance")

    fig, axes = plt.subplots(1, len(families), figsize=(5 * len(families), 4.8), squeeze=False)
    variance_budget = 0.03 if 0.03 in budgets else budgets[0]
    for ax, family in zip(axes[0], families, strict=True):
        data = table[table.held_family.eq(family) & table.budget.eq(variance_budget)]
        seeds = sorted(data.seed.unique())
        for method in selected:
            rows = data[data.model.eq(method)].set_index("seed").reindex(seeds)
            ax.plot(np.arange(len(seeds)), rows.recall, "o-", label=method, color=COLORS[method])
        ax.set(
            title=family,
            xticks=np.arange(len(seeds)),
            xticklabels=seeds,
            xlabel="Seed",
            ylabel="Recall",
        )
    axes[0, -1].legend(fontsize=7)
    fig.suptitle(f"Seed variation at {variance_budget:.0%} calibration budget", y=1.02)
    finish(fig, "05_seed_variance")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, metric in zip(axes, ("recall", "fpr"), strict=True):
        for method in ("learned_fusion", "selective_fusion"):
            rows = table[table.model.eq(method)]
            ax.scatter(
                rows[f"baseline_{metric}"],
                rows[metric],
                label=method,
                color=COLORS[method],
                alpha=0.7,
            )
        bounds = ax.get_xlim(), ax.get_ylim()
        low, high = min(b[0] for b in bounds), max(b[1] for b in bounds)
        ax.plot([low, high], [low, high], "k--", linewidth=0.8)
        ax.set(
            xlabel=f"Paired LightGBM {metric}",
            ylabel=f"Fusion {metric}",
            title="Above diagonal is better" if metric == "recall" else "Below diagonal is better",
        )
        ax.legend()
    fig.suptitle("LightGBM vs fusion: each point is one family/budget/seed")
    finish(fig, "06_lightgbm_vs_fusion")


def reconnaissance_analysis(inputs, train_csv, output):
    """Describe actual metadata and feature slices, without inferring subtypes."""
    raw = pd.read_csv(train_csv)
    dataset_hash = hashlib.sha256(train_csv.read_bytes()).hexdigest()
    recon = raw[raw.attack_cat.astype(str).str.strip().eq("Reconnaissance")]
    subtype = next(
        (
            column
            for column in raw
            if column.lower().replace(" ", "_") in {"attack_subcategory", "attack_subtype"}
        ),
        None,
    )
    report = {
        "benchmark_reconnaissance_rows": len(recon),
        "benchmark_columns": raw.columns.tolist(),
        "subtype_performance_supported": subtype is not None,
        "reason": "Per-row subtype labels available"
        if subtype
        else (
            "Benchmark rows contain attack_cat only; no subtype or reliable IP/port/time join "
            "to auxiliary ground truth. Protocol/service/state are feature slices, not subtypes."
        ),
    }
    events = train_csv.parent.parent / "UNSW-NB15_LIST_EVENTS.csv"
    if events.is_file():
        catalog = pd.read_csv(events, encoding="cp1252")
        catalog.columns = catalog.columns.str.strip()
        entries = catalog[
            catalog["Attack category"].astype(str).str.strip().str.casefold().eq("reconnaissance")
        ]
        report["auxiliary_event_subcategories_not_joined_to_benchmark"] = entries.fillna(
            ""
        ).to_dict(orient="records")
    (output / "reconnaissance_metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    slices = []
    for root in inputs:
        for archive in sorted((root / "Reconnaissance").glob("seed_*/evaluation_scores.npz")):
            directory = archive.parent
            manifest = json.loads((directory / "manifest.json").read_text())
            if manifest["data_sha256"] != dataset_hash:
                raise ValueError("Reconnaissance dataset hash mismatch")
            with np.load(archive, allow_pickle=False) as saved:
                indices = saved["row_indices"]
                keep = saved["families"] == "Reconnaissance"
                frame = raw.iloc[indices].reset_index(drop=True)
                decisions_path = directory / "evaluation_decisions.npz"
                if decisions_path.is_file():
                    with np.load(decisions_path) as predictions:
                        if not np.array_equal(predictions["row_indices"], indices):
                            raise ValueError("Decision/score row identities differ")
                        decisions = {
                            key: predictions[key]
                            for key in predictions.files
                            if key != "row_indices"
                        }
                else:
                    # Legacy v1 has no decisions archive; recover its existing cutoffs.
                    results = json.loads((directory / "metrics.json").read_text())
                    thresholds = {(r["model"], r["budget"]): r.get("threshold") for r in results}
                    decisions = {}
                    for result in results:
                        method, budget = result["model"], result["budget"]
                        if method == "naive_or":
                            prediction = (saved["lightgbm"] >= thresholds["lightgbm", budget]) | (
                                saved["ae_denoising"] >= thresholds["ae_denoising", budget]
                            )
                        else:
                            prediction = saved[method] >= result["threshold"]
                        decisions[f"{method}__{budget}"] = prediction
            for key, predictions in decisions.items():
                method, budget = key.split("__")
                baseline = decisions[f"lightgbm__{budget}"]
                for column in ([subtype] if subtype else []) + ["proto", "service", "state"]:
                    for value, rows in frame.loc[keep].groupby(column, dropna=False).groups.items():
                        selected = np.asarray(list(rows), dtype=int)
                        slices.append(
                            {
                                "seed": manifest["seed"],
                                "model": method,
                                "budget": float(budget),
                                "slice_kind": "subtype" if column == subtype else "feature_slice",
                                "feature": column,
                                "value": str(value),
                                "rows": len(selected),
                                "recall": float(predictions[selected].mean()),
                                "recovered": int(
                                    (predictions[selected] & ~baseline[selected]).sum()
                                ),
                                "lost": int((~predictions[selected] & baseline[selected]).sum()),
                            }
                        )
    if slices:
        pd.DataFrame(slices).to_csv(output / "reconnaissance_slices.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--reconnaissance", action="store_true")
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("data/raw/CSV_Files/Training and Testing Sets/UNSW_NB15_training-set.csv"),
    )
    args = parser.parse_args()
    frames = []
    sources = []
    for directory in args.input_dirs:
        path = directory / "comparison.csv"
        if not (directory / "completed.json").is_file():
            print(
                f"WARNING: {directory} has no suite completion marker; aggregating available rows",
                flush=True,
            )
        data = pd.read_csv(path, keep_default_na=False, na_values=[""])
        frames.append(data)
        sources.append(
            {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    table = enrich(pd.concat(frames, ignore_index=True))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    table.to_csv(args.output_dir / "comparison.csv", index=False)
    statistical_summary(table).to_csv(args.output_dir / "summary.csv", index=False)
    table[table.model.isin(["learned_fusion", "selective_fusion"])].to_csv(
        args.output_dir / "paired_fusion_vs_lightgbm.csv", index=False
    )
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "sources": sources,
                "rows": len(table),
                "interpretation": "Descriptive seed variation; no statistical significance claim. "
                "NaN SD means fewer than two seeds; undefined ratios remain empty. "
                "Seed repeats share data; pooled ratios describe repeats, not unique attacks.",
            },
            indent=2,
        )
        + "\n"
    )
    if args.reconnaissance:
        reconnaissance_analysis(args.input_dirs, args.train_csv, args.output_dir)
    if args.plots:
        generate_plots(table, args.output_dir / "plots")
    print(
        f"Wrote {len(table)} result rows, summaries and requested analyses to {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
