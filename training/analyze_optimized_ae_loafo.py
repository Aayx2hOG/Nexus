"""Report which LOAFO false negatives the optimized autoencoder recovers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training_utils import PROJECT_ROOT

DEFAULT_RESULTS_DIR = (
    PROJECT_ROOT / "experiments" / "leave_one_attack_family_out" / "optimized_autoencoder"
)
DEFAULT_FAMILIES = (
    "Fuzzers",
    "Exploits",
    "DoS",
    "Reconnaissance",
    "Analysis",
    "Backdoor",
    "Shellcode",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--families", nargs="+", default=list(DEFAULT_FAMILIES))
    return parser.parse_args()


def family_slug(family: str) -> str:
    return family.lower().replace(" ", "_")


def miss_recovery_row(predictions: pd.DataFrame, family: str) -> dict[str, Any]:
    """Count held-family attack misses recovered by the optimized AE."""
    required = {
        "label",
        "optimized_ae_prediction",
        "lightgbm_prediction",
        "random_forest_prediction",
        "isolation_forest_prediction",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction data lacks columns: {sorted(missing)}")
    attacks = predictions["label"].to_numpy(dtype=np.int64) == 1
    ae = predictions.loc[attacks, "optimized_ae_prediction"].to_numpy(dtype=bool)
    model_columns = {
        "lightgbm": "lightgbm_prediction",
        "random_forest": "random_forest_prediction",
        "isolation_forest": "isolation_forest_prediction",
    }
    row: dict[str, Any] = {
        "held_out_family": family,
        "attack_samples": int(attacks.sum()),
        "optimized_ae_detected": int(ae.sum()),
    }
    all_model_misses = np.ones(ae.size, dtype=bool)
    for name, column in model_columns.items():
        model_prediction = predictions.loc[attacks, column].to_numpy(dtype=bool)
        misses = ~model_prediction
        caught = misses & ae
        all_model_misses &= misses
        miss_count = int(misses.sum())
        row[f"{name}_misses"] = miss_count
        row[f"{name}_misses_caught_by_ae"] = int(caught.sum())
        row[f"{name}_miss_recovery_rate"] = (
            float(caught.sum() / miss_count) if miss_count else None
        )
        row[f"{name}_misses_remaining_after_ae"] = int((misses & ~ae).sum())
    row["all_three_models_miss"] = int(all_model_misses.sum())
    row["all_three_models_miss_ae_catches"] = int((all_model_misses & ae).sum())
    row["all_four_miss"] = int((all_model_misses & ~ae).sum())
    return row


def total_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total: dict[str, Any] = {
        "held_out_family": "TOTAL",
        "attack_samples": sum(row["attack_samples"] for row in rows),
        "optimized_ae_detected": sum(row["optimized_ae_detected"] for row in rows),
    }
    for model in ("lightgbm", "random_forest", "isolation_forest"):
        misses = sum(row[f"{model}_misses"] for row in rows)
        caught = sum(row[f"{model}_misses_caught_by_ae"] for row in rows)
        total[f"{model}_misses"] = misses
        total[f"{model}_misses_caught_by_ae"] = caught
        total[f"{model}_miss_recovery_rate"] = caught / misses if misses else None
        total[f"{model}_misses_remaining_after_ae"] = misses - caught
    for field in (
        "all_three_models_miss",
        "all_three_models_miss_ae_catches",
        "all_four_miss",
    ):
        total[field] = sum(row[field] for row in rows)
    return total


def write_report(rows: list[dict[str, Any]], total: dict[str, Any], path: Path) -> None:
    lines = [
        "# Optimized AE recovery of LOAFO model errors",
        "",
        "Here, an error means a false negative: a held-out-family attack missed by the "
        "named model. A recovery means the optimized autoencoder detected that attack.",
        "",
        "| Family | LGB misses | AE catches | RF misses | AE catches | IF misses | "
        "AE catches | All 3 miss | AE catches | All 4 miss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in [*rows, total]:
        lines.append(
            f"| {row['held_out_family']} | {row['lightgbm_misses']} | "
            f"{row['lightgbm_misses_caught_by_ae']} | {row['random_forest_misses']} | "
            f"{row['random_forest_misses_caught_by_ae']} | "
            f"{row['isolation_forest_misses']} | "
            f"{row['isolation_forest_misses_caught_by_ae']} | "
            f"{row['all_three_models_miss']} | "
            f"{row['all_three_models_miss_ae_catches']} | {row['all_four_miss']} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate recovery",
            "",
            f"- LightGBM: **{total['lightgbm_misses_caught_by_ae']} of "
            f"{total['lightgbm_misses']}** misses recovered "
            f"(**{total['lightgbm_miss_recovery_rate']:.2%}**).",
            f"- Random Forest: **{total['random_forest_misses_caught_by_ae']} of "
            f"{total['random_forest_misses']}** misses recovered "
            f"(**{total['random_forest_miss_recovery_rate']:.2%}**).",
            f"- Isolation Forest: **{total['isolation_forest_misses_caught_by_ae']} of "
            f"{total['isolation_forest_misses']}** misses recovered "
            f"(**{total['isolation_forest_miss_recovery_rate']:.2%}**).",
            f"- All three comparison models missed **{total['all_three_models_miss']}** "
            f"attacks. The optimized AE uniquely recovered "
            f"**{total['all_three_models_miss_ae_catches']}**; all four models missed "
            f"**{total['all_four_miss']}**.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = []
    for family in args.families:
        path = args.results_dir / family_slug(family) / "predictions.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed LOAFO predictions: {path}")
        rows.append(miss_recovery_row(pd.read_csv(path), family))
    total = total_row(rows)
    all_rows = [*rows, total]
    pd.DataFrame(all_rows).to_csv(args.results_dir / "ae_miss_recovery.csv", index=False)
    (args.results_dir / "ae_miss_recovery.json").write_text(
        json.dumps(
            {"definition": "attack false negatives caught by optimized AE", "rows": all_rows},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_report(rows, total, args.results_dir / "ae_miss_recovery.md")
    print(json.dumps(total, indent=2, sort_keys=True))
    print(f"Saved reports to {args.results_dir.resolve()}")


if __name__ == "__main__":
    main()
