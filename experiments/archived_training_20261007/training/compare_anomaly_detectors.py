"""Compare frozen Autoencoder and Isolation Forest test metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from training_utils import PROJECT_ROOT

DEFAULT_AUTOENCODER_METRICS = PROJECT_ROOT / "models" / "autoencoder" / "autoencoder_metrics.json"
DEFAULT_IF_METRICS = (
    PROJECT_ROOT / "models" / "isolation_forest" / "isolation_forest_metrics.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "anomaly_detector_comparison.csv"
METRICS = [
    "precision",
    "recall",
    "f1",
    "false_positive_rate",
    "specificity",
    "balanced_accuracy",
    "roc_auc",
    "pr_auc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--autoencoder-metrics", type=Path, default=DEFAULT_AUTOENCODER_METRICS)
    parser.add_argument("--isolation-forest-metrics", type=Path, default=DEFAULT_IF_METRICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    autoencoder = json.loads(args.autoencoder_metrics.read_text(encoding="utf-8"))
    isolation_forest = json.loads(
        args.isolation_forest_metrics.read_text(encoding="utf-8")
    )
    rows = []
    for metric in METRICS:
        ae_value = float(autoencoder[metric])
        if_value = float(isolation_forest[metric])
        rows.append(
            {
                "metric": metric,
                "autoencoder": ae_value,
                "isolation_forest": if_value,
                "autoencoder_minus_isolation_forest": ae_value - if_value,
            }
        )
    comparison = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output, index=False)
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Saved comparison to {args.output.resolve()}")


if __name__ == "__main__":
    main()
