"""Evaluate both trained multiclass models on the official test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib

from multiclass_training_utils import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    PROJECT_ROOT,
    evaluate_multiclass_classifier,
    load_multiclass_tree_data,
)

MODEL_FILES = {
    "multiclass_random_forest": "multiclass_random_forest.joblib",
    "multiclass_lightgbm": "multiclass_lightgbm.joblib",
}
SUMMARY_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "f1_weighted",
    "roc_auc_ovr_macro",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "multiclass_evaluation_metrics.json",
    )
    return parser.parse_args()


def print_summary(results: dict[str, dict[str, Any]]) -> None:
    model_width = max(len("model"), *(len(name) for name in results))
    metric_width = 22
    header = f"{'model':<{model_width}}" + "".join(
        f" {metric:>{metric_width}}" for metric in SUMMARY_METRICS
    )
    print(header)
    print("-" * len(header))
    for model_name, metrics in results.items():
        print(
            f"{model_name:<{model_width}}"
            + "".join(
                f" {metrics[metric]:>{metric_width}.6f}" for metric in SUMMARY_METRICS
            )
        )


def main() -> None:
    args = parse_args()
    _, X_test, _, y_test, class_names = load_multiclass_tree_data(args.data_dir)
    results = {}
    for model_name, filename in MODEL_FILES.items():
        model_path = args.model_dir / filename
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}. Train it first.")
        model = joblib.load(model_path)
        results[model_name] = evaluate_multiclass_classifier(
            model, X_test, y_test, class_names
        )

    print_summary(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved full evaluation report to {args.output}")


if __name__ == "__main__":
    main()
