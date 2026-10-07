"""Evaluate the trained Random Forest and LightGBM models on the test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib

from training_utils import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    PROJECT_ROOT,
    evaluate_binary_classifier,
    load_tree_data,
)

MODEL_FILES = {
    "random_forest": "random_forest.joblib",
    "lightgbm": "lightgbm.joblib",
}
SUMMARY_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "false_positive_rate",
    "specificity",
    "roc_auc",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing random_forest.joblib and lightgbm.joblib; "
            "auto-detected when omitted"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON report path (default: reports/evaluation_metrics.json)",
    )
    return parser.parse_args()


def find_model_dir(requested_dir: Path | None, data_dir: Path) -> Path:
    if requested_dir is not None:
        return requested_dir

    candidates = (DEFAULT_OUTPUT_DIR, PROJECT_ROOT / "models", data_dir)
    for candidate in candidates:
        if all((candidate / filename).is_file() for filename in MODEL_FILES.values()):
            return candidate

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not find both trained models. Searched: {searched}. "
        "Train them first or specify --model-dir."
    )


def load_model(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(
            f"Model not found: {path}. Train it first or provide the correct --model-dir."
        )
    return joblib.load(path)


def print_summary(results: dict[str, dict[str, Any]]) -> None:
    model_width = max(len("model"), *(len(name) for name in results))
    metric_width = 19
    header = f"{'model':<{model_width}}" + "".join(
        f" {metric:>{metric_width}}" for metric in SUMMARY_METRICS
    )
    print(header)
    print("-" * len(header))
    for model_name, metrics in results.items():
        row = f"{model_name:<{model_width}}" + "".join(
            f" {metrics[metric]:>{metric_width}.6f}" for metric in SUMMARY_METRICS
        )
        print(row)

    for model_name, metrics in results.items():
        matrix = metrics["confusion_matrix"]
        print(f"\n{model_name} confusion matrix [[TN, FP], [FN, TP]]: {matrix}")


def main() -> None:
    args = parse_args()
    _, X_test, _, y_test = load_tree_data(args.data_dir)
    model_dir = find_model_dir(args.model_dir, args.data_dir)
    print(f"Evaluating {len(MODEL_FILES)} models on {X_test.shape[0]:,} test rows\n")
    print(f"Loading models from {model_dir}\n")

    results = {}
    for model_name, filename in MODEL_FILES.items():
        model = load_model(model_dir / filename)
        results[model_name] = evaluate_binary_classifier(model, X_test, y_test)

    print_summary(results)

    output_path = args.output or PROJECT_ROOT / "reports" / "evaluation_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved full evaluation report to {output_path}")


if __name__ == "__main__":
    main()
