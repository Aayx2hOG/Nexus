"""Train and evaluate a Random Forest on the preprocessed UNSW-NB15 data."""

from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

from training_utils import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    evaluate_binary_classifier,
    load_tree_data,
    save_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    X_train, X_test, y_train, y_test = load_tree_data(args.data_dir)
    print(f"Training Random Forest on {X_train.shape[0]:,} rows and {X_train.shape[1]:,} features")

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced_subsample",
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        verbose=1,
    )
    model.fit(X_train, y_train)

    metrics = evaluate_binary_classifier(model, X_test, y_test)
    metrics["model"] = "random_forest"
    metrics["train_rows"] = X_train.shape[0]
    metrics["test_rows"] = X_test.shape[0]
    metrics["feature_count"] = X_train.shape[1]
    metrics["parameters"] = model.get_params()
    save_results(model, metrics, args.output_dir, "random_forest")


if __name__ == "__main__":
    main()
