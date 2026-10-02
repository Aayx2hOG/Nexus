"""Train and evaluate LightGBM on the preprocessed UNSW-NB15 data."""

from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
from sklearn.model_selection import train_test_split

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
    parser.add_argument("--n-estimators", type=int, default=1_000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--early-stopping-rounds", type=int, default=75)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def make_model(args: argparse.Namespace, n_estimators: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        class_weight="balanced",
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        verbosity=-1,
    )


def main() -> None:
    args = parse_args()
    if not 0 < args.validation_size < 1:
        raise ValueError("--validation-size must be between 0 and 1")
    if args.early_stopping_rounds < 1:
        raise ValueError("--early-stopping-rounds must be positive")

    X_train, X_test, y_train, y_test = load_tree_data(args.data_dir)
    print(f"Training LightGBM on {X_train.shape[0]:,} rows and {X_train.shape[1]:,} features")

    X_fit, X_valid, y_fit, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=args.validation_size,
        stratify=y_train,
        random_state=args.random_state,
    )
    selection_model = make_model(args, args.n_estimators)
    selection_model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_valid, y_valid)],
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(args.early_stopping_rounds),
            lgb.log_evaluation(25),
        ],
    )

    best_iteration = selection_model.best_iteration_ or args.n_estimators
    print(f"Refitting on all training rows with {best_iteration} boosting rounds")
    model = make_model(args, best_iteration)
    model.fit(X_train, y_train)

    metrics = evaluate_binary_classifier(model, X_test, y_test)
    metrics["model"] = "lightgbm"
    metrics["train_rows"] = X_train.shape[0]
    metrics["test_rows"] = X_test.shape[0]
    metrics["feature_count"] = X_train.shape[1]
    metrics["best_iteration"] = best_iteration
    metrics["parameters"] = model.get_params()
    save_results(model, metrics, args.output_dir, "lightgbm")


if __name__ == "__main__":
    main()
