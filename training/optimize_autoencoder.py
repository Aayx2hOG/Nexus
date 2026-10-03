"""Validation-only controlled search for the UNSW-NB15 autoencoder.

This program deliberately never opens the official test arrays.  It trains on
normal rows from the existing internal training split, selects each threshold
on the labelled validation split, and freezes one winner for later evaluation
by ``evaluate_optimized_autoencoder.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from train_autoencoder import fit_autoencoder, make_autoencoder_features, reconstruction_errors
from train_isolation_forest import select_operating_point
from training_utils import DEFAULT_DATA_DIR, PROJECT_ROOT

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "autoencoder_optimization"
DEFAULT_BASELINE_DIR = PROJECT_ROOT / "models" / "autoencoder"
ARCHITECTURES = (
    (128, 32, 128),
    (128, 64, 32, 64, 128),
    (128, 64, 128),
    (256, 64, 256),
    (256, 128, 32, 128, 256),
    (256, 128, 64, 128, 256),
)
LEARNING_RATES = (1e-3, 5e-4)
BATCH_SIZES = (256, 512, 1024)
FPR_LIMITS = (0.05, 0.10, 0.15)
BASE_ALPHA = 1e-4
ALPHA_SENSITIVITY = 1e-5
EXPECTED_BASELINE = {
    "roc_auc": 0.9117195981937121,
    "pr_auc": 0.9615426520139674,
    "recall": 0.7958439817336294,
    "f1": 0.8638472032742156,
    "balanced_accuracy": 0.8481452051525289,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--skip-alpha-sensitivity",
        action="store_true",
        help="Run only the 36 architecture/LR/batch candidates (not recommended).",
    )
    return parser.parse_args()


def json_dump(value: Any, path: Path) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_training_data_only(data_dir: Path) -> tuple[sparse.csr_matrix, np.ndarray, dict[str, Any]]:
    """Load only official-training artifacts; test files are intentionally untouched."""
    x_path = data_dir / "X_train_tree.npz"
    y_path = data_dir / "y_train.npy"
    metadata_path = data_dir / "preprocessing_metadata.json"
    missing = [str(path) for path in (x_path, y_path, metadata_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing training artifact(s): " + ", ".join(missing))
    X = sparse.load_npz(x_path).tocsr()
    y = np.load(y_path, allow_pickle=False)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if X.shape[0] != y.size or not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Training matrix/labels are misaligned or non-binary")
    numerical_count = len(metadata["numerical_columns"])
    if not 0 < numerical_count <= X.shape[1]:
        raise ValueError("Invalid numerical feature count in preprocessing metadata")
    return X, y, metadata


def load_and_check_baseline(baseline_dir: Path) -> dict[str, Any]:
    metrics_path = baseline_dir / "autoencoder_validation_metrics.json"
    config_path = baseline_dir / "autoencoder_config.json"
    if not metrics_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            "Existing baseline AE metrics/config are required and will only be read: "
            f"{baseline_dir}"
        )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["parameters"]["hidden_layers"] != [128, 32, 128]:
        raise RuntimeError("Baseline architecture is not the expected [128, 32, 128]")
    selected = metrics["selected_operating_point"]
    actual = {
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "recall": selected["recall"],
        "f1": selected["f1"],
        "balanced_accuracy": selected["balanced_accuracy"],
    }
    mismatches = {
        name: {"expected": expected, "actual": actual[name]}
        for name, expected in EXPECTED_BASELINE.items()
        if not np.isclose(actual[name], expected, rtol=0.0, atol=1e-9)
    }
    if mismatches:
        raise RuntimeError(f"Baseline metrics do not match the expected run: {mismatches}")
    return {"metrics": metrics, "config": config, "summary": actual}


def candidate_id(
    layers: tuple[int, ...], learning_rate: float, batch_size: int, alpha: float
) -> str:
    architecture = "-".join(map(str, layers))
    return f"layers_{architecture}__lr_{learning_rate:g}__batch_{batch_size}__alpha_{alpha:g}"


def evaluate_candidate(
    *,
    X_fit: sparse.csr_matrix,
    X_early_stop: sparse.csr_matrix,
    X_validation: sparse.csr_matrix,
    y_validation: np.ndarray,
    layers: tuple[int, ...],
    learning_rate: float,
    batch_size: int,
    alpha: float,
    epochs: int,
    patience: int,
    random_state: int,
) -> tuple[Any, dict[str, Any]]:
    started = time.perf_counter()
    model, history, best_epoch = fit_autoencoder(
        X_fit,
        X_early_stop,
        layers,
        epochs,
        batch_size,
        learning_rate,
        patience,
        random_state,
        alpha,
    )
    duration = time.perf_counter() - started
    scores = reconstruction_errors(model, X_validation, batch_size)
    operating_points = {
        f"fpr_{int(limit * 100):02d}": select_operating_point(y_validation, scores, limit)
        for limit in FPR_LIMITS
    }
    for point in operating_points.values():
        (tn, fp), (fn, tp) = point["confusion_matrix"]
        point.update({"tn": tn, "fp": fp, "fn": fn, "tp": tp})
    result = {
        "candidate_id": candidate_id(layers, learning_rate, batch_size, alpha),
        "architecture": list(layers),
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "alpha": alpha,
        "maximum_epochs": epochs,
        "patience": patience,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "training_seconds": duration,
        "roc_auc": float(roc_auc_score(y_validation, scores)),
        "pr_auc": float(average_precision_score(y_validation, scores)),
        "operating_points": operating_points,
        "training_history": history,
    }
    return model, result


def add_baseline_differences(result: dict[str, Any], baseline: dict[str, float]) -> None:
    point = result["operating_points"]["fpr_10"]
    result["improvement_over_existing_baseline"] = {
        "recall": point["recall"] - baseline["recall"],
        "f1": point["f1"] - baseline["f1"],
        "pr_auc": result["pr_auc"] - baseline["pr_auc"],
        "balanced_accuracy": point["balanced_accuracy"] - baseline["balanced_accuracy"],
    }


def selection_key(result: dict[str, Any]) -> tuple[float, ...]:
    """Primary recall-at-10%-FPR objective followed by requested tie-breakers."""
    point = result["operating_points"]["fpr_10"]
    return (
        point["recall"],
        result["pr_auc"],
        point["f1"],
        point["balanced_accuracy"],
        result["roc_auc"],
        -sum(result["architecture"]),
        -len(result["architecture"]),
    )


def flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    nested_keys = {
        "operating_points",
        "training_history",
        "improvement_over_existing_baseline",
    }
    flat = {key: value for key, value in result.items() if key not in nested_keys}
    flat["architecture"] = json.dumps(result["architecture"])
    for name, point in result["operating_points"].items():
        for metric, value in point.items():
            if metric != "confusion_matrix":
                flat[f"{name}_{metric}"] = value
        (tn, fp), (fn, tp) = point["confusion_matrix"]
        flat.update({f"{name}_tn": tn, f"{name}_fp": fp, f"{name}_fn": fn, f"{name}_tp": tp})
    for metric, value in result["improvement_over_existing_baseline"].items():
        flat[f"improvement_{metric}"] = value
    return flat


def main() -> None:
    args = parse_args()
    if args.output_dir.resolve() == args.baseline_dir.resolve():
        raise ValueError("Optimization output must not be the existing baseline directory")
    if args.epochs != 50 or args.patience != 7 or args.random_state != 42:
        raise ValueError(
            "This controlled experiment requires epochs=50, patience=7, random-state=42"
        )
    if (args.output_dir / "official_test_evaluation_record.json").exists():
        raise RuntimeError(
            "This output directory already contains an official-test evaluation. "
            "Use a new --output-dir for another search."
        )
    baseline = load_and_check_baseline(args.baseline_dir)
    X_train, y_train, metadata = load_training_data_only(args.data_dir)
    train_indices, validation_indices = train_test_split(
        np.arange(y_train.size), test_size=0.2, stratify=y_train, random_state=42
    )
    normal_fit_indices = train_indices[y_train[train_indices] == 0]
    normal_early_stop_indices = validation_indices[y_train[validation_indices] == 0]
    expected_split = baseline["config"]["split"]
    actual_split = {
        "official_training_rows": int(y_train.size),
        "internal_training_rows": int(train_indices.size),
        "fit_normal_rows": int(normal_fit_indices.size),
        "validation_rows": int(validation_indices.size),
    }
    split_mismatches = {
        name: {"expected": expected_split[name], "actual": actual}
        for name, actual in actual_split.items()
        if expected_split.get(name) != actual
    }
    if baseline["config"].get("random_state") != 42 or split_mismatches:
        raise RuntimeError(
            f"Current data split does not match the random-state-42 baseline: {split_mismatches}"
        )
    numerical_count = len(metadata["numerical_columns"])
    scaler = StandardScaler().fit(X_train[normal_fit_indices, :numerical_count].toarray())
    X_fit = make_autoencoder_features(X_train[normal_fit_indices], numerical_count, scaler)
    X_early_stop = make_autoencoder_features(
        X_train[normal_early_stop_indices], numerical_count, scaler
    )
    X_validation = make_autoencoder_features(X_train[validation_indices], numerical_count, scaler)
    y_validation = y_train[validation_indices]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    first_stage_best_model: Any | None = None
    first_stage_winner: dict[str, Any] | None = None
    grid = [
        (layers, learning_rate, batch_size, BASE_ALPHA)
        for layers in ARCHITECTURES
        for learning_rate in LEARNING_RATES
        for batch_size in BATCH_SIZES
    ]
    for index, (layers, learning_rate, batch_size, alpha) in enumerate(grid, 1):
        name = candidate_id(layers, learning_rate, batch_size, alpha)
        print(f"\nCandidate {index}/{len(grid)}: {name}")
        model, result = evaluate_candidate(
            X_fit=X_fit,
            X_early_stop=X_early_stop,
            X_validation=X_validation,
            y_validation=y_validation,
            layers=layers,
            learning_rate=learning_rate,
            batch_size=batch_size,
            alpha=alpha,
            epochs=args.epochs,
            patience=args.patience,
            random_state=args.random_state,
        )
        add_baseline_differences(result, baseline["summary"])
        results.append(result)
        json_dump(result, args.output_dir / f"validation_metrics__{result['candidate_id']}.json")
        if first_stage_winner is None or selection_key(result) > selection_key(first_stage_winner):
            first_stage_winner = result
            first_stage_best_model = model
        json_dump(results, args.output_dir / "search_results.json")
        pd.DataFrame(map(flatten_result, results)).to_csv(
            args.output_dir / "search_results.csv", index=False
        )

    if first_stage_winner is None or first_stage_best_model is None:
        raise RuntimeError("Search did not produce a candidate")
    winner = first_stage_winner
    winner_model = first_stage_best_model
    if not args.skip_alpha_sensitivity:
        model, result = evaluate_candidate(
            X_fit=X_fit,
            X_early_stop=X_early_stop,
            X_validation=X_validation,
            y_validation=y_validation,
            layers=tuple(first_stage_winner["architecture"]),
            learning_rate=first_stage_winner["learning_rate"],
            batch_size=first_stage_winner["batch_size"],
            alpha=ALPHA_SENSITIVITY,
            epochs=args.epochs,
            patience=args.patience,
            random_state=args.random_state,
        )
        add_baseline_differences(result, baseline["summary"])
        results.append(result)
        json_dump(result, args.output_dir / f"validation_metrics__{result['candidate_id']}.json")
        if selection_key(result) > selection_key(winner):
            winner = result
            winner_model = model

    selected_point = winner["operating_points"]["fpr_10"]
    fit_scores = reconstruction_errors(winner_model, X_fit, winner["batch_size"])
    validation_scores = reconstruction_errors(
        winner_model, X_validation, winner["batch_size"]
    )
    config = {
        "model": "MLPRegressorAutoencoder",
        "candidate_id": winner["candidate_id"],
        "parameters": {
            "hidden_layers": winner["architecture"],
            "learning_rate": winner["learning_rate"],
            "batch_size": winner["batch_size"],
            "alpha": winner["alpha"],
            "maximum_epochs": args.epochs,
            "patience": args.patience,
            "activation": "relu",
            "optimizer": "adam",
        },
        "random_state": 42,
        "selected_threshold": selected_point["threshold"],
        "threshold_selection": "maximum validation attack recall subject to FPR <= 0.10",
        "feature_scaling": "numeric StandardScaler fitted on normal fit rows; one-hot unchanged",
        "anomaly_score": "per-row mean squared reconstruction error",
        "numerical_feature_count": numerical_count,
        "feature_count": int(X_train.shape[1]),
        "split": {
            "official_training_rows": int(y_train.size),
            "internal_training_rows": int(train_indices.size),
            "fit_normal_rows": int(normal_fit_indices.size),
            "validation_rows": int(validation_indices.size),
            "validation_attack_rows": int(y_validation.sum()),
            "validation_normal_rows": int((y_validation == 0).sum()),
            "validation_fraction": 0.2,
            "stratified_on": "binary label",
        },
        "normalization": {
            "method": "min-max with clipping",
            "fit_data": "normal internal training and complete validation scores",
            "minimum": float(min(fit_scores.min(), validation_scores.min())),
            "maximum": float(max(fit_scores.max(), validation_scores.max())),
        },
        "training": {
            "best_epoch": winner["best_epoch"],
            "epochs_completed": winner["epochs_completed"],
            "training_seconds": winner["training_seconds"],
        },
        "validation_metrics": winner,
    }
    fingerprint_source = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    fingerprint = hashlib.sha256(fingerprint_source).hexdigest()
    frozen_record = {
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_test_evaluated": False,
        "selection_data": (
            "official training split only (internal validation); official test not loaded"
        ),
        "selection_objective": "maximize validation attack recall subject to FPR <= 0.10",
        "candidate_count": len(results),
        "winner_candidate_id": winner["candidate_id"],
        "selected_threshold": selected_point["threshold"],
        "config_sha256": fingerprint,
    }
    json_dump(results, args.output_dir / "search_results.json")
    pd.DataFrame(map(flatten_result, results)).sort_values(
        ["fpr_10_recall", "pr_auc", "fpr_10_f1", "fpr_10_balanced_accuracy", "roc_auc"],
        ascending=False,
    ).to_csv(args.output_dir / "search_results.csv", index=False)
    json_dump(config, args.output_dir / "best_validation_config.json")
    json_dump(winner, args.output_dir / "best_validation_metrics.json")
    json_dump(frozen_record, args.output_dir / "selected_before_test.json")
    joblib.dump(
        {"model": winner_model, "numeric_scaler": scaler},
        args.output_dir / "best_autoencoder.joblib",
    )
    print(json.dumps(frozen_record, indent=2, sort_keys=True))
    print("Winner frozen. Run evaluate_optimized_autoencoder.py explicitly to touch test data.")


if __name__ == "__main__":
    main()
