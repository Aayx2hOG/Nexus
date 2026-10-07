"""Evaluate the single pre-test-frozen optimized AE on the official test split."""

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
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from evaluation_isolation_forest import DEFAULT_RAW_TEST, load_attack_categories
from train_autoencoder import make_autoencoder_features, reconstruction_errors
from training_utils import DEFAULT_DATA_DIR, PROJECT_ROOT, load_tree_data

DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "experiments" / "autoencoder_optimization"
DEFAULT_BASELINE_DIR = PROJECT_ROOT / "models" / "autoencoder"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--raw-test-csv", type=Path, default=DEFAULT_RAW_TEST)
    parser.add_argument("--inference-batch-size", type=int, default=2048)
    return parser.parse_args()


def json_dump(value: Any, path: Path) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def metric_row(metric: str, baseline: float, optimized: float) -> dict[str, Any]:
    return {
        "metric": metric,
        "baseline_ae": baseline,
        "optimized_ae": optimized,
        "difference": optimized - baseline,
    }


def percentage(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.2f}%"


def main() -> None:
    args = parse_args()
    config_path = args.artifact_dir / "best_validation_config.json"
    frozen_path = args.artifact_dir / "selected_before_test.json"
    model_path = args.artifact_dir / "best_autoencoder.joblib"
    for path in (config_path, frozen_path, model_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen optimization artifact: {path}")
    evaluation_record_path = args.artifact_dir / "official_test_evaluation_record.json"
    if evaluation_record_path.exists():
        raise RuntimeError(
            "The frozen winner has already been evaluated on the official test set. "
            "Refusing to evaluate it again."
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if fingerprint != frozen["config_sha256"]:
        raise RuntimeError("Frozen winner config changed after validation selection")
    if frozen["official_test_evaluated"] is not False:
        raise RuntimeError("Pre-test selection record is invalid")
    if config["candidate_id"] != frozen["winner_candidate_id"]:
        raise RuntimeError("Model config and pre-test winner record disagree")

    _, X_test, _, y_test = load_tree_data(args.data_dir)
    attack_categories = load_attack_categories(args.raw_test_csv, y_test)
    bundle = joblib.load(model_path)
    X_autoencoder = make_autoencoder_features(
        X_test, int(config["numerical_feature_count"]), bundle["numeric_scaler"]
    )
    threshold = float(config["selected_threshold"])
    started = time.perf_counter()
    scores = reconstruction_errors(bundle["model"], X_autoencoder, args.inference_batch_size)
    predictions = (scores >= threshold).astype(np.int64)
    inference_seconds = time.perf_counter() - started
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if fp + tn else 0.0
    metrics = {
        "model": "Optimized MLPRegressorAutoencoder",
        "candidate_id": config["candidate_id"],
        "test_rows": int(y_test.size),
        "frozen_validation_threshold": threshold,
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "false_positive_rate": float(fpr),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "pr_auc": float(average_precision_score(y_test, scores)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "timing": {
            "inference_seconds": float(inference_seconds),
            "records_per_second": float(y_test.size / inference_seconds),
            "milliseconds_per_record": float(1000 * inference_seconds / y_test.size),
            "includes": "reconstruction and thresholding",
        },
    }
    per_attack: dict[str, Any] = {}
    for category in sorted(np.unique(attack_categories)):
        mask = attack_categories == category
        labels = y_test[mask]
        category_predictions = predictions[mask]
        total = int(labels.sum())
        detected = int(((labels == 1) & (category_predictions == 1)).sum())
        per_attack[category] = {
            "total": total,
            "detected": detected,
            "missed": total - detected,
            "recall": float(detected / total) if total else None,
            "mean_reconstruction_error": float(scores[mask].mean()),
            "median_reconstruction_error": float(np.median(scores[mask])),
        }

    baseline_validation = json.loads(
        (args.baseline_dir / "autoencoder_validation_metrics.json").read_text(encoding="utf-8")
    )
    baseline_test = json.loads(
        (args.baseline_dir / "autoencoder_metrics.json").read_text(encoding="utf-8")
    )
    baseline_per_attack = json.loads(
        (args.baseline_dir / "autoencoder_per_attack_metrics.json").read_text(encoding="utf-8")
    )
    winner_validation = config["validation_metrics"]
    baseline_point = baseline_validation["selected_operating_point"]
    winner_point = winner_validation["operating_points"]["fpr_10"]
    comparison = [
        metric_row(
            "Validation recall @ <=10% FPR",
            baseline_point["recall"],
            winner_point["recall"],
        ),
        metric_row("Validation F1", baseline_point["f1"], winner_point["f1"]),
        metric_row("Validation PR-AUC", baseline_validation["pr_auc"], winner_validation["pr_auc"]),
        metric_row(
            "Validation ROC-AUC", baseline_validation["roc_auc"], winner_validation["roc_auc"]
        ),
        metric_row("Test recall", baseline_test["recall"], metrics["recall"]),
        metric_row("Test precision", baseline_test["precision"], metrics["precision"]),
        metric_row("Test F1", baseline_test["f1"], metrics["f1"]),
        metric_row(
            "Test FPR", baseline_test["false_positive_rate"], metrics["false_positive_rate"]
        ),
        metric_row("Test PR-AUC", baseline_test["pr_auc"], metrics["pr_auc"]),
        metric_row("Test ROC-AUC", baseline_test["roc_auc"], metrics["roc_auc"]),
    ]
    family_comparison = []
    for family, optimized in per_attack.items():
        if optimized["recall"] is None:
            continue
        old = baseline_per_attack.get(family, {})
        baseline_recall = old.get("attack_recall", old.get("recall"))
        family_comparison.append(
            {
                "attack_family": family,
                "baseline_recall": baseline_recall,
                "optimized_recall": optimized["recall"],
                "difference": (
                    optimized["recall"] - baseline_recall
                    if baseline_recall is not None
                    else None
                ),
            }
        )

    test_record = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pre_test_record": frozen_path.name,
        "config_sha256_verified": fingerprint,
        "threshold_source": "frozen validation operating point; test labels were not used",
    }
    score_frame = pd.DataFrame(
        {
            "true_label": y_test,
            "attack_cat": attack_categories,
            "reconstruction_error": scores,
            "prediction": predictions,
        }
    )
    json_dump(metrics, args.artifact_dir / "official_test_metrics.json")
    json_dump(per_attack, args.artifact_dir / "official_test_per_attack_metrics.json")
    json_dump(
        {"metrics": comparison, "per_attack_recall": family_comparison},
        args.artifact_dir / "baseline_comparison.json",
    )
    pd.DataFrame(comparison).to_csv(args.artifact_dir / "baseline_comparison.csv", index=False)
    pd.DataFrame(family_comparison).to_csv(
        args.artifact_dir / "per_attack_recall_comparison.csv", index=False
    )
    score_frame.to_csv(args.artifact_dir / "official_test_scores.csv", index=False)
    search_results = json.loads((args.artifact_dir / "search_results.json").read_text())
    top = sorted(search_results, key=lambda item: (
        item["operating_points"]["fpr_10"]["recall"],
        item["pr_auc"],
        item["operating_points"]["fpr_10"]["f1"],
    ), reverse=True)[:5]
    recall_gain = winner_point["recall"] - baseline_point["recall"]
    complexity = sum(config["parameters"]["hidden_layers"])
    baseline_complexity = sum([128, 32, 128])
    substantial = recall_gain >= 0.01
    lines = [
        "# Autoencoder optimization final report",
        "",
        f"Configurations tested: **{len(search_results)}**",
        f"Winning architecture: **{config['parameters']['hidden_layers']}**",
        f"Learning rate: **{config['parameters']['learning_rate']}**",
        f"Batch size: **{config['parameters']['batch_size']}**",
        f"L2 alpha: **{config['parameters']['alpha']}**",
        f"Epochs completed / best epoch: **{config['training']['epochs_completed']} / "
        f"{config['training']['best_epoch']}**",
        f"Frozen reconstruction threshold: **{threshold:.12g}**",
        "",
        "## Top validation candidates",
        "",
        "| Architecture | LR | Batch | Alpha | Recall @10% FPR | Actual FPR | F1 | "
        "PR-AUC | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in top:
        point = item["operating_points"]["fpr_10"]
        lines.append(
            f"| {item['architecture']} | {item['learning_rate']:g} | {item['batch_size']} | "
            f"{item['alpha']:g} | {point['recall']:.6f} | "
            f"{point['false_positive_rate']:.6f} | {point['f1']:.6f} | "
            f"{item['pr_auc']:.6f} | {item['roc_auc']:.6f} |"
        )
    lines.extend([
        "",
        "## Baseline comparison",
        "",
        "| Metric | Baseline AE | Optimized AE | Difference |",
        "|---|---:|---:|---:|",
    ])
    for row in comparison:
        lines.append(
            f"| {row['metric']} | {row['baseline_ae']:.6f} | "
            f"{row['optimized_ae']:.6f} | {row['difference']:+.6f} |"
        )
    lines.extend([
        "",
        "## Official-test attack-family recall",
        "",
        "| Family | Total | Detected | Missed | Recall | Baseline recall | Difference |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    family_lookup = {row["attack_family"]: row for row in family_comparison}
    for family, item in per_attack.items():
        if item["total"] == 0:
            continue
        family_row = family_lookup[family]
        lines.append(
            f"| {family} | {item['total']} | {item['detected']} | {item['missed']} | "
            f"{percentage(item['recall'])} | {percentage(family_row['baseline_recall'])} | "
            f"{percentage(family_row['difference'])} |"
        )
    assessment = (
        "The validation recall gain is at least 1 percentage point, the report's predefined "
        "threshold for a substantial improvement."
        if substantial
        else "The validation recall gain is below 1 percentage point, so it is not substantial "
        "under the report's predefined heuristic. Prefer the baseline if operational simplicity "
        "matters and the secondary metrics do not show a meaningful gain."
    )
    lines.extend([
        "",
        "## Complexity assessment",
        "",
        f"Validation recall improvement: **{recall_gain:+.2%}**. Hidden-unit sum changed "
        f"from **{baseline_complexity}** to **{complexity}**. {assessment}",
        "",
        "The winner and threshold were selected solely on validation data. The official test "
        "was evaluated once afterward and was not used to retune or replace the winner.",
    ])
    (args.artifact_dir / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Write this completion marker last so an interrupted evaluation can be rerun.
    json_dump(test_record, evaluation_record_path)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Saved final evaluation under {args.artifact_dir.resolve()}")


if __name__ == "__main__":
    main()
