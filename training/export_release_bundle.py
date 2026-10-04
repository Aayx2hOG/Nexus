"""Export release bundle from validated training outputs.

Builds artifacts/bundles/<version>/ containing:
- manifest.json (provenance, metrics, thresholds, schemas, hashes)
- preprocessor.joblib (fitted OneHotFeatures)
- model.joblib (fitted LightGBM classifier with decision threshold)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "training"))


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export_bundle(
    experiment_dir: Path,
    output_dir: Path,
    bundle_version: str = "v1.0.0",
) -> Path:
    manifest_path = experiment_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing experiment manifest: {manifest_path}")

    with manifest_path.open() as f:
        exp_manifest = json.load(f)

    splits_path = experiment_dir / "split_indices.npz"
    if not splits_path.is_file():
        raise FileNotFoundError(f"Missing split indices: {splits_path}")

    # Package the frozen validation winner, never retrain during export.
    for name in ("binary/model.joblib", "binary/selected_validation.json", "split_indices.npz"):
        expected = exp_manifest["artifact_hashes"][name]
        if sha256_file(experiment_dir / name) != expected:
            raise ValueError(f"Experiment artifact hash mismatch: {name}")
    with (experiment_dir / "binary/selected_validation.json").open() as f:
        sel_val = json.load(f)
    frozen = joblib.load(experiment_dir / "binary/model.joblib")
    preprocessor = frozen.features
    model = frozen.estimator
    decision_threshold = float(frozen.decision_threshold_)
    if decision_threshold != float(sel_val["decision_threshold"]):
        raise ValueError("Saved model threshold does not match selected validation")
    if list(preprocessor.native.columns) != exp_manifest["feature_columns"]:
        raise ValueError("Saved preprocessing schema does not match experiment")
    csv_hash = exp_manifest["train_sha256"]

    # Create destination bundle directory
    bundle_path = output_dir / bundle_version
    bundle_path.mkdir(parents=True, exist_ok=True)

    model_target = bundle_path / "model.joblib"
    preprocessor_target = bundle_path / "preprocessor.joblib"

    print(f"Saving model to {model_target}...", flush=True)
    joblib.dump(model, model_target)

    print(f"Saving preprocessor to {preprocessor_target}...", flush=True)
    joblib.dump(preprocessor, preprocessor_target)

    # Build manifest
    source_files = [
        PROJECT_ROOT / "training" / "fast_lightgbm_models.py",
        PROJECT_ROOT / "training" / "train_validated_lightgbm.py",
        PROJECT_ROOT / "training" / "tune_lightgbm_fast.py",
    ]
    source_hashes = {p.name: sha256_file(p) for p in source_files if p.exists()}

    artifact_hashes = {
        "model.joblib": sha256_file(model_target),
        "preprocessor.joblib": sha256_file(preprocessor_target),
    }

    manifest = {
        "bundle_version": bundle_version,
        "created_utc": datetime.now(UTC).isoformat(),
        "code_version": "0.2.0",
        "export_method": "frozen_checkpoint_no_refit",
        "checkpoint_sha256": exp_manifest["artifact_hashes"]["binary/model.joblib"],
        "feature_columns": exp_manifest["feature_columns"],
        "dtypes": {
            col: "string" if col in preprocessor.categorical else "float64"
            for col in exp_manifest["feature_columns"]
        },
        "categorical_columns": preprocessor.categorical,
        "decision_threshold": decision_threshold,
        "calibration_provenance": (
            "Pointwise 95% Wilson lower bound on recall >= 0.95 calibrated "
            "on held-out calibration partition of UNSW-NB15"
        ),
        "validated_metrics": {
            **sel_val["validation_metrics"],
            "caveat": "selection estimate, not independent confirmation",
        },
        "data_sha256": csv_hash,
        "split_sha256": sha256_file(splits_path),
        "source_hashes": source_hashes,
        "artifact_hashes": artifact_hashes,
        "limits": {
            "single_threshold": True,
            "severity_levels": ["alert"],
        },
    }

    manifest_target = bundle_path / "manifest.json"
    with manifest_target.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Successfully exported release bundle {bundle_version} to {bundle_path}", flush=True)
    return bundle_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=PROJECT_ROOT / "models/lightgbm_validated_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/bundles",
    )
    parser.add_argument("--version", type=str, default="v1.0.0")
    args = parser.parse_args()

    export_bundle(args.experiment_dir, args.output_dir, args.version)


if __name__ == "__main__":
    main()
