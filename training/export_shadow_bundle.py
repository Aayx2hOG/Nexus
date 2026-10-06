"""Package one explicitly chosen frozen selective policy after raw-input parity checks.

No fitting, calibration, threshold search, or independent-evaluation claim.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nexus.bundle import PROJECT_ROOT, sha256_file
from nexus.shadow import (
    INFERENCE_SOURCES,
    SERVING_SOURCES,
    ShadowBundle,
    ShadowManifest,
    load_shadow,
)


def export_shadow(source: Path, csv: Path, output: Path, version: str, budget: float) -> Path:
    if output.exists():
        raise FileExistsError("Shadow destination already exists; choose a new immutable version")
    provenance = json.loads((source / "manifest.json").read_text())
    if provenance["status"] != "complete" or sha256_file(csv) != provenance["data_sha256"]:
        raise ValueError("Incomplete research run or mismatched source data")
    names = (
        "models.joblib",
        "split_indices.npz",
        "calibration.json",
        "evaluation_scores.npz",
        "evaluation_decisions.npz",
    )
    for name in names:
        if sha256_file(source / name) != provenance["artifact_hashes"][name]:
            raise ValueError(f"Research artifact hash mismatch: {name}")
    for name in INFERENCE_SOURCES:
        if sha256_file(PROJECT_ROOT / "training" / name) != provenance["source_hashes"][name]:
            raise ValueError(f"Research inference implementation changed: {name}")
    checkpoint = joblib.load(source / "models.joblib")
    policy = checkpoint["selective_policies"][str(budget)]
    calibration = json.loads((source / "calibration.json").read_text())
    if calibration["selective_policies"][str(budget)] != policy:
        raise ValueError("Checkpoint and calibration disagree")
    manifest = ShadowManifest(
        mode="shadow",
        bundle_version=version,
        feature_columns=list(checkpoint["tree_features"].native.columns),
        budget=budget,
        held_family=provenance["held_family"],
        seed=provenance["seed"],
        data_sha256=provenance["data_sha256"],
        split_sha256=provenance["artifact_hashes"]["split_indices.npz"],
        source_manifest_sha256=sha256_file(source / "manifest.json"),
        artifact_hashes={},
        source_hashes={k: provenance["source_hashes"][k] for k in INFERENCE_SOURCES},
        serving_source_hashes={k: sha256_file(PROJECT_ROOT / k) for k in SERVING_SOURCES},
        policy=policy,
        evaluation_note="Raw-input parity on existing development evaluation rows only. "
        "Independent confirmation deferred. No live-model budget guarantee.",
    )
    runtime = ShadowBundle(manifest, checkpoint, "export-in-progress")
    raw = pd.read_csv(csv)
    with np.load(source / "split_indices.npz") as partitions:
        indices = partitions["evaluation"].copy()
    errors = {k: 0.0 for k in ("lightgbm", "ae_denoising", "ae_latent", "learned_fusion")}
    mismatch = baseline_lost = 0
    # Check every archived evaluation row, including the raw-input serving frame conversion.
    # Numeric flow fields arrive as Python numbers; float64 mirrors flows_to_dataframe.
    with (
        np.load(source / "evaluation_scores.npz") as scores,
        np.load(source / "evaluation_decisions.npz") as decisions,
    ):
        if not (
            np.array_equal(indices, scores["row_indices"])
            and np.array_equal(indices, decisions["row_indices"])
        ):
            raise ValueError("Evaluation rows are misaligned")
        expected = decisions[f"selective_fusion__{budget}"]
        for start in range(0, len(indices), 100):
            stop = min(start + 100, len(indices))
            frame = raw.iloc[indices[start:stop]][manifest.feature_columns].copy()
            for name in frame:
                frame[name] = (
                    frame[name].astype("string")
                    if name in checkpoint["tree_features"].categorical
                    else frame[name].astype(np.float64)
                )
            actual = runtime.score_frame(frame)
            for name in errors:
                reference = scores[name][start:stop]
                errors[name] = max(errors[name], float(np.max(np.abs(actual[name] - reference))))
                if not np.allclose(actual[name], reference, rtol=1e-5, atol=1e-6):
                    raise ValueError(f"Raw-input score parity failed: {name}")
            mismatch += int(np.count_nonzero(actual["candidate"] != expected[start:stop]))
            baseline_lost += int(np.count_nonzero(actual["baseline"] & ~actual["candidate"]))
            if start % 10000 == 0:
                print(f"Verified {stop}/{len(indices)} raw evaluation rows", flush=True)
    if mismatch or baseline_lost or not len(indices):
        raise ValueError("Raw-input selective decision parity failed")
    parity = {
        "status": "passed",
        "rows": len(indices),
        "batch_size": 100,
        "score_rtol": 1e-5,
        "score_atol": 1e-6,
        "max_absolute_score_errors": errors,
        "decision_mismatches": mismatch,
        "lost_reference_detections": baseline_lost,
        "checkpoint_sha256": provenance["artifact_hashes"]["models.joblib"],
        "data_sha256": manifest.data_sha256,
        "split_sha256": manifest.split_sha256,
        "budget": budget,
        "note": manifest.evaluation_note,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # The destination appears only after the complete bundle verifies successfully.
    with tempfile.TemporaryDirectory(prefix="shadow-export-", dir=output.parent) as temp:
        stage = Path(temp) / "bundle"
        stage.mkdir()
        for name in ("models.joblib", "calibration.json"):
            shutil.copyfile(source / name, stage / name)
        shutil.copyfile(source / "manifest.json", stage / "source_manifest.json")
        (stage / "parity.json").write_text(json.dumps(parity, indent=2) + "\n")
        manifest.artifact_hashes = {
            name: sha256_file(stage / name)
            for name in ("models.joblib", "calibration.json", "parity.json", "source_manifest.json")
        }
        (stage / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
        load_shadow(stage)
        stage.rename(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--budget", required=True, type=float)
    args = parser.parse_args()
    print(export_shadow(args.source, args.csv, args.output, args.version, args.budget))


if __name__ == "__main__":
    main()
