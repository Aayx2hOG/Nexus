"""Capture existing calibration scores and audit frozen boundaries without model loading."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from complementary_fusion import SelectiveFusion

ARCHIVE = "calibration_evidence.npz"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_rows(rows, split):
    require(
        set(split) == {"fit", "early", "fusion", "calibration", "evaluation"},
        "Expected all five partition index arrays",
    )
    for name, values in split.items():
        require(
            values.ndim == 1 and np.issubdtype(values.dtype, np.integer) and np.all(values >= 0),
            f"Invalid {name} indices",
        )
    combined = np.concatenate(list(split.values()))
    require(len(np.unique(combined)) == len(combined), "Overlapping or duplicate partition rows")
    require(np.array_equal(rows, split["calibration"]), "Calibration partition mismatch")


def validate_arrays(arrays, metadata):
    rows, labels = arrays["row_indices"], arrays["labels"]
    require(rows.ndim == 1 and np.issubdtype(rows.dtype, np.integer), "Invalid row indices")
    require(len(rows) > 0 and len(np.unique(rows)) == len(rows), "Empty/duplicate calibration rows")
    require(
        labels.shape == rows.shape and np.isin(labels, [0, 1]).all() and np.any(labels == 0),
        "Need aligned binary labels including benign rows",
    )
    families = arrays["families"]
    require(families.shape == rows.shape, "Family alignment mismatch")
    require(np.array_equal(families == "Normal", labels == 0), "Family/label mismatch")
    if metadata["held_family"] != "none":
        require(not np.any(families == metadata["held_family"]), "Held family in calibration")
    for name, values in arrays.items():
        require(values.shape == rows.shape, f"Score/row alignment mismatch: {name}")
        if name.startswith(("score__", "rank__")):
            require(np.isfinite(values).all(), f"Non-finite values: {name}")
        if name.startswith("rank__"):
            require(np.all((values >= 0) & (values <= 1)), "Anomaly ranks outside [0,1]")
        if name.startswith("route__"):
            require(values.dtype == np.bool_, "Route masks must be boolean")


def write_evidence(
    output,
    *,
    split,
    labels,
    families,
    scores,
    anomaly_rank,
    policies,
    budgets,
    seed,
    held_family,
    data_sha256,
    model_sha256,
):
    """Serialize arrays already computed by an authorized run; never fit or score models."""
    output = Path(output)
    rows = np.asarray(split["calibration"])
    validate_rows(rows, split)
    arrays = {
        "row_indices": rows,
        "labels": np.asarray(labels),
        "families": np.asarray(families, dtype=str),
        "rank__ae_denoising": np.asarray(anomaly_rank),
    }
    arrays.update({f"score__{key}": np.asarray(values) for key, values in scores.items()})
    entries = {}
    for key, state in policies.items():
        budget = float(key.rsplit("__", 1)[-1])
        require(budget in budgets, "Policy budget missing from run configuration")
        policy = SelectiveFusion(**state)
        arrays[f"route__{key}"] = policy.route(arrays["score__lightgbm"], anomaly_rank)
        entries[key] = {
            "budget": budget,
            "policy": state,
            "recovery_score": "score__learned_fusion",
            "anomaly_rank": "rank__ae_denoising",
        }
    metadata = dict(
        schema_version=1,
        partition="calibration",
        seed=seed,
        held_family=held_family,
        data_sha256=data_sha256,
        partition_sha256=digest(output / "split_indices.npz"),
        model_sha256=model_sha256,
        policies=entries,
        calibration_budgets=list(budgets),
        rank_reference_partition="fusion benign rows",
        threshold_selection="Existing conservative >= policy; nextafter of order "
        "statistic when capacity binds, subject to configured minimum score",
    )
    validate_arrays(arrays, metadata)
    for key in entries:
        boundary_report(arrays, metadata, key)
    with (output / ARCHIVE).open("xb") as stream:
        np.savez_compressed(stream, metadata_json=np.array(json.dumps(metadata)), **arrays)


def tail_statistics(values, cutoff, window):
    """Describe a fixed absolute window; never select or suggest a threshold."""
    below = np.unique(values[values < cutoff])
    above = values[values > cutoff]
    distinct = np.unique(values)
    return {
        "rows": len(values),
        "above_cutoff": int((values > cutoff).sum()),
        "equal_cutoff": int((values == cutoff).sum()),
        "passing_ge_cutoff": int((values >= cutoff).sum()),
        "window_absolute": window,
        "within_window": int((np.abs(values - cutoff) <= window).sum()),
        "predecessor_float_ties": int((values == np.nextafter(cutoff, -np.inf)).sum()),
        "nearest_below": float(below[-1]) if len(below) else None,
        "nearest_above": float(above.min()) if len(above) else None,
        "maximum": float(distinct[-1]) if len(distinct) else None,
        "maximum_ties": int((values == distinct[-1]).sum()) if len(distinct) else 0,
        "maximum_to_next_distinct_gap": float(distinct[-1] - distinct[-2])
        if len(distinct) > 1
        else None,
        "quantiles": dict(
            zip(
                ["min", "median", "p90", "p95", "p99", "max"],
                np.quantile(values, [0, 0.5, 0.9, 0.95, 0.99, 1]).tolist(),
                strict=True,
            )
        )
        if len(values)
        else None,
    }


def boundary_report(arrays, metadata, policy_key, window=1e-4):
    require(np.isfinite(window) and window >= 0, "Window must be finite and nonnegative")
    require(policy_key in metadata["policies"], f"No archived policy: {policy_key}")
    entry = metadata["policies"][policy_key]
    policy = SelectiveFusion(**entry["policy"])
    require(np.isfinite(policy.recovery_threshold), "Non-finite frozen recovery cutoff")
    budget = entry["budget"]
    require(0 < budget < 1, "Invalid calibration budget")
    benign = arrays["labels"] == 0
    tree, fusion = arrays["score__lightgbm"], arrays[entry["recovery_score"]]
    routed = policy.route(tree, arrays[entry["anomaly_rank"]])
    require(np.array_equal(routed, arrays[f"route__{policy_key}"]), "Archived route mismatch")
    primary = benign & (tree >= policy.baseline_threshold)
    recovery = benign & routed & (fusion >= policy.recovery_threshold)
    n, primary_fp, recovery_fp = int(benign.sum()), int(primary.sum()), int(recovery.sum())
    allowed = int(np.floor(budget * n))
    require(n == policy.calibration_rows, "Benign calibration count mismatch")
    require(primary_fp == policy.baseline_false_positives, "Primary FP count mismatch")
    require(
        primary_fp + recovery_fp == policy.calibration_false_positives,
        "Total calibration FP count mismatch",
    )
    require(
        0 <= policy.allowed_additional_false_positives <= allowed - primary_fp,
        "Invalid recovery allowance",
    )
    require(recovery_fp <= policy.allowed_additional_false_positives, "Recovery allowance exceeded")
    return dict(
        seed=metadata["seed"],
        held_family=metadata["held_family"],
        policy_key=policy_key,
        calibration_budget=budget,
        benign_calibration_rows=n,
        frozen_recovery_cutoff=policy.recovery_threshold,
        primary_calibration_fps=primary_fp,
        recovery_calibration_fps=recovery_fp,
        total_calibration_fps=primary_fp + recovery_fp,
        observed_calibration_fpr=(primary_fp + recovery_fp) / n,
        total_allowed_calibration_fps=allowed,
        recovery_allowance=policy.allowed_additional_false_positives,
        unused_recovery_allowance=policy.allowed_additional_false_positives - recovery_fp,
        all_benign=tail_statistics(fusion[benign], policy.recovery_threshold, window),
        routed_benign=tail_statistics(fusion[benign & routed], policy.recovery_threshold, window),
        evaluation_fps=None,
        interpretation="Calibration evidence only; evaluation FPs not analyzed. "
        "Routed benign scores constrain recovery, not all benign scores. "
        "Tail counts alone do not establish causality or a need to change thresholds.",
    )


def audit_directory(directory, policy_key, window=1e-4):
    directory = Path(directory)
    if not (directory / ARCHIVE).is_file():
        raise FileNotFoundError(
            f"Missing {ARCHIVE}; calibration evidence was not archived. "
            "No inference or regeneration is permitted by this auditor."
        )
    manifest = json.loads((directory / "manifest.json").read_text())
    require(manifest["status"] == "complete", "Run is incomplete")
    for name in (ARCHIVE, "split_indices.npz", "calibration.json"):
        require(
            digest(directory / name) == manifest["artifact_hashes"].get(name),
            f"Artifact hash mismatch: {name}",
        )
    with np.load(directory / ARCHIVE, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        arrays = {key: archive[key] for key in archive.files if key != "metadata_json"}
    require(
        metadata["schema_version"] == 1 and metadata["partition"] == "calibration",
        "Unsupported evidence schema or partition",
    )
    for key in ("seed", "held_family", "data_sha256"):
        require(metadata[key] == manifest[key], f"Evidence/manifest mismatch: {key}")
    require(
        metadata["partition_sha256"] == manifest["artifact_hashes"]["split_indices.npz"],
        "Evidence partition hash mismatch",
    )
    require(
        metadata["model_sha256"] == manifest["artifact_hashes"]["models.joblib"],
        "Evidence checkpoint hash mismatch",
    )
    with np.load(directory / "split_indices.npz", allow_pickle=False) as archive:
        validate_rows(arrays["row_indices"], {key: archive[key] for key in archive.files})
    validate_arrays(arrays, metadata)
    calibration = json.loads((directory / "calibration.json").read_text())
    require(
        calibration["partition"] == "calibration" and not calibration["uses_evaluation_labels"],
        "Invalid calibration provenance",
    )
    require(
        {key: value["policy"] for key, value in metadata["policies"].items()}
        == calibration["selective_policies"],
        "Frozen policy mismatch",
    )
    result = boundary_report(arrays, metadata, policy_key, window)
    result["evidence_sha256"] = manifest["artifact_hashes"][ARCHIVE]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--policy-key", required=True, help="Saved policy key, e.g. 0.01")
    parser.add_argument(
        "--window",
        type=float,
        default=1e-4,
        help="Descriptive absolute score neighborhood; never changes the cutoff",
    )
    parser.add_argument("--output-json", type=Path, help="Optional new file; otherwise stdout")
    args = parser.parse_args()
    result = audit_directory(args.run_dir, args.policy_key, args.window)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output_json:
        with args.output_json.open("x") as stream:
            stream.write(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
