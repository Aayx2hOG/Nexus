"""Create explicitly curated archived examples for a selective-recovery demonstration.

Selection uses existing evaluation decisions, never fitting or threshold tuning.
This fixture must not be presented as a representative accuracy benchmark.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from nexus.bundle import sha256_file
from nexus.replay import row_to_flow
from nexus.shadow import load_shadow


def build_demo(bundle_dir: Path, source: Path, csv: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("Demo output already exists")
    bundle = load_shadow(bundle_dir)
    manifest = bundle.manifest
    if sha256_file(csv) != manifest.data_sha256:
        raise ValueError("Demo dataset hash mismatch")
    if sha256_file(source / "manifest.json") != manifest.source_manifest_sha256:
        raise ValueError("Demo source manifest mismatch")
    provenance = json.loads((source / "manifest.json").read_text())
    for name in ("evaluation_scores.npz", "evaluation_decisions.npz"):
        if sha256_file(source / name) != provenance["artifact_hashes"][name]:
            raise ValueError("Demo archive hash mismatch")
    budget = str(manifest.budget)
    with (
        np.load(source / "evaluation_scores.npz") as scores,
        np.load(source / "evaluation_decisions.npz") as decisions,
    ):
        if not np.array_equal(scores["row_indices"], decisions["row_indices"]):
            raise ValueError("Demo archive rows differ")
        baseline = decisions[f"lightgbm__{budget}"]
        candidate = decisions[f"selective_fusion__{budget}"]
        attack = scores["labels"] == 1
        cohorts = {
            "recovered_attack": attack & ~baseline & candidate,
            "added_false_positive": ~attack & ~baseline & candidate,
            "preserved_attack": attack & baseline,
            "preserved_benign": ~attack & ~candidate,
            "persistent_miss": attack & ~candidate,
        }
        selected = [
            (name, int(index))
            for name, mask in cohorts.items()
            for index in np.flatnonzero(mask)[:3]
        ]
        indices = [int(scores["row_indices"][i]) for _, i in selected]
        expected = [bool(candidate[i]) for _, i in selected]
    raw = pd.read_csv(csv).iloc[indices]
    event_time = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    flows = [row_to_flow(row, str(uuid4()), event_time) for row in raw.to_dict("records")]
    actual = bundle.predict(flows)
    if [r["decision"] == "alert" for r in actual] != expected:
        raise ValueError("Curated fixture failed serving parity")
    evidence = {
        "note": "Deliberately selected archived examples, not representative performance. "
        "Event timestamps mark fixture creation, not packet capture time.",
        "bundle_version": manifest.bundle_version,
        "manifest_sha256": bundle.manifest_sha256,
        "data_sha256": manifest.data_sha256,
        "cases": [
            {
                "cohort": name,
                "source_row": row_index,
                "flow_id": flow.flow_id,
                "expected_shadow": result,
            }
            for (name, _), row_index, flow, result in zip(
                selected, indices, flows, actual, strict=True
            )
        ],
    }
    output.mkdir(parents=True)
    payload = {
        "schema_version": "unsw-nb15.v0",
        "flows": [flow.model_dump(mode="json") for flow in flows],
    }
    truth = {
        "items": [
            {"flow_id": flow.flow_id, "label": int(row["label"]), "attack_cat": row["attack_cat"]}
            for flow, row in zip(flows, raw.to_dict("records"), strict=True)
        ]
    }
    for name, value in (
        ("predictions.json", payload),
        ("truth.json", truth),
        ("evidence.json", evidence),
    ):
        (output / name).write_text(json.dumps(value, indent=2) + "\n")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    evidence = build_demo(args.bundle, args.source, args.csv, args.output)
    print(
        json.dumps(
            {"output": str(args.output), "cases": len(evidence["cases"]), "note": evidence["note"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
