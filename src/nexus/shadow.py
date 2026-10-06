"""Frozen selective-fusion shadow inference, independent of operational decisions."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from nexus.bundle import PROJECT_ROOT, sha256_file
from nexus.inference import flows_to_dataframe
from nexus.schemas import Flow, FlowFeatures

# Bundle imports establish the trusted local training module path for joblib.
from anomaly_detection_models import benign_rank  # isort: skip
from complementary_fusion import SelectiveFusion  # isort: skip

INFERENCE_SOURCES = (
    "anomaly_detection_models.py",
    "complementary_fusion.py",
    "fast_lightgbm_models.py",
)


SERVING_SOURCES = ("src/nexus/shadow.py", "src/nexus/inference.py", "src/nexus/schemas.py")


class ShadowManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    format_version: int = Field(default=1, ge=1, le=1)
    mode: str = Field(pattern="^shadow$")
    bundle_version: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    feature_columns: list[str]
    budget: float = Field(gt=0, lt=1)
    held_family: str
    seed: int
    data_sha256: str
    split_sha256: str
    source_manifest_sha256: str
    artifact_hashes: dict[str, str]
    source_hashes: dict[str, str]
    serving_source_hashes: dict[str, str]
    policy: dict[str, Any]
    evaluation_note: str


class ShadowBundle:
    def __init__(self, manifest: ShadowManifest, checkpoint: dict, manifest_sha256: str):
        self.manifest = manifest
        self.checkpoint = checkpoint
        self.manifest_sha256 = manifest_sha256
        self.policy = SelectiveFusion(**manifest.policy)
        self.parity: dict = {}

    def score_frame(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        saved = self.checkpoint
        frame = frame.loc[:, self.manifest.feature_columns]
        tree = saved["lightgbm"].predict_proba(saved["tree_features"].transform(frame))[:, 1]
        features, ae = saved["anomaly_models"]["denoising"]
        values = features.transform(frame)
        reconstruction = ae.score(values)
        latent = ae.score(values, latent=True)
        fusion = saved["fusion"].predict_proba(
            np.column_stack([tree, np.log1p(reconstruction), np.log1p(latent)])
        )[:, 1]
        rank = benign_rank(saved["rank_references"]["ae_denoising"], reconstruction)
        scores = {
            "lightgbm": tree,
            "ae_denoising": reconstruction,
            "ae_latent": latent,
            "learned_fusion": fusion,
            "anomaly_rank": rank,
        }
        if not all(np.isfinite(v).all() and len(v) == len(frame) for v in scores.values()):
            raise ValueError("Invalid shadow scores")
        scores["baseline"] = tree >= self.policy.baseline_threshold
        scores["eligible"] = self.policy.route(tree, rank)
        scores["candidate"] = self.policy.predict(tree, fusion, rank)
        if np.any(scores["baseline"] & ~scores["candidate"]):
            raise ValueError("Shadow policy removed baseline detections")
        return scores

    def predict(self, flows: list[Flow]) -> list[dict]:
        values = self.score_frame(flows_to_dataframe(flows, self.manifest.feature_columns))
        result = []
        for i, flow in enumerate(flows):
            baseline, candidate = bool(values["baseline"][i]), bool(values["candidate"][i])
            result.append(
                {
                    "flow_id": flow.flow_id,
                    "status": "scored",
                    "reference_score": float(values["lightgbm"][i]),
                    "reference_threshold": self.policy.baseline_threshold,
                    "reference_decision": "alert" if baseline else "normal",
                    "reconstruction_error": float(values["ae_denoising"][i]),
                    "latent_distance": float(values["ae_latent"][i]),
                    "anomaly_rank": float(values["anomaly_rank"][i]),
                    "fusion_score": float(values["learned_fusion"][i]),
                    "recovery_threshold": self.policy.recovery_threshold,
                    "eligible": bool(values["eligible"][i]),
                    "decision": "alert" if candidate else "normal",
                    "reason": "reference_alert"
                    if baseline
                    else ("selective_recovery" if candidate else "kept_normal"),
                }
            )
        return result


def load_shadow(path: Path, project_root: Path = PROJECT_ROOT) -> ShadowBundle:
    manifest = ShadowManifest.model_validate_json((path / "manifest.json").read_text())
    required = {"models.joblib", "calibration.json", "parity.json", "source_manifest.json"}
    if set(manifest.artifact_hashes) != required:
        raise ValueError("Incomplete shadow bundle")
    if set(manifest.source_hashes) != set(INFERENCE_SOURCES):
        raise ValueError("Incomplete inference source provenance")
    for name, expected in manifest.artifact_hashes.items():
        if sha256_file(path / name) != expected:
            raise ValueError(f"Shadow artifact hash mismatch: {name}")
    for name, expected in manifest.source_hashes.items():
        if sha256_file(project_root / "training" / name) != expected:
            raise ValueError(f"Shadow inference source mismatch: {name}")
    if set(manifest.serving_source_hashes) != set(SERVING_SOURCES):
        raise ValueError("Incomplete serving source provenance")
    for name, expected in manifest.serving_source_hashes.items():
        if sha256_file(project_root / name) != expected:
            raise ValueError(f"Shadow serving implementation changed: {name}")
    source = json.loads((path / "source_manifest.json").read_text())
    if sha256_file(path / "source_manifest.json") != manifest.source_manifest_sha256:
        raise ValueError("Shadow source manifest mismatch")
    if any(
        manifest.source_hashes[name] != source["source_hashes"][name] for name in INFERENCE_SOURCES
    ):
        raise ValueError("Shadow inference source differs from frozen research")
    for name in ("models.joblib", "calibration.json"):
        if manifest.artifact_hashes[name] != source["artifact_hashes"][name]:
            raise ValueError("Shadow artifacts differ from frozen research")
    if (
        manifest.data_sha256 != source["data_sha256"]
        or manifest.split_sha256 != source["artifact_hashes"]["split_indices.npz"]
        or manifest.held_family != source["held_family"]
        or manifest.seed != source["seed"]
    ):
        raise ValueError("Shadow research provenance mismatch")
    parity = json.loads((path / "parity.json").read_text())
    if (
        parity.get("status") != "passed"
        or parity.get("rows", 0) < 1
        or parity.get("decision_mismatches") != 0
        or parity.get("lost_reference_detections") != 0
        or parity.get("checkpoint_sha256") != manifest.artifact_hashes["models.joblib"]
        or parity.get("budget") != manifest.budget
        or parity.get("data_sha256") != manifest.data_sha256
        or parity.get("split_sha256") != manifest.split_sha256
    ):
        raise ValueError("Missing matching raw-input parity evidence")
    if set(manifest.feature_columns) != set(FlowFeatures.model_fields) or len(
        manifest.feature_columns
    ) != len(FlowFeatures.model_fields):
        raise ValueError("Shadow flow schema mismatch")
    calibration = json.loads((path / "calibration.json").read_text())
    if calibration["selective_policies"][str(manifest.budget)] != manifest.policy:
        raise ValueError("Shadow calibration policy mismatch")
    saved = joblib.load(path / "models.joblib")
    if (
        saved["selective_policies"][str(manifest.budget)] != manifest.policy
        or list(saved["tree_features"].native.columns) != manifest.feature_columns
        or tuple(saved["fusion_columns"]) != ("lightgbm", "ae_denoising", "ae_latent")
    ):
        raise ValueError("Shadow checkpoint configuration mismatch")
    policy = SelectiveFusion(**manifest.policy)
    if (
        policy.baseline_threshold != saved["thresholds"][str(manifest.budget)]["lightgbm"]
        or not 0 <= policy.uncertain_lower_ratio <= 1
        or not 0 <= policy.suspicious_quantile <= 1
    ):
        raise ValueError("Invalid frozen routing policy")
    if policy.primary_budget_fraction != 1.0 or not all(
        np.isfinite(v)
        for v in (
            policy.baseline_threshold,
            policy.recovery_threshold,
            policy.uncertain_lower_ratio,
            policy.suspicious_quantile,
        )
    ):
        raise ValueError("Only finite baseline-preserving policies support shadow serving")
    bundle = ShadowBundle(manifest, saved, sha256_file(path / "manifest.json"))
    bundle.parity = parity
    return bundle


class ShadowService:
    """Optional bounded synchronous shadow work; failure never changes live decisions."""

    def __init__(self, path: Path | None):
        self.path = path
        self.bundle: ShadowBundle | None = None
        self.load_error: str | None = None
        self.lock = threading.Lock()

    def load(self) -> None:
        self.bundle = None
        self.load_error = None
        if self.path is None:
            return
        try:
            self.bundle = load_shadow(self.path)
        except Exception:
            import logging

            logging.getLogger(__name__).exception("Shadow bundle failed verification")
            self.load_error = "bundle_verification_failed"

    def status(self) -> dict:
        bundle = self.bundle
        return {
            "status": "ready" if bundle else ("unavailable" if self.path else "disabled"),
            "mode": "shadow",
            "policy": bundle.manifest.policy if bundle else None,
            "parity": bundle.parity if bundle else None,
            "error_code": self.load_error,
            "bundle_version": bundle.manifest.bundle_version if bundle else None,
            "manifest_sha256": bundle.manifest_sha256 if bundle else None,
            "budget": bundle.manifest.budget if bundle else None,
            "held_family": bundle.manifest.held_family if bundle else None,
            "seed": bundle.manifest.seed if bundle else None,
            "note": "Research baseline differs from live LightGBM. Shadow cannot create alerts; "
            "its calibration budget does not apply to the live model or future traffic.",
        }

    def evaluate(self, flows: list[Flow]) -> dict:
        status = self.status()
        if self.bundle is None:
            return {**status, "results": []}
        start = perf_counter()
        if not self.lock.acquire(blocking=False):
            results = [
                {"flow_id": f.flow_id, "status": "error", "error_code": "shadow_busy"}
                for f in flows
            ]
        else:
            try:
                results = self.bundle.predict(flows)
            except Exception:
                import logging

                logging.getLogger(__name__).exception("Shadow inference failed")
                results = [
                    {
                        "flow_id": f.flow_id,
                        "status": "error",
                        "error_code": "shadow_inference_failed",
                    }
                    for f in flows
                ]
            finally:
                self.lock.release()
        return {
            **status,
            "status": "scored" if all(r["status"] == "scored" for r in results) else "error",
            "batch_elapsed_ms": (perf_counter() - start) * 1000,
            "results": results,
        }
