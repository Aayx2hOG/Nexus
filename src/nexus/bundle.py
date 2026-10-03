"""Model release bundle loading and cryptographic integrity verification."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = PROJECT_ROOT / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class BundleManifest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    bundle_version: str
    code_version: str
    feature_columns: list[str]
    dtypes: dict[str, str]
    categorical_columns: list[str]
    decision_threshold: float
    calibration_provenance: str
    validated_metrics: dict[str, Any]
    data_sha256: str
    split_sha256: str
    source_hashes: dict[str, str]
    artifact_hashes: dict[str, str]
    limits: dict[str, Any]


@dataclass
class LoadedBundle:
    manifest: BundleManifest
    bundle_dir: Path
    model: Any
    preprocessor: Any
    decision_threshold: float
    feature_columns: list[str]
    categorical_columns: list[str]


class BundleLoader:
    def __init__(
        self,
        bundles_dir: Path,
        version: str = "v1.0.0",
        project_root: Path = PROJECT_ROOT,
    ) -> None:
        self.bundles_dir = bundles_dir
        self.version = version
        self.project_root = project_root
        self.loaded_bundle: LoadedBundle | None = None
        self.load_error: str | None = None

    @property
    def is_ready(self) -> bool:
        return self.loaded_bundle is not None

    def load(self) -> LoadedBundle:
        bundle_dir = self.bundles_dir / self.version
        manifest_path = bundle_dir / "manifest.json"

        if not manifest_path.is_file():
            err = f"Bundle manifest not found at {manifest_path}"
            self.load_error = err
            logger.error(err)
            raise FileNotFoundError(err)

        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            manifest = BundleManifest.model_validate(manifest_data)
        except Exception as exc:
            err = f"Failed to parse bundle manifest: {exc}"
            self.load_error = err
            logger.error(err)
            raise ValueError(err) from exc

        # 1. Verify artifact hashes
        for filename, expected_hash in manifest.artifact_hashes.items():
            artifact_path = bundle_dir / filename
            if not artifact_path.is_file():
                err = f"Bundle artifact missing: {filename}"
                self.load_error = err
                logger.error(err)
                raise FileNotFoundError(err)
            computed_hash = sha256_file(artifact_path)
            if computed_hash != expected_hash:
                err = (
                    f"Bundle artifact hash mismatch for {filename}: "
                    f"{computed_hash} != {expected_hash}"
                )
                self.load_error = err
                logger.error(err)
                raise ValueError(err)

        # 2. Verify source hashes against project_root / "training"
        training_dir = self.project_root / "training"
        for src_name, expected_src_hash in manifest.source_hashes.items():
            src_path = training_dir / src_name
            if src_path.is_file():
                computed_src_hash = sha256_file(src_path)
                if computed_src_hash != expected_src_hash:
                    err = (
                        f"Source code hash mismatch for {src_name}: "
                        f"{computed_src_hash} != {expected_src_hash}"
                    )
                    self.load_error = err
                    logger.error(err)
                    raise ValueError(err)

        # 3. Load artifacts
        try:
            model = joblib.load(bundle_dir / "model.joblib")
            preprocessor = joblib.load(bundle_dir / "preprocessor.joblib")
        except Exception as exc:
            err = f"Failed to load bundle artifacts into memory: {exc}"
            self.load_error = err
            logger.error(err)
            raise RuntimeError(err) from exc

        self.loaded_bundle = LoadedBundle(
            manifest=manifest,
            bundle_dir=bundle_dir,
            model=model,
            preprocessor=preprocessor,
            decision_threshold=manifest.decision_threshold,
            feature_columns=manifest.feature_columns,
            categorical_columns=manifest.categorical_columns,
        )
        self.load_error = None
        logger.info(f"Loaded and verified release bundle {self.version} successfully.")
        return self.loaded_bundle
