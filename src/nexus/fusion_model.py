"""Load the selected frozen binary/AE packages without training-module imports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

MODE_NAMES = np.array(["no_anomaly", "low", "mid", "high"])


def decide(probability, error, config):
    """Apply exact saved boundaries; equality enters the higher anomaly mode."""
    p, error = np.asarray(probability, dtype=float), np.asarray(error, dtype=float)
    if (
        p.shape != error.shape
        or p.ndim != 1
        or not np.isfinite(p).all()
        or not np.isfinite(error).all()
        or np.any((p < 0) | (p > 1))
    ):
        raise ValueError("Expected aligned finite 1-D scores with probabilities in [0, 1]")
    bounds = np.asarray(config["ae"]["boundaries"], dtype=float)
    if bounds.shape != (3,) or not np.isfinite(bounds).all() or not np.all(np.diff(bounds) > 0):
        raise ValueError("Expected three strictly increasing AE boundaries")
    modes = np.searchsorted(bounds, error, side="right")
    binary = p >= config["binary"]["decision_threshold"]
    rule = config["decision_rule"]
    if rule["kind"] == "uncertainty_band_four_mode":
        lower, upper = rule["confidence_band"]
        inside = (p >= lower) & (p <= upper)
        hard = (modes >= 2) | ((modes == 1) & binary)
        final = np.where(inside, hard, binary)
    elif rule["kind"] == "mode_confidence":
        final = p >= np.asarray(rule["mode_attack_cutoffs"])[modes]
    else:
        raise ValueError(f"Unknown fusion rule: {rule['kind']}")
    return final.astype(np.int64), modes, binary.astype(np.int64)


class FrozenFusionModel:
    """Raw UNSW-NB15 DataFrame -> binary labels and inspectable detector scores."""

    def __init__(self, weights, config):
        self.weights = weights
        self.config = config
        self.classes_ = np.array([0, 1])
        self.feature_names_in_ = np.asarray(config["feature_columns"])

    def _frame(self, frame):
        if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
            raise ValueError("Input must be a DataFrame with unique columns")
        missing = set(self.feature_names_in_) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing flow features: {sorted(missing)}")
        # Extra columns such as id/label/attack_cat are excluded from prediction.
        return frame.loc[:, self.feature_names_in_].copy()

    def _binary_inputs(self, frame):
        w = self.weights["binary"]
        native = frame.copy()
        for column in w["columns"]:
            if column in w["categories"]:
                values = native[column].astype("string")
                values = values.where(values.isin(w["categories"][column]))
                native[column] = pd.Categorical(values, categories=w["categories"][column])
            else:
                native[column] = (
                    pd.to_numeric(native[column], errors="raise")
                    .replace([np.inf, -np.inf], np.nan)
                    .astype(np.float32)
                )
        numeric = sparse.csr_matrix(native[w["numeric"]].to_numpy(dtype=np.float32))
        categorical = native[w["categorical"]].astype(object)
        categorical = categorical.where(categorical.notna(), None)
        return sparse.hstack([numeric, w["encoder"].transform(categorical)], format="csr")

    def _ae_inputs(self, frame):
        w = self.weights["ae"]
        # Match training's float32 cast before numeric standardization.
        numeric = w["imputer"].transform(frame[w["numerical_columns"]]).astype(np.float32)
        numeric = w["scaler"].transform(numeric).astype(np.float32)
        categorical = w["encoder"].transform(
            frame[w["categorical_columns"]].fillna("__MISSING__").astype(str)
        )
        return sparse.hstack([sparse.csr_matrix(numeric), categorical], format="csr")

    def predict_details(self, frame):
        frame = self._frame(frame)
        if not len(frame):
            return pd.DataFrame(
                index=frame.index,
                columns=[
                    "prediction",
                    "lightgbm_attack_score",
                    "binary_prediction",
                    "ae_error",
                    "ae_mode",
                ],
            )
        p = self.weights["binary"]["estimator"].predict_proba(self._binary_inputs(frame))[:, 1]
        inputs = self._ae_inputs(frame)
        errors = np.empty(len(frame), dtype=np.float64)
        batch_size = self.config["inference_batch_size"]
        for start in range(0, len(frame), batch_size):
            stop = min(start + batch_size, len(frame))
            batch = inputs[start:stop]
            reconstructed = self.weights["ae"]["model"].predict(batch)
            errors[start:stop] = np.square(reconstructed - batch.toarray()).mean(axis=1)
        pred, modes, binary = decide(p, errors, self.config)
        return pd.DataFrame(
            dict(
                prediction=pred,
                lightgbm_attack_score=p,
                binary_prediction=binary,
                ae_error=errors,
                ae_mode=MODE_NAMES[modes],
            ),
            index=frame.index,
        )

    def predict(self, frame):
        return self.predict_details(frame)["prediction"].to_numpy(dtype=np.int64)


def load_fusion_model(directory):
    """Verify a local package manifest and load its fitted estimators/preprocessors."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    for name in ["config.json", "weights.joblib", "metrics.json"]:
        expected = manifest["sha256"][name]
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if expected != actual:
            raise ValueError(f"Model package integrity check failed: {name}")
    config = json.loads((directory / "config.json").read_text())
    if config["format_version"] != 1:
        raise ValueError("Unsupported model package format")
    return FrozenFusionModel(joblib.load(directory / "weights.joblib"), config)
