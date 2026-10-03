"""Preprocessing, reconstruction, and latent-distance anomaly detectors."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, RobustScaler


class AnomalyFeatures:
    """Fit on benign fitting rows; balance numeric and categorical loss blocks.

    Log transforms apply only to predeclared nonnegative traffic measurements.
    Unknown categorical values get an explicit extra indicator per source field.
    """

    LOG_COLUMNS = {
        "dur",
        "spkts",
        "dpkts",
        "sbytes",
        "dbytes",
        "rate",
        "sload",
        "dload",
        "sloss",
        "dloss",
        "sinpkt",
        "dinpkt",
        "sjit",
        "djit",
        "tcprtt",
        "synack",
        "ackdat",
        "smean",
        "dmean",
        "response_body_len",
        "trans_depth",
    }

    def __init__(self, improved=True):
        self.improved = improved

    def _numeric(self, frame):
        values = frame[self.numeric].apply(pd.to_numeric, errors="raise").copy()
        values = values.replace([np.inf, -np.inf], np.nan)
        if self.improved:
            for name in set(self.numeric) & self.LOG_COLUMNS:
                if (values[name].dropna() < 0).any():
                    raise ValueError(f"Negative value in nonnegative feature {name}")
                values[name] = np.log1p(values[name])
        return values

    def fit(self, frame):
        self.columns = list(frame.columns)
        self.categorical = [c for c in frame if not pd.api.types.is_numeric_dtype(frame[c])]
        self.numeric = [c for c in frame if c not in self.categorical]
        values = self._numeric(frame)
        self.medians = values.median().fillna(0)
        if self.improved:
            self.scaler = RobustScaler()
        else:
            from sklearn.preprocessing import StandardScaler

            self.scaler = StandardScaler()
        self.scaler.fit(values.fillna(self.medians))
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        if self.categorical:
            self.encoder.fit(frame[self.categorical].fillna("__MISSING__").astype(str))
        return self

    def transform(self, frame):
        numeric = self.scaler.transform(self._numeric(frame).fillna(self.medians))
        if self.improved:
            numeric = numeric / np.sqrt(max(1, len(self.numeric)))
        parts = [numeric]
        if self.categorical:
            categories = frame[self.categorical].fillna("__MISSING__").astype(str)
            encoded = self.encoder.transform(categories)
            unknown = np.column_stack(
                [
                    ~categories[c].isin(vocabulary).to_numpy()
                    for c, vocabulary in zip(
                        self.categorical, self.encoder.categories_, strict=True
                    )
                ]
            )
            if self.improved:
                encoded = encoded / np.sqrt(len(self.categorical))
                unknown = unknown / np.sqrt(len(self.categorical))
            parts.extend([encoded, unknown])
        result = np.hstack(parts).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Anomaly features contain non-finite values")
        return result


class Autoencoder:
    """Compact AE with optional input masking and normal-only early stopping."""

    def __init__(self, seed=42, epochs=40, patience=5, batch_size=512, noise=0.0):
        self.seed = seed
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.noise = noise

    def fit(self, fit, early):
        rng = np.random.default_rng(self.seed)
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 16, 64),
            activation="relu",
            solver="adam",
            learning_rate_init=1e-3,
            alpha=1e-4,
            max_iter=1,
            random_state=self.seed,
            shuffle=False,
        )
        best_loss, stale = float("inf"), 0
        self.history = []
        best = None
        for epoch in range(self.epochs):
            order = rng.permutation(len(fit))
            for start in range(0, len(fit), self.batch_size):
                clean = fit[order[start : start + self.batch_size]]
                noisy = clean * (rng.random(clean.shape) >= self.noise)
                self.model.batch_size = len(clean)
                self.model.partial_fit(noisy, clean)
            loss = float(self.score(early).mean())
            if not np.isfinite(loss):
                raise ValueError("Autoencoder diverged; inspect input scaling")
            self.history.append({"epoch": epoch + 1, "normal_early_mse": loss})
            print(
                f"  AE epoch {epoch + 1}/{self.epochs}: normal validation MSE={loss:.8g}",
                flush=True,
            )
            if loss < best_loss:
                best_loss, stale = loss, 0
                best = copy.deepcopy(self.model)
                self.best_epoch = epoch + 1
            else:
                stale += 1
            if stale >= self.patience:
                break
        if best is None:
            raise ValueError("Autoencoder did not complete any epochs")
        self.model = best
        self.reference = LedoitWolf().fit(self.encode(fit))
        return self

    def encode(self, values):
        hidden = values
        for weight, bias in zip(self.model.coefs_[:2], self.model.intercepts_[:2], strict=True):
            hidden = np.maximum(hidden @ weight + bias, 0)
        return hidden

    def score(self, values, latent=False):
        output = []
        for start in range(0, len(values), self.batch_size):
            batch = values[start : start + self.batch_size]
            if latent:
                scores = self.reference.mahalanobis(self.encode(batch))
            else:
                scores = np.square(self.model.predict(batch) - batch).mean(axis=1)
            output.append(scores)
        return np.concatenate(output)


def benign_rank(reference, scores):
    """Empirical benign percentile; this is not an attack probability."""
    return np.searchsorted(np.sort(reference), scores, side="right") / len(reference)


def fpr_threshold(benign_scores, budget):
    """Conservative empirical cutoff, including ties, for a >= decision rule."""
    values = np.asarray(benign_scores, dtype=float)
    if not len(values) or not np.isfinite(values).all() or not 0 < budget < 1:
        raise ValueError("Need finite benign scores and an FPR budget in (0, 1)")
    allowed = int(np.floor(budget * len(values)))
    boundary = np.sort(values)[len(values) - allowed - 1]
    return float(np.nextafter(boundary, np.inf))
