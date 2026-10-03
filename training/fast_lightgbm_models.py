"""Serializable raw-frame preprocessing and a gated LightGBM classifier."""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder


class NativeFeatures:
    """Freeze categorical vocabularies on fitting rows; LightGBM handles numeric NaNs."""

    def fit(self, frame):
        self.columns = list(frame.columns)
        self.categories = {
            column: sorted(frame[column].dropna().astype(str).unique().tolist())
            for column in frame
            if not pd.api.types.is_numeric_dtype(frame[column])
        }
        return self

    def transform(self, frame):
        result = frame.loc[:, self.columns].copy()
        for column in self.columns:
            if column in self.categories:
                values = result[column].astype("string")
                values = values.where(values.isin(self.categories[column]))
                result[column] = pd.Categorical(values, categories=self.categories[column])
            else:
                result[column] = (
                    pd.to_numeric(result[column], errors="raise")
                    .replace([np.inf, -np.inf], np.nan)
                    .astype(np.float32)
                )
        return result


class NativeModel:
    """Accept raw feature DataFrames and preserve the model's decision rule."""

    def __init__(self, features, estimator):
        self.features = features
        self.estimator = estimator
        self.classes_ = estimator.classes_
        if hasattr(estimator, "decision_threshold_"):
            self.decision_threshold_ = estimator.decision_threshold_

    def predict_proba(self, frame):
        return self.estimator.predict_proba(self.features.transform(frame))

    def predict(self, frame):
        probabilities = self.predict_proba(frame)
        if hasattr(self, "decision_threshold_"):
            return (probabilities[:, 1] >= self.decision_threshold_).astype(int)
        return self.estimator.predict(self.features.transform(frame))


class OneHotFeatures:
    """Same numeric processing as NativeFeatures, with fit-only one-hot categories."""

    def fit(self, frame):
        self.native = NativeFeatures().fit(frame)
        self.categorical = list(self.native.categories)
        self.numeric = [c for c in self.native.columns if c not in self.categorical]
        self.encoder = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
        if self.categorical:
            self.encoder.fit(self._categories(self.native.transform(frame)))
        return self

    def _categories(self, frame):
        values = frame[self.categorical].astype(object)
        return values.where(values.notna(), None)

    def transform(self, frame):
        native = self.native.transform(frame)
        numeric = sparse.csr_matrix(native[self.numeric].to_numpy(dtype=np.float32))
        if not self.categorical:
            return numeric
        return sparse.hstack(
            [numeric, self.encoder.transform(self._categories(native))], format="csr"
        )


class AttackCascade:
    """Normal=0; a binary gate followed by a model trained only on attack families."""

    def __init__(self, gate, attack_model, threshold, class_count):
        self.gate = gate
        self.attack_model = attack_model
        self.decision_threshold = float(threshold)
        self.classes_ = np.arange(class_count)
        if not np.array_equal(attack_model.classes_, np.arange(class_count - 1)):
            raise ValueError("Attack model classes must be contiguous IDs starting at zero")

    def predict_proba(self, frame):
        attack_probability = self.gate.predict_proba(frame)[:, 1]
        family_probability = self.attack_model.predict_proba(frame)
        return np.column_stack(
            [1 - attack_probability, attack_probability[:, None] * family_probability]
        )

    def predict(self, frame):
        attack = self.gate.predict_proba(frame)[:, 1] >= self.decision_threshold
        family = self.attack_model.predict_proba(frame).argmax(axis=1) + 1
        return np.where(attack, family, 0)
