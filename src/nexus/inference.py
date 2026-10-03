"""Serving inference engine with TreeSHAP explanations and preprocessing parity."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from nexus.alerts import FeatureContribution
from nexus.bundle import LoadedBundle
from nexus.schemas import Flow

logger = logging.getLogger(__name__)


@dataclass
class FlowPrediction:
    flow_id: str
    event_time: str
    score: float
    threshold: float
    decision: str  # "alert" or "normal"
    top_features: list[FeatureContribution]
    raw_features: dict[str, Any]


def flows_to_dataframe(flows: list[Flow], feature_columns: list[str]) -> pd.DataFrame:
    """Extract raw flow features into a DataFrame matching training schema and dtypes."""
    rows = []
    for flow in flows:
        feat_dict = flow.features.model_dump()
        rows.append(feat_dict)
    frame = pd.DataFrame(rows)
    # Ensure all feature columns exist and are ordered identically to training
    frame = frame.loc[:, feature_columns].copy()
    for col in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[col]):
            frame[col] = frame[col].astype("string")
        else:
            frame[col] = frame[col].astype(np.float64)
    return frame


def get_transformed_feature_names(preprocessor: Any) -> list[str]:
    """Retrieve column names of the transformed CSR matrix from OneHotFeatures."""
    numeric_names = list(preprocessor.numeric)
    cat_names = preprocessor.encoder.get_feature_names_out(preprocessor.categorical).tolist()
    return numeric_names + cat_names


def explain_alert_tree_shap(
    booster: Any,
    X_row: sparse.csr_matrix,
    transformed_feature_names: list[str],
    flow_features: dict[str, Any],
    top_k: int = 10,
) -> list[FeatureContribution]:
    """Compute TreeSHAP contributions for an alerted flow row via LightGBM.

    Returns the top-k features ranked by absolute contribution |contribution|,
    with feature name, actual feature value, and signed contribution.
    """
    # booster.predict with pred_contrib=True returns (1, n_features + 1)
    contribs = booster.predict(X_row, pred_contrib=True)
    if sparse.issparse(contribs):
        contrib_row = contribs.toarray()[0, :-1]  # Exclude expected value (bias)
    else:
        contrib_row = contribs[0, :-1]

    # Rank by |contribution| descending
    abs_indices = np.argsort(-np.abs(contrib_row))
    top_indices = abs_indices[:top_k]

    contributions = []
    for idx in top_indices:
        feat_name = transformed_feature_names[idx]
        signed_contrib = float(contrib_row[idx])

        # Resolve display value:
        # If feat_name is in raw features (numeric), use its raw value
        if feat_name in flow_features:
            val = flow_features[feat_name]
        else:
            # For one-hot feature like proto_tcp, check CSR matrix value
            val = float(X_row[0, idx])

        contributions.append(
            FeatureContribution(
                feature=feat_name,
                contribution=signed_contrib,
                value=val if isinstance(val, (int, float, str)) else float(val),
            )
        )
    return contributions


def predict_batch(
    bundle: LoadedBundle,
    flows: list[Flow],
) -> list[FlowPrediction]:
    """Score a batch of flows using the loaded release bundle and return predictions."""
    if not flows:
        return []

    frame = flows_to_dataframe(flows, bundle.feature_columns)
    X = bundle.preprocessor.transform(frame)

    scores = bundle.model.predict_proba(X)[:, 1]
    threshold = float(bundle.decision_threshold)
    decisions = ["alert" if s >= threshold else "normal" for s in scores]

    transformed_feature_names = get_transformed_feature_names(bundle.preprocessor)

    predictions = []
    for i, flow in enumerate(flows):
        score = float(scores[i])
        decision = decisions[i]
        top_features: list[FeatureContribution] = []

        if decision == "alert":
            X_row = X[i : i + 1]
            raw_dict = flow.features.model_dump()
            top_features = explain_alert_tree_shap(
                booster=bundle.model.booster_,
                X_row=X_row,
                transformed_feature_names=transformed_feature_names,
                flow_features=raw_dict,
                top_k=10,
            )

        predictions.append(
            FlowPrediction(
                flow_id=flow.flow_id,
                event_time=flow.event_time,
                score=score,
                threshold=threshold,
                decision=decision,
                top_features=top_features,
                raw_features=flow.features.model_dump(),
            )
        )

    return predictions
