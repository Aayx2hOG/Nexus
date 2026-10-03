"""Inference and serving parity test against offline training/evaluation path."""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "training"))

from train_validated_lightgbm import canonical_features  # noqa: E402
from tune_lightgbm_fast import read_data  # noqa: E402

from nexus.bundle import BundleLoader  # noqa: E402
from nexus.inference import flows_to_dataframe, predict_batch  # noqa: E402
from nexus.schemas import Flow, FlowFeatures  # noqa: E402

BUNDLES_DIR = PROJECT_ROOT / "artifacts" / "bundles"
CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "CSV_Files"
    / "Training and Testing Sets"
    / "UNSW_NB15_testing-set.csv"
)


@pytest.fixture(scope="module")
def bundle():
    loader = BundleLoader(BUNDLES_DIR, "v1.0.0")
    return loader.load()


def test_serving_offline_parity(bundle):
    if not CSV_PATH.is_file():
        pytest.skip(f"CSV dataset not available at {CSV_PATH}")

    raw, y, _ = read_data(CSV_PATH)
    raw = canonical_features(raw)

    splits_path = PROJECT_ROOT / "experiments" / "lightgbm_validated_v1" / "split_indices.npz"
    splits = np.load(splits_path)
    sel = splits["selection"][:100]  # Take 100 selection rows

    sample_df = raw.iloc[sel].copy()

    # 1. Offline path
    offline_X = bundle.preprocessor.transform(sample_df)
    offline_scores = bundle.model.predict_proba(offline_X)[:, 1]
    offline_decisions = [
        "alert" if s >= bundle.decision_threshold else "normal" for s in offline_scores
    ]

    # 2. Serving path: convert each row to a Flow schema instance
    int_fields = {
        "spkts",
        "dpkts",
        "sbytes",
        "dbytes",
        "sttl",
        "dttl",
        "sloss",
        "dloss",
        "swin",
        "stcpb",
        "dtcpb",
        "dwin",
        "smean",
        "dmean",
        "trans_depth",
        "response_body_len",
        "ct_srv_src",
        "ct_state_ttl",
        "ct_dst_ltm",
        "ct_src_dport_ltm",
        "ct_dst_sport_ltm",
        "ct_dst_src_ltm",
        "is_ftp_login",
        "ct_ftp_cmd",
        "ct_flw_http_mthd",
        "ct_src_ltm",
        "ct_srv_dst",
        "is_sm_ips_ports",
    }
    flows = []
    for _, row in sample_df.iterrows():
        feat_dict = {}
        for col in bundle.feature_columns:
            if col in bundle.categorical_columns:
                feat_dict[col] = str(row[col])
            elif col in int_fields:
                feat_dict[col] = int(round(float(row[col])))
            else:
                feat_dict[col] = float(row[col])

        flow = Flow(
            flow_id=str(uuid4()),
            event_time="2026-10-02T10:00:00Z",
            features=FlowFeatures(**feat_dict),
        )
        flows.append(flow)

    serving_df = flows_to_dataframe(flows, bundle.feature_columns)
    serving_X = bundle.preprocessor.transform(serving_df)
    predictions = predict_batch(bundle, flows)

    # 3. Parity assertions
    np.testing.assert_allclose(
        serving_X.toarray(),
        offline_X.toarray(),
        atol=1e-6,
        err_msg="Serving transformed feature matrix does not match offline matrix!",
    )

    serving_scores = np.array([p.score for p in predictions])
    np.testing.assert_allclose(
        serving_scores,
        offline_scores,
        atol=1e-6,
        err_msg="Serving prediction scores do not match offline scores!",
    )

    serving_decisions = [p.decision for p in predictions]
    assert serving_decisions == offline_decisions, "Serving decisions do not match offline!"

    # 4. Check TreeSHAP explanations for alerted flows
    alerted_predictions = [p for p in predictions if p.decision == "alert"]
    assert len(alerted_predictions) > 0, "Expected at least one alert in 100 sample flows"

    for alert_pred in alerted_predictions:
        assert len(alert_pred.top_features) == 10
        # Verify ranking: absolute contributions must be monotonically non-increasing
        abs_contribs = [abs(f.contribution) for f in alert_pred.top_features]
        assert abs_contribs == sorted(abs_contribs, reverse=True)
        # Verify feature names are distinct
        names = [f.feature for f in alert_pred.top_features]
        assert len(names) == len(set(names))
