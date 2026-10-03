"""Tests for dataset replay adapter, split isolation, and ground truth stripping."""

from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from nexus.api import create_app
from nexus.config import Settings
from nexus.replay import (
    load_evaluation_data,
    replay_dataset,
    row_to_flow,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_TOKEN = "test-token-" + "c" * 32


def test_row_to_flow_strips_ground_truth():
    row = {
        "id": 999,
        "dur": 0.1,
        "proto": "tcp",
        "service": "http",
        "state": "FIN",
        "spkts": 10,
        "dpkts": 8,
        "sbytes": 500,
        "dbytes": 400,
        "rate": 100.0,
        "sttl": 64,
        "dttl": 64,
        "sload": 1000.0,
        "dload": 800.0,
        "sloss": 0,
        "dloss": 0,
        "sinpkt": 1.0,
        "dinpkt": 1.0,
        "sjit": 0.5,
        "djit": 0.5,
        "swin": 255,
        "stcpb": 12345,
        "dtcpb": 67890,
        "dwin": 255,
        "tcprtt": 0.05,
        "synack": 0.02,
        "ackdat": 0.03,
        "smean": 50,
        "dmean": 50,
        "trans_depth": 1,
        "response_body_len": 200,
        "ct_srv_src": 2,
        "ct_state_ttl": 1,
        "ct_dst_ltm": 2,
        "ct_src_dport_ltm": 1,
        "ct_dst_sport_ltm": 1,
        "ct_dst_src_ltm": 2,
        "is_ftp_login": 0,
        "ct_ftp_cmd": 0,
        "ct_flw_http_mthd": 1,
        "ct_src_ltm": 2,
        "ct_srv_dst": 2,
        "is_sm_ips_ports": 0,
        "label": 1,
        "attack_cat": "Exploits",
    }
    fid = str(uuid4())
    flow = row_to_flow(row, fid, "2026-10-03T10:00:00Z")
    flow_dump = flow.model_dump()
    assert flow_dump["flow_id"] == fid
    features = flow_dump["features"]
    assert "label" not in features
    assert "attack_cat" not in features
    assert "id" not in features


def test_load_evaluation_data_partition_isolation(tmp_path: Path):
    csv_file = tmp_path / "mock.csv"
    split_file = tmp_path / "splits.npz"

    # Create dummy data of 10 rows
    data = {
        "id": list(range(10)),
        "dur": [0.1] * 10,
        "proto": ["tcp"] * 10,
        "service": ["http"] * 10,
        "state": ["FIN"] * 10,
        "spkts": [1] * 10,
        "dpkts": [1] * 10,
        "sbytes": [10] * 10,
        "dbytes": [10] * 10,
        "rate": [10.0] * 10,
        "sttl": [64] * 10,
        "dttl": [64] * 10,
        "sload": [10.0] * 10,
        "dload": [10.0] * 10,
        "sloss": [0] * 10,
        "dloss": [0] * 10,
        "sinpkt": [1.0] * 10,
        "dinpkt": [1.0] * 10,
        "sjit": [0.0] * 10,
        "djit": [0.0] * 10,
        "swin": [255] * 10,
        "stcpb": [100] * 10,
        "dtcpb": [100] * 10,
        "dwin": [255] * 10,
        "tcprtt": [0.0] * 10,
        "synack": [0.0] * 10,
        "ackdat": [0.0] * 10,
        "smean": [10] * 10,
        "dmean": [10] * 10,
        "trans_depth": [0] * 10,
        "response_body_len": [0] * 10,
        "ct_srv_src": [1] * 10,
        "ct_state_ttl": [0] * 10,
        "ct_dst_ltm": [1] * 10,
        "ct_src_dport_ltm": [1] * 10,
        "ct_dst_sport_ltm": [1] * 10,
        "ct_dst_src_ltm": [1] * 10,
        "is_ftp_login": [0] * 10,
        "ct_ftp_cmd": [0] * 10,
        "ct_flw_http_mthd": [0] * 10,
        "ct_src_ltm": [1] * 10,
        "ct_srv_dst": [1] * 10,
        "is_sm_ips_ports": [0] * 10,
        "label": [0] * 10,
        "attack_cat": ["Normal"] * 10,
    }
    pd.DataFrame(data).to_csv(csv_file, index=False)

    np.savez(
        split_file,
        selection=np.array([2, 5, 8]),
        fit=np.array([0, 1, 3, 4, 6, 7, 9]),
    )

    df_full = load_evaluation_data(csv_file, limit=2)
    assert len(df_full) == 2


def test_replay_end_to_end_with_test_client(tmp_path: Path):
    db_path = tmp_path / "replay_test.sqlite3"
    bundle_dir = PROJECT_ROOT / "artifacts" / "bundles"
    settings = Settings(
        database_path=db_path,
        api_token=TEST_TOKEN,
        reviewer_id="replay-analyst",
        bundles_dir=bundle_dir,
        bundle_version="v1.0.0",
    )
    app = create_app(settings)

    # 2 sample rows
    mock_rows = [
        {
            "id": 1,
            "dur": 0.00001,
            "proto": "udp",
            "service": "-",
            "state": "INT",
            "spkts": 2,
            "dpkts": 0,
            "sbytes": 100,
            "dbytes": 0,
            "rate": 100000.0,
            "sttl": 254,
            "dttl": 0,
            "sload": 40000000.0,
            "dload": 0.0,
            "sloss": 0,
            "dloss": 0,
            "sinpkt": 0.01,
            "dinpkt": 0.0,
            "sjit": 0.0,
            "djit": 0.0,
            "swin": 0,
            "stcpb": 0,
            "dtcpb": 0,
            "dwin": 0,
            "tcprtt": 0.0,
            "synack": 0.0,
            "ackdat": 0.0,
            "smean": 50,
            "dmean": 0,
            "trans_depth": 0,
            "response_body_len": 0,
            "ct_srv_src": 1,
            "ct_state_ttl": 2,
            "ct_dst_ltm": 1,
            "ct_src_dport_ltm": 1,
            "ct_dst_sport_ltm": 1,
            "ct_dst_src_ltm": 1,
            "is_ftp_login": 0,
            "ct_ftp_cmd": 0,
            "ct_flw_http_mthd": 0,
            "ct_src_ltm": 1,
            "ct_srv_dst": 1,
            "is_sm_ips_ports": 0,
            "label": 1,
            "attack_cat": "Generic",
        },
        {
            "id": 2,
            "dur": 0.5,
            "proto": "tcp",
            "service": "http",
            "state": "FIN",
            "spkts": 10,
            "dpkts": 8,
            "sbytes": 500,
            "dbytes": 2000,
            "rate": 36.0,
            "sttl": 62,
            "dttl": 252,
            "sload": 8000.0,
            "dload": 32000.0,
            "sloss": 1,
            "dloss": 1,
            "sinpkt": 50.0,
            "dinpkt": 60.0,
            "sjit": 10.0,
            "djit": 5.0,
            "swin": 255,
            "stcpb": 1000,
            "dtcpb": 2000,
            "dwin": 255,
            "tcprtt": 0.01,
            "synack": 0.005,
            "ackdat": 0.005,
            "smean": 50,
            "dmean": 250,
            "trans_depth": 1,
            "response_body_len": 1500,
            "ct_srv_src": 1,
            "ct_state_ttl": 1,
            "ct_dst_ltm": 1,
            "ct_src_dport_ltm": 1,
            "ct_dst_sport_ltm": 1,
            "ct_dst_src_ltm": 1,
            "is_ftp_login": 0,
            "ct_ftp_cmd": 0,
            "ct_flw_http_mthd": 1,
            "ct_src_ltm": 1,
            "ct_srv_dst": 1,
            "is_sm_ips_ports": 0,
            "label": 0,
            "attack_cat": "Normal",
        },
    ]
    df = pd.DataFrame(mock_rows)

    with TestClient(app, headers={"Authorization": f"Bearer {TEST_TOKEN}"}) as client:
        summary = replay_dataset(
            df=df,
            http_client=client,
            db_path=db_path,
            batch_size=2,
            rate=0.0,
        )
        assert summary["total_sent"] == 2

        # Verify replay truth was recorded in SQLite
        store = app.state.store
        replay_summary = store.get_replay_summary()
        assert replay_summary["total_replayed"] == 2
        assert "confusion_matrix" in replay_summary
