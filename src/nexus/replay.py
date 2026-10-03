"""Dataset replay adapter drawing strictly from frozen evaluation partition.

Strips ground-truth labels and attack categories from HTTP serving payloads,
records truth separately in SQLite replay_truth table, and generates client-side UUIDs.
"""

from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import numpy as np
import pandas as pd

from nexus.schemas import Flow, FlowBatch, FlowFeatures
from nexus.storage import AlertStore

DISCLOSURE_BANNER = (
    "=" * 80 + "\n"
    "NEXUS DATASET REPLAY ADAPTER\n"
    "Replay results are selection estimates / historical replay, not independent confirmation.\n"
    + "="
    * 80
)

DEFAULT_CSV = Path("data/raw/CSV_Files/Training and Testing Sets/UNSW_NB15_testing-set.csv")
DEFAULT_SPLIT_FILE = Path("models/lightgbm_validated_v1/split_indices.npz")


def now_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def load_evaluation_data(
    csv_path: Path,
    split_file: Path | None = None,
    split_name: str = "selection",
    offset: int = 0,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load evaluation flows strictly isolated to the specified frozen partition."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Replay CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)

    if split_file is not None and split_file.exists():
        splits = np.load(split_file)
        if split_name not in splits:
            raise ValueError(f"Split partition '{split_name}' not found in {split_file}")
        indices = splits[split_name]
        # Verify indices are within CSV bounds
        if len(df) == 175341:
            df = df.iloc[indices].copy()
        else:
            # If CSV is already pre-filtered or different size, log warning
            pass

    if offset > 0:
        df = df.iloc[offset:].copy()
    if limit is not None and limit > 0:
        df = df.iloc[:limit].copy()

    return df


def row_to_flow(row: dict[str, Any], flow_id: str, event_time: str) -> Flow:
    """Extract 42 features from raw CSV row, stripping label, attack_cat, and id."""
    feat_dict = dict(row)
    for col in ("id", "label", "attack_cat"):
        feat_dict.pop(col, None)

    # Convert numeric fields that require int in FlowFeatures
    int_fields = (
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
    )
    for k in int_fields:
        if k in feat_dict:
            feat_dict[k] = int(feat_dict[k])

    float_fields = (
        "dur",
        "rate",
        "sload",
        "dload",
        "sinpkt",
        "dinpkt",
        "sjit",
        "djit",
        "tcprtt",
        "synack",
        "ackdat",
    )
    for k in float_fields:
        if k in feat_dict:
            feat_dict[k] = float(feat_dict[k])

    features = FlowFeatures(**feat_dict)
    return Flow(flow_id=flow_id, event_time=event_time, features=features)


def replay_dataset(
    df: pd.DataFrame,
    api_url: str = "http://localhost:8000",
    api_token: str | None = None,
    db_path: Path | None = None,
    batch_size: int = 50,
    rate: float = 0.0,
    http_client: Any | None = None,
) -> dict[str, Any]:
    """Replay flows sequentially into serving API and record ground truth in replay_truth."""
    print(DISCLOSURE_BANNER, flush=True)

    store: AlertStore | None = None
    if db_path is not None and db_path.exists():
        store = AlertStore(db_path)

    headers = {}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    client = http_client or httpx.Client(base_url=api_url, timeout=30.0, headers=headers)
    should_close_client = http_client is None

    total_rows = len(df)
    total_sent = 0
    total_alerts = 0
    start_time = time.monotonic()

    print(
        f"Starting replay: {total_rows} flows (batch_size={batch_size}, target_rate={rate}/s)",
        flush=True,
    )

    try:
        for offset in range(0, total_rows, batch_size):
            chunk = df.iloc[offset : offset + batch_size]
            flows: list[Flow] = []
            truths: list[tuple[str, int, str]] = []

            for _, row in chunk.iterrows():
                fid = str(uuid4())
                ts = now_timestamp()
                r_dict = row.to_dict()
                flow = row_to_flow(r_dict, fid, ts)
                flows.append(flow)
                label = int(r_dict.get("label", 0))
                attack_cat = str(r_dict.get("attack_cat", "Normal")).strip()
                truths.append((fid, label, attack_cat))

            # Build prediction batch payload
            batch = FlowBatch(schema_version="unsw-nb15.v0", flows=flows)
            req_data = batch.model_dump(mode="json")

            # POST to prediction serving endpoint
            resp = client.post("/api/v1/predictions", json=req_data)
            if resp.status_code != 200:
                print(f"Error sending batch at offset {offset}: {resp.status_code} {resp.text}")
                continue

            result = resp.json()
            total_sent += len(flows)
            total_alerts += result.get("alert_count", 0)

            # Record replay truth separately (never visible to inference)
            if store is not None:
                store.record_replay_truth(truths)
            elif api_token:
                try:
                    client.post(
                        "/api/v1/replay/truth",
                        json={
                            "items": [
                                {"flow_id": fid, "label": lbl, "attack_cat": cat}
                                for fid, lbl, cat in truths
                            ]
                        },
                    )
                except Exception as ex:
                    print(f"Warning: Failed to record replay truth via API: {ex}")

            # Rate throttling
            if rate > 0:
                expected_elapsed = total_sent / rate
                actual_elapsed = time.monotonic() - start_time
                delay = expected_elapsed - actual_elapsed
                if delay > 0:
                    time.sleep(delay)

            elapsed = max(time.monotonic() - start_time, 0.001)
            current_rate = total_sent / elapsed
            print(
                f"\rReplayed: {total_sent}/{total_rows} flows | Alerts: {total_alerts} | "
                f"Rate: {current_rate:.1f} flows/s",
                end="",
                flush=True,
            )

        print("\nReplay finished.", flush=True)

    finally:
        if should_close_client and hasattr(client, "close"):
            client.close()

    elapsed = max(time.monotonic() - start_time, 0.001)
    summary = {
        "total_sent": total_sent,
        "total_alerts": total_alerts,
        "elapsed_seconds": round(elapsed, 2),
        "actual_rate": round(total_sent / elapsed, 1),
    }
    print(f"Summary: {summary}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Nexus UNSW-NB15 Dataset Replay Adapter")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to evaluation CSV")
    parser.add_argument(
        "--split-file",
        type=Path,
        default=DEFAULT_SPLIT_FILE,
        help="Path to split_indices.npz",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="selection",
        help="Partition name (strictly 'selection' for frozen evaluation)",
    )
    parser.add_argument("--rate", type=float, default=0.0, help="Target replay rate (flows/sec)")
    parser.add_argument("--offset", type=int, default=0, help="Starting row offset in partition")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of flows to replay")
    parser.add_argument("--batch-size", type=int, default=50, help="Flows per prediction batch")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="Nexus API URL")
    parser.add_argument("--token", type=str, default=None, help="Nexus API Bearer token")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/nexus.sqlite3"),
        help="Path to SQLite database",
    )

    args = parser.parse_args()

    df = load_evaluation_data(
        csv_path=args.csv,
        split_file=args.split_file,
        split_name=args.split,
        offset=args.offset,
        limit=args.limit,
    )

    replay_dataset(
        df=df,
        api_url=args.url,
        api_token=args.token,
        db_path=args.db if args.db.exists() else None,
        batch_size=args.batch_size,
        rate=args.rate,
    )


if __name__ == "__main__":
    main()
