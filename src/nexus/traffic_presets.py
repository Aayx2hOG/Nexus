"""Traffic presets and live website attack simulation endpoints."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from nexus.errors import APIError
from nexus.inference import predict_batch
from nexus.replay import row_to_flow
from nexus.routes import require_analyst, store_for
from nexus.schemas import Flow, FlowFeatures
from nexus.storage import AlertData, AlertStore

Store = Annotated[AlertStore, Depends(store_for)]
Analyst = Annotated[str, Depends(require_analyst)]

router = APIRouter(prefix="/api/v1/traffic", tags=["traffic"])
PRESETS_DIR = Path(__file__).resolve().parents[2] / "data" / "presets"

PROFILES = [
    {
        "id": "normal_web",
        "name": "Normal Web Browsing",
        "category": "Normal",
        "description": "Legitimate HTTP GET / index.html browsing traffic with typical TCP handshake and session teardown.",
        "service": "http",
        "expected_verdict": "normal",
        "template_row_index": 4,  # row 4 in normal UNSW-NB15 testing partition
        "features": {
            "dur": 0.449454,
            "proto": "tcp",
            "service": "http",
            "state": "FIN",
            "spkts": 10,
            "dpkts": 6,
            "sbytes": 534,
            "dbytes": 268,
            "rate": 33.373826,
            "sttl": 254,
            "dttl": 252,
            "sload": 8561.499023,
            "dload": 3987.059814,
            "sloss": 2,
            "dloss": 1,
            "sinpkt": 47.750333,
            "dinpkt": 75.659602,
            "sjit": 2415.837634,
            "djit": 115.807,
            "swin": 255,
            "stcpb": 2436137549,
            "dtcpb": 1977154190,
            "dwin": 255,
            "tcprtt": 0.128381,
            "synack": 0.071147,
            "ackdat": 0.057234,
            "smean": 53,
            "dmean": 45,
            "trans_depth": 0,
            "response_body_len": 0,
            "ct_srv_src": 43,
            "ct_state_ttl": 1,
            "ct_dst_ltm": 2,
            "ct_src_dport_ltm": 2,
            "ct_dst_sport_ltm": 1,
            "ct_dst_src_ltm": 40,
            "is_ftp_login": 0,
            "ct_ftp_cmd": 0,
            "ct_flw_http_mthd": 0,
            "ct_src_ltm": 2,
            "ct_srv_dst": 39,
            "is_sm_ips_ports": 0,
        },
    },
    {
        "id": "http_exploit",
        "name": "HTTP Web Exploit / SQLi",
        "category": "Exploits",
        "description": "Web application SQL injection and remote code execution payload with deep transaction depth and anomalous payload size.",
        "service": "http",
        "expected_verdict": "alert",
        "features": {
            "dur": 0.505999,
            "proto": "tcp",
            "service": "http",
            "state": "FIN",
            "spkts": 12,
            "dpkts": 26,
            "sbytes": 860,
            "dbytes": 26136,
            "rate": 73.12267,
            "sttl": 62,
            "dttl": 252,
            "sload": 12474.33203,
            "dload": 397328.8125,
            "sloss": 2,
            "dloss": 10,
            "sinpkt": 45.999909,
            "dinpkt": 17.784881,
            "sjit": 3178.403187,
            "djit": 2219.545984,
            "swin": 255,
            "stcpb": 1381170896,
            "dtcpb": 3232491580,
            "dwin": 255,
            "tcprtt": 0.12007,
            "synack": 0.054775,
            "ackdat": 0.065295,
            "smean": 72,
            "dmean": 1005,
            "trans_depth": 1,
            "response_body_len": 12289,
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
        },
    },
    {
        "id": "http_dos",
        "name": "HTTP DoS Flood",
        "category": "DoS",
        "description": "Denial of service connection flood overwhelming web application worker threads with high traffic rates.",
        "service": "http",
        "expected_verdict": "alert",
        "features": {
            "dur": 0.354228,
            "proto": "tcp",
            "service": "http",
            "state": "FIN",
            "spkts": 10,
            "dpkts": 8,
            "sbytes": 832,
            "dbytes": 1106,
            "rate": 47.991688,
            "sttl": 62,
            "dttl": 252,
            "sload": 16904.36719,
            "dload": 21861.625,
            "sloss": 2,
            "dloss": 2,
            "sinpkt": 39.358665,
            "dinpkt": 49.336426,
            "sjit": 1978.851929,
            "djit": 87.218,
            "swin": 255,
            "stcpb": 3535933800,
            "dtcpb": 1133246419,
            "dwin": 255,
            "tcprtt": 0.082531,
            "synack": 0.041284,
            "ackdat": 0.041247,
            "smean": 83,
            "dmean": 138,
            "trans_depth": 1,
            "response_body_len": 268,
            "ct_srv_src": 2,
            "ct_state_ttl": 1,
            "ct_dst_ltm": 1,
            "ct_src_dport_ltm": 1,
            "ct_dst_sport_ltm": 1,
            "ct_dst_src_ltm": 1,
            "is_ftp_login": 0,
            "ct_ftp_cmd": 0,
            "ct_flw_http_mthd": 1,
            "ct_src_ltm": 1,
            "ct_srv_dst": 2,
            "is_sm_ips_ports": 0,
        },
    },
    {
        "id": "http_recon",
        "name": "Port & Web Vulnerability Reconnaissance",
        "category": "Reconnaissance",
        "description": "Rapid port sweeps and path discovery requests inspecting target server HTTP endpoints for unpatched flaws.",
        "service": "http",
        "expected_verdict": "alert",
        "features": {
            "dur": 0.884141,
            "proto": "tcp",
            "service": "http",
            "state": "FIN",
            "spkts": 10,
            "dpkts": 8,
            "sbytes": 804,
            "dbytes": 354,
            "rate": 19.227701,
            "sttl": 62,
            "dttl": 252,
            "sload": 6542.497559,
            "dload": 2841.488037,
            "sloss": 2,
            "dloss": 2,
            "sinpkt": 98.237885,
            "dinpkt": 124.975426,
            "sjit": 5012.392578,
            "djit": 218.491,
            "swin": 255,
            "stcpb": 3535933800,
            "dtcpb": 1133246419,
            "dwin": 255,
            "tcprtt": 0.198371,
            "synack": 0.099185,
            "ackdat": 0.099186,
            "smean": 80,
            "dmean": 44,
            "trans_depth": 1,
            "response_body_len": 0,
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
        },
    },
    {
        "id": "http_fuzzer",
        "name": "HTTP Protocol Fuzzing",
        "category": "Fuzzers",
        "description": "Transmission of non-standard, malformed HTTP headers and fragmented request chunks aimed at causing parser crashes.",
        "service": "http",
        "expected_verdict": "alert",
        "features": {
            "dur": 0.370517,
            "proto": "tcp",
            "service": "http",
            "state": "FIN",
            "spkts": 10,
            "dpkts": 8,
            "sbytes": 804,
            "dbytes": 2912,
            "rate": 45.881837,
            "sttl": 62,
            "dttl": 252,
            "sload": 15632.21191,
            "dload": 55015.02344,
            "sloss": 2,
            "dloss": 2,
            "sinpkt": 41.168556,
            "dinpkt": 49.12143,
            "sjit": 3087.785371,
            "djit": 112.496508,
            "swin": 255,
            "stcpb": 987669403,
            "dtcpb": 3823328964,
            "dwin": 255,
            "tcprtt": 0.02143,
            "synack": 0.010163,
            "ackdat": 0.011267,
            "smean": 80,
            "dmean": 364,
            "trans_depth": 1,
            "response_body_len": 1049,
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
        },
    },
]


class ProbeRequest(BaseModel):
    target_url: str = Field(min_length=1, max_length=256)
    profile_id: str = Field(min_length=1, max_length=64)


class TopFeatureItem(BaseModel):
    feature: str
    contribution: float
    value: Any = None


class ProbeResult(BaseModel):
    target_url: str
    profile_id: str
    profile_name: str
    category: str
    flow_id: str
    target_hit: bool
    score: float
    threshold: float
    decision: Literal["alert", "normal"]
    threat_level: Literal["CRITICAL", "HIGH", "MEDIUM", "CLEAN"]
    alert_id: str | None = None
    top_features: list[TopFeatureItem] = Field(default_factory=list)
    recommended_action: str
    event_time: str


@router.get("/profiles")
def get_simulation_profiles():
    """List available traffic profiles for website attack simulation."""
    return [
        {
            "id": p["id"],
            "name": p["name"],
            "category": p["category"],
            "description": p["description"],
            "service": p["service"],
            "expected_verdict": p["expected_verdict"],
        }
        for p in PROFILES
    ]


@router.post("/simulate-probe", dependencies=[Depends(require_analyst)])
def simulate_probe(
    payload: ProbeRequest,
    request: Request,
    store: Store,
) -> ProbeResult:
    """Run an interactive attack simulation against a target website and check if it got hit."""
    loader = getattr(request.app.state, "bundle_loader", None)
    if loader is None or not loader.is_ready or loader.loaded_bundle is None:
        raise APIError(503, "model_unavailable", "No evaluated model bundle is configured.")

    profile = next((p for p in PROFILES if p["id"] == payload.profile_id), None)
    if not profile:
        raise APIError(400, "unknown_profile", f"Unknown profile: {payload.profile_id}")

    bundle = loader.loaded_bundle
    event_time = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    flow_id = str(uuid4())

    features_dict = dict(profile["features"])
    features = FlowFeatures(**features_dict)
    flow = Flow(flow_id=flow_id, event_time=event_time, features=features)

    predictions = predict_batch(bundle, [flow])
    p = predictions[0]

    alert_id = None
    alerts_to_create = []
    if p.decision == "alert":
        alert_id = str(uuid4())
        alerts_to_create.append(
            AlertData(
                alert_id=alert_id,
                flow_id=p.flow_id,
                event_time=p.event_time,
                source="model",
                bundle_version=bundle.manifest.bundle_version,
                predicted_class=profile["category"],
                score=p.score,
                threshold=p.threshold,
                severity="alert",
                top_features=p.top_features,
            )
        )

    store.record_predictions_and_alerts(
        predictions, bundle.manifest.bundle_version, alerts_to_create
    )

    is_hit = p.decision == "alert"
    threat_level: Literal["CRITICAL", "HIGH", "MEDIUM", "CLEAN"] = (
        "CRITICAL" if p.score >= 0.95 else ("HIGH" if p.score >= p.threshold else "CLEAN")
    )

    if is_hit:
        rec_action = (
            f"ALERT: Malicious {profile['category']} activity detected targeting {payload.target_url}. "
            f"Review Alert #{alert_id[:8]} in the SOC queue. Inspect target web server access logs for anomalous "
            f"request patterns and block the originating IP."
        )
    else:
        rec_action = (
            f"CLEAN: Traffic pattern targeting {payload.target_url} falls well below the operational decision "
            f"threshold ({p.threshold:.4f}). Traffic appears to be benign browsing behavior."
        )

    return ProbeResult(
        target_url=payload.target_url,
        profile_id=profile["id"],
        profile_name=profile["name"],
        category=profile["category"],
        flow_id=flow_id,
        target_hit=is_hit,
        score=p.score,
        threshold=p.threshold,
        decision=p.decision,
        threat_level=threat_level,
        alert_id=alert_id,
        top_features=[
            TopFeatureItem(feature=tf.feature, contribution=tf.contribution, value=tf.value)
            for tf in p.top_features
        ],
        recommended_action=rec_action,
        event_time=event_time,
    )


@router.get("/presets/{preset_id}", dependencies=[Depends(require_analyst)])
def get_preset_data(preset_id: str):
    """Retrieve pre-extracted UNSW-NB15 flow batches (normal_50, attack_50, mixed_100)."""
    csv_file = PRESETS_DIR / f"{preset_id}.csv"
    json_file = PRESETS_DIR / f"{preset_id}.json"

    if not csv_file.exists() or not json_file.exists():
        raise APIError(404, "preset_not_found", f"Preset '{preset_id}' not found.")

    with csv_file.open("r", encoding="utf-8") as f:
        csv_text = f.read()

    with json_file.open("r", encoding="utf-8") as f:
        flows_raw = json.load(f)

    return {
        "preset_id": preset_id,
        "flow_count": len(flows_raw),
        "csv_text": csv_text,
        "flows": flows_raw,
    }
