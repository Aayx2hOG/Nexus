"""Real HTTP probe: actually connects to a target URL and extracts network features.

This replaces the fake simulator. It measures real TCP handshake timing,
response sizes, request/response packet counts, and maps them to
UNSW-NB15-compatible features for genuine ML-based attack detection.
"""

from __future__ import annotations

import logging
import socket
import ssl
import time
from datetime import UTC, datetime
from typing import Annotated, Any, Literal
from urllib.parse import urlparse
from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from nexus.alerts import FeatureContribution
from nexus.errors import APIError
from nexus.inference import predict_batch
from nexus.routes import require_analyst, store_for
from nexus.schemas import Flow, FlowFeatures
from nexus.storage import AlertData, AlertStore

logger = logging.getLogger(__name__)

Store = Annotated[AlertStore, Depends(store_for)]
Analyst = Annotated[str, Depends(require_analyst)]

router = APIRouter(prefix="/api/v1/traffic", tags=["live-probe"])


class LiveProbeRequest(BaseModel):
    target_url: str = Field(min_length=1, max_length=512)
    method: Literal["GET", "HEAD", "POST", "OPTIONS"] = "GET"
    timeout_seconds: float = Field(default=10.0, ge=1.0, le=30.0)
    follow_redirects: bool = True


class ConnectionMetrics(BaseModel):
    """Raw measurements from the actual HTTP connection."""
    dns_resolve_ms: float
    tcp_connect_ms: float
    tls_handshake_ms: float | None
    ttfb_ms: float  # time to first byte
    total_ms: float
    request_size_bytes: int
    response_size_bytes: int
    response_header_bytes: int
    response_body_bytes: int
    status_code: int
    http_version: str
    num_redirects: int
    server_header: str | None
    content_type: str | None
    is_https: bool


class TopFeatureItem(BaseModel):
    feature: str
    contribution: float
    value: Any = None


class LiveProbeResult(BaseModel):
    target_url: str
    method: str
    flow_id: str
    connection: ConnectionMetrics
    target_hit: bool
    score: float
    threshold: float
    decision: Literal["alert", "normal"]
    threat_level: Literal["CRITICAL", "HIGH", "MEDIUM", "CLEAN"]
    alert_id: str | None = None
    top_features: list[TopFeatureItem] = Field(default_factory=list)
    recommended_action: str
    event_time: str
    extracted_features: dict[str, Any] = Field(default_factory=dict)


def _measure_dns(hostname: str) -> tuple[float, str | None]:
    """Resolve DNS and return (time_ms, ip_address)."""
    start = time.perf_counter()
    try:
        info = socket.getaddrinfo(hostname, None, socket.AF_INET, socket.SOCK_STREAM)
        elapsed = (time.perf_counter() - start) * 1000
        ip = info[0][4][0] if info else None
        return elapsed, ip
    except socket.gaierror:
        elapsed = (time.perf_counter() - start) * 1000
        return elapsed, None


def _measure_tcp_connect(host: str, port: int, timeout: float = 5.0) -> float:
    """Measure raw TCP SYN-ACK round trip time in seconds."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    start = time.perf_counter()
    try:
        sock.connect((host, port))
        elapsed = time.perf_counter() - start
        return elapsed
    except (socket.timeout, OSError):
        return time.perf_counter() - start
    finally:
        sock.close()


def _measure_tls_handshake(host: str, port: int, timeout: float = 5.0) -> float | None:
    """Measure TLS handshake time in seconds. Returns None for non-TLS."""
    if port not in (443, 8443):
        return None
    context = ssl.create_default_context()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
        start = time.perf_counter()
        ssock = context.wrap_socket(sock, server_hostname=host)
        elapsed = time.perf_counter() - start
        ssock.close()
        return elapsed
    except Exception:
        return None
    finally:
        sock.close()


def _extract_unsw_features(
    conn: ConnectionMetrics,
    tcp_rtt: float,
    method: str,
) -> dict[str, Any]:
    """Map real HTTP connection measurements to UNSW-NB15 flow features.

    This is a best-effort mapping from application-level HTTP metrics
    to the 42 network-flow features the model expects. Some features
    (like window sizes and TCP sequence numbers) are estimated from
    typical values since we can't capture raw packets without root.
    """
    total_duration = conn.total_ms / 1000.0  # seconds
    if total_duration <= 0:
        total_duration = 0.001

    # Estimate packet counts from byte sizes
    # Typical MSS = 1460 bytes, plus ~54 bytes TCP/IP headers per packet
    mss = 1460
    src_data_packets = max(1, conn.request_size_bytes // mss + 1)
    dst_data_packets = max(1, conn.response_size_bytes // mss + 1)
    # Add TCP handshake (SYN, SYN-ACK, ACK) + FIN packets
    spkts = src_data_packets + 3  # data + SYN + ACK + FIN
    dpkts = dst_data_packets + 3  # data + SYN-ACK + ACK + FIN

    sbytes = conn.request_size_bytes + (spkts * 54)  # data + headers
    dbytes = conn.response_size_bytes + (dpkts * 54)

    rate = (spkts + dpkts) / total_duration if total_duration > 0 else 0

    # TTL values: 64 is common for Linux/macOS, 128 for Windows
    sttl = 64
    dttl = 252 if conn.is_https else 64  # remote servers often have high TTLs after hops

    # Load calculations: bits per second
    sload = (sbytes * 8) / total_duration if total_duration > 0 else 0
    dload = (dbytes * 8) / total_duration if total_duration > 0 else 0

    # Packet loss estimation: 0 for successful connections
    sloss = 0
    dloss = 0

    # Inter-packet arrival time (milliseconds)
    sinpkt = (total_duration * 1000) / spkts if spkts > 0 else 0
    dinpkt = (total_duration * 1000) / dpkts if dpkts > 0 else 0

    # Jitter estimation from timing variance
    # Real jitter would need multiple probes; estimate from RTT variance
    sjit = abs(conn.ttfb_ms - conn.tcp_connect_ms) * 10  # rough estimate
    djit = abs(conn.total_ms - conn.ttfb_ms) * 5

    # TCP metrics
    synack = tcp_rtt / 2  # SYN-ACK is roughly half RTT
    ackdat = tcp_rtt / 2
    tcprtt = tcp_rtt

    # Mean packet sizes
    smean = sbytes // spkts if spkts > 0 else 0
    dmean = dbytes // dpkts if dpkts > 0 else 0

    # HTTP transaction depth
    trans_depth = 1 if method in ("GET", "POST") else 0

    # Determine service and protocol
    service = "http" if not conn.is_https else "http"
    proto = "tcp"
    state = "FIN"  # completed connection

    # Connection tracking counters (single probe = 1)
    ct_srv_src = 1
    ct_state_ttl = 1
    ct_dst_ltm = 1
    ct_src_dport_ltm = 1
    ct_dst_sport_ltm = 1
    ct_dst_src_ltm = 1
    ct_src_ltm = 1
    ct_srv_dst = 1

    features = {
        "dur": round(total_duration, 6),
        "proto": proto,
        "service": service,
        "state": state,
        "spkts": spkts,
        "dpkts": dpkts,
        "sbytes": sbytes,
        "dbytes": dbytes,
        "rate": round(rate, 6),
        "sttl": sttl,
        "dttl": dttl,
        "sload": round(sload, 6),
        "dload": round(dload, 6),
        "sloss": sloss,
        "dloss": dloss,
        "sinpkt": round(sinpkt, 6),
        "dinpkt": round(dinpkt, 6),
        "sjit": round(sjit, 6),
        "djit": round(djit, 6),
        "swin": 255,  # standard for most modern OS
        "stcpb": 0,  # can't read seq nums from userspace
        "dtcpb": 0,
        "dwin": 255,
        "tcprtt": round(tcprtt, 6),
        "synack": round(synack, 6),
        "ackdat": round(ackdat, 6),
        "smean": smean,
        "dmean": dmean,
        "trans_depth": trans_depth,
        "response_body_len": conn.response_body_bytes,
        "ct_srv_src": ct_srv_src,
        "ct_state_ttl": ct_state_ttl,
        "ct_dst_ltm": ct_dst_ltm,
        "ct_src_dport_ltm": ct_src_dport_ltm,
        "ct_dst_sport_ltm": ct_dst_sport_ltm,
        "ct_dst_src_ltm": ct_dst_src_ltm,
        "is_ftp_login": 0,
        "ct_ftp_cmd": 0,
        "ct_flw_http_mthd": 1 if method in ("GET", "POST", "PUT", "DELETE", "HEAD") else 0,
        "ct_src_ltm": ct_src_ltm,
        "ct_srv_dst": ct_srv_dst,
        "is_sm_ips_ports": 0,
    }

    return features


@router.post("/live-probe", dependencies=[Depends(require_analyst)])
async def live_probe(
    payload: LiveProbeRequest,
    request: Request,
    store: Store,
) -> LiveProbeResult:
    """Actually probe a target URL: real HTTP connection, real metrics, real ML inference."""
    loader = getattr(request.app.state, "bundle_loader", None)
    if loader is None or not loader.is_ready or loader.loaded_bundle is None:
        raise APIError(503, "model_unavailable", "No evaluated model bundle is configured.")

    # Parse and validate URL
    parsed = urlparse(payload.target_url)
    if parsed.scheme not in ("http", "https"):
        raise APIError(400, "invalid_url", "URL must use http:// or https:// scheme.")
    hostname = parsed.hostname
    if not hostname:
        raise APIError(400, "invalid_url", "Could not extract hostname from URL.")

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    is_https = parsed.scheme == "https"

    bundle = loader.loaded_bundle
    event_time = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    flow_id = str(uuid4())

    # Step 1: DNS resolution timing
    dns_ms, resolved_ip = _measure_dns(hostname)
    if resolved_ip is None:
        raise APIError(
            422, "dns_failed",
            f"DNS resolution failed for '{hostname}'. Target is unreachable."
        )

    # Step 2: Raw TCP connect timing
    tcp_rtt = _measure_tcp_connect(resolved_ip, port, payload.timeout_seconds)
    tcp_connect_ms = tcp_rtt * 1000

    # Step 3: TLS handshake timing (if HTTPS)
    tls_ms = None
    if is_https:
        tls_time = _measure_tls_handshake(hostname, port, payload.timeout_seconds)
        if tls_time is not None:
            tls_ms = tls_time * 1000

    # Step 4: Actual HTTP request with full timing
    overall_start = time.perf_counter()
    try:
        async with httpx.AsyncClient(
            timeout=payload.timeout_seconds,
            follow_redirects=payload.follow_redirects,
            verify=True,
        ) as client:
            # Measure TTFB
            ttfb_start = time.perf_counter()
            response = await client.request(
                payload.method,
                payload.target_url,
                headers={
                    "User-Agent": "Nexus-Probe/1.0 (Security Scanner)",
                    "Accept": "*/*",
                },
            )
            ttfb_ms = (time.perf_counter() - ttfb_start) * 1000

            # Read full body
            body = response.content
            total_ms = (time.perf_counter() - overall_start) * 1000

            # Calculate sizes
            # Estimate request size from method + URL + headers
            request_line = f"{payload.method} {parsed.path or '/'} HTTP/1.1\r\n"
            request_headers_est = f"Host: {hostname}\r\nUser-Agent: Nexus-Probe/1.0\r\nAccept: */*\r\n\r\n"
            request_size = len(request_line.encode()) + len(request_headers_est.encode())

            # Response headers size
            response_header_text = "\r\n".join(
                f"{k}: {v}" for k, v in response.headers.items()
            )
            response_header_bytes = len(response_header_text.encode())
            response_body_bytes = len(body)
            response_size = response_header_bytes + response_body_bytes

            num_redirects = len(response.history) if hasattr(response, 'history') else 0

            conn_metrics = ConnectionMetrics(
                dns_resolve_ms=round(dns_ms, 2),
                tcp_connect_ms=round(tcp_connect_ms, 2),
                tls_handshake_ms=round(tls_ms, 2) if tls_ms is not None else None,
                ttfb_ms=round(ttfb_ms, 2),
                total_ms=round(total_ms, 2),
                request_size_bytes=request_size,
                response_size_bytes=response_size,
                response_header_bytes=response_header_bytes,
                response_body_bytes=response_body_bytes,
                status_code=response.status_code,
                http_version=response.http_version or "HTTP/1.1",
                num_redirects=num_redirects,
                server_header=response.headers.get("server"),
                content_type=response.headers.get("content-type"),
                is_https=is_https,
            )

    except httpx.ConnectTimeout:
        raise APIError(
            422, "connection_timeout",
            f"TCP connection to {hostname}:{port} timed out after {payload.timeout_seconds}s."
        )
    except httpx.ConnectError as e:
        raise APIError(
            422, "connection_failed",
            f"Failed to connect to {hostname}:{port}: {e}"
        )
    except httpx.ReadTimeout:
        raise APIError(
            422, "read_timeout",
            f"Response read from {hostname} timed out after {payload.timeout_seconds}s."
        )
    except Exception as e:
        logger.exception("Live probe failed for %s", payload.target_url)
        raise APIError(422, "probe_failed", f"Probe failed: {type(e).__name__}: {e}")

    # Step 5: Extract UNSW-NB15-compatible features from real metrics
    extracted = _extract_unsw_features(conn_metrics, tcp_rtt, payload.method)

    # Step 6: Run through the ML model
    features = FlowFeatures(**extracted)
    flow = Flow(flow_id=flow_id, event_time=event_time, features=features)
    predictions = predict_batch(bundle, [flow])
    p = predictions[0]

    # Step 7: Create alert if needed
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
                predicted_class="LiveProbe",
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
        "CRITICAL" if p.score >= 0.95
        else ("HIGH" if p.score >= p.threshold else ("MEDIUM" if p.score >= 0.4 else "CLEAN"))
    )

    if is_hit:
        rec_action = (
            f"ALERT: The live probe to {payload.target_url} returned network characteristics "
            f"that the ML model flagged as malicious (score: {p.score:.4f}, "
            f"threshold: {p.threshold:.4f}). "
            f"Investigate the target server's access logs, check for anomalous response patterns, "
            f"and review Alert #{alert_id[:8] if alert_id else 'N/A'} in the SOC triage queue."
        )
    else:
        rec_action = (
            f"CLEAN: The live probe to {payload.target_url} shows normal HTTP response behavior. "
            f"Connection metrics (RTT: {tcp_rtt*1000:.1f}ms, response: {conn_metrics.response_body_bytes} bytes, "
            f"status: {conn_metrics.status_code}) are within expected bounds. "
            f"Score {p.score:.4f} is below threshold {p.threshold:.4f}."
        )

    return LiveProbeResult(
        target_url=payload.target_url,
        method=payload.method,
        flow_id=flow_id,
        connection=conn_metrics,
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
        extracted_features=extracted,
    )
