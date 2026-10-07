"""Provisional UNSW-NB15 contract; freeze a successor after the dataset audit."""

import re
from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "unsw-nb15.v0"
Count = Annotated[int, Field(ge=0, le=2**53 - 1)]
NonNegative = Annotated[float, Field(ge=0, le=1e15)]
Byte = Annotated[int, Field(ge=0, le=255)]
Window = Annotated[int, Field(ge=0, le=65535)]
Sequence = Annotated[int, Field(ge=0, le=2**32 - 1)]
Flag = Annotated[int, Field(ge=0, le=1)]
Category = Annotated[str, Field(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9_-]+$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)


class FlowFeatures(StrictModel):
    """42 predictor columns; labels, attack_cat, and dataset row IDs are forbidden.

    Numeric limits are API safety bounds, not learned dataset ranges. Category
    membership and training compatibility require the future release bundle.
    Preserve the dataset's units; no conversion or preprocessing occurs here.
    """

    dur: NonNegative
    proto: Category
    service: Category
    state: Category
    spkts: Count
    dpkts: Count
    sbytes: Count
    dbytes: Count
    rate: NonNegative
    sttl: Byte
    dttl: Byte
    sload: NonNegative
    dload: NonNegative
    sloss: Count
    dloss: Count
    sinpkt: NonNegative
    dinpkt: NonNegative
    sjit: NonNegative
    djit: NonNegative
    swin: Window
    stcpb: Sequence
    dtcpb: Sequence
    dwin: Window
    tcprtt: NonNegative
    synack: NonNegative
    ackdat: NonNegative
    smean: Count
    dmean: Count
    trans_depth: Count
    response_body_len: Count
    ct_srv_src: Count
    ct_state_ttl: Count
    ct_dst_ltm: Count
    ct_src_dport_ltm: Count
    ct_dst_sport_ltm: Count
    ct_dst_src_ltm: Count
    is_ftp_login: Flag
    ct_ftp_cmd: Count
    ct_flw_http_mthd: Count
    ct_src_ltm: Count
    ct_srv_dst: Count
    is_sm_ips_ports: Flag


def canonical_uuid(value: str) -> str:
    try:
        parsed = UUID(value)
    except ValueError:
        raise ValueError("must be a canonical nonzero UUID") from None
    if str(parsed) != value or parsed.int == 0:
        raise ValueError("must be a canonical nonzero UUID")
    return value


def timestamp(value: str) -> str:
    if not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-](?:[01]\d|2[0-3]):[0-5]\d)",
        value,
        flags=re.ASCII,
    ):
        raise ValueError("must be an RFC3339 timestamp with timezone")
    try:
        datetime.fromisoformat(value)
    except ValueError:
        raise ValueError("must be a valid calendar timestamp") from None
    return value


Identifier = Annotated[str, Field(min_length=36, max_length=36), AfterValidator(canonical_uuid)]
Timestamp = Annotated[str, Field(min_length=20, max_length=35), AfterValidator(timestamp)]


class Flow(StrictModel):
    flow_id: Identifier
    event_time: Timestamp
    features: FlowFeatures


class FlowBatch(StrictModel):
    schema_version: Literal["unsw-nb15.v0"]
    flows: Annotated[list[Flow], Field(min_length=1, max_length=100)]
    model_id: str | None = None

    @model_validator(mode="after")
    def unique_ids(self) -> "FlowBatch":
        ids = [flow.flow_id for flow in self.flows]
        if len(ids) != len(set(ids)):
            raise ValueError("flow_id must be unique within a batch")
        return self


class ValidationResult(StrictModel):
    valid: Literal[True] = True
    schema_version: Literal["unsw-nb15.v0"] = SCHEMA_VERSION
    flow_count: int
    model_compatible: Literal[False] = False


class Health(StrictModel):
    status: Literal["alive", "not_ready", "ready"]
    bundle_version: str | None = None


class PredictionSummaryItem(StrictModel):
    flow_id: Identifier
    score: float
    threshold: float
    decision: Literal["alert", "normal"]
    alert_id: Identifier | None = None
    ae_mode: str | None = None
    ae_error: float | None = None


class BatchPredictionResponse(StrictModel):
    bundle_version: str
    predictions: list[PredictionSummaryItem]
    alert_count: int
    total_count: int
    model_id: str | None = None


class ModelOptionItem(StrictModel):
    id: str
    name: str
    description: str
    architecture: str
    decision_threshold: float
    focus: str
    ae_budget: str | None = None
    is_fusion: bool = False


class ModelSummaryResponse(StrictModel):
    bundle_version: str
    algorithm: str
    decision_threshold: float
    bundle_hashes: dict[str, str]
    source_hashes: dict[str, str]
    selection_metrics: dict[str, Any]
    metrics_note: Literal["selection estimate, not independent confirmation"] = (
        "selection estimate, not independent confirmation"
    )


class StatsResponse(StrictModel):
    alerts_24h: int
    awaiting_review: int
    confirmed_attacks: int
    false_positives: int
    needs_investigation: int
    total_predictions_24h: int


class NonAlertSampleItem(StrictModel):
    sequence: int
    flow_id: Identifier
    bundle_version: str
    event_time: Timestamp | None = None
    ingest_time: Timestamp
    features: dict[str, Any]
    score: float
    threshold: float
    decision: Literal["normal"]
    review: dict[str, Any] | None = None


class NonAlertSamplePage(StrictModel):
    items: list[NonAlertSampleItem]
    next_after: int | None = None


class ReplayTruthItem(StrictModel):
    flow_id: Identifier
    label: Flag
    attack_cat: str


class ReplayTruthBatch(StrictModel):
    items: Annotated[list[ReplayTruthItem], Field(min_length=1, max_length=1000)]


class ReplaySummaryResponse(StrictModel):
    total_replayed: int
    confusion_matrix: list[list[int]]
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    accuracy: float
    recall: float
    false_positive_rate: float
    false_alerts_per_1000_normal: float
    summary_note: Literal[
        "historical replay on evaluation partition, not independent confirmation"
    ] = "historical replay on evaluation partition, not independent confirmation"
