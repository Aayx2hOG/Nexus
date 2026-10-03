"""HTTP routes and dependency wiring; persistence lives in AlertStore."""

from secrets import compare_digest
from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from nexus.alerts import (
    AlertData,
    AlertPage,
    AlertQuery,
    AlertRecord,
    FeedbackInput,
    FeedbackPage,
    FeedbackRecord,
    PageQuery,
)
from nexus.config import Settings
from nexus.errors import APIError
from nexus.inference import predict_batch
from nexus.schemas import (
    BatchPredictionResponse,
    FlowBatch,
    Health,
    Identifier,
    ModelSummaryResponse,
    NonAlertSampleItem,
    NonAlertSamplePage,
    PredictionSummaryItem,
    ReplaySummaryResponse,
    ReplayTruthBatch,
    StatsResponse,
    ValidationResult,
)
from nexus.storage import AlertStore


async def no_query(request: Request) -> None:
    if request.query_params:
        raise APIError(400, "unexpected_query", "This endpoint does not accept query parameters.")


async def unique_query(request: Request) -> None:
    keys = [key for key, _ in request.query_params.multi_items()]
    if len(keys) != len(set(keys)):
        raise APIError(400, "duplicate_query", "Query parameters must be unique.")


def store_for(request: Request) -> AlertStore:
    return request.app.state.store


bearer = HTTPBearer(auto_error=False)


async def require_analyst(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> str:
    settings: Settings = request.app.state.settings
    if settings.api_token is None:
        raise APIError(503, "auth_not_configured", "Configure an API token to enable alert access.")
    if len(request.headers.getlist("authorization")) != 1 or credentials is None:
        raise APIError(401, "unauthorized", "A valid bearer token is required.")
    if not compare_digest(credentials.credentials.encode(), settings.api_token.encode()):
        raise APIError(401, "unauthorized", "A valid bearer token is required.")
    return settings.reviewer_id


Store = Annotated[AlertStore, Depends(store_for)]
Analyst = Annotated[str, Depends(require_analyst)]

health = APIRouter(prefix="/health", tags=["health"], dependencies=[Depends(no_query)])
flows = APIRouter(prefix="/api/v1", tags=["flows"], dependencies=[Depends(no_query)])
alerts = APIRouter(
    prefix="/api/v1/alerts",
    tags=["alerts"],
    dependencies=[Depends(require_analyst)],
)
predictions = APIRouter(prefix="/api/v1/predictions", tags=["predictions"])
stats = APIRouter(
    prefix="/api/v1/stats",
    tags=["stats"],
    dependencies=[Depends(require_analyst), Depends(no_query)],
)
replay = APIRouter(
    prefix="/api/v1/replay",
    tags=["replay"],
    dependencies=[Depends(require_analyst)],
)


@health.get("/live", response_model_exclude_none=True)
async def live() -> Health:
    return Health(status="alive")


@health.get("/ready", response_model_exclude_none=True)
async def ready(request: Request, response: Response) -> Health:
    loader = getattr(request.app.state, "bundle_loader", None)
    if loader is not None and loader.is_ready and loader.loaded_bundle is not None:
        return Health(
            status="ready",
            bundle_version=loader.loaded_bundle.manifest.bundle_version,
        )
    response.status_code = 503
    return Health(status="not_ready")


@flows.get("/schema")
async def schema() -> dict:
    return FlowBatch.model_json_schema()


@flows.post("/flows/validate")
async def validate_flows(batch: FlowBatch) -> ValidationResult:
    return ValidationResult(flow_count=len(batch.flows))


@flows.get("/model")
async def get_model_summary(request: Request) -> ModelSummaryResponse:
    loader = getattr(request.app.state, "bundle_loader", None)
    if loader is None or not loader.is_ready or loader.loaded_bundle is None:
        raise APIError(503, "model_unavailable", "No evaluated model bundle is configured.")
    manifest = loader.loaded_bundle.manifest
    return ModelSummaryResponse(
        bundle_version=manifest.bundle_version,
        algorithm=getattr(manifest, "algorithm", "LightGBM"),
        decision_threshold=manifest.decision_threshold,
        bundle_hashes=manifest.artifact_hashes,
        source_hashes=manifest.source_hashes,
        selection_metrics=manifest.validated_metrics,
        metrics_note="selection estimate, not independent confirmation",
    )


@predictions.post("", dependencies=[Depends(no_query)])
async def predict(batch: FlowBatch, request: Request, store: Store) -> BatchPredictionResponse:
    loader = getattr(request.app.state, "bundle_loader", None)
    if loader is None or not loader.is_ready or loader.loaded_bundle is None:
        raise APIError(503, "model_unavailable", "No evaluated model bundle is configured.")

    bundle = loader.loaded_bundle
    flow_predictions = predict_batch(bundle, batch.flows)

    alerts_to_create = []
    for p in flow_predictions:
        if p.decision == "alert":
            alerts_to_create.append(
                AlertData(
                    alert_id=str(uuid4()),
                    flow_id=p.flow_id,
                    event_time=p.event_time,
                    source="model",
                    bundle_version=bundle.manifest.bundle_version,
                    predicted_class="Generic",
                    score=p.score,
                    threshold=p.threshold,
                    severity="alert",
                    top_features=p.top_features,
                )
            )

    recorded = store.record_predictions_and_alerts(
        flow_predictions, bundle.manifest.bundle_version, alerts_to_create
    )
    items = [
        PredictionSummaryItem(
            flow_id=p.flow_id,
            score=p.score,
            threshold=p.threshold,
            decision=p.decision,
            alert_id=rec.alert_id if rec else None,
        )
        for p, rec in recorded
    ]
    return BatchPredictionResponse(
        bundle_version=bundle.manifest.bundle_version,
        predictions=items,
        alert_count=len(alerts_to_create),
        total_count=len(flow_predictions),
    )


@predictions.get("/sample", dependencies=[Depends(require_analyst)])
def sample_predictions(
    store: Store,
    query: Annotated[PageQuery, Query()],
) -> NonAlertSamplePage:
    items, next_after = store.sample_non_alert_predictions(query)
    return NonAlertSamplePage(
        items=[NonAlertSampleItem(**item) for item in items],
        next_after=next_after,
    )


@predictions.post(
    "/{flow_id}/feedback",
    dependencies=[Depends(require_analyst), Depends(no_query)],
)
def submit_sample_feedback(
    flow_id: Identifier,
    feedback: FeedbackInput,
    store: Store,
    reviewer: Analyst,
) -> dict[str, Any]:
    return store.add_sample_feedback(flow_id, feedback, reviewer)


@stats.get("")
def get_stats(store: Store) -> StatsResponse:
    data = store.get_stats()
    return StatsResponse(**data)


@replay.post("/truth", dependencies=[Depends(no_query)])
def record_replay_truth(
    batch: ReplayTruthBatch,
    store: Store,
) -> dict[str, int]:
    store.record_replay_truth([(item.flow_id, item.label, item.attack_cat) for item in batch.items])
    return {"recorded": len(batch.items)}


@replay.get("/summary", dependencies=[Depends(no_query)])
def get_replay_summary(store: Store) -> ReplaySummaryResponse:
    data = store.get_replay_summary()
    return ReplaySummaryResponse(**data)


@alerts.get("")
def list_alerts(store: Store, query: Annotated[AlertQuery, Query()]) -> AlertPage:
    return store.list_alerts(query)


@alerts.get("/{alert_id}", dependencies=[Depends(no_query)])
def get_alert(alert_id: Identifier, store: Store) -> AlertRecord:
    return store.get_alert(alert_id)


@alerts.get("/{alert_id}/feedback")
def list_feedback(
    alert_id: Identifier,
    store: Store,
    query: Annotated[PageQuery, Query()],
) -> FeedbackPage:
    return store.list_feedback(alert_id, query)


@alerts.post("/{alert_id}/feedback", dependencies=[Depends(no_query)])
def submit_feedback(
    alert_id: Identifier,
    feedback: FeedbackInput,
    store: Store,
    reviewer: Analyst,
) -> FeedbackRecord:
    return store.add_feedback(alert_id, feedback, reviewer)
