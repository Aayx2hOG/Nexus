"""HTTP routes and dependency wiring; persistence lives in AlertStore."""

from secrets import compare_digest
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from nexus.alerts import (
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
from nexus.schemas import FlowBatch, Health, Identifier, ValidationResult
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


@flows.post("/predictions", status_code=503)
async def predict(batch: FlowBatch) -> None:
    raise APIError(503, "model_unavailable", "No evaluated model bundle is configured.")


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
