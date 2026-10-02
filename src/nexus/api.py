"""Application factory and lifecycle. Start with uvicorn nexus.api:app."""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from starlette.concurrency import run_in_threadpool

from nexus import routes
from nexus.config import Settings
from nexus.errors import register_error_handlers
from nexus.storage import AlertStore
from nexus.transport import RequestLimits, StrictRequests


def create_app(settings: Settings | None = None, *, limits: RequestLimits | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    store = AlertStore(settings.database_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await run_in_threadpool(store.initialize)
        yield

    app = FastAPI(
        title="Nexus API",
        version="0.2.0",
        redirect_slashes=False,
        lifespan=lifespan,
        dependencies=[Depends(routes.unique_query)],
    )
    app.state.settings = settings
    app.state.store = store
    app.add_middleware(StrictRequests, limits=limits or RequestLimits())
    register_error_handlers(app)
    for router in (routes.health, routes.flows, routes.alerts):
        app.include_router(router)
    return app


app = create_app()
