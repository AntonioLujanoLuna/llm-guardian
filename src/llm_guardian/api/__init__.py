"""HTTP routers, all reachable in-process through the providers in ``deps.py``."""

from __future__ import annotations

from fastapi import APIRouter, Depends, FastAPI

from . import health, incidents, memory, owasp, policy, scan
from .security import require_service_token

__all__ = ["register"]


def register(app: FastAPI) -> None:
    """Attach every route to ``app``. ``/v1`` routes are gated by the optional service token.

    The OpenAI-compatible proxy is mounted at ``/openai/v1`` only when ``proxy.upstream_url`` is
    set, behind the same token.
    """
    v1 = APIRouter(prefix="/v1", dependencies=[Depends(require_service_token)])
    for module in (scan, policy, incidents, memory, health, owasp):
        v1.include_router(module.router)
    app.include_router(v1)
    app.include_router(health.prometheus_router)
    if app.state.settings.proxy.upstream_url:
        from . import proxy

        app.include_router(
            proxy.router,
            prefix="/openai/v1",
            dependencies=[Depends(require_service_token)],
        )
