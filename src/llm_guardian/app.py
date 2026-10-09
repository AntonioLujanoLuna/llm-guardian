"""The FastAPI application.

``create_app`` is the only entry point. The routers live in ``api/`` and the providers they take
live in ``deps.py``: nothing here reads configuration or builds a client by hand, so a gateway can
integrate by overriding the dependencies instead of reimplementing the service.
"""

from __future__ import annotations

from fastapi import FastAPI

from .settings import GuardianSettings

__all__ = ["create_app"]


def create_app(settings: GuardianSettings | None = None) -> FastAPI:
    """Build the guardian application.

    Shadow by default: verdicts are returned and counted, and nothing is blocked until the
    operator sets ``GUARDIAN_MODE=enforce``.
    """
    from . import __version__

    resolved = settings if settings is not None else GuardianSettings()
    app = FastAPI(
        title="LLM Guardian",
        version=__version__,
        description=(
            "Incremental pre-inference inspection with reviewed risk memory. Shadow by default."
        ),
    )
    app.state.settings = resolved
    return app
