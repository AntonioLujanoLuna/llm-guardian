"""Adaptive LLM Guardian with risk memory.

A self-hosted security layer that sits ahead of model routing: it inspects only the new or
modified segments of a request, decides ``allow`` / ``escalate`` / ``block`` with deterministic
checks first and reviewed precedents only on the borderline, and enforces real permissions at
the action boundary instead of asking a model for them.

Four architectural rules, from the design (`ideas/adaptive_llm_guardian_risk_memory.md`):

1. Detection is not vector similarity: new attacks need not resemble known ones, and security
   documentation closely resembles malicious instructions.
2. Retrieval stays off the normal path: incremental scanning plus an exact-content feature
   cache, querying the vector index only when the verdict is genuinely uncertain.
3. Authorization is never delegated to a model: tool execution, cross-tenant retrieval, egress
   and quotas are decided by hard policy checks.
4. Only verified evidence is learned from: nothing an attacker submitted is promoted into the
   risk memory without independent review.
"""

from typing import TYPE_CHECKING, Any

from .settings import GuardianSettings

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from fastapi import FastAPI

__all__ = ["GuardianSettings", "__version__", "create_app"]

__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    """Expose ``create_app`` lazily so importing the package never imports the service."""
    if name == "create_app":
        from .app import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:  # pragma: no cover - the real signature lives in app.py

    def create_app(*args: Any, **kwargs: Any) -> FastAPI: ...
