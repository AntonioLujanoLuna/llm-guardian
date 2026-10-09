"""Risk memory: reviewed precedents, conditional retrieval, no automatic whitelisting."""

from __future__ import annotations

from ..settings import MemorySettings
from .base import (
    MemoryBackendUnavailable,
    RiskMemory,
    VerifiedExample,
    not_queried,
)
from .inmemory import InMemoryRiskMemory

__all__ = [
    "InMemoryRiskMemory",
    "MemoryBackendUnavailable",
    "RiskMemory",
    "VerifiedExample",
    "build_risk_memory",
    "not_queried",
]


def build_risk_memory(settings: MemorySettings) -> RiskMemory:
    """Construct the configured backend, falling back to in-memory (with a recorded reason).

    A misconfiguration degrades loudly rather than silently: the fallback keeps the service up and
    the degraded reason is reported by ``/v1/health`` and as a scan warning.
    """
    if settings.backend == "elasticsearch":
        from .elastic import ElasticsearchRiskMemory

        try:
            return ElasticsearchRiskMemory(settings)
        except MemoryBackendUnavailable as exc:
            fallback = InMemoryRiskMemory(settings)
            fallback.degraded_reason = f"elasticsearch unavailable, using in-memory: {exc}"
            return fallback
    return InMemoryRiskMemory(settings)
