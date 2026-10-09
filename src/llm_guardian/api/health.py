"""``GET /v1/health``, ``GET /v1/metrics`` and the Prometheus ``GET /metrics``."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from .. import __version__
from ..deps import (
    FeatureCacheDep,
    MetricsDep,
    PolicyEngineDep,
    RiskMemoryDep,
    ScannerPipelineDep,
    SettingsDep,
)
from ..models import HealthReport, MetricsReport, RiskMemoryHealth

router = APIRouter(tags=["health"])
prometheus_router = APIRouter(tags=["metrics"])


@router.get("/health", response_model=HealthReport)
def health(
    settings: SettingsDep,
    pipeline: ScannerPipelineDep,
    cache: FeatureCacheDep,
    memory: RiskMemoryDep,
    policy: PolicyEngineDep,
    metrics: MetricsDep,
) -> HealthReport:
    """Readiness: mode, scanner versions, cache and risk memory, plus any degraded reason."""
    mem_stats = memory.stats()
    reasons: list[str] = []
    if not policy.loaded():
        reasons.append(f"policy_unavailable:{policy.load_error}")
    if not mem_stats.reachable:
        reasons.append("risk_memory_unreachable")
    if memory.degraded_reason:
        reasons.append(f"risk_memory_degraded:{memory.degraded_reason}")
    reasons.extend(pipeline.warnings)
    degraded = any(
        reason.startswith(("policy_unavailable", "risk_memory_unreachable")) for reason in reasons
    )
    return HealthReport(
        status="degraded" if degraded else "ok",
        guardian_version=__version__,
        mode=settings.mode,
        shadow=settings.shadow,
        uptime_seconds=round(metrics.window_seconds(), 3),
        scanner_versions=pipeline.versions(),
        risk_memory=RiskMemoryHealth(
            backend=mem_stats.backend,
            reachable=mem_stats.reachable,
            detail=mem_stats.detail,
        ),
        cache=cache.stats(),
        reasons=tuple(reasons),
    )


@router.get("/metrics", response_model=MetricsReport)
def metrics_report(cache: FeatureCacheDep, metrics: MetricsDep) -> MetricsReport:
    """Per-stage latency percentiles and counters."""
    return metrics.report(cache.stats())


@prometheus_router.get("/metrics", response_class=PlainTextResponse)
def prometheus(cache: FeatureCacheDep, metrics: MetricsDep) -> PlainTextResponse:
    """The same counters in Prometheus text exposition format."""
    return PlainTextResponse(metrics.prometheus(cache.stats()))


__all__ = ["prometheus_router", "router"]
