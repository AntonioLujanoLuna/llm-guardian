"""The dependency-injection layer -- the intended integration point.

Every provider reads a singleton off ``request.app.state`` so a gateway route can ask for
``ScanServiceDep`` and get a fully-wired service without an HTTP hop, and so
``app.dependency_overrides`` works in tests. ``ScanServiceDep`` is composed from the individual
providers rather than a prebuilt object, so overriding, say, ``get_risk_memory`` actually changes
what the scan service uses -- that is why the override test proves the graph is real.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, cast

from fastapi import Depends, Request

from .cache import FeatureCache
from .incidents import IncidentRepository, IncidentStore
from .memory import RiskMemory, build_risk_memory
from .metrics import Metrics
from .policy import PolicyEngine
from .scanners import ScannerPipeline
from .service import ScanService
from .session import SessionRiskTracker
from .settings import GuardianSettings
from .verdicts import VerdictEngine

#: A clock is injected everywhere a timestamp is produced, so tests are deterministic.
Clock = Callable[[], datetime]


def utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass
class ServiceContainer:
    """The singletons a running guardian holds. Built once by :func:`create_service`."""

    settings: GuardianSettings
    clock: Clock
    pipeline: ScannerPipeline
    cache: FeatureCache
    memory: RiskMemory
    incidents: IncidentStore
    policy: PolicyEngine
    metrics: Metrics
    verdicts: VerdictEngine
    sessions: SessionRiskTracker


def create_service(settings: GuardianSettings) -> ServiceContainer:
    """Build the whole graph for ``create_app``. No I/O beyond reading the policy file lazily."""
    clock = utcnow
    pipeline = ScannerPipeline.from_settings(settings.scanners)
    cache = FeatureCache(settings.cache)
    memory = build_risk_memory(settings.memory)
    incidents = IncidentStore(settings, memory, clock, _incident_repository(settings))
    metrics = Metrics()
    verdicts = VerdictEngine(settings)
    policy = PolicyEngine(settings.policy)
    return ServiceContainer(
        settings=settings,
        clock=clock,
        pipeline=pipeline,
        cache=cache,
        memory=memory,
        incidents=incidents,
        policy=policy,
        metrics=metrics,
        verdicts=verdicts,
        sessions=SessionRiskTracker(settings.session),
    )


def _incident_repository(settings: GuardianSettings) -> IncidentRepository | None:
    if settings.incidents.backend == "sqlite":
        from .incidents_sqlite import SqliteIncidentRepository

        return SqliteIncidentRepository(settings.incidents.sqlite_path)
    return None


# -- providers --------------------------------------------------------------------------------
def get_settings(request: Request) -> GuardianSettings:
    return cast(GuardianSettings, request.app.state.settings)


def get_clock(request: Request) -> Clock:
    return cast(Clock, request.app.state.clock)


def get_scanner_pipeline(request: Request) -> ScannerPipeline:
    return cast(ScannerPipeline, request.app.state.scanner_pipeline)


def get_feature_cache(request: Request) -> FeatureCache:
    return cast(FeatureCache, request.app.state.feature_cache)


def get_risk_memory(request: Request) -> RiskMemory:
    return cast(RiskMemory, request.app.state.risk_memory)


def get_incident_store(request: Request) -> IncidentStore:
    return cast(IncidentStore, request.app.state.incident_store)


def get_policy_engine(request: Request) -> PolicyEngine:
    return cast(PolicyEngine, request.app.state.policy_engine)


def get_metrics(request: Request) -> Metrics:
    return cast(Metrics, request.app.state.metrics)


def get_verdict_engine(request: Request) -> VerdictEngine:
    return cast(VerdictEngine, request.app.state.verdict_engine)


def get_session_tracker(request: Request) -> SessionRiskTracker:
    return cast(SessionRiskTracker, request.app.state.session_tracker)


# -- Annotated aliases ------------------------------------------------------------------------
SettingsDep = Annotated[GuardianSettings, Depends(get_settings)]
ClockDep = Annotated[Clock, Depends(get_clock)]
ScannerPipelineDep = Annotated[ScannerPipeline, Depends(get_scanner_pipeline)]
FeatureCacheDep = Annotated[FeatureCache, Depends(get_feature_cache)]
RiskMemoryDep = Annotated[RiskMemory, Depends(get_risk_memory)]
IncidentStoreDep = Annotated[IncidentStore, Depends(get_incident_store)]
PolicyEngineDep = Annotated[PolicyEngine, Depends(get_policy_engine)]
MetricsDep = Annotated[Metrics, Depends(get_metrics)]
VerdictEngineDep = Annotated[VerdictEngine, Depends(get_verdict_engine)]
SessionTrackerDep = Annotated[SessionRiskTracker, Depends(get_session_tracker)]


def get_scan_service(
    settings: SettingsDep,
    clock: ClockDep,
    pipeline: ScannerPipelineDep,
    cache: FeatureCacheDep,
    memory: RiskMemoryDep,
    incidents: IncidentStoreDep,
    metrics: MetricsDep,
    sessions: SessionTrackerDep,
) -> ScanService:
    """Compose a scan service from the current providers, so overrides propagate."""
    return ScanService(settings, clock, pipeline, cache, memory, incidents, metrics, sessions)


ScanServiceDep = Annotated[ScanService, Depends(get_scan_service)]
