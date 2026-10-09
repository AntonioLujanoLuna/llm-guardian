"""Opt-in: a reviewed-benign case for identical bytes lowers an escalate -- and nothing more."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from llm_guardian.cache import FeatureCache
from llm_guardian.incidents import IncidentStore
from llm_guardian.memory import InMemoryRiskMemory
from llm_guardian.metrics import Metrics
from llm_guardian.models import (
    GuardMode,
    IncidentStatus,
    ReviewRequest,
    ScanRequest,
    ScanResponse,
    SegmentInput,
    VerdictDecision,
)
from llm_guardian.provenance import Provenance
from llm_guardian.scanners.pipeline import ScannerPipeline
from llm_guardian.service import ScanService
from llm_guardian.settings import GuardianSettings, MemorySettings
from tests.support import TOOL_RESULT_INJECTION

#: A benign question the detectors escalate (inside the uncertain band).
ESCALATING = "what is the api key for the staging service?"
T0 = datetime(2026, 3, 1, tzinfo=UTC)


class Clock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        return self.now


def build(enabled: bool = True, clock: Clock | None = None) -> tuple[ScanService, IncidentStore]:
    clock = clock or Clock()
    settings = GuardianSettings(
        mode=GuardMode.ENFORCE, memory=MemorySettings(reviewed_benign_allows=enabled)
    )
    memory = InMemoryRiskMemory(settings.memory, clock=clock)
    store = IncidentStore(settings, memory, clock=clock)
    service = ScanService(
        settings,
        clock,
        ScannerPipeline.from_settings(settings.scanners),
        FeatureCache(settings.cache),
        memory,
        store,
        Metrics(),
    )
    return service, store


def scan(
    service: ScanService,
    text: str = ESCALATING,
    provenance: Provenance = Provenance.USER,
    tenant: str | None = "acme",
) -> ScanResponse:
    return service.scan(
        ScanRequest(tenant=tenant, segments=(SegmentInput(text=text, provenance=provenance),))
    )


def review_latest(store: IncidentStore, label: IncidentStatus = IncidentStatus.BENIGN) -> str:
    case = store.list()[-1]
    store.review(case.case_id, ReviewRequest(reviewer="ana", label=label))
    return case.case_id


def test_reviewed_false_positive_stops_escalating() -> None:
    service, store = build()
    assert scan(service).decision is VerdictDecision.ESCALATE
    case_id = review_latest(store)
    response = scan(service)
    assert response.decision is VerdictDecision.ALLOW
    assert f"reviewed_benign_precedent_allow:{case_id}" in response.segments[0].notes


def test_off_by_default() -> None:
    assert GuardianSettings().memory.reviewed_benign_allows is False
    service, store = build(enabled=False)
    scan(service)
    review_latest(store)
    assert scan(service).decision is VerdictDecision.ESCALATE


def test_not_across_tenants() -> None:
    service, store = build()
    scan(service, tenant="acme")
    review_latest(store)
    assert scan(service, tenant="globex").decision is VerdictDecision.ESCALATE
    assert scan(service, tenant=None).decision is VerdictDecision.ESCALATE


def test_not_across_provenance() -> None:
    service, store = build()
    scan(service, provenance=Provenance.USER)
    review_latest(store)
    response = scan(service, provenance=Provenance.RETRIEVED_CHUNK)
    assert response.decision is not VerdictDecision.ALLOW


def test_not_on_similarity() -> None:
    service, store = build()
    scan(service)
    review_latest(store)
    near = "what is the api key for the production service?"
    assert scan(service, near).decision is VerdictDecision.ESCALATE


def test_confirmed_attack_review_does_not_allow() -> None:
    service, store = build()
    scan(service)
    review_latest(store, IncidentStatus.CONFIRMED_ATTACK)
    assert scan(service).decision is VerdictDecision.ESCALATE


def test_re_review_to_attack_revokes_the_allow() -> None:
    service, store = build()
    scan(service)
    case_id = review_latest(store)
    assert scan(service).decision is VerdictDecision.ALLOW
    store.review(case_id, ReviewRequest(reviewer="bob", label=IncidentStatus.CONFIRMED_ATTACK))
    assert scan(service).decision is VerdictDecision.ESCALATE


def test_expired_review_no_longer_applies() -> None:
    clock = Clock()
    service, store = build(clock=clock)
    scan(service)
    review_latest(store)
    clock.now = T0 + timedelta(days=400)
    assert scan(service).decision is VerdictDecision.ESCALATE


def test_never_lowers_a_block() -> None:
    service, store = build()
    assert scan(service, TOOL_RESULT_INJECTION).decision is VerdictDecision.BLOCK
    review_latest(store)  # a careless review cannot unblock it
    assert scan(service, TOOL_RESULT_INJECTION).decision is VerdictDecision.BLOCK


def test_never_on_high_impact_provenance() -> None:
    service, store = build()
    first = scan(service, provenance=Provenance.TOOL_RESULT)
    assert first.decision is not VerdictDecision.ALLOW
    review_latest(store)
    again = scan(service, provenance=Provenance.TOOL_RESULT)
    assert again.decision is first.decision
    assert not any("reviewed_benign" in note for note in again.segments[0].notes)


class MarkerScanner:
    """``ZZ`` is a medium finding (escalates); ``QQ`` is one only a seam can complete."""

    name = "marker"
    version = "marker-1"

    def scan(self, text, provenance):
        from llm_guardian.models import Severity
        from llm_guardian.scanners.base import build_finding

        findings = []
        for kind, token in (("zz", "ZZ"), ("qq", "QQ")):
            at = text.find(token)
            if at >= 0:
                findings.append(
                    build_finding(
                        scanner=self.name,
                        scanner_version=self.version,
                        kind=kind,
                        severity=Severity.MEDIUM,
                        message=kind,
                        span=(at, at + 2),
                    )
                )
        return findings


def test_not_when_a_seam_contributed() -> None:
    clock = Clock()
    settings = GuardianSettings(
        mode=GuardMode.ENFORCE, memory=MemorySettings(reviewed_benign_allows=True)
    )
    memory = InMemoryRiskMemory(settings.memory, clock=clock)
    store = IncidentStore(settings, memory, clock=clock)
    service = ScanService(
        settings,
        clock,
        ScannerPipeline([MarkerScanner()]),
        FeatureCache(settings.cache),
        memory,
        store,
        Metrics(),
    )
    assert scan(service, "Q ZZ").decision is VerdictDecision.ESCALATE
    review_latest(store)
    assert scan(service, "Q ZZ").decision is VerdictDecision.ALLOW

    response = service.scan(
        ScanRequest(
            tenant="acme",
            segments=(
                SegmentInput(text="xQ", provenance=Provenance.USER),
                SegmentInput(text="Q ZZ", provenance=Provenance.USER),
            ),
        )
    )
    second = response.segments[1]
    assert any(outcome.scanner == "seam" for outcome in second.scanners)
    assert second.decision is VerdictDecision.ESCALATE
