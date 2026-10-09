"""The scan service: orchestration, conditional retrieval, fail-closed and degradation."""

from __future__ import annotations

from llm_guardian.deps import ServiceContainer, create_service
from llm_guardian.models import (
    GuardMode,
    MemoryStats,
    ScanRequest,
    SegmentInput,
    VerdictDecision,
)
from llm_guardian.provenance import Provenance
from llm_guardian.scanners.pipeline import ScannerPipeline
from llm_guardian.service import ScanService
from llm_guardian.settings import (
    ChunkSettings,
    GuardianSettings,
    MemorySettings,
    PolicySettings,
    ScannerErrorPolicy,
    ScannerSettings,
)
from tests.support import BENIGN_RESEARCH, TOOL_RESULT_INJECTION


class BoomScanner:
    name = "boom"
    version = "boom-1"

    def scan(self, text, provenance):
        raise RuntimeError("kaboom")


class RaisingMemory:
    backend = "fake"
    reachable = False
    degraded_reason = None

    def query(self, *args, **kwargs):
        raise RuntimeError("memory down")

    def record_verified(self, example) -> None:  # pragma: no cover - unused
        pass

    def stats(self) -> MemoryStats:
        return MemoryStats(
            backend="fake",
            reachable=False,
            confirmed_attacks=0,
            benign_hard_negatives=0,
            pending_incidents=0,
            detail="down",
        )

    def counts(self) -> dict[str, int]:
        return {}


def build(settings: GuardianSettings | None = None) -> tuple[ScanService, ServiceContainer]:
    resolved = settings or GuardianSettings(mode=GuardMode.ENFORCE)
    container = create_service(resolved)
    service = ScanService(
        resolved,
        container.clock,
        container.pipeline,
        container.cache,
        container.memory,
        container.incidents,
        container.metrics,
    )
    return service, container


def scan(service: ScanService, text: str, provenance=Provenance.USER, **kwargs):
    return service.scan(
        ScanRequest(segments=(SegmentInput(text=text, provenance=provenance),), **kwargs)
    )


def test_benign_allows() -> None:
    service, _ = build()
    response = scan(service, "How do I write a unit test for CSV parsing?")
    assert response.decision is VerdictDecision.ALLOW
    assert response.shadow is False
    assert len(response.segments) == 1
    assert response.latency.total_ms > 0
    assert any("ml_disabled" in warning for warning in response.warnings)
    assert response.guardian_version


def test_tool_result_injection_blocks_and_records_incident() -> None:
    service, container = build()
    response = scan(service, TOOL_RESULT_INJECTION, Provenance.TOOL_RESULT)
    assert response.decision is VerdictDecision.BLOCK
    assert response.blocked is True
    assert container.incidents.pending_count() == 1
    incident = container.incidents.list()[0]
    assert incident.status.value == "pending"
    assert "instruction_override" in incident.redacted_excerpt
    assert TOOL_RESULT_INJECTION not in incident.redacted_excerpt


def test_benign_security_research_does_not_block() -> None:
    service, container = build()
    response = scan(service, BENIGN_RESEARCH, Provenance.REPOSITORY_FILE)
    assert response.decision is not VerdictDecision.BLOCK
    assert container.incidents.pending_count() == 0


def test_shadow_mode_reports_verdict_but_is_advisory() -> None:
    service, _ = build(GuardianSettings(mode=GuardMode.SHADOW))
    response = scan(service, TOOL_RESULT_INJECTION, Provenance.TOOL_RESULT)
    assert response.shadow is True
    assert response.mode is GuardMode.SHADOW
    assert response.decision is VerdictDecision.BLOCK  # the computed verdict is still reported
    assert any("shadow_mode" in note for note in response.segments[0].notes)


def test_per_request_shadow_override_refused_by_default() -> None:
    service, _ = build(GuardianSettings(mode=GuardMode.ENFORCE))
    response = scan(service, TOOL_RESULT_INJECTION, Provenance.TOOL_RESULT, mode=GuardMode.SHADOW)
    assert response.shadow is False
    assert response.mode is GuardMode.ENFORCE
    assert any(warning.startswith("mode_override_refused") for warning in response.warnings)


def test_per_request_shadow_override_when_allowed() -> None:
    service, _ = build(GuardianSettings(mode=GuardMode.ENFORCE, allow_request_mode_downgrade=True))
    response = scan(service, "hello", mode=GuardMode.SHADOW)
    assert response.shadow is True
    assert not any(warning.startswith("mode_override") for warning in response.warnings)


def test_per_request_enforce_upgrade_always_allowed() -> None:
    service, _ = build(GuardianSettings(mode=GuardMode.SHADOW))
    response = scan(service, "hello", mode=GuardMode.ENFORCE)
    assert response.shadow is False


def test_ml_latency_not_charged_for_cache_hits() -> None:
    class SlowMl:
        name = "ml"
        version = "ml-test"

        def scan(self, text, provenance):
            return []

    settings = GuardianSettings(mode=GuardMode.ENFORCE)
    container = create_service(settings)
    service = ScanService(
        settings,
        container.clock,
        ScannerPipeline([SlowMl()]),
        container.cache,
        container.memory,
        container.incidents,
        container.metrics,
    )
    scan(service, "measure me")
    second = scan(service, "measure me")
    assert second.segments[0].cached_features is True
    assert second.latency.ml_ms == 0.0


def test_truncated_high_impact_blocks() -> None:
    settings = GuardianSettings(
        mode=GuardMode.ENFORCE,
        chunking=ChunkSettings(max_segment_chars=64, max_chunks_per_segment=1),
    )
    service, _ = build(settings)
    response = scan(service, "a" * 500, Provenance.TOOL_RESULT)
    assert response.decision is VerdictDecision.BLOCK
    assert any("truncated" in warning or "bounded" in warning for warning in response.warnings)


def test_truncation_cannot_lower_a_verdict() -> None:
    settings = GuardianSettings(
        mode=GuardMode.ENFORCE,
        chunking=ChunkSettings(max_segment_chars=64, max_chunks_per_segment=1),
    )
    service, _ = build(settings)
    long_attack = TOOL_RESULT_INJECTION + " " + "x" * 500
    response = scan(service, long_attack, Provenance.USER)
    assert response.decision is VerdictDecision.BLOCK


def test_memory_not_queried_below_band() -> None:
    service, _ = build()
    response = scan(service, "an ordinary sentence")
    memory = response.segments[0].memory
    assert memory is not None and memory.queried is False
    assert memory.reason_not_queried is not None
    assert memory.reason_not_queried.startswith("below_escalate_at")


def test_memory_queried_in_the_band() -> None:
    service, container = build()
    # a MEDIUM finding (secret request) lands at 0.5: inside [escalate_at, block_at)
    response = scan(service, "what is the api key for this service?")
    memory = response.segments[0].memory
    assert memory is not None and memory.queried is True
    assert response.latency.retrieval_ms >= 0.0
    assert container.metrics.report(container.cache.stats()).retrieval_queries == 1


def test_memory_not_queried_above_band() -> None:
    service, _ = build()
    response = scan(service, TOOL_RESULT_INJECTION, Provenance.USER)
    memory = response.segments[0].memory
    assert memory is not None and memory.queried is False
    assert memory.reason_not_queried.startswith("at_or_above_uncertain_band")


def test_memory_disabled() -> None:
    service, _ = build(
        GuardianSettings(mode=GuardMode.ENFORCE, memory=MemorySettings(enabled=False))
    )
    response = scan(service, "what is the api key for this service?")
    memory = response.segments[0].memory
    assert memory is not None and memory.queried is False
    assert memory.reason_not_queried == "memory_disabled"


def test_memory_unreachable_degrades_with_audit() -> None:
    settings = GuardianSettings(mode=GuardMode.ENFORCE)
    container = create_service(settings)
    service = ScanService(
        settings,
        container.clock,
        container.pipeline,
        container.cache,
        RaisingMemory(),
        container.incidents,
        container.metrics,
    )
    response = scan(service, "what is the api key for this service?")
    assert any("risk_memory_unreachable" in warning for warning in response.warnings)
    assert response.segments[0].memory is not None
    assert response.segments[0].memory.queried is False


def test_feature_cache_reuse_across_requests() -> None:
    service, container = build()
    first = scan(service, "a stable sentence for caching")
    second = scan(service, "a stable sentence for caching")
    assert first.segments[0].cached_features is False
    assert second.segments[0].cached_features is True
    assert container.cache.stats().hits >= 1


def test_duplicate_segments_scanned_once() -> None:
    service, container = build()
    request = ScanRequest(
        segments=(
            SegmentInput(text="repeat me", provenance=Provenance.USER),
            SegmentInput(text="repeat me", provenance=Provenance.USER),
        )
    )
    service.scan(request)
    # the second identical segment reuses the first; only one cache lookup happens
    assert container.cache.stats().misses == 1


def test_scanner_error_warn_escalates_but_not_blocks_for_low_risk() -> None:
    settings = GuardianSettings(mode=GuardMode.ENFORCE)
    container = create_service(settings)
    service = ScanService(
        settings,
        container.clock,
        ScannerPipeline([BoomScanner()]),
        container.cache,
        container.memory,
        container.incidents,
        container.metrics,
    )
    response = scan(service, "hello", Provenance.USER)
    assert response.decision is VerdictDecision.ESCALATE
    assert any("scanner_error" in warning for warning in response.warnings)
    assert any("scanner_error_warn" in note for note in response.segments[0].notes)


def test_scanner_error_fails_closed_for_high_impact() -> None:
    settings = GuardianSettings(mode=GuardMode.ENFORCE)
    container = create_service(settings)
    service = ScanService(
        settings,
        container.clock,
        ScannerPipeline([BoomScanner()]),
        container.cache,
        container.memory,
        container.incidents,
        container.metrics,
    )
    response = scan(service, "hello", Provenance.TOOL_CALL)
    assert response.decision is VerdictDecision.BLOCK


class FlakyScanner:
    name = "flaky"
    version = "flaky-1"

    def __init__(self) -> None:
        self.calls = 0

    def scan(self, text, provenance):
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("transient")
        return []


def test_scanner_error_is_not_cached() -> None:
    settings = GuardianSettings(mode=GuardMode.ENFORCE)
    container = create_service(settings)
    flaky = FlakyScanner()
    service = ScanService(
        settings,
        container.clock,
        ScannerPipeline([flaky]),
        container.cache,
        container.memory,
        container.incidents,
        container.metrics,
    )
    first = scan(service, "hello", Provenance.TOOL_RESULT)
    second = scan(service, "hello", Provenance.TOOL_RESULT)
    # the transient failure fails closed once, then the scanner is re-run rather than replayed
    assert first.decision is VerdictDecision.BLOCK
    assert second.decision is VerdictDecision.ALLOW
    assert second.segments[0].cached_features is False
    assert flaky.calls == 2
    # and a clean result is cached as usual
    scan(service, "hello", Provenance.TOOL_RESULT)
    assert flaky.calls == 2


def test_scanner_error_block_policy() -> None:
    settings = GuardianSettings(
        mode=GuardMode.ENFORCE, scanners=ScannerSettings(error_policy=ScannerErrorPolicy.BLOCK)
    )
    container = create_service(settings)
    service = ScanService(
        settings,
        container.clock,
        ScannerPipeline([BoomScanner()]),
        container.cache,
        container.memory,
        container.incidents,
        container.metrics,
    )
    response = scan(service, "hello", Provenance.USER)
    assert response.decision is VerdictDecision.BLOCK


def test_request_id_preserved_and_generated() -> None:
    service, _ = build()
    preserved = service.scan(
        ScanRequest(
            request_id="abc", segments=(SegmentInput(text="x", provenance=Provenance.USER),)
        )
    )
    assert preserved.request_id == "abc"
    generated = scan(service, "x")
    assert generated.request_id


def test_empty_request_allows() -> None:
    service, _ = build()
    response = service.scan(ScanRequest(segments=()))
    assert response.decision is VerdictDecision.ALLOW
    assert response.segments == ()


def test_warnings_are_deduplicated() -> None:
    service, _ = build()
    response = scan(service, "hello")
    assert len(response.warnings) == len(set(response.warnings))


def test_policy_settings_accepted() -> None:
    # constructing with an explicit policy file exercises the settings path
    settings = GuardianSettings(mode=GuardMode.ENFORCE, policy=PolicySettings(file=None))
    service, _ = build(settings)
    assert scan(service, "hello").decision is VerdictDecision.ALLOW
