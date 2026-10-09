"""Verdict engine: thresholds, floor, category overrides, error policy and fail-closed."""

from __future__ import annotations

from llm_guardian.models import (
    GuardMode,
    ScannerErrorPolicy,
    Severity,
    VerdictDecision,
)
from llm_guardian.provenance import Provenance
from llm_guardian.settings import GuardianSettings, ScannerSettings, ThresholdSettings
from llm_guardian.verdicts import VerdictEngine, worst_decision
from tests.helpers import make_finding, make_outcome, make_segment


def make_engine(
    *,
    thresholds: ThresholdSettings | None = None,
    scanners: ScannerSettings | None = None,
) -> VerdictEngine:
    return VerdictEngine(
        GuardianSettings(
            thresholds=thresholds or ThresholdSettings(),
            scanners=scanners or ScannerSettings(),
        )
    )


def decide(engine: VerdictEngine, outcomes, segment=None, mode=None):
    from llm_guardian.aggregate import aggregate

    seg = segment or make_segment()
    signals = aggregate(outcomes, engine._settings.scanners)
    return engine.decide(seg, signals, outcomes, mode=mode)


def test_clean_allows() -> None:
    verdict = decide(make_engine(), [make_outcome()])
    assert verdict.decision is VerdictDecision.ALLOW
    assert verdict.score == 0.0


def test_medium_escalates() -> None:
    verdict = decide(make_engine(), [make_outcome(make_finding(severity=Severity.MEDIUM))])
    assert verdict.decision is VerdictDecision.ESCALATE
    assert verdict.threshold == 0.35


def test_high_blocks_and_reports_block_threshold() -> None:
    verdict = decide(make_engine(), [make_outcome(make_finding(severity=Severity.HIGH))])
    assert verdict.decision is VerdictDecision.BLOCK
    assert verdict.threshold == 0.75


def test_critical_hits_severity_floor() -> None:
    verdict = decide(make_engine(), [make_outcome(make_finding(severity=Severity.CRITICAL))])
    assert verdict.decision is VerdictDecision.BLOCK
    assert any(note.startswith("severity_floor_block") for note in verdict.notes)


def test_category_override_blocks_low_score() -> None:
    engine = make_engine(
        thresholds=ThresholdSettings(category_overrides={"LLM01": {"block_at": 0.4}})
    )
    verdict = decide(engine, [make_outcome(make_finding(severity=Severity.MEDIUM))])
    assert verdict.decision is VerdictDecision.BLOCK
    assert verdict.threshold == 0.4
    assert "category_block:LLM01" in verdict.notes


def test_category_override_escalates() -> None:
    engine = make_engine(
        thresholds=ThresholdSettings(category_overrides={"LLM01": {"escalate_at": 0.2}})
    )
    verdict = decide(engine, [make_outcome(make_finding(severity=Severity.MEDIUM))])
    assert verdict.decision is VerdictDecision.ESCALATE
    assert verdict.threshold == 0.2


def test_scanner_error_warn_lifts_allow_to_escalate() -> None:
    verdict = decide(make_engine(), [make_outcome(error="RuntimeError: boom")])
    assert verdict.decision is VerdictDecision.ESCALATE
    assert any(note.startswith("scanner_error_warn_escalate") for note in verdict.notes)


def test_scanner_error_warn_keeps_existing_block() -> None:
    outcomes = [make_outcome(make_finding(severity=Severity.HIGH), error="boom")]
    verdict = decide(make_engine(), outcomes)
    assert verdict.decision is VerdictDecision.BLOCK
    assert any(note.startswith("scanner_error_warn:") for note in verdict.notes)


def test_scanner_error_block_policy() -> None:
    engine = make_engine(scanners=ScannerSettings(error_policy=ScannerErrorPolicy.BLOCK))
    outcomes = [make_outcome(make_finding(severity=Severity.LOW), error="boom")]
    verdict = decide(engine, outcomes)
    assert verdict.decision is VerdictDecision.BLOCK
    assert "scanner_error_block" in verdict.notes


def test_high_impact_provenance_fails_closed_on_error() -> None:
    engine = make_engine(scanners=ScannerSettings(error_policy=ScannerErrorPolicy.WARN))
    segment = make_segment(provenance=Provenance.TOOL_RESULT)
    verdict = decide(engine, [make_outcome(error="boom")], segment=segment)
    assert verdict.decision is VerdictDecision.BLOCK
    assert "high_impact_scanner_error_fail_closed" in verdict.notes


def test_truncated_low_risk_does_not_lower_or_block() -> None:
    segment = make_segment(truncated=True, provenance=Provenance.USER)
    verdict = decide(make_engine(), [make_outcome(make_finding(severity=Severity.MEDIUM))], segment)
    assert verdict.decision is VerdictDecision.ESCALATE
    assert "truncated_segment_not_evidence_of_safety" in verdict.notes


def test_truncated_high_impact_blocks() -> None:
    segment = make_segment(truncated=True, provenance=Provenance.TOOL_CALL)
    verdict = decide(make_engine(), [make_outcome()], segment)
    assert verdict.decision is VerdictDecision.BLOCK
    assert "high_impact_truncated_fail_closed" in verdict.notes


def test_cached_features_flag() -> None:
    all_cached = decide(make_engine(), [make_outcome(make_finding(), cached=True)])
    assert all_cached.cached_features is True
    fresh = decide(make_engine(), [make_outcome(make_finding(), cached=False)])
    assert fresh.cached_features is False
    assert decide(make_engine(), [make_outcome()]).cached_features is False


def test_shadow_mode_adds_advisory_note_but_computes_verdict() -> None:
    verdict = decide(
        make_engine(), [make_outcome(make_finding(severity=Severity.HIGH))], mode=GuardMode.SHADOW
    )
    assert verdict.decision is VerdictDecision.BLOCK
    assert "shadow_mode:verdict_is_advisory" in verdict.notes


def test_memory_evidence_is_advisory() -> None:
    from llm_guardian.memory.base import Neighbor, evidence_from
    from llm_guardian.models import IncidentStatus

    evidence = evidence_from(
        [Neighbor(case_id="a", label=IncidentStatus.CONFIRMED_ATTACK, similarity=0.9, digest="d")],
        k=5,
        query_ms=0.1,
    )
    engine = make_engine()
    signals = engine._settings
    from llm_guardian.aggregate import aggregate

    seg = make_segment()
    verdict = engine.decide(
        seg, aggregate([make_outcome()], signals.scanners), [make_outcome()], evidence
    )
    assert verdict.decision is VerdictDecision.ALLOW  # cosine never blocks, never whitelists
    assert any(note.startswith("memory_consulted") for note in verdict.notes)
    assert "cosine_is_not_a_permission" in verdict.notes


def test_worst_decision() -> None:
    allow = decide(make_engine(), [make_outcome()])
    escalate = decide(make_engine(), [make_outcome(make_finding(severity=Severity.MEDIUM))])
    block = decide(make_engine(), [make_outcome(make_finding(severity=Severity.HIGH))])
    assert worst_decision([allow, escalate, block]) is VerdictDecision.BLOCK
    assert worst_decision([allow, escalate]) is VerdictDecision.ESCALATE
    assert worst_decision([allow]) is VerdictDecision.ALLOW
    assert worst_decision([]) is VerdictDecision.ALLOW
