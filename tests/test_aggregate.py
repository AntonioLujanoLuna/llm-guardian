"""Aggregation: per-category scoring, weights, dedupe and scanner errors."""

from __future__ import annotations

from llm_guardian.aggregate import (
    SEVERITY_WEIGHT,
    TOP_FINDINGS_LIMIT,
    aggregate,
    all_findings,
    scanner_errors,
)
from llm_guardian.models import OwaspCategory, Severity
from llm_guardian.settings import ScannerSettings
from tests.helpers import make_finding, make_outcome


def test_severity_weight_ladder() -> None:
    assert SEVERITY_WEIGHT[Severity.LOW] < SEVERITY_WEIGHT[Severity.MEDIUM]
    assert SEVERITY_WEIGHT[Severity.MEDIUM] < SEVERITY_WEIGHT[Severity.HIGH]
    assert SEVERITY_WEIGHT[Severity.HIGH] < SEVERITY_WEIGHT[Severity.CRITICAL]
    assert SEVERITY_WEIGHT[Severity.CRITICAL] == 1.0


def test_category_score_is_max_not_sum() -> None:
    outcomes = [
        make_outcome(
            make_finding(severity=Severity.LOW),
            make_finding(kind="other", severity=Severity.LOW),
        )
    ]
    signals = aggregate(outcomes, ScannerSettings())
    assert signals.category_scores[OwaspCategory.LLM01] == 0.25


def test_overall_score_is_the_worst_category() -> None:
    outcomes = [
        make_outcome(make_finding(severity=Severity.HIGH)),
        make_outcome(make_finding(kind="x", severity=Severity.LOW, owasp=(OwaspCategory.LLM02,))),
    ]
    signals = aggregate(outcomes, ScannerSettings())
    assert signals.score == 0.75


def test_uncategorised_finding_contributes_to_score() -> None:
    outcomes = [make_outcome(make_finding(severity=Severity.HIGH, owasp=()))]
    signals = aggregate(outcomes, ScannerSettings())
    assert signals.score == 0.75
    assert signals.category_scores == {}


def test_scanner_weight_multiplies() -> None:
    outcomes = [make_outcome(make_finding(severity=Severity.HIGH))]
    signals = aggregate(outcomes, ScannerSettings(weights={"rules": 0.5}))
    assert signals.category_scores[OwaspCategory.LLM01] == 0.375


def test_score_capped_at_one() -> None:
    outcomes = [make_outcome(make_finding(severity=Severity.CRITICAL))]
    signals = aggregate(outcomes, ScannerSettings(weights={"rules": 5.0}))
    assert signals.score == 1.0


def test_top_findings_dedupe_by_kind() -> None:
    outcomes = [
        make_outcome(
            make_finding(kind="dup", severity=Severity.LOW),
            make_finding(kind="dup", severity=Severity.HIGH),
            make_finding(kind="unique", severity=Severity.MEDIUM),
        )
    ]
    signals = aggregate(outcomes, ScannerSettings())
    kinds = [f.kind for f in signals.top_findings]
    assert kinds.count("dup") == 1
    assert signals.top_findings[0].kind == "dup"
    assert signals.top_findings[0].severity is Severity.HIGH


def test_top_findings_is_bounded() -> None:
    findings = tuple(make_finding(kind=f"k{i}", severity=Severity.LOW) for i in range(10))
    signals = aggregate([make_outcome(*findings)], ScannerSettings())
    assert len(signals.top_findings) == TOP_FINDINGS_LIMIT


def test_matched_rules_sorted_and_unique() -> None:
    outcomes = [make_outcome(make_finding(kind="b"), make_finding(kind="a"))]
    signals = aggregate(outcomes, ScannerSettings())
    assert signals.matched_rules == ("a", "b")


def test_scanner_errors_reported() -> None:
    outcomes = [make_outcome(error="RuntimeError: boom", scanner="rules")]
    signals = aggregate(outcomes, ScannerSettings())
    assert signals.scanner_errors == ("rules: RuntimeError: boom",)


def test_helpers() -> None:
    outcomes = [make_outcome(make_finding(), error="x", scanner="rules")]
    assert len(all_findings(outcomes)) == 1
    assert scanner_errors(outcomes) == ["rules: x"]


def test_no_findings_is_zero() -> None:
    signals = aggregate([make_outcome()], ScannerSettings())
    assert signals.score == 0.0
    assert signals.category_scores == {}
    assert signals.matched_rules == ()
