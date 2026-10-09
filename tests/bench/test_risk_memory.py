"""The risk-memory corpus: coverage, hygiene, and the measured result it is co-designed with.

Mirrors ``tests/bench/test_tool_calls.py``. Two of these are load-bearing beyond hygiene:

- **no real secret or personal data** -- checked with an explicit regex set and, when importable,
  the project's own secrets scanner;
- **the measured result is pinned** -- zero decisions changed by a precedent, and every hostile
  scenario refused, so that wiring the memory into a verdict is a deliberate change that must also
  update ``docs/evaluations/risk-memory.md``, not an accident.

The last tests run the real runner and the CLI dispatch.
"""

from __future__ import annotations

import json
import re
from collections import Counter

import pytest

from llm_guardian.bench import risk_memory

# -------------------------------------------------------------------------------------------------
# A minimal regex set for real-looking secrets, used always.
# -------------------------------------------------------------------------------------------------
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern)
    for pattern in (
        r"AKIA[0-9A-Z]{16}",
        r"ASIA[0-9A-Z]{16}",
        r"gh[pousr]_[A-Za-z0-9]{36,}",
        r"github_pat_[A-Za-z0-9_]{22,}",
        r"xox[baprs]-[A-Za-z0-9-]{10,}",
        r"sk_live_[0-9a-zA-Z]{20,}",
        r"sk-[A-Za-z0-9]{20,}",
        r"AIza[0-9A-Za-z_\-]{35}",
        r"glpat-[A-Za-z0-9_\-]{20,}",
        r"ya29\.[A-Za-z0-9_\-]{20,}",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    )
)

_THIRD_PARTY_MARKERS: tuple[str, ...] = ("PIGuard", "NotInject", "Prompt Armor", "deepset", "vigil")

_PUBLIC_EMAIL_DOMAINS: tuple[str, ...] = (
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "outlook.com",
    "protonmail.com",
    "icloud.com",
)

_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(r"\b\d{3}-\d{3}-\d{4}\b")

_REQUIRED_FAMILIES: frozenset[str] = frozenset(
    {
        risk_memory.FAMILY_REPEAT_FALSE_POSITIVE,
        risk_memory.FAMILY_REPEAT_TRUE_POSITIVE,
        risk_memory.FAMILY_FIRST_SIGHT,
        risk_memory.FAMILY_NEAR_MISS,
        risk_memory.FAMILY_POISONING,
    }
)


def _scenario_text(scenario: risk_memory.Scenario) -> str:
    parts = [scenario.id, scenario.family, scenario.segment_text, scenario.why]
    if scenario.attack:
        parts.append(scenario.attack)
    if scenario.refusal:
        parts.append(scenario.refusal)
    for precedent in scenario.precedents:
        parts.extend([precedent.precedent_id, precedent.text, precedent.family or ""])
    return " ".join(parts)


# =================================================================================================
# Coverage
# =================================================================================================
def test_has_at_least_forty_scenarios() -> None:
    assert len(risk_memory.CASES) >= 40


def test_ids_are_unique_and_prefixed() -> None:
    ids = [scenario.id for scenario in risk_memory.CASES]
    assert len(ids) == len(set(ids))
    assert all(scenario_id.startswith("rm-") for scenario_id in ids)


def test_required_families_are_covered() -> None:
    families = {scenario.family for scenario in risk_memory.CASES}
    assert families == _REQUIRED_FAMILIES


def test_every_scenario_has_a_reason_and_family() -> None:
    from llm_guardian.provenance import Provenance

    for scenario in risk_memory.CASES:
        assert scenario.why.strip(), scenario.id
        assert scenario.family.strip(), scenario.id
        assert scenario.expectation.strip(), scenario.id
        assert scenario.provenance in {item.value for item in Provenance}, scenario.id


def test_hostile_scenarios_name_the_refusal_path() -> None:
    hostile = [scenario for scenario in risk_memory.CASES if scenario.hostile]
    assert len(hostile) >= 5
    for scenario in hostile:
        assert scenario.attack and scenario.attack.strip(), scenario.id
        assert scenario.refusal and scenario.refusal.strip(), scenario.id


def test_counts_breakdown_is_consistent() -> None:
    breakdown = risk_memory.counts()
    assert breakdown["total"] == len(risk_memory.CASES)
    assert breakdown["hostile"] >= 5
    assert breakdown["declared_changes"] >= 1


# =================================================================================================
# Hygiene
# =================================================================================================
def test_no_scenario_contains_a_real_looking_secret() -> None:
    for scenario in risk_memory.CASES:
        text = _scenario_text(scenario)
        for pattern in _SECRET_PATTERNS:
            assert not pattern.search(text), f"{scenario.id} matches {pattern.pattern}"


def test_no_scenario_carries_third_party_text() -> None:
    for scenario in risk_memory.CASES:
        text = _scenario_text(scenario)
        for marker in _THIRD_PARTY_MARKERS:
            assert marker not in text, f"{scenario.id} references {marker}"


def test_no_scenario_carries_real_personal_data() -> None:
    for scenario in risk_memory.CASES:
        text = _scenario_text(scenario)
        assert not _PHONE.search(text), f"{scenario.id} looks like a phone number"
        for match in _EMAIL.finditer(text):
            domain = match.group(0).rsplit("@", 1)[1].lower()
            assert domain not in _PUBLIC_EMAIL_DOMAINS, f"{scenario.id} uses a public email domain"


def test_module_documents_its_synthesis() -> None:
    docstring = risk_memory.__doc__ or ""
    assert "synthesized" in docstring
    assert "no real secret" in docstring
    assert "design-cost result, not a proof" in docstring


# =================================================================================================
# The measured result (regression guard on the co-designed corpus + memory)
# =================================================================================================
def test_no_scenario_weakened_a_decision_without_a_review() -> None:
    results = risk_memory.run_corpus()
    report = risk_memory.measure(results, generated_at=risk_memory.FIXED_NOW.isoformat())
    assert report.weakened_without_review == 0, [
        attempt.case_id for attempt in report.security_attempts if attempt.weakened
    ]
    assert report.weakened_total == 0


def test_no_precedent_changed_a_verdict_on_this_corpus() -> None:
    results = risk_memory.run_corpus()
    report = risk_memory.measure(results, generated_at=risk_memory.FIXED_NOW.isoformat())
    # The pinned finding: the risk memory is advisory-only, so it changes nothing. Wiring it into
    # a verdict is a deliberate change that must update docs/evaluations/risk-memory.md.
    assert report.decisions_changed_by_a_precedent == 0
    assert report.escalations_removed == 0
    assert report.blocks_removed == 0
    assert report.design_changes_delivered == 0
    assert report.design_expected_changes == len(report.design_changes_missed)


def test_every_hostile_attempt_is_reported_as_refused() -> None:
    results = risk_memory.run_corpus()
    report = risk_memory.measure(results, generated_at=risk_memory.FIXED_NOW.isoformat())
    assert report.security_attempts
    assert all(not attempt.weakened for attempt in report.security_attempts)
    assert all(attempt.refusal for attempt in report.security_attempts)


def test_first_sight_scenarios_have_no_precedent() -> None:
    first_sight = [
        scenario
        for scenario in risk_memory.CASES
        if scenario.family == risk_memory.FAMILY_FIRST_SIGHT
    ]
    assert first_sight
    assert all(not scenario.precedents for scenario in first_sight)


def test_repeat_false_positive_cases_are_reviewed_benign() -> None:
    repeats = [
        scenario
        for scenario in risk_memory.CASES
        if scenario.family == risk_memory.FAMILY_REPEAT_FALSE_POSITIVE
    ]
    assert len(repeats) >= 5
    for scenario in repeats:
        assert scenario.precedents
        assert all(precedent.label == "benign" for precedent in scenario.precedents)
        assert all(precedent.mode == "reviewed" for precedent in scenario.precedents)


def test_corpus_is_deterministic() -> None:
    first = {result.id: result.seeded_decision for result in risk_memory.run_corpus()}
    second = {result.id: result.seeded_decision for result in risk_memory.run_corpus()}
    assert first == second


# =================================================================================================
# CLI
# =================================================================================================
def test_run_cli_emits_json(capsys: pytest.CaptureFixture[str]) -> None:
    code = risk_memory.run_cli(["--json", "--no-latency"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["corpus_total"] == len(risk_memory.CASES)
    assert payload["weakened_without_review"] == 0
    assert payload["decisions_changed_by_a_precedent"] == 0


def test_run_cli_emits_markdown(capsys: pytest.CaptureFixture[str]) -> None:
    code = risk_memory.run_cli(["--no-latency"])
    assert code == 0
    out = capsys.readouterr().out
    assert "## Decision impact" in out
    assert "## The security assertion, measured" in out
    assert "## Latency (in-process)" in out


def test_run_cli_measures_latency_by_default(capsys: pytest.CaptureFixture[str]) -> None:
    code = risk_memory.run_cli(["--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["latency"]) == 4


def test_bench_main_dispatches_to_risk_memory(capsys: pytest.CaptureFixture[str]) -> None:
    from llm_guardian.bench import __main__ as bench_main

    code = bench_main.main(["--risk-memory", "--json", "--no-latency"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["corpus_total"] == len(risk_memory.CASES)


def test_family_impact_covers_every_family() -> None:
    results = risk_memory.run_corpus()
    report = risk_memory.measure(results, generated_at=risk_memory.FIXED_NOW.isoformat())
    assert {impact.family for impact in report.family_impact} == set(risk_memory.FAMILIES)
    by_family = Counter(result.family for result in results)
    assert all(impact.total == by_family[impact.family] for impact in report.family_impact)
