"""The tool-call corpus: coverage, hygiene, and the measured result it is co-designed with.

Mirrors ``tests/bench/test_corpus.py`` for the tool boundary. Two of these are load-bearing beyond
hygiene:

- **no real secret or personal data** -- if a case embedded a credential or a real address the
  corpus would be a liability, so it is checked with the project's own secrets scanner when that
  module is importable, and always with an explicit regex set as a backstop;
- **no third-party text** -- the corpus must be self-authored, so we assert the module documents
  synthesis and carries no other project's text.

The last test runs the real runner and pins the co-designed result (0 false denials, every
malicious case denied). It is a regression guard: if someone adds a case or relaxes a rule, the
corpus and the policy must be brought back into agreement deliberately.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest

from llm_guardian.bench import policy_report, tool_calls

# -------------------------------------------------------------------------------------------------
# A minimal regex set for real-looking secrets, used always (and as a fallback for the scanner).
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
        r"glpat-[A-Za-z0-9_\-]{20}",
        r"ya29\.[A-Za-z0-9_\-]{20,}",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        r"eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}",
    )
)

# Strings that would indicate copied third-party dataset material rather than synthesis.
_THIRD_PARTY_MARKERS: tuple[str, ...] = (
    "PIGuard",
    "NotInject",
    "Prompt Armor",
    "deepset",
    "jackhhao",
    "leolee99",
)

#: Domains that would make a case carry a real person's address rather than a synthetic one.
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

_REQUIRED_MALICIOUS_FAMILIES: frozenset[str] = frozenset(
    {
        "exfiltration_egress",
        "external_recipient",
        "host_pivot",
        "sensitive_file_read",
        "path_traversal_write",
        "cross_tenant_read",
        "cross_tenant_write",
        "destructive_shell",
        "curl_pipe_shell",
        "shell_metacharacter",
        "capability_escalation",
        "plugin_mass_install",
        "confused_deputy",
        "undeclared_argument",
        "argument_too_long",
        "missing_required_argument",
        "argument_pattern_violation",
        "unauthorized_tool",
    }
)

_REQUIRED_LEGITIMATE_FAMILIES: frozenset[str] = frozenset(
    {
        "file_read_workspace",
        "search_workspace",
        "run_test_suite",
        "own_database_query",
        "internal_channel_post",
        "internal_mail",
        "open_pull_request",
        "deploy_to_staging",
        "fetch_dependency",
        "update_owned_ticket",
        "awkward_legitimate",
    }
)

_INJECTION_KINDS: frozenset[str] = frozenset(
    {
        "agent_directive_control_override",
        "agent_directive_adopt_as_instruction",
        "tool_call_instruction",
    }
)

_SEVERITIES: frozenset[str] = frozenset({"info", "low", "medium", "high", "critical"})


def _case_text(case: tool_calls.ToolCallCase) -> str:
    """Every string a case carries, so a hygiene check sees all of it."""
    parts = [case.id, case.principal, case.tool, case.family, case.why, case.proposed_by]
    parts.extend(str(value) for value in case.tool_arguments.values())
    for item in case.evidence:
        parts.extend([item.segment_id, item.finding_kind, item.severity, item.provenance])
    return " ".join(parts)


def _project_secrets_scan(text: str) -> list[str] | None:
    try:
        from llm_guardian.scanners import secrets as secrets_module
    except Exception:
        return None
    for name in ("scan_secrets", "find_secrets", "scan", "detect"):
        function = getattr(secrets_module, name, None)
        if callable(function):
            try:
                result = function(text)
            except Exception:
                return None
            return [str(item) for item in result] if result else []
    return None


# =================================================================================================
# Coverage
# =================================================================================================
def test_has_at_least_150_cases() -> None:
    assert len(tool_calls.CASES) >= 150


def test_ids_are_unique_and_prefixed() -> None:
    ids = [case.id for case in tool_calls.CASES]
    assert len(ids) == len(set(ids))
    assert all(case_id.startswith("tc-") for case_id in ids)


def test_both_labels_are_well_represented() -> None:
    by_label = Counter(case.label for case in tool_calls.CASES)
    assert by_label[tool_calls.LEGITIMATE] >= 60
    assert by_label[tool_calls.MALICIOUS] >= 60
    assert set(by_label) <= tool_calls.LABELS


def test_at_least_ten_malicious_families() -> None:
    families = {case.family for case in tool_calls.CASES if case.label == tool_calls.MALICIOUS}
    assert len(families) >= 10


def test_required_malicious_families_are_covered() -> None:
    families = {case.family for case in tool_calls.CASES if case.label == tool_calls.MALICIOUS}
    assert families >= _REQUIRED_MALICIOUS_FAMILIES


def test_required_legitimate_families_are_covered() -> None:
    families = {case.family for case in tool_calls.CASES if case.label == tool_calls.LEGITIMATE}
    assert families >= _REQUIRED_LEGITIMATE_FAMILIES


def test_every_case_has_a_reason_and_family() -> None:
    for case in tool_calls.CASES:
        assert case.why.strip(), case.id
        assert case.family.strip(), case.id
        assert case.principal.strip(), case.id


def test_actions_and_provenance_are_known() -> None:
    for case in tool_calls.CASES:
        assert case.action in {"tool_call", "retrieval", "egress", "budget"}, case.id
        assert case.proposed_by in tool_calls.PROVENANCE_VALUES, case.id
        for item in case.evidence:
            assert item.provenance in tool_calls.PROVENANCE_VALUES, case.id
            assert item.severity in _SEVERITIES, case.id
            assert item.finding_kind.strip(), case.id
            assert item.segment_id.strip(), case.id


def test_confused_deputy_cases_carry_a_triggering_finding() -> None:
    deputies = [case for case in tool_calls.CASES if case.family == "confused_deputy"]
    assert len(deputies) >= 6
    for case in deputies:
        assert case.evidence, case.id
        assert any(
            item.finding_kind in _INJECTION_KINDS
            and item.provenance in {"tool_result", "retrieved_chunk", "repository_file"}
            for item in case.evidence
        ), case.id
        assert case.tool_arguments, case.id  # arguments are valid; only the provenance is poisoned


def test_counts_breakdown_is_consistent() -> None:
    breakdown = tool_calls.counts()
    assert breakdown["total"] == len(tool_calls.CASES)
    assert breakdown["by_label"][tool_calls.LEGITIMATE] + breakdown["by_label"][
        tool_calls.MALICIOUS
    ] == len(tool_calls.CASES)
    assert breakdown["malicious_family_count"] >= 10


# =================================================================================================
# Hygiene
# =================================================================================================
def test_no_case_contains_a_real_looking_secret() -> None:
    for case in tool_calls.CASES:
        text = _case_text(case)
        for pattern in _SECRET_PATTERNS:
            assert not pattern.search(text), f"{case.id} matches {pattern.pattern}"
        scanned = _project_secrets_scan(text)
        if scanned is not None:
            assert not scanned, f"{case.id}: project secrets scanner flagged {scanned}"


def test_no_case_carries_third_party_dataset_text() -> None:
    for case in tool_calls.CASES:
        text = _case_text(case)
        for marker in _THIRD_PARTY_MARKERS:
            assert marker not in text, f"{case.id} references {marker}"


def test_no_case_carries_real_personal_data() -> None:
    for case in tool_calls.CASES:
        text = _case_text(case)
        assert not _PHONE.search(text), f"{case.id} looks like a phone number"
        for match in _EMAIL.finditer(text):
            domain = match.group(0).rsplit("@", 1)[1].lower()
            assert domain not in _PUBLIC_EMAIL_DOMAINS, f"{case.id} uses a public email domain"


def test_module_documents_its_synthesis() -> None:
    docstring = tool_calls.__doc__ or ""
    assert "synthesized" in docstring
    assert "No text is copied" in docstring
    assert "no real secret" in docstring


# =================================================================================================
# The measured result (regression guard on the co-designed corpus + policy)
# =================================================================================================
def _example_policy() -> Path:
    return Path(__file__).resolve().parents[2] / "policies" / "example.toml"


def test_runner_holds_the_co_designed_result() -> None:
    policy_file = _example_policy()
    assert policy_file.is_file(), f"missing policy file: {policy_file}"
    outcomes = policy_report.evaluate(tool_calls.CASES, policy_report.build_engine(policy_file))
    report = policy_report.measure(outcomes, policy_file)
    assert report.corpus_total == len(tool_calls.CASES)
    assert report.false_denials == 0, [o.case_id for o in report.false_denial_cases]
    assert report.malicious_caught == report.malicious
    assert report.missed_cases == ()


def test_runner_cli_emits_json(capsys: pytest.CaptureFixture[str]) -> None:
    code = policy_report.run_cli(["--policy", str(_example_policy()), "--json"])
    assert code == 0
    import json

    payload = json.loads(capsys.readouterr().out)
    assert payload["corpus_total"] == len(tool_calls.CASES)
    assert payload["false_denials"] == 0


def test_runner_cli_missing_policy_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = policy_report.run_cli(["--policy", str(tmp_path / "nope.toml")])
    assert code == 2
    assert "not found" in capsys.readouterr().err


def test_runner_cli_emits_markdown(capsys: pytest.CaptureFixture[str]) -> None:
    code = policy_report.run_cli(["--policy", str(_example_policy())])
    assert code == 0
    out = capsys.readouterr().out
    assert "## Headline" in out
    assert "False denials per 1,000 legitimate tool calls" in out
    assert "Against the naive alternatives" in out


def test_bench_main_dispatches_to_policy_mode(capsys: pytest.CaptureFixture[str]) -> None:
    from llm_guardian.bench import __main__ as bench_main

    code = bench_main.main(["--tool-boundary", "--json", "--policy", str(_example_policy())])
    assert code == 0
    import json

    payload = json.loads(capsys.readouterr().out)
    assert payload["false_denials"] == 0


def test_bench_main_dispatches_to_detector_harness(capsys: pytest.CaptureFixture[str]) -> None:
    from llm_guardian.bench import __main__ as bench_main

    code = bench_main.main(
        ["--stub", "allow", "--json", "--warmup", "0", "--family", "instruction_override"]
    )
    assert code == 0
    import json

    assert json.loads(capsys.readouterr().out)["corpus_size"] >= 1
