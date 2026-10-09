"""Corpus integrity: labels, coverage, provenance, and the honesty claims the design requires.

Two of these tests are load-bearing beyond hygiene:

- **no real secret** -- if a case accidentally embedded a credential shape the corpus would be a
  liability, so it is checked with the project's own secrets scanner when that module is
  importable, and always with an explicit regex set as a backstop;
- **no third-party text** -- the corpus must be self-authored, so we assert the module documents
  synthesis and that no case carries another project's text or links.
"""

from __future__ import annotations

import re
from collections import Counter
from importlib.util import find_spec
from pathlib import Path

import pytest

from llm_guardian.bench import corpus

# -------------------------------------------------------------------------------------------------
# A minimal regex set for real-looking secrets, used always (and as a fallback for the scanner).
# -------------------------------------------------------------------------------------------------
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern)
    for pattern in (
        r"AKIA[0-9A-Z]{16}",  # AWS access key id
        r"ASIA[0-9A-Z]{16}",  # AWS temporary key id
        r"gh[pousr]_[A-Za-z0-9]{36,}",  # GitHub token
        r"github_pat_[A-Za-z0-9_]{22,}",  # GitHub fine-grained PAT
        r"xox[baprs]-[A-Za-z0-9-]{10,}",  # Slack token
        r"sk_live_[0-9a-zA-Z]{20,}",  # Stripe live secret
        r"sk-[A-Za-z0-9]{20,}",  # OpenAI-style key
        r"AIza[0-9A-Za-z_\-]{35}",  # Google API key
        r"glpat-[A-Za-z0-9_\-]{20}",  # GitLab PAT
        r"ya29\.[A-Za-z0-9_\-]{20,}",  # Google OAuth token
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",  # PEM private key
        r"eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}",  # JWT
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

_SPANISH_MARKERS: tuple[str, ...] = (
    "á",
    "é",
    "í",
    "ó",
    "ú",
    "ñ",
    "¿",
    "¡",
    " que ",
    " para ",
    " con ",
    " una ",
    " los ",
    " las ",
    " del ",
    " que ",
)

_ENGLISH_MARKERS: tuple[str, ...] = (" the ", " and ", " for ", " with ", " this ", " that ")


def _project_secrets_scan(text: str) -> list[str] | None:
    """Use the project's secrets scanner if its module is importable, else return None.

    The scanner is another module's responsibility and its exact API is not ours to pin; if it is
    absent or does not expose a usable callable we fall back to the regex set above.
    """

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


def test_has_at_least_120_cases() -> None:
    assert len(corpus.CASES) >= 120


def test_ids_are_unique_and_prefixed() -> None:
    ids = [case.id for case in corpus.CASES]
    assert len(ids) == len(set(ids))
    assert all(case_id.startswith("bench-") for case_id in ids)


def test_both_labels_are_well_represented() -> None:
    by_label = Counter(case.label for case in corpus.CASES)
    assert by_label["attack"] >= 60
    assert by_label["benign"] >= 60
    assert set(by_label) == {"attack", "benign"}


def test_both_languages_are_present_in_both_labels() -> None:
    attacks = [case for case in corpus.CASES if case.label == "attack"]
    benign = [case for case in corpus.CASES if case.label == "benign"]
    for group in (attacks, benign):
        languages = {case.language for case in group}
        assert {"en", "es"} <= languages


def test_at_least_twelve_attack_families() -> None:
    families = {case.family for case in corpus.CASES if case.label == "attack"}
    assert len(families) >= 12


def test_required_attack_families_are_covered() -> None:
    families = {case.family for case in corpus.CASES if case.label == "attack"}
    required = {
        "instruction_override",
        "system_prompt_exfiltration",
        "delimiter_role_spoofing",
        "tool_call_injection",
        "indirect_injection_repo",
        "invisible_text_smuggling",
        "obfuscated_payload",
        "exfiltration_request",
        "agent_tool_misuse",
        "jailbreak_persona",
        "long_context_buried",
    }
    assert required <= families


def test_required_benign_families_are_covered() -> None:
    families = {case.family for case in corpus.CASES if case.label == "benign"}
    required = {
        "benign_coding",
        "benign_repo_readme",
        "benign_spanish_engineering",
        "benign_security_explainer",
        "benign_incident_writeup",
        "benign_owasp_checklist",
        "benign_redteam_template",
        "benign_concept_discussion",
        "benign_cve_description",
    }
    assert required <= families


def test_every_case_has_a_reason_and_family() -> None:
    for case in corpus.CASES:
        assert case.why.strip(), case.id
        assert case.family.strip(), case.id
        assert case.text.strip(), case.id


def test_provenance_values_are_known() -> None:
    for case in corpus.CASES:
        assert case.provenance in corpus.PROVENANCE_VALUES, case.id


def test_attack_cases_carry_owasp_categories() -> None:
    for case in corpus.CASES:
        if case.label == "attack":
            assert case.owasp, case.id
            assert all(category.startswith("LLM") for category in case.owasp), case.id


def test_long_context_cases_exceed_three_kilobytes() -> None:
    long_cases = [case for case in corpus.CASES if case.family == "long_context_buried"]
    assert long_cases
    for case in long_cases:
        assert len(case.text) >= 2500, case.id


def test_language_labels_are_not_swapped() -> None:
    """A mislabeled corpus is a bug. An 'es' case that reads as plain English (or the reverse) is
    caught here; code and JSON snippets belong to neither language, so they are left alone."""

    es_with_markers = 0
    en_with_markers = 0
    for case in corpus.CASES:
        lowered = f" {case.text.lower()} "
        looks_es = any(marker in lowered for marker in _SPANISH_MARKERS)
        looks_en = any(marker in lowered for marker in _ENGLISH_MARKERS)
        if case.language == "es":
            assert not (looks_en and not looks_es), case.id
            es_with_markers += int(looks_es)
        else:
            assert not (looks_es and not looks_en), case.id
            en_with_markers += int(looks_en)
    # Natural prose exists in both languages, not just labels.
    assert es_with_markers >= 20
    assert en_with_markers >= 20


def test_no_case_contains_a_real_looking_secret() -> None:
    for case in corpus.CASES:
        for pattern in _SECRET_PATTERNS:
            assert not pattern.search(case.text), f"{case.id} matches {pattern.pattern}"
        scanned = _project_secrets_scan(case.text)
        if scanned is not None:
            assert not scanned, f"{case.id}: project secrets scanner flagged {scanned}"


def test_no_case_carries_third_party_dataset_text() -> None:
    for case in corpus.CASES:
        for marker in _THIRD_PARTY_MARKERS:
            assert marker not in case.text, f"{case.id} references {marker}"
        assert "github.com/prompt-armor" not in case.text
        assert "github.com/leolee99" not in case.text


def test_module_documents_its_synthesis() -> None:
    docstring = corpus.__doc__ or ""
    assert "synthesized" in docstring
    assert "No text is copied" in docstring
    assert "no real secret" in docstring


def test_counts_breakdown_is_complete() -> None:
    breakdown = corpus.counts()
    assert breakdown["total"] == len(corpus.CASES)
    assert breakdown["by_label"]["attack"] + breakdown["by_label"]["benign"] == len(corpus.CASES)
    assert breakdown["attack_family_count"] >= 12
    assert breakdown["benign_family_count"] >= 4


def test_filter_by_family_language_label_and_tags() -> None:
    english_attacks = corpus.filter_cases(label="attack", language="en")
    assert english_attacks
    assert all(case.label == "attack" and case.language == "en" for case in english_attacks)

    security = corpus.filter_cases(tags=["overdefense"])
    assert security
    assert all("overdefense" in case.tags for case in security)

    a_family = corpus.CASES[0].family
    assert all(case.family == a_family for case in corpus.filter_cases(family=a_family))

    by_provenance = corpus.filter_cases(provenance="tool_result")
    assert by_provenance
    assert all(case.provenance == "tool_result" for case in by_provenance)

    by_owasp = corpus.filter_cases(owasp=["LLM02"])
    assert by_owasp
    assert all("LLM02" in case.owasp for case in by_owasp)


def test_read_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    path.write_text("\n\n" + corpus.dumps_jsonl(corpus.CASES[:2]) + "\n", encoding="utf-8")
    assert corpus.load(path) == list(corpus.CASES[:2])


def test_jsonl_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "corpus.jsonl"
    corpus.export_jsonl(target)
    reloaded = corpus.load(target)
    assert reloaded == list(corpus.CASES)


def test_load_empty_path_is_builtin_corpus() -> None:
    assert corpus.load(None, label="benign") == [
        case for case in corpus.CASES if case.label == "benign"
    ]


def test_load_rejects_malformed_jsonl(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        corpus.load(bad)


def test_dumps_jsonl_is_one_object_per_line() -> None:
    text = corpus.dumps_jsonl(corpus.CASES[:3])
    lines = [line for line in text.splitlines() if line]
    assert len(lines) == 3


def test_service_integration_skips_cleanly_when_absent() -> None:
    if find_spec("llm_guardian.service") is None:
        pytest.skip("llm_guardian.service is not present in this checkout")
    from llm_guardian.app import create_app
    from llm_guardian.bench import harness

    scan = harness.make_app_scan_callable(create_app())
    report = harness.run(corpus.CASES[:1], scan, warmup=0)
    assert report.corpus_size == 1
