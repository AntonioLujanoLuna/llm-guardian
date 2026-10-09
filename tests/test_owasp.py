"""The OWASP coverage table and its markdown rendering."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from llm_guardian.models import OwaspCategory
from llm_guardian.owasp import coverage_markdown, coverage_report, coverage_table

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_all_categories_present_and_unique() -> None:
    table = coverage_table()
    categories = [row.category for row in table]
    assert len(categories) == 10
    assert set(categories) == set(OwaspCategory)


def test_honest_implemented_flags() -> None:
    by_category = {row.category: row for row in coverage_table()}
    # Supply chain and misinformation are not a detector's job.
    assert by_category[OwaspCategory.LLM04].implemented is False
    assert by_category[OwaspCategory.LLM07].implemented is False
    assert by_category[OwaspCategory.LLM05].implemented is False
    # Prompt injection and disclosure are.
    assert by_category[OwaspCategory.LLM01].implemented is True
    assert by_category[OwaspCategory.LLM02].implemented is True


def test_implemented_rows_name_real_test_files() -> None:
    for row in coverage_table():
        if not row.implemented:
            continue
        assert row.tests, f"{row.category} claims implementation with no tests"
        for relative in row.tests:
            assert (REPO_ROOT / relative).is_file(), f"missing test file {relative}"


def test_enforced_by_is_set() -> None:
    assert all(row.enforced_by for row in coverage_table())


def test_coverage_report_timestamp() -> None:
    fixed = datetime(2021, 1, 1, tzinfo=UTC)
    report = coverage_report(lambda: fixed)
    assert report.generated_at == fixed
    assert "no aggregate" in report.note


def test_coverage_markdown_lists_rows() -> None:
    text = coverage_markdown(lambda: datetime(2021, 1, 1, tzinfo=UTC))
    assert "| LLM01 |" in text
    assert "| LLM04 |" in text
    assert text.startswith("# OWASP LLM Top 10")
