"""The per-category OWASP coverage table served by ``/v1/owasp``.

``implemented`` means *this service* implements a real control for the category -- it is not a
claim that the category is fully addressed by the guardian. Supply chain (LLM04), misinformation
(LLM07) and much of data/model poisoning (LLM05) are the gateway's and the build pipeline's job,
so they are marked ``False`` with an honest ``enforced_by``. There is deliberately no aggregate
"OWASP safety score".
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from .models import OwaspCategory as C
from .models import OwaspCategoryCoverage, OwaspCoverageReport

_GUARDIAN = "guardian"


def coverage_table() -> tuple[OwaspCategoryCoverage, ...]:
    """The hand-maintained coverage table. Tests name real files under ``tests/``."""
    return (
        OwaspCategoryCoverage(
            category=C.LLM01,
            title="Prompt Injection",
            safeguards=(
                "instruction-override, exfiltration and delimiter/role rules (EN + ES)",
                "quoted/discussion context guard to avoid blocking security research",
                "deobfuscation of base64/hex/rot13/percent/reversed payloads",
                "invisible-text and homoglyph detection",
                "server-assigned provenance; instructions in data are a signal, not an instruction",
            ),
            implemented=True,
            enforced_by=_GUARDIAN,
            tests=(
                "tests/scanners/test_rules.py",
                "tests/scanners/test_deobfuscate.py",
                "tests/scanners/test_invisible.py",
            ),
            notes="Detection is textual and deterministic; it never replaces tool authorization.",
        ),
        OwaspCategoryCoverage(
            category=C.LLM02,
            title="Sensitive Information Disclosure",
            safeguards=(
                "secret scanner: API keys, PEM blocks, JWTs, connection strings, .env assignments",
                "findings carry a redacted hint, never the secret value",
                "exfil-shaped long-base64 query params and paste/webhook destinations",
            ),
            implemented=True,
            enforced_by=_GUARDIAN,
            tests=("tests/scanners/test_secrets.py", "tests/scanners/test_urls.py"),
            notes="Retrieval ACLs and egress enforcement remain the gateway's responsibility.",
        ),
        OwaspCategoryCoverage(
            category=C.LLM03,
            title="Excessive Agency",
            safeguards=(
                "hard policy checks for tool capabilities, egress hosts and budgets",
                "cross-tenant retrieval requires an explicit grant",
                "tool-call injection markers detected in data",
                "dangerous command shapes in tool calls: credential-to-network, pipe-to-shell, "
                "reverse shell, destructive commands, persistence",
            ),
            implemented=True,
            enforced_by="policy.py + gateway tool executor",
            tests=("tests/test_policy.py", "tests/scanners/test_command.py"),
            notes="The guardian decides; the executor enforces at the point the tool runs.",
        ),
        OwaspCategoryCoverage(
            category=C.LLM04,
            title="Supply Chain",
            safeguards=("dependency/model provenance, pinning and artifact scanning",),
            implemented=False,
            enforced_by="build pipeline / gateway",
            tests=(),
            notes="Not a detector concern: belongs to CI, artifact pinning and model provenance.",
        ),
        OwaspCategoryCoverage(
            category=C.LLM05,
            title="Data and Model Poisoning",
            safeguards=(
                "ingestion review of incidents before any label is trusted",
                "risk memory stores only reviewed evidence",
            ),
            implemented=False,
            enforced_by="gateway ingestion + incident review",
            tests=("tests/test_incidents.py",),
            notes="Guardian contributes the review gate; index provenance is the gateway's job.",
        ),
        OwaspCategoryCoverage(
            category=C.LLM06,
            title="Unbounded Consumption",
            safeguards=(
                "declared budgets (tokens, concurrency, requests_per_minute) enforced by policy",
                "bounded inspection: max_chunks_per_segment caps work per request",
            ),
            implemented=True,
            enforced_by="policy.py + gateway rate limiting",
            tests=("tests/test_policy.py", "tests/test_segments.py"),
            notes="Budgets are refused before inference; the gateway performs the accounting.",
        ),
        OwaspCategoryCoverage(
            category=C.LLM07,
            title="Misinformation",
            safeguards=("optional grounded verification, cited evidence and human oversight",),
            implemented=False,
            enforced_by="gateway / application",
            tests=(),
            notes="Not a textual detector concern.",
        ),
        OwaspCategoryCoverage(
            category=C.LLM08,
            title="Hidden Context Exposure",
            safeguards=(
                "invisible-text, bidi, tag-block and hidden-HTML-comment detection",
                "risk memory isolates tenants; global precedents carry no tenant scope",
                "system-prompt exfiltration rules",
            ),
            implemented=True,
            enforced_by=_GUARDIAN,
            tests=(
                "tests/scanners/test_invisible.py",
                "tests/test_memory.py",
                "tests/scanners/test_rules.py",
            ),
            notes="Prompt minimisation and tenant separation at retrieval are the gateway's.",
        ),
        OwaspCategoryCoverage(
            category=C.LLM09,
            title="Vector and Embedding Weaknesses",
            safeguards=(
                "retrieval stays off the normal path; the index is queried only in the band",
                "cosine similarity never whitelists content",
                "only reviewed evidence is indexed",
            ),
            implemented=True,
            enforced_by="memory/ + gateway retrieval ACLs",
            tests=("tests/test_memory.py",),
            notes="Pre-retrieval ACL filters and index isolation remain the gateway's job.",
        ),
        OwaspCategoryCoverage(
            category=C.LLM10,
            title="Improper Output Handling",
            safeguards=(
                "dangerous URI schemes (javascript:, data:, file:) flagged",
                "destructive shell and curl-pipe-to-shell patterns detected",
                "delimiter/control-token spoofing detected",
            ),
            implemented=True,
            enforced_by="guardian (input) + gateway output escaping",
            tests=("tests/scanners/test_urls.py", "tests/scanners/test_rules.py"),
            notes="Output escaping and command validation at release time are the gateway's.",
        ),
    )


def coverage_report(now: Callable[[], datetime]) -> OwaspCoverageReport:
    return OwaspCoverageReport(generated_at=now(), categories=coverage_table())


def coverage_markdown(now: Callable[[], datetime]) -> str:
    """Render the same table as markdown, so ``docs/owasp-coverage.md`` cannot drift from it."""
    lines = [
        "# OWASP LLM Top 10 (2026) coverage",
        "",
        "Generated from `owasp.coverage_table()`, the same source as `GET /v1/owasp`. "
        "`implemented` means this service implements a control for the category; it is not a claim "
        "that the category is fully addressed here. There is deliberately no aggregate score.",
        "",
        "| Category | Title | Implemented | Enforced by | Tests |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in coverage_table():
        tests = ", ".join(f"`{t}`" for t in row.tests) or "-"
        implemented = "yes" if row.implemented else "no"
        lines.append(
            f"| {row.category.value} | {row.title} | {implemented} | {row.enforced_by} | {tests} |"
        )
    return "\n".join(lines) + "\n"
