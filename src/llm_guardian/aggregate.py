"""Turn a pile of scanner findings into per-OWASP-category signals.

There is no single "safety score": a number that mixes an instruction-override signal with an
egress request cannot be compared to a threshold meaningfully. So scoring is *per category*, and
the overall figure a verdict thresholds against is the worst category -- one strong, unambiguous
signal is not diluted by the absence of others.

Within a category the worst finding wins rather than the findings summing: three low-severity
signals must not add up to a block. Severity maps to a weight, and each scanner's weight is
multiplied in from settings, so a noisy detector can be turned down without editing a branch.
"""

from __future__ import annotations

from collections.abc import Sequence

from .models import Finding, OwaspCategory, ScannerOutcome, Severity, Signals
from .scanners.base import severity_rank
from .settings import ScannerSettings

#: Severity -> score contribution. Chosen so the default thresholds line up with the ladder:
#: 0.35 escalate (above ``low``), 0.75 block (at ``high``), 1.0 for ``critical``.
SEVERITY_WEIGHT: dict[Severity, float] = {
    Severity.INFO: 0.1,
    Severity.LOW: 0.25,
    Severity.MEDIUM: 0.5,
    Severity.HIGH: 0.75,
    Severity.CRITICAL: 1.0,
}

#: How many findings the aggregate keeps. The full list stays in the scanner outcomes.
TOP_FINDINGS_LIMIT = 5


def all_findings(outcomes: Sequence[ScannerOutcome]) -> list[Finding]:
    return [finding for outcome in outcomes for finding in outcome.findings]


def scanner_errors(outcomes: Sequence[ScannerOutcome]) -> list[str]:
    return [f"{o.scanner}: {o.error}" for o in outcomes if o.error]


def aggregate(outcomes: Sequence[ScannerOutcome], settings: ScannerSettings) -> Signals:
    """Score findings per category, dedupe by kind and pick the top findings."""
    findings = all_findings(outcomes)
    category_scores: dict[OwaspCategory, float] = {}
    uncategorised = 0.0

    for finding in findings:
        weight = _weight(finding, settings)
        if finding.owasp:
            for category in finding.owasp:
                category_scores[category] = max(category_scores.get(category, 0.0), weight)
        else:
            uncategorised = max(uncategorised, weight)

    capped = {category: min(1.0, score) for category, score in category_scores.items()}
    score = max([uncategorised, *capped.values(), 0.0])

    return Signals(
        score=min(1.0, score),
        category_scores=capped,
        top_findings=tuple(_top(findings)),
        matched_rules=tuple(sorted({finding.kind for finding in findings})),
        scanner_errors=tuple(scanner_errors(outcomes)),
    )


def _weight(finding: Finding, settings: ScannerSettings) -> float:
    scanner_weight = settings.weights.get(finding.scanner, 1.0)
    return SEVERITY_WEIGHT.get(finding.severity, 0.0) * scanner_weight


def _top(findings: Sequence[Finding]) -> list[Finding]:
    """Dedupe by kind (keeping the most severe) then rank by severity, scanner and kind."""
    by_kind: dict[str, Finding] = {}
    for finding in findings:
        current = by_kind.get(finding.kind)
        if current is None or severity_rank(finding.severity) > severity_rank(current.severity):
            by_kind[finding.kind] = finding
    ranked = sorted(
        by_kind.values(),
        key=lambda f: (-severity_rank(f.severity), f.scanner, f.kind),
    )
    return ranked[:TOP_FINDINGS_LIMIT]
