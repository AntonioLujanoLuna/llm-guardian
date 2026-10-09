"""The scanner contract and the one helper every scanner builds findings with.

A scanner is deterministic, offline and self-describing: it has a stable ``name`` and ``version``
(the version participates in the feature-cache key, so changing a pattern is a cache miss rather
than a stale hit) and it reports *signals*, never a verdict. Authorization is never a scanner's
job.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ..models import Finding, OwaspCategory, Severity
from ..provenance import Provenance

#: Severity, weakest to strongest. The index is the rank used for comparisons; the numeric weight
#: used for scoring lives in ``aggregate.py`` so the two never drift.
SEVERITY_ORDER: tuple[Severity, ...] = (
    Severity.INFO,
    Severity.LOW,
    Severity.MEDIUM,
    Severity.HIGH,
    Severity.CRITICAL,
)


def severity_rank(severity: Severity) -> int:
    """Position of a severity in the ordered scale."""
    return SEVERITY_ORDER.index(severity)


def raise_severity(severity: Severity, steps: int = 1) -> Severity:
    """Move a severity up the scale, saturating at ``critical``."""
    return SEVERITY_ORDER[min(len(SEVERITY_ORDER) - 1, severity_rank(severity) + steps)]


def lower_severity(severity: Severity, steps: int = 1) -> Severity:
    """Move a severity down the scale, saturating at ``info``."""
    return SEVERITY_ORDER[max(0, severity_rank(severity) - steps)]


def cap_severity(severity: Severity, ceiling: Severity) -> Severity:
    """Clamp a severity to at most ``ceiling`` (used to downgrade quoted/discussed matches)."""
    return severity if severity_rank(severity) <= severity_rank(ceiling) else ceiling


def parse_severity(value: str, default: Severity = Severity.CRITICAL) -> Severity:
    """Parse a configured severity name, defaulting on anything unrecognised."""
    try:
        return Severity(value.strip().lower())
    except ValueError:
        return default


def build_finding(
    *,
    scanner: str,
    scanner_version: str,
    kind: str,
    severity: Severity,
    message: str,
    owasp: Sequence[OwaspCategory] = (),
    span: tuple[int, int] | None = None,
    attributes: dict[str, str] | None = None,
) -> Finding:
    """Build a :class:`Finding`, normalising the optional collections."""
    return Finding(
        scanner=scanner,
        scanner_version=scanner_version,
        kind=kind,
        severity=severity,
        message=message,
        owasp=tuple(owasp),
        span=span,
        attributes=dict(attributes or {}),
    )


class Scanner(Protocol):
    """One deterministic check over one segment.

    ``scan`` takes the text *and* the server-assigned provenance: the same bytes mean different
    things depending on the boundary they crossed, and only the scanner layer is allowed to weigh
    that (the verdict layer enforces the consequences).
    """

    name: str
    version: str

    def scan(self, text: str, provenance: Provenance) -> Sequence[Finding]: ...
