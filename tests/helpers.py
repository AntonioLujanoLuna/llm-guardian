"""Small builders shared across the core tests."""

from __future__ import annotations

from llm_guardian.models import (
    Finding,
    OwaspCategory,
    ScannerOutcome,
    Segment,
    Severity,
)
from llm_guardian.provenance import Provenance, trust_of
from llm_guardian.scanners.base import build_finding
from llm_guardian.segments import digest_of, make_segment_id


def make_finding(
    kind: str = "instruction_override",
    severity: Severity = Severity.HIGH,
    scanner: str = "rules",
    owasp: tuple[OwaspCategory, ...] = (OwaspCategory.LLM01,),
    message: str = "a finding",
) -> Finding:
    return build_finding(
        scanner=scanner,
        scanner_version="test-1",
        kind=kind,
        severity=severity,
        message=message,
        owasp=owasp,
    )


def make_outcome(
    *findings: Finding,
    scanner: str = "rules",
    error: str | None = None,
    cached: bool = False,
    duration_ms: float = 0.0,
) -> ScannerOutcome:
    return ScannerOutcome(
        scanner=scanner,
        scanner_version="test-1",
        findings=tuple(findings),
        error=error,
        cached=cached,
        duration_ms=duration_ms,
    )


def make_segment(
    text: str = "hello world",
    provenance: Provenance = Provenance.USER,
    index: int = 0,
    truncated: bool = False,
) -> Segment:
    digest = digest_of(text)
    return Segment(
        segment_id=make_segment_id("conv", index, digest),
        index=index,
        digest=digest,
        text=text,
        provenance=provenance,
        trust=trust_of(provenance),
        truncated=truncated,
    )
