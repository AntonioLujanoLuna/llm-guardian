"""``GET /v1/owasp`` -- per-category implementation and test status, with no aggregate score."""

from __future__ import annotations

from fastapi import APIRouter

from ..deps import ClockDep
from ..models import OwaspCoverageReport
from ..owasp import coverage_report

router = APIRouter(tags=["owasp"])


@router.get("/owasp", response_model=OwaspCoverageReport)
def owasp_report(clock: ClockDep) -> OwaspCoverageReport:
    """The coverage table. Honest about what the guardian does not own."""
    return coverage_report(clock)
