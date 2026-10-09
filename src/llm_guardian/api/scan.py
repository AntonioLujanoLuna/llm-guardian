"""``POST /v1/scan`` -- incremental inspection of a request's new segments."""

from __future__ import annotations

from fastapi import APIRouter

from ..deps import ScanServiceDep
from ..models import ScanRequest, ScanResponse

router = APIRouter(tags=["scan"])


@router.post("/scan", response_model=ScanResponse)
def scan(body: ScanRequest, service: ScanServiceDep) -> ScanResponse:
    """Scan the submitted segments and return per-segment verdicts and stage latencies."""
    return service.scan(body)
