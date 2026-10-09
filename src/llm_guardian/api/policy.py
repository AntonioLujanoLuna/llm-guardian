"""``POST /v1/policy/check`` -- hard authorization for a tool call, retrieval, egress or budget."""

from __future__ import annotations

from fastapi import APIRouter

from .. import __version__
from ..deps import ClockDep, PolicyEngineDep
from ..models import PolicyCheckRequest, PolicyCheckResponse

router = APIRouter(tags=["policy"])


@router.post("/policy/check", response_model=PolicyCheckResponse)
def policy_check(
    body: PolicyCheckRequest,
    engine: PolicyEngineDep,
    clock: ClockDep,
) -> PolicyCheckResponse:
    """Evaluate the request against the declared policy. No model output is consulted."""
    check = engine.check(body, now=clock)
    return PolicyCheckResponse(check=check, guardian_version=__version__)
