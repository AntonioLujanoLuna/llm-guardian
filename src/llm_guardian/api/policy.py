"""``POST /v1/policy/check`` -- hard authorization for a tool call, retrieval, egress or budget.

``POST /v1/agent/authorize`` is the same decision with the argument and evidence detail a tool
executor needs at the point of execution: it is what an executor calls when it is about to run a
tool, and it carries the findings a gateway echoed from a prior scan.
"""

from __future__ import annotations

from fastapi import APIRouter

from .. import __version__
from ..deps import ClockDep, PolicyEngineDep
from ..models import Authorization, PolicyCheckRequest, PolicyCheckResponse

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


@router.post("/agent/authorize", response_model=Authorization)
def agent_authorize(
    body: PolicyCheckRequest,
    engine: PolicyEngineDep,
    clock: ClockDep,
) -> Authorization:
    """The tool-boundary decision for an executor: policy plus argument and evidence detail."""
    check = engine.check(body, now=clock)
    return Authorization(
        check=check,
        decision=check.decision,
        denied_by=check.denied_by,
        argument_failures=check.argument_failures,
        evidence_used=check.evidence_used,
        guardian_version=__version__,
    )
