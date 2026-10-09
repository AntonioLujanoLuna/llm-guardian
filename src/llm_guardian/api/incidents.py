"""``POST /v1/incidents`` and friends -- untrusted alerts and the human review that labels them."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from ..deps import IncidentStoreDep
from ..models import (
    IncidentCreateRequest,
    IncidentRecord,
    IncidentStatus,
    ReviewRequest,
)

router = APIRouter(tags=["incidents"])


@router.post("/incidents", response_model=IncidentRecord, status_code=status.HTTP_201_CREATED)
def create_incident(body: IncidentCreateRequest, store: IncidentStoreDep) -> IncidentRecord:
    """Record a pending incident. It is an observation, never a label."""
    return store.create(body)


@router.get("/incidents", response_model=list[IncidentRecord])
def list_incidents(
    store: IncidentStoreDep,
    status: IncidentStatus | None = None,
) -> list[IncidentRecord]:
    """List incidents, optionally filtered by status."""
    return store.list(status)


@router.post("/incidents/{case_id}/review", response_model=IncidentRecord)
def review_incident(case_id: str, body: ReviewRequest, store: IncidentStoreDep) -> IncidentRecord:
    """The only path that produces a trusted label. Requires a reviewer identity."""
    try:
        return store.review(case_id, body)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"unknown case_id: {case_id}"
        ) from exc
