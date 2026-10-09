"""``POST /v1/incidents`` and friends -- untrusted alerts and the human review that labels them."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Response, status

from ..deps import IncidentStoreDep
from ..models import (
    IncidentCreateRequest,
    IncidentRecord,
    IncidentStatus,
    ReviewRequest,
)
from ..provenance import Provenance

router = APIRouter(tags=["incidents"])


@router.post("/incidents", response_model=IncidentRecord, status_code=status.HTTP_201_CREATED)
def create_incident(body: IncidentCreateRequest, store: IncidentStoreDep) -> IncidentRecord:
    """Record a pending incident. It is an observation, never a label."""
    return store.create(body)


@router.get("/incidents", response_model=list[IncidentRecord])
def list_incidents(
    store: IncidentStoreDep,
    response: Response,
    status: IncidentStatus | None = None,
    tenant: str | None = None,
    provenance: Provenance | None = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[IncidentRecord]:
    """List incidents oldest first, filtered and paged; ``X-Total-Count`` has the full count."""
    response.headers["X-Total-Count"] = str(
        store.count(status, tenant=tenant, provenance=provenance)
    )
    return store.list(status, tenant=tenant, provenance=provenance, limit=limit, offset=offset)


@router.post("/incidents/{case_id}/review", response_model=IncidentRecord)
def review_incident(case_id: str, body: ReviewRequest, store: IncidentStoreDep) -> IncidentRecord:
    """The only path that produces a trusted label. Requires a reviewer identity."""
    try:
        return store.review(case_id, body)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"unknown case_id: {case_id}"
        ) from exc
