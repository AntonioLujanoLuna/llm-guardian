"""``POST /v1/memory/query`` and ``GET /v1/memory/stats`` -- reviewed precedents as evidence.

The explicit query endpoint always performs retrieval (it is a direct request, not the normal
scan path); the similarity it returns is cosine, never a probability of intent.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from ..deps import RiskMemoryDep
from ..models import MemoryQueryRequest, MemoryQueryResponse, MemoryStats

router = APIRouter(tags=["memory"])


@router.post("/memory/query", response_model=MemoryQueryResponse)
def memory_query(body: MemoryQueryRequest, memory: RiskMemoryDep) -> MemoryQueryResponse:
    """Query reviewed precedents for a text, returning attack/benign similarities and margin."""
    try:
        evidence = memory.query(body.text, body.provenance, body.k)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"risk memory unavailable: {type(exc).__name__}: {exc}",
        ) from exc
    return MemoryQueryResponse(
        attack_sim=evidence.attack_sim or 0.0,
        benign_sim=evidence.benign_sim or 0.0,
        margin=evidence.margin or 0.0,
        neighbors=evidence.neighbors,
        backend=memory.backend,
        took_ms=evidence.query_ms,
    )


@router.get("/memory/stats", response_model=MemoryStats)
def memory_stats(memory: RiskMemoryDep) -> MemoryStats:
    """Partition sizes and backend reachability."""
    return memory.stats()
