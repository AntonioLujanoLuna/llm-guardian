"""Elasticsearch risk memory over three separately permissioned indexes.

The design keeps trusted labels and untrusted alerts apart on purpose: ``confirmed_attacks`` and
``benign_hard_negatives`` hold reviewed evidence, and ``pending_incidents`` holds alerts that must
never be read as labels. The client is loaded lazily, so the service runs without the ``es`` extra
and only fails -- with an actionable message -- when this backend is actually configured.

Retrieval is kNN when an embedder is provided and BM25 otherwise; either way the returned
similarity is bounded to ``[0, 1)`` and is never presented as a probability of intent.
"""

from __future__ import annotations

import contextlib
import importlib
import threading
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from ..models import IncidentStatus, MemoryEvidence, MemoryStats, OwaspCategory
from ..provenance import Provenance
from ..settings import MemorySettings
from .base import (
    MemoryBackendUnavailable,
    Neighbor,
    VerifiedExample,
    assert_trusted_label,
    evidence_from,
)

BACKEND = "elasticsearch"

_SUFFIX_ATTACKS = "confirmed_attacks"
_SUFFIX_BENIGN = "benign_hard_negatives"
_SUFFIX_PENDING = "pending_incidents"


def _load_elasticsearch() -> Any:
    """Import the optional client lazily so importing this module never needs the extra."""
    try:
        return importlib.import_module("elasticsearch")
    except ImportError as exc:
        raise MemoryBackendUnavailable(
            "The Elasticsearch risk memory needs the optional 'es' extra. Install it with "
            "`uv sync --extra es`, or set GUARDIAN_MEMORY__BACKEND=inmemory."
        ) from exc


class ElasticsearchRiskMemory:
    """Reviewed precedents in Elasticsearch; kNN when an embedder is set, else BM25."""

    backend = BACKEND

    def __init__(
        self,
        settings: MemorySettings,
        *,
        client: Any = None,
        embedder: Callable[[str], Sequence[float]] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._prefix = settings.es_index_prefix
        self._default_k = settings.retrieval_k
        self._embedder = embedder
        self._clock = clock or _utcnow
        self.reachable = True
        self.degraded_reason: str | None = None
        self._lock = threading.Lock()
        self._client: Any = client if client is not None else self._build_client(settings)

    @staticmethod
    def _build_client(settings: MemorySettings) -> Any:
        module = _load_elasticsearch()
        client_class = getattr(module, "Elasticsearch", None)
        if client_class is None:  # pragma: no cover - defensive against a broken install
            raise MemoryBackendUnavailable("elasticsearch package has no Elasticsearch class")
        kwargs: dict[str, Any] = {
            "hosts": list(settings.es_urls) or ["http://localhost:9200"],
            "request_timeout": settings.es_request_timeout_seconds,
        }
        if settings.es_username is not None:
            password = settings.es_password.get_secret_value() if settings.es_password else ""
            kwargs["basic_auth"] = (settings.es_username, password)
        return client_class(**kwargs)

    # -- index names --------------------------------------------------------------------------
    @property
    def attacks_index(self) -> str:
        return f"{self._prefix}_{_SUFFIX_ATTACKS}"

    @property
    def benign_index(self) -> str:
        return f"{self._prefix}_{_SUFFIX_BENIGN}"

    @property
    def pending_index(self) -> str:
        return f"{self._prefix}_{_SUFFIX_PENDING}"

    @property
    def searchable_indices(self) -> str:
        """Only the two trusted partitions are ever searchable as evidence."""
        return f"{self.attacks_index},{self.benign_index}"

    # -- contract -----------------------------------------------------------------------------
    def record_verified(self, example: VerifiedExample) -> None:
        assert_trusted_label(example.label)
        index = (
            self.attacks_index
            if example.label is IncidentStatus.CONFIRMED_ATTACK
            else self.benign_index
        )
        document: dict[str, Any] = {
            "case_id": example.case_id,
            "label": example.label.value,
            "digest": example.digest,
            "text": example.text,
            "owasp": [category.value for category in example.owasp],
            "family": example.family,
            "tenant": example.tenant,
            "review_status": "verified",
            "detector_version": "guardian",
        }
        if example.retention_deadline is not None:
            document["retention_deadline"] = example.retention_deadline.isoformat()
        if self._embedder is not None:
            document["embedding"] = list(self._embedder(example.text))
        other = self.benign_index if index == self.attacks_index else self.attacks_index
        with self._lock:
            # A re-review supersedes the previous label for the same case id: best-effort delete
            # of the opposite partition's document so a stale label is not returned alongside it.
            with contextlib.suppress(Exception):
                self._client.delete(index=other, id=example.case_id, ignore=[404])
            self._client.index(index=index, id=example.case_id, document=document)

    def query(
        self,
        text: str,
        provenance: Provenance,
        k: int,
        tenant: str | None = None,
    ) -> MemoryEvidence:
        started = time.perf_counter()
        size = k or self._default_k
        try:
            hits = self._search(text, size, tenant)
        except Exception as exc:
            self.reachable = False
            self.degraded_reason = f"{type(exc).__name__}: {exc}"
            raise
        now = self._clock()
        fresh: list[dict[str, Any]] = []
        expired = 0
        for hit in hits:
            if _hit_expired(hit, now):
                expired += 1
                continue
            fresh.append(hit)
        neighbors = [_hit_to_neighbor(hit) for hit in fresh]
        query_ms = (time.perf_counter() - started) * 1000.0
        return evidence_from(neighbors, k=size, query_ms=query_ms, expired_examples=expired)

    def _search(self, text: str, size: int, tenant: str | None) -> list[dict[str, Any]]:
        filters: list[dict[str, Any]] = []
        if tenant is not None:
            filters.append(
                {
                    "bool": {
                        "should": [
                            {"term": {"tenant": tenant}},
                            {"bool": {"must_not": {"exists": {"field": "tenant"}}}},
                        ]
                    }
                }
            )
        if self._embedder is not None:
            body: dict[str, Any] = {
                "knn": {
                    "field": "embedding",
                    "query_vector": list(self._embedder(text)),
                    "k": size,
                    "num_candidates": max(size * 10, 50),
                    "filter": filters,
                },
                "size": size,
            }
        else:
            body = {
                "query": {
                    "bool": {
                        "must": [{"multi_match": {"query": text, "fields": ["text"]}}],
                        "filter": filters,
                    }
                },
                "size": size,
            }
        response = self._client.search(index=self.searchable_indices, **body)
        self.reachable = True
        self.degraded_reason = None
        return list(response.get("hits", {}).get("hits", []))

    def stats(self) -> MemoryStats:
        try:
            return MemoryStats(
                backend=BACKEND,
                reachable=True,
                confirmed_attacks=self._count(self.attacks_index),
                benign_hard_negatives=self._count(self.benign_index),
                pending_incidents=self._count(self.pending_index),
                detail=None,
            )
        except Exception as exc:
            self.reachable = False
            self.degraded_reason = f"{type(exc).__name__}: {exc}"
            return MemoryStats(
                backend=BACKEND,
                reachable=False,
                confirmed_attacks=0,
                benign_hard_negatives=0,
                pending_incidents=0,
                detail=self.degraded_reason,
            )

    def counts(self) -> dict[str, int]:
        return {
            "confirmed_attacks": self._count(self.attacks_index),
            "benign_hard_negatives": self._count(self.benign_index),
            "pending_incidents": self._count(self.pending_index),
        }

    def _count(self, index: str) -> int:
        response = self._client.count(index=index)
        return int(response.get("count", 0))


def _hit_to_neighbor(hit: dict[str, Any]) -> Neighbor:
    source = hit.get("_source", {})
    label = IncidentStatus(source.get("label", IncidentStatus.CONFIRMED_ATTACK.value))
    raw_score = float(hit.get("_score") or 0.0)
    similarity = _bounded_similarity(raw_score, source)
    owasp = tuple(OwaspCategory(item) for item in source.get("owasp", []) if item)
    return Neighbor(
        case_id=str(source.get("case_id", "")),
        label=label,
        similarity=similarity,
        owasp=owasp,
        family=source.get("family"),
        digest=str(source.get("digest", "")),
    )


def _bounded_similarity(score: float, source: dict[str, Any]) -> float:
    """kNN returns cosine directly; BM25 scores are squashed into ``[0, 1)``."""
    if "embedding" in source:
        return max(0.0, min(1.0, score))
    if score <= 0.0:
        return 0.0
    return score / (score + 5.0)


def _hit_expired(hit: dict[str, Any], now: datetime) -> bool:
    """A hit whose stored retention deadline has passed must not be returned as evidence."""
    source = hit.get("_source", {})
    raw = source.get("retention_deadline")
    if not raw:
        return False
    try:
        deadline = datetime.fromisoformat(str(raw))
    except ValueError:
        return False
    return deadline <= now


def _utcnow() -> datetime:
    return datetime.now(UTC)
