"""The scan service: where segmentation, cache, scanners, memory, verdicts and metrics meet.

This is the only place the stages are wired together, and it is where the invariants that cross
stage boundaries are enforced:

* Retrieval stays off the normal path -- risk memory is queried only while the deterministic score
  sits in the uncertain band, and every other lookup returns ``queried=False`` with its reason.
* A truncated segment, a scanner error, a disabled classifier and an unreachable risk memory are
  all surfaced as warnings; degradation is audited, never a silent allow.
* Every escalate or block auto-records a *pending* incident. A pending record is an observation,
  not a label, and only a later human review can promote it. It keeps the segment with secrets
  masked (``incidents.excerpt``), so that once reviewed it is a precedent an identical later
  segment actually matches.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Sequence
from datetime import datetime

from . import __version__
from .aggregate import aggregate
from .cache import FeatureCache
from .incidents import IncidentStore
from .memory import RiskMemory, not_queried
from .memory.base import MemoryBackendUnavailable
from .metrics import Metrics
from .models import (
    MAX_EXCERPT_CHARS,
    GuardMode,
    IncidentCreateRequest,
    LatencyBreakdown,
    MemoryEvidence,
    OwaspCategory,
    ScannerOutcome,
    ScanRequest,
    ScanResponse,
    Segment,
    SegmentVerdict,
    Signals,
    VerdictDecision,
)
from .redaction import redact_secrets
from .scanners import PREPROCESSING_VERSION, ScannerPipeline
from .segments import Segmenter
from .settings import GuardianSettings
from .verdicts import VerdictEngine, worst_decision

#: Finding summaries are built only from finding *messages*, which never carry secret material.
_MAX_SUMMARY = 2000


class ScanService:
    """Orchestrates one scan from input segments to a response."""

    def __init__(
        self,
        settings: GuardianSettings,
        clock: Callable[[], datetime],
        pipeline: ScannerPipeline,
        cache: FeatureCache,
        memory: RiskMemory,
        incidents: IncidentStore,
        metrics: Metrics,
    ) -> None:
        self._settings = settings
        self._clock = clock
        self._pipeline = pipeline
        self._cache = cache
        self._memory = memory
        self._incidents = incidents
        self._metrics = metrics
        self._segmenter = Segmenter(settings.chunking)
        self._verdicts = VerdictEngine(settings)
        self._fingerprint = pipeline.fingerprint()

    def scan(self, request: ScanRequest) -> ScanResponse:
        warnings: list[str] = []
        mode = self._resolve_mode(request.mode, warnings)
        shadow = mode is GuardMode.SHADOW

        started = time.perf_counter()
        preprocess_started = time.perf_counter()
        segments = self._segmenter.segment(request.conversation_id, request.segments)
        preprocess_ms = _elapsed_ms(preprocess_started)

        scan_started = time.perf_counter()
        outcomes_per_segment, ml_ms = self._scan_segments(segments, warnings)
        # The pipeline runs the classifier inline; report it as its own stage, not twice.
        deterministic_ms = max(0.0, _elapsed_ms(scan_started) - ml_ms)

        retrieval_ms = 0.0
        verdicts: list[SegmentVerdict] = []
        for segment, outcomes in zip(segments, outcomes_per_segment, strict=True):
            signals = aggregate(outcomes, self._settings.scanners)
            memory, lookup_ms = self._maybe_query(segment, signals, request.tenant, warnings)
            retrieval_ms += lookup_ms
            verdicts.append(self._verdicts.decide(segment, signals, outcomes, memory, mode=mode))

        self._record_incidents(request, segments, verdicts)

        expired_examples = sum(
            verdict.memory.expired_examples for verdict in verdicts if verdict.memory is not None
        )
        if expired_examples:
            warnings.append(f"risk_memory_expired_dropped:{expired_examples}")

        if any(segment.truncated for segment in segments):
            warnings.append(
                "bounded_inspection: content was truncated by max_chunks_per_segment; "
                "a truncated scan is not evidence of safety."
            )
        for outcome in (o for outcomes in outcomes_per_segment for o in outcomes):
            if outcome.error:
                warnings.append(f"scanner_error:{outcome.scanner}:{outcome.error}")
        warnings.extend(self._pipeline.warnings)
        if self._memory.degraded_reason:
            warnings.append(f"risk_memory_degraded:{self._memory.degraded_reason}")

        total_ms = _elapsed_ms(started)
        self._metrics.observe("preprocess", preprocess_ms)
        self._metrics.observe("deterministic", deterministic_ms)
        self._metrics.observe("ml", ml_ms)
        self._metrics.observe("retrieval", retrieval_ms)
        self._metrics.observe("total", total_ms)

        response = ScanResponse(
            request_id=request.request_id or uuid.uuid4().hex,
            conversation_id=request.conversation_id,
            mode=mode,
            shadow=shadow,
            decision=worst_decision(verdicts),
            segments=tuple(verdicts),
            latency=LatencyBreakdown(
                preprocess_ms=round(preprocess_ms, 4),
                deterministic_ms=round(deterministic_ms, 4),
                ml_ms=round(ml_ms, 4),
                retrieval_ms=round(retrieval_ms, 4),
                total_ms=round(total_ms, 4),
            ),
            warnings=tuple(dict.fromkeys(warnings)),
            scanner_versions=self._pipeline.versions(),
            guardian_version=__version__,
        )
        self._metrics.record_scan(response)
        return response

    # -- mode -----------------------------------------------------------------------------------
    def _resolve_mode(self, requested: GuardMode | None, warnings: list[str]) -> GuardMode:
        """A request may opt into enforcement, but may only opt out when the operator allows it."""
        configured = self._settings.mode
        if requested is None or requested is configured:
            return configured
        if (
            configured is GuardMode.ENFORCE
            and requested is GuardMode.SHADOW
            and not self._settings.allow_request_mode_downgrade
        ):
            warnings.append(
                "mode_override_refused: the request asked for shadow but the guardian is "
                "configured to enforce (allow_request_mode_downgrade is false)."
            )
            return configured
        return requested

    # -- scanning + cache ---------------------------------------------------------------------
    def _scan_segments(
        self, segments: Sequence[Segment], warnings: list[str]
    ) -> tuple[list[tuple[ScannerOutcome, ...]], float]:
        """Outcomes per segment, plus the classifier time actually spent (cache hits cost none)."""
        seen: dict[tuple[str, str], tuple[ScannerOutcome, ...]] = {}
        results: list[tuple[ScannerOutcome, ...]] = []
        ml_ms = 0.0
        for segment in segments:
            key_signature = (segment.digest, segment.provenance.value)
            if key_signature in seen:
                results.append(seen[key_signature])
                continue
            outcomes = self._scan_one(segment)
            ml_ms += _ml_ms(outcomes)
            seen[key_signature] = outcomes
            results.append(outcomes)
        return results, ml_ms

    def _scan_one(self, segment: Segment) -> tuple[ScannerOutcome, ...]:
        key = FeatureCache.make_key(
            segment.digest,
            segment.provenance,
            self._fingerprint,
            PREPROCESSING_VERSION,
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        outcomes = self._pipeline.scan(segment.text, segment.provenance)
        # A scanner error is a fact about this run, not a feature of the content: caching it
        # would replay a transient failure (and its fail-closed block) for the whole TTL.
        if not any(outcome.error for outcome in outcomes):
            self._cache.put(key, outcomes)
        return outcomes

    # -- conditional retrieval ----------------------------------------------------------------
    def _maybe_query(
        self,
        segment: Segment,
        signals: Signals,
        tenant: str | None,
        warnings: list[str],
    ) -> tuple[MemoryEvidence, float]:
        memory_settings = self._settings.memory
        if not memory_settings.enabled:
            return not_queried("memory_disabled"), 0.0
        escalate_at = self._settings.thresholds.escalate_at
        upper = memory_settings.query_only_level_below
        score = signals.score
        if score < escalate_at:
            return not_queried(f"below_escalate_at:{score:.3f}"), 0.0
        if score >= upper:
            return not_queried(f"at_or_above_uncertain_band:{score:.3f}"), 0.0
        try:
            evidence = self._memory.query(
                segment.text, segment.provenance, memory_settings.retrieval_k, tenant
            )
        except (MemoryBackendUnavailable, RuntimeError, OSError, ValueError) as exc:
            warnings.append(f"risk_memory_unreachable:{type(exc).__name__}: {exc}")
            return not_queried("risk_memory_unreachable"), 0.0
        return evidence, evidence.query_ms

    def _excerpt(self, segment: Segment, verdict: SegmentVerdict) -> str:
        if self._settings.incidents.excerpt == "findings":
            return _findings_summary(verdict)
        return redact_secrets(segment.text, segment.provenance, limit=MAX_EXCERPT_CHARS)

    # -- incidents ----------------------------------------------------------------------------
    def _record_incidents(
        self,
        request: ScanRequest,
        segments: Sequence[Segment],
        verdicts: Sequence[SegmentVerdict],
    ) -> None:
        for segment, verdict in zip(segments, verdicts, strict=True):
            if verdict.decision is VerdictDecision.ALLOW:
                continue
            self._incidents.create(
                IncidentCreateRequest(
                    conversation_id=request.conversation_id,
                    tenant=request.tenant,
                    provenance=segment.provenance,
                    content_digest=segment.digest,
                    redacted_excerpt=self._excerpt(segment, verdict),
                    findings_summary=_findings_summary(verdict),
                    owasp=_categories(verdict.signals),
                    detector_versions=self._pipeline.versions(),
                    score=verdict.score,
                    retention_days=self._settings.retention_days_default,
                )
            )


def _findings_summary(verdict: SegmentVerdict) -> str:
    parts = [
        f"{finding.kind} ({finding.severity.value}): {finding.message}"
        for finding in verdict.signals.top_findings
    ]
    text = "; ".join(parts) or f"decision={verdict.decision.value} score={verdict.score:.3f}"
    return text[:_MAX_SUMMARY]


def _categories(signals: Signals) -> tuple[OwaspCategory, ...]:
    return tuple(sorted(signals.category_scores, key=lambda category: category.value))


def _ml_ms(outcomes: Sequence[ScannerOutcome]) -> float:
    """Classifier time spent producing ``outcomes``; a replayed cache entry spent none."""
    return sum(o.duration_ms for o in outcomes if o.scanner == "ml" and not o.cached)


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0
