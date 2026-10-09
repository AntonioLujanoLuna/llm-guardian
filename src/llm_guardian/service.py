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
    Finding,
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
from .provenance import Provenance, is_high_impact
from .redaction import redact_secrets
from .scanners import PREPROCESSING_VERSION, ScannerPipeline
from .scanners.base import severity_rank
from .segments import Segmenter, digest_of
from .session import SessionRiskTracker
from .settings import GuardianSettings
from .verdicts import VerdictEngine, worst_decision

#: Finding summaries are built only from finding *messages*, which never carry secret material.
_MAX_SUMMARY = 2000

#: The outcome name a cross-segment finding is reported under.
SEAM_SCANNER = "seam"

#: Rule contexts (``scanners/rules.py``) that mark a match as talked about rather than issued.
_DISCUSSION_CONTEXTS = frozenset({"quoted", "discussion", "quoted_discussion"})


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
        sessions: SessionRiskTracker | None = None,
    ) -> None:
        self._settings = settings
        self._clock = clock
        self._pipeline = pipeline
        self._cache = cache
        self._memory = memory
        self._incidents = incidents
        self._metrics = metrics
        self._sessions = sessions
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
            verdicts.append(
                self._verdicts.decide(
                    segment,
                    signals,
                    outcomes,
                    memory,
                    mode=mode,
                    benign_precedent=self._benign_precedent(
                        segment, signals, outcomes, request.tenant
                    ),
                )
            )

        session_risk = self._apply_session_risk(request, verdicts)
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
            session_risk=None if session_risk is None else round(session_risk, 4),
        )
        self._metrics.record_scan(response)
        return response

    # -- cross-turn risk ------------------------------------------------------------------------
    def _apply_session_risk(
        self, request: ScanRequest, verdicts: list[SegmentVerdict]
    ) -> float | None:
        """Fold this request into its conversation's risk; escalate an allow that tips it over."""
        if self._sessions is None or not self._sessions.settings.enabled:
            return None
        if request.conversation_id is None or not verdicts:
            return None
        effective = [self._session_contribution(v) for v in verdicts]
        worst = max(range(len(verdicts)), key=lambda i: effective[i])
        risk = self._sessions.observe(
            request.tenant, request.conversation_id, effective[worst], request.request_id
        )
        threshold = self._sessions.settings.escalate_at
        if (
            risk >= threshold
            and effective[worst] > 0.0
            and worst_decision(verdicts) is VerdictDecision.ALLOW
        ):
            verdict = verdicts[worst]
            verdicts[worst] = verdict.model_copy(
                update={
                    "decision": VerdictDecision.ESCALATE,
                    "threshold": threshold,
                    "notes": (*verdict.notes, f"session_risk_escalate:{risk:.3f}"),
                }
            )
        return risk

    def _session_contribution(self, verdict: SegmentVerdict) -> float:
        """The part of a segment's score that is evidence against the conversation.

        A segment a human reviewed as benign contributes nothing, and neither does a match the
        rules judged to be quoted or discussed: a security team talking about prompt injection
        produces exactly those, turn after turn, and must not accumulate into an escalation.
        """
        if any(note.startswith("reviewed_benign_precedent_allow") for note in verdict.notes):
            return 0.0
        kept = tuple(
            outcome.model_copy(
                update={
                    "findings": tuple(
                        f
                        for f in outcome.findings
                        if f.attributes.get("context") not in _DISCUSSION_CONTEXTS
                    )
                }
            )
            for outcome in verdict.scanners
        )
        return aggregate(kept, self._settings.scanners).score

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
            outcomes = self._scan_text(segment.text, segment.digest, segment.provenance)
            ml_ms += _ml_ms(outcomes)
            seen[key_signature] = outcomes
            results.append(outcomes)

        if self._settings.chunking.scan_seams and self._settings.chunking.overlap_chars > 0:
            for position in range(1, len(segments)):
                seam = self._scan_seam(
                    segments[position - 1],
                    segments[position],
                    results[position - 1],
                    results[position],
                )
                if seam is None:
                    continue
                ml_ms += seam.duration_ms if not seam.cached else 0.0
                results[position] = (*results[position], seam)
                if seam.findings:
                    warnings.append(
                        f"seam_finding:{segments[position - 1].index}-{segments[position].index}:"
                        + ",".join(sorted({finding.kind for finding in seam.findings}))
                    )
        return results, ml_ms

    def _scan_seam(
        self,
        before: Segment,
        after: Segment,
        before_outcomes: Sequence[ScannerOutcome],
        after_outcomes: Sequence[ScannerOutcome],
    ) -> ScannerOutcome | None:
        """Scan the join of two adjacent submitted segments; keep only what neither side showed.

        Chunks of one input already overlap, and a seam between different provenances would mix
        trust levels, so only a boundary between two inputs of the same provenance is scanned.
        The result is attached to the later segment -- the one that completes the instruction.
        """
        if before.source_index == after.source_index or before.provenance != after.provenance:
            return None
        window = self._settings.chunking.overlap_chars
        tail, head = before.text[-window:], after.text[:window]
        if not tail or not head:
            return None
        contiguous = before.char_end is not None and after.char_start == before.char_end
        joins = [(tail + head, len(tail), len(tail))]
        if not contiguous and not (tail[-1].isspace() or head[0].isspace()):
            # Separate messages glued without a space would fuse the boundary words.
            joins.append((f"{tail} {head}", len(tail), len(tail) + 1))

        known: dict[str, int] = {}
        for outcome in (*before_outcomes, *after_outcomes):
            for finding in outcome.findings:
                rank = severity_rank(finding.severity)
                known[finding.kind] = max(known.get(finding.kind, -1), rank)

        findings: dict[str, Finding] = {}
        errors: list[str] = []
        duration = 0.0
        all_cached = True
        for text, tail_end, head_start in joins:
            for outcome in self._scan_text(text, digest_of(text), after.provenance):
                duration += outcome.duration_ms
                all_cached = all_cached and outcome.cached
                if outcome.error:
                    errors.append(f"{outcome.scanner}: {outcome.error}")
                for finding in outcome.findings:
                    rank = severity_rank(finding.severity)
                    if rank <= known.get(finding.kind, -1):
                        continue
                    # A match wholly on one side was already judged there, with its full
                    # context; the window would only lose that context and re-rate it. Keep a
                    # spanned finding only when it straddles the boundary, and a spanless one
                    # only when its kind is new to both sides.
                    if finding.span is not None:
                        start, end = finding.span
                        if not (start < tail_end and end > head_start):
                            continue
                    elif finding.kind in known:
                        continue
                    current = findings.get(finding.kind)
                    if current is None or rank > severity_rank(current.severity):
                        findings[finding.kind] = finding.model_copy(
                            update={
                                "span": None,
                                "attributes": {
                                    **finding.attributes,
                                    "seam_with": before.segment_id,
                                },
                            }
                        )
        if not findings and not errors:
            return None
        return ScannerOutcome(
            scanner=SEAM_SCANNER,
            scanner_version=self._fingerprint,
            findings=tuple(findings.values()),
            error="; ".join(errors) or None,
            duration_ms=duration,
            cached=all_cached,
        )

    def _scan_text(
        self, text: str, digest: str, provenance: Provenance
    ) -> tuple[ScannerOutcome, ...]:
        key = FeatureCache.make_key(digest, provenance, self._fingerprint, PREPROCESSING_VERSION)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        outcomes = self._pipeline.scan(text, provenance)
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

    def _benign_precedent(
        self,
        segment: Segment,
        signals: Signals,
        outcomes: Sequence[ScannerOutcome],
        tenant: str | None,
    ) -> str | None:
        """The case id of a reviewed-benign exact match, looked up only where it could matter."""
        if not self._settings.memory.reviewed_benign_allows:
            return None
        if any(outcome.scanner == SEAM_SCANNER for outcome in outcomes):
            return None  # the review covered these bytes alone, not their join with a neighbour
        thresholds = self._settings.thresholds
        if not thresholds.escalate_at <= signals.score < thresholds.block_at:
            return None
        if segment.truncated or is_high_impact(segment.provenance):
            return None
        record = self._incidents.reviewed_benign(tenant, segment.provenance, segment.digest)
        return None if record is None else record.case_id

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
