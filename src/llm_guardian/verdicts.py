"""The verdict engine: findings + policy inputs -> ``allow`` / ``escalate`` / ``block``.

Every rule that changes a decision writes a note explaining itself, so a result is never an
unexplained state. The hard boundaries from the design surface here:

* A lone deterministic finding at or above ``severity_floor`` blocks on its own.
* A scanner error follows the declared error policy -- ``warn`` can lift a verdict to *escalate*
  but never to *block*; ``block`` blocks.
* High-impact provenance (a tool call or tool result) fails **closed**: any scanner error or a
  truncated segment blocks, because a mistake there reaches a tool, a filesystem or the network.
* A truncated segment can never *lower* a verdict.
* Risk-memory evidence is advisory: cosine similarity never whitelists content, so it records a
  note and changes nothing.

Shadow vs enforce lives one level up, in the service: this engine always computes the real
verdict, and the response marks it advisory when the mode is shadow.
"""

from __future__ import annotations

from collections.abc import Sequence

from .models import (
    GuardMode,
    MemoryEvidence,
    ScannerErrorPolicy,
    ScannerOutcome,
    Segment,
    SegmentVerdict,
    Severity,
    Signals,
    VerdictDecision,
)
from .provenance import is_high_impact
from .scanners.base import parse_severity, severity_rank
from .settings import GuardianSettings

#: Order of severity for "worst wins" across segments.
_DECISION_RANK: dict[VerdictDecision, int] = {
    VerdictDecision.ALLOW: 0,
    VerdictDecision.ESCALATE: 1,
    VerdictDecision.BLOCK: 2,
}


def worst_decision(verdicts: Sequence[SegmentVerdict]) -> VerdictDecision:
    """The most severe decision across a request's segments."""
    worst = VerdictDecision.ALLOW
    for verdict in verdicts:
        if _DECISION_RANK[verdict.decision] > _DECISION_RANK[worst]:
            worst = verdict.decision
    return worst


class VerdictEngine:
    """Decides one segment's verdict from its signals, scanner outcomes and provenance."""

    def __init__(self, settings: GuardianSettings) -> None:
        self._settings = settings
        thresholds = settings.thresholds
        self._escalate_at = thresholds.escalate_at
        self._block_at = thresholds.block_at
        self._severity_floor = parse_severity(thresholds.severity_floor, Severity.CRITICAL)
        self._overrides = thresholds.category_overrides

    def decide(
        self,
        segment: Segment,
        signals: Signals,
        outcomes: Sequence[ScannerOutcome],
        memory: MemoryEvidence | None = None,
        *,
        mode: GuardMode | None = None,
    ) -> SegmentVerdict:
        notes: list[str] = []
        threshold = self._block_at

        decision, threshold = self._apply_scores(signals, notes)

        if self._floor_hit(outcomes):
            if decision is not VerdictDecision.BLOCK:
                decision = VerdictDecision.BLOCK
                threshold = self._block_at
            notes.append(f"severity_floor_block:{self._severity_floor.value}")

        decision, threshold = self._apply_scanner_errors(
            segment, outcomes, decision, threshold, notes
        )

        if segment.truncated:
            if is_high_impact(segment.provenance):
                if decision is not VerdictDecision.BLOCK:
                    decision = VerdictDecision.BLOCK
                    threshold = self._block_at
                notes.append("high_impact_truncated_fail_closed")
            else:
                notes.append("truncated_segment_not_evidence_of_safety")

        if memory is not None and memory.queried:
            margin = memory.margin if memory.margin is not None else 0.0
            notes.append(f"memory_consulted:margin={margin:.3f}")
            notes.append("cosine_is_not_a_permission")

        if mode is GuardMode.SHADOW:
            notes.append("shadow_mode:verdict_is_advisory")

        cached_features = bool(outcomes) and all(outcome.cached for outcome in outcomes)

        return SegmentVerdict(
            segment_id=segment.segment_id,
            index=segment.index,
            provenance=segment.provenance,
            digest=segment.digest,
            decision=decision,
            score=signals.score,
            threshold=threshold,
            signals=signals,
            scanners=tuple(outcomes),
            memory=memory,
            policy=None,
            cached_features=cached_features,
            notes=tuple(notes),
        )

    # -- internals ----------------------------------------------------------------------------
    def _apply_scores(self, signals: Signals, notes: list[str]) -> tuple[VerdictDecision, float]:
        blocked_by_category: float | None = None
        escalated_by_category: float | None = None
        for category, score in signals.category_scores.items():
            override = self._overrides.get(category.value, {})
            block_at = float(override.get("block_at", self._block_at))
            escalate_at = float(override.get("escalate_at", self._escalate_at))
            if score >= block_at:
                blocked_by_category = max(blocked_by_category or 0.0, block_at)
                notes.append(f"category_block:{category.value}")
            elif score >= escalate_at:
                escalated_by_category = max(escalated_by_category or 0.0, escalate_at)
                notes.append(f"category_escalate:{category.value}")

        overall_block = signals.score >= self._block_at
        overall_escalate = signals.score >= self._escalate_at

        if blocked_by_category is not None or overall_block:
            threshold = blocked_by_category if blocked_by_category is not None else self._block_at
            return VerdictDecision.BLOCK, threshold
        if escalated_by_category is not None or overall_escalate:
            threshold = (
                escalated_by_category if escalated_by_category is not None else self._escalate_at
            )
            return VerdictDecision.ESCALATE, threshold
        return VerdictDecision.ALLOW, self._block_at

    def _floor_hit(self, outcomes: Sequence[ScannerOutcome]) -> bool:
        floor = severity_rank(self._severity_floor)
        for outcome in outcomes:
            for finding in outcome.findings:
                if severity_rank(finding.severity) >= floor:
                    return True
        return False

    def _apply_scanner_errors(
        self,
        segment: Segment,
        outcomes: Sequence[ScannerOutcome],
        decision: VerdictDecision,
        threshold: float,
        notes: list[str],
    ) -> tuple[VerdictDecision, float]:
        errors = [outcome for outcome in outcomes if outcome.error]
        if not errors:
            return decision, threshold

        high_impact = is_high_impact(segment.provenance)
        if high_impact:
            notes.append("high_impact_scanner_error_fail_closed")
            return VerdictDecision.BLOCK, self._block_at

        policy = self._settings.scanners.error_policy
        if policy is ScannerErrorPolicy.BLOCK:
            notes.append("scanner_error_block")
            return VerdictDecision.BLOCK, self._block_at

        # WARN: an error may lift a verdict to escalate, never to block on its own.
        names = ",".join(outcome.scanner for outcome in errors)
        if decision is VerdictDecision.ALLOW:
            notes.append(f"scanner_error_warn_escalate:{names}")
            return VerdictDecision.ESCALATE, self._escalate_at
        notes.append(f"scanner_error_warn:{names}")
        return decision, threshold
