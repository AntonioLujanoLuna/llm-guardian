"""Metrics: nearest-rank percentiles, counters and Prometheus exposition."""

from __future__ import annotations

from llm_guardian.metrics import LatencyRecorder, Metrics, percentile
from llm_guardian.models import (
    CacheStats,
    GuardMode,
    LatencyBreakdown,
    MemoryEvidence,
    OwaspCategory,
    ScanResponse,
    SegmentVerdict,
    Signals,
    VerdictDecision,
)
from llm_guardian.provenance import Provenance
from tests.helpers import make_outcome


def test_percentile_nearest_rank() -> None:
    sample = [float(i) for i in range(1, 101)]
    assert percentile(sample, 50) == 50.0
    assert percentile(sample, 95) == 95.0
    assert percentile(sample, 99) == 99.0
    assert percentile([], 50) == 0.0
    assert percentile([7.0], 99) == 7.0


def test_latency_recorder_report() -> None:
    recorder = LatencyRecorder(max_samples=100)
    for value in (1.0, 2.0, 3.0, 4.0):
        recorder.observe("stage", value)
    report = recorder.report()["stage"]
    assert report.count == 4
    assert report.max_ms == 4.0
    assert report.p50_ms == 2.0
    assert report.p95_ms == 4.0


def test_latency_recorder_bounded() -> None:
    recorder = LatencyRecorder(max_samples=2)
    for value in (1.0, 2.0, 3.0):
        recorder.observe("s", value)
    assert recorder.report()["s"].count == 2


def make_verdict(
    decision: VerdictDecision,
    *,
    categories: tuple[OwaspCategory, ...] = (OwaspCategory.LLM01,),
    queried: bool = False,
    error: str | None = None,
) -> SegmentVerdict:
    signals = Signals(score=0.5, category_scores={category: 0.5 for category in categories})
    memory = MemoryEvidence(queried=True, margin=0.1) if queried else None
    return SegmentVerdict(
        segment_id="s",
        index=0,
        provenance=Provenance.USER,
        digest="d",
        decision=decision,
        score=0.5,
        threshold=0.35,
        signals=signals,
        scanners=(make_outcome(error=error),),
        memory=memory,
    )


def make_response(*verdicts: SegmentVerdict) -> ScanResponse:
    return ScanResponse(
        request_id="r",
        conversation_id=None,
        mode=GuardMode.ENFORCE,
        shadow=False,
        decision=VerdictDecision.BLOCK,
        segments=tuple(verdicts),
        latency=LatencyBreakdown(),
        guardian_version="0.1.0",
    )


def test_record_scan_counts() -> None:
    metrics = Metrics()
    metrics.observe("total", 1.0)
    response = make_response(
        make_verdict(VerdictDecision.BLOCK, queried=True, error="boom"),
        make_verdict(VerdictDecision.ESCALATE, categories=(OwaspCategory.LLM02,)),
        make_verdict(VerdictDecision.ALLOW, categories=()),
    )
    metrics.record_scan(response)
    report = metrics.report(
        CacheStats(entries=0, hits=0, misses=0, evictions=0, max_entries=1, ttl_seconds=1.0)
    )
    assert report.scans == 1
    assert report.segments == 3
    assert report.verdicts[VerdictDecision.BLOCK] == 1
    assert report.verdicts[VerdictDecision.ESCALATE] == 1
    assert report.escalations == 1
    assert report.retrieval_queries == 1
    assert report.scanner_errors == 1
    assert report.decisions_by_category[OwaspCategory.LLM01] == 1
    assert report.decisions_by_category[OwaspCategory.LLM02] == 1
    assert report.window_seconds >= 0.0


def test_note_scanner_error() -> None:
    metrics = Metrics()
    metrics.note_scanner_error()
    metrics.note_scanner_error()
    report = metrics.report(
        CacheStats(entries=0, hits=0, misses=0, evictions=0, max_entries=1, ttl_seconds=1.0)
    )
    assert report.scanner_errors == 2


def test_prometheus_text() -> None:
    metrics = Metrics()
    metrics.record_scan(make_response(make_verdict(VerdictDecision.BLOCK)))
    cache = CacheStats(entries=2, hits=3, misses=4, evictions=0, max_entries=10, ttl_seconds=60.0)
    text = metrics.prometheus(cache)
    assert "guardian_scans_total 1" in text
    assert 'guardian_verdicts_total{decision="block"} 1' in text
    assert 'guardian_decisions_by_category_total{category="LLM01"}' in text
    assert "guardian_cache_entries 2" in text
    assert "guardian_latency_budget_ms" in text
    assert text.endswith("\n")
