"""Latency percentiles and counters, plus the Prometheus exposition of the same numbers.

Percentiles use the nearest-rank method, which is what the small synthetic samples in tests pin:
for ``[1..100]`` the p50 is 50 and the p95 is 95. Samples are bounded per stage so a long-running
service keeps a fixed memory footprint, and every mutation and read holds a lock because the
endpoints are synchronous and may be called from a thread pool.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque

from .models import (
    CacheStats,
    MetricsReport,
    OwaspCategory,
    ScanResponse,
    StageLatency,
    VerdictDecision,
)
from .settings import DEFAULT_LATENCY_BUDGET_MS

DEFAULT_MAX_SAMPLES = 10_000

#: Live latency budget used only for the ``_budget`` gauges in the Prometheus text.
LATENCY_BUDGET_MS = DEFAULT_LATENCY_BUDGET_MS


def percentile(sorted_values: list[float], percent: float) -> float:
    """Nearest-rank percentile: the value at ``ceil(p/100 * n)`` in a sorted list."""
    if not sorted_values:
        return 0.0
    rank = max(1, math.ceil(percent / 100.0 * len(sorted_values)))
    return sorted_values[min(rank, len(sorted_values)) - 1]


class LatencyRecorder:
    """Per-stage latency samples with bounded memory."""

    def __init__(self, max_samples: int = DEFAULT_MAX_SAMPLES) -> None:
        self._max_samples = max_samples
        self._samples: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def observe(self, stage: str, milliseconds: float) -> None:
        with self._lock:
            samples = self._samples.get(stage)
            if samples is None:
                samples = deque(maxlen=self._max_samples)
                self._samples[stage] = samples
            samples.append(max(0.0, milliseconds))

    def report(self) -> dict[str, StageLatency]:
        with self._lock:
            snapshot = {stage: list(values) for stage, values in self._samples.items()}
        report: dict[str, StageLatency] = {}
        for stage, values in snapshot.items():
            ordered = sorted(values)
            report[stage] = StageLatency(
                count=len(ordered),
                p50_ms=round(percentile(ordered, 50), 4),
                p95_ms=round(percentile(ordered, 95), 4),
                p99_ms=round(percentile(ordered, 99), 4),
                max_ms=round(ordered[-1], 4) if ordered else 0.0,
            )
        return report


class Metrics:
    """The gateway-local counters and latency recorder behind ``/v1/metrics`` and ``/metrics``."""

    def __init__(self, max_samples: int = DEFAULT_MAX_SAMPLES) -> None:
        self._latency = LatencyRecorder(max_samples)
        self._lock = threading.Lock()
        self._start = time.monotonic()
        self._scans = 0
        self._segments = 0
        self._verdicts: dict[VerdictDecision, int] = {d: 0 for d in VerdictDecision}
        self._categories: dict[str, int] = {}
        self._escalations = 0
        self._retrieval_queries = 0
        self._scanner_errors = 0

    # -- recording ----------------------------------------------------------------------------
    def observe(self, stage: str, milliseconds: float) -> None:
        self._latency.observe(stage, milliseconds)

    def record_scan(self, response: ScanResponse) -> None:
        with self._lock:
            self._scans += 1
            self._segments += len(response.segments)
            for verdict in response.segments:
                self._verdicts[verdict.decision] = self._verdicts.get(verdict.decision, 0) + 1
                if verdict.decision is VerdictDecision.ESCALATE:
                    self._escalations += 1
                if verdict.memory is not None and verdict.memory.queried:
                    self._retrieval_queries += 1
                for category in verdict.signals.category_scores:
                    self._categories[category.value] = self._categories.get(category.value, 0) + 1
                for outcome in verdict.scanners:
                    if outcome.error:
                        self._scanner_errors += 1

    def note_scanner_error(self) -> None:
        with self._lock:
            self._scanner_errors += 1

    # -- reporting ----------------------------------------------------------------------------
    def window_seconds(self) -> float:
        return time.monotonic() - self._start

    def report(self, cache: CacheStats) -> MetricsReport:
        with self._lock:
            scans = self._scans
            segments = self._segments
            verdicts = dict(self._verdicts)
            categories = dict(self._categories)
            escalations = self._escalations
            retrieval_queries = self._retrieval_queries
            scanner_errors = self._scanner_errors

        return MetricsReport(
            window_seconds=round(self.window_seconds(), 3),
            scans=scans,
            segments=segments,
            verdicts=verdicts,
            decisions_by_category={
                OwaspCategory(category): count for category, count in categories.items()
            },
            stages=self._latency.report(),
            escalations=escalations,
            retrieval_queries=retrieval_queries,
            scanner_errors=scanner_errors,
            cache=cache,
        )

    def prometheus(self, cache: CacheStats) -> str:
        report = self.report(cache)
        lines: list[str] = []
        lines += _counter("guardian_scans_total", "Requests scanned.", report.scans)
        lines += _counter("guardian_segments_total", "Segments inspected.", report.segments)
        for decision, count in sorted(report.verdicts.items(), key=lambda item: item[0].value):
            lines += _counter(
                "guardian_verdicts_total",
                "Verdicts by decision.",
                count,
                {"decision": decision.value},
            )
        for category, count in sorted(
            report.decisions_by_category.items(), key=lambda i: i[0].value
        ):
            lines += _counter(
                "guardian_decisions_by_category_total",
                "Segment decisions that touched an OWASP category.",
                count,
                {"category": category.value},
            )
        lines += _counter("guardian_escalations_total", "Escalated requests.", report.escalations)
        lines += _counter(
            "guardian_retrieval_queries_total",
            "Risk-memory lookups inside the uncertain band.",
            report.retrieval_queries,
        )
        lines += _counter(
            "guardian_scanner_errors_total", "Scanner failures observed.", report.scanner_errors
        )
        lines += _gauge("guardian_cache_entries", "Feature-cache entries.", report.cache.entries)
        lines += _counter("guardian_cache_hits_total", "Feature-cache hits.", report.cache.hits)
        lines += _counter(
            "guardian_cache_misses_total", "Feature-cache misses.", report.cache.misses
        )
        for stage, latency in sorted(report.stages.items()):
            lines += _gauge(
                "guardian_stage_latency_p50_ms",
                "Per-stage p50 latency.",
                latency.p50_ms,
                {"stage": stage},
            )
            lines += _gauge(
                "guardian_stage_latency_p95_ms",
                "Per-stage p95 latency.",
                latency.p95_ms,
                {"stage": stage},
            )
            lines += _gauge(
                "guardian_stage_latency_p99_ms",
                "Per-stage p99 latency.",
                latency.p99_ms,
                {"stage": stage},
            )
        lines += _gauge(
            "guardian_latency_budget_ms",
            "The design's experimental p95 target for short requests.",
            LATENCY_BUDGET_MS,
        )
        return "\n".join(lines) + "\n"


def _counter(
    name: str, help_text: str, value: int, labels: dict[str, str] | None = None
) -> list[str]:
    return [
        f"# HELP {name} {help_text}",
        f"# TYPE {name} counter",
        f"{name}{_labels(labels)} {value}",
    ]


def _gauge(
    name: str, help_text: str, value: float, labels: dict[str, str] | None = None
) -> list[str]:
    return [
        f"# HELP {name} {help_text}",
        f"# TYPE {name} gauge",
        f"{name}{_labels(labels)} {value}",
    ]


def _labels(labels: dict[str, str] | None) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{key}="{value}"' for key, value in sorted(labels.items()))
    return "{" + inner + "}"
