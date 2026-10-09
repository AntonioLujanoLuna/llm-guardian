"""Latency percentiles and counters, plus the Prometheus exposition of the same numbers.

The Prometheus text groups samples by metric family -- one ``# HELP`` and one ``# TYPE`` per
family, however many labelled samples it has -- because a second HELP line for the same name makes
a Prometheus scrape fail outright.

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
        self._events: dict[str, int] = {}

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
                    if outcome.scanner == "seam" and outcome.findings:
                        self._bump_locked("seam_findings")
                for note in verdict.notes:
                    for prefix, event in _NOTE_EVENTS:
                        if note.startswith(prefix):
                            self._bump_locked(event)
            for warning in response.warnings:
                if warning.startswith("mode_override_refused"):
                    self._bump_locked("mode_overrides_refused")

    def count_event(self, event: str, amount: int = 1) -> None:
        """Count a named event recorded outside a scan (the proxy's blocks and denials)."""
        with self._lock:
            self._bump_locked(event, amount)

    def _bump_locked(self, event: str, amount: int = 1) -> None:
        self._events[event] = self._events.get(event, 0) + amount

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
            events = dict(self._events)

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
            events=events,
            cache=cache,
        )

    def prometheus(self, cache: CacheStats) -> str:
        report = self.report(cache)
        out = _Exposition()
        out.counter("guardian_scans_total", "Requests scanned.", report.scans)
        out.counter("guardian_segments_total", "Segments inspected.", report.segments)
        for decision, count in sorted(report.verdicts.items(), key=lambda item: item[0].value):
            out.counter(
                "guardian_verdicts_total",
                "Verdicts by decision.",
                count,
                {"decision": decision.value},
            )
        for category, count in sorted(
            report.decisions_by_category.items(), key=lambda i: i[0].value
        ):
            out.counter(
                "guardian_decisions_by_category_total",
                "Segment decisions that touched an OWASP category.",
                count,
                {"category": category.value},
            )
        out.counter("guardian_escalations_total", "Escalated requests.", report.escalations)
        out.counter(
            "guardian_retrieval_queries_total",
            "Risk-memory lookups inside the uncertain band.",
            report.retrieval_queries,
        )
        out.counter(
            "guardian_scanner_errors_total", "Scanner failures observed.", report.scanner_errors
        )
        for event, count in sorted(report.events.items()):
            out.counter(
                "guardian_events_total",
                "Named guardian events: seams, session escalations, proxy blocks, ...",
                count,
                {"event": event},
            )
        out.gauge("guardian_cache_entries", "Feature-cache entries.", report.cache.entries)
        out.counter("guardian_cache_hits_total", "Feature-cache hits.", report.cache.hits)
        out.counter("guardian_cache_misses_total", "Feature-cache misses.", report.cache.misses)
        for stage, latency in sorted(report.stages.items()):
            for name, value in (
                ("p50", latency.p50_ms),
                ("p95", latency.p95_ms),
                ("p99", latency.p99_ms),
            ):
                out.gauge(
                    f"guardian_stage_latency_{name}_ms",
                    f"Per-stage {name} latency.",
                    value,
                    {"stage": stage},
                )
        out.gauge(
            "guardian_latency_budget_ms",
            "The design's experimental p95 target for short requests.",
            LATENCY_BUDGET_MS,
        )
        return out.render()


#: Verdict notes that are themselves events worth counting.
_NOTE_EVENTS: tuple[tuple[str, str], ...] = (
    ("session_risk_escalate", "session_escalations"),
    ("reviewed_benign_precedent_allow", "reviewed_benign_allows"),
)


class _Exposition:
    """Prometheus text format, grouped so each family has exactly one HELP and one TYPE line."""

    def __init__(self) -> None:
        self._families: dict[str, tuple[str, str, list[str]]] = {}

    def counter(
        self, name: str, help_text: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        self._add("counter", name, help_text, value, labels)

    def gauge(
        self, name: str, help_text: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        self._add("gauge", name, help_text, value, labels)

    def _add(
        self,
        kind: str,
        name: str,
        help_text: str,
        value: float,
        labels: dict[str, str] | None,
    ) -> None:
        family = self._families.setdefault(name, (help_text, kind, []))
        family[2].append(f"{name}{_labels(labels)} {value}")

    def render(self) -> str:
        lines: list[str] = []
        for name, (help_text, kind, samples) in self._families.items():
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} {kind}")
            lines.extend(samples)
        return "\n".join(lines) + "\n"


def _labels(labels: dict[str, str] | None) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{key}="{_escape(value)}"' for key, value in sorted(labels.items()))
    return "{" + inner + "}"


def _escape(value: str) -> str:
    """Label-value escaping from the exposition format: backslash, double quote, newline."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
