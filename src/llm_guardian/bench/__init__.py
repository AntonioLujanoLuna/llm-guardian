"""Phase-1 evaluation: a labelled corpus and a black-box measurement harness.

The design (``ideas/adaptive_llm_guardian_risk_memory.md``, sections 6 and 7) makes Phase 1 an
*empirical* question: before the guardian is allowed to block anything, measure its comparative
recall, latency, error handling and false-positive rate on held-out traffic and benign developer
requests. This package is that measurement.

Two halves, deliberately separated:

- :mod:`llm_guardian.bench.corpus` is the labelled data: attacks *and* the benign lookalikes that
  decide whether a guardian is usable. Everything here is synthesized for this repository; no
  third-party dataset text is copied and no real secret appears (a test enforces both).
- :mod:`llm_guardian.bench.harness` is the instrument: it drives a scan callable (in-process or
  over HTTP), computes the metrics the design names, and prints every false positive and false
  negative by name, because a metric without the failing cases cannot be acted on.
"""

from __future__ import annotations

from .corpus import CASES, Case, filter_cases, load
from .harness import Report, ScanObservation, run, run_cli

__all__ = [
    "CASES",
    "Case",
    "Report",
    "ScanObservation",
    "filter_cases",
    "load",
    "run",
    "run_cli",
]
