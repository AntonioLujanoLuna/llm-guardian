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

Run it with ``python -m llm_guardian.bench`` (or ``llm-guardian bench``). The exports below are
lazy: importing this package must not import the harness, or running the harness as a module
loads it twice and runpy warns about it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - names are resolved lazily at runtime
    from .corpus import CASES as CASES
    from .corpus import Case as Case
    from .corpus import filter_cases as filter_cases
    from .corpus import load as load
    from .harness import Report as Report
    from .harness import ScanObservation as ScanObservation
    from .harness import run as run
    from .harness import run_cli as run_cli

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

_CORPUS_NAMES = frozenset({"CASES", "Case", "filter_cases", "load"})
_HARNESS_NAMES = frozenset({"Report", "ScanObservation", "run", "run_cli"})


def __getattr__(name: str) -> Any:
    if name in _CORPUS_NAMES:
        from . import corpus

        return getattr(corpus, name)
    if name in _HARNESS_NAMES:
        from . import harness

        return getattr(harness, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
