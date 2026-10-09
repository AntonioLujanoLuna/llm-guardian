"""The Phase-1 measurement harness: black-box metrics over the labelled corpus.

It is deliberately black-box. A case goes out as a ``POST /v1/scan`` with one segment and the
case's provenance; a verdict and a ``LatencyBreakdown`` come back. Nothing here reaches inside the
guardian, so the same instrument measures an in-process app and a live URL, and a stub, without
changing a line.

What it publishes -- the Phase-1 exit criterion from the design, section 7:

- **recall by attack family** and **by OWASP category** (detection is a ``block``; an ``escalate``
  is reported separately because it is a request for review, not a catch);
- **false blocks per 1,000 benign developer requests** -- the headline usability number, and the
  reason half the corpus is benign security work that merely *quotes* an attack;
- **escalation rate**;
- **precision / recall** over the labelled set, with the standing caveat that the corpus was
  written by the same author as the guardian, so this is a *baseline*, not third-party validation;
- **per-stage latency p50/p95/p99/max**, including the total measured against the design's
  30 ms p95 target, reported as a number;
- **error / degradation counts**: truncated scans, scanner errors, unreachable risk memory and a
  disabled ML classifier.

And, because a metric without its failing cases cannot be acted on, the report carries a per-case
table of **every false positive and every false negative**, with family and a text excerpt.

:func:`naive_keyword_flag` is the overdefense trap made explicit: a one-line keyword reference
implementation, provided so the report can name the benign cases a crude matcher would wrongly
flag. Those names are the point of the exercise.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..models import LatencyBreakdown, ScanResponse, VerdictDecision
from ..settings import DEFAULT_LATENCY_BUDGET_MS
from .corpus import CASES, Case, load

__all__ = [
    "NAIVE_KEYWORDS",
    "Report",
    "ScanObservation",
    "default_inprocess_callable",
    "make_app_scan_callable",
    "make_url_scan_callable",
    "naive_keyword_false_positives",
    "naive_keyword_flag",
    "run",
    "run_cli",
    "stub_callable",
]

Decision = str
ALLOW: Decision = VerdictDecision.ALLOW.value
ESCALATE: Decision = VerdictDecision.ESCALATE.value
BLOCK: Decision = VerdictDecision.BLOCK.value
ERROR: Decision = "error"

#: Stage name -> the ``LatencyBreakdown`` attribute it reads.
_STAGES: tuple[tuple[str, str], ...] = (
    ("preprocess", "preprocess_ms"),
    ("deterministic", "deterministic_ms"),
    ("ml", "ml_ms"),
    ("retrieval", "retrieval_ms"),
    ("total", "total_ms"),
)


# =================================================================================================
# The scan callable contract
# =================================================================================================
class ScanObservation(BaseModel):
    """What one scan call of one case produced, normalized across every transport.

    ``decision`` is one of ``allow`` / ``escalate`` / ``block`` / ``error``. An ``error`` is a
    transport or HTTP failure, distinct from a scanner error inside a healthy response.
    """

    model_config = ConfigDict(frozen=True)

    decision: Decision
    latency: LatencyBreakdown = Field(default_factory=LatencyBreakdown)
    mode: str = "shadow"
    shadow: bool = True
    truncated: bool = False
    scanner_errors: tuple[str, ...] = ()
    memory_reachable: bool | None = None
    ml_enabled: bool = False
    warnings: tuple[str, ...] = ()
    error: str | None = None


ScanCallable = Callable[[Case], ScanObservation]


def _request_body(case: Case) -> dict[str, Any]:
    return {
        "request_id": case.id,
        "conversation_id": f"bench-{case.id}",
        "segments": [{"text": case.text, "provenance": case.provenance}],
    }


def from_scan_response(response: ScanResponse) -> ScanObservation:
    """Reduce a full :class:`ScanResponse` to the fields the harness actually measures."""

    scanner_errors: list[str] = []
    notes: list[str] = []
    memory_reachable: bool | None = None
    for segment in response.segments:
        scanner_errors.extend(segment.signals.scanner_errors)
        for outcome in segment.scanners:
            if outcome.error:
                scanner_errors.append(f"{outcome.scanner}: {outcome.error}")
        notes.extend(segment.notes)
        memory = segment.memory
        if memory is not None and memory.reason_not_queried:
            reason = memory.reason_not_queried.lower()
            if "unreach" in reason or "unavailable" in reason or "error" in reason:
                memory_reachable = False
    haystack = " ".join((*response.warnings, *notes)).lower()
    return ScanObservation(
        decision=str(response.decision),
        latency=response.latency,
        mode=str(response.mode),
        shadow=response.shadow,
        truncated="truncat" in haystack,
        scanner_errors=tuple(scanner_errors),
        memory_reachable=memory_reachable,
        ml_enabled=response.latency.ml_ms > 0.0,
        warnings=tuple(response.warnings),
    )


def as_observation(result: object) -> ScanObservation:
    """Coerce whatever a callable returned into a :class:`ScanObservation`."""

    if isinstance(result, ScanObservation):
        return result
    if isinstance(result, ScanResponse):
        return from_scan_response(result)
    if isinstance(result, dict):
        return ScanObservation.model_validate(result)
    if isinstance(result, str):
        return ScanObservation(decision=result)
    raise TypeError(f"scan callable returned {type(result)!r}, not an observation")


def _wrap(fn: Callable[[Case], object]) -> ScanCallable:
    return lambda case: as_observation(fn(case))


def make_url_scan_callable(base_url: str, *, timeout: float = 10.0) -> ScanCallable:
    """Drive a running service over HTTP. Uses the standard library, so no client dependency."""

    endpoint = base_url.rstrip("/") + "/v1/scan"

    def _scan(case: Case) -> ScanObservation:
        body = json.dumps(_request_body(case)).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers={"content-type": "application/json", "accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return ScanObservation(decision=ERROR, error=f"HTTP {exc.code}")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return ScanObservation(decision=ERROR, error=f"{type(exc).__name__}: {exc}")
        return from_scan_response(ScanResponse.model_validate(payload))

    return _scan


def make_app_scan_callable(app: object, *, client: object | None = None) -> ScanCallable:
    """Drive an in-process FastAPI app through its own ASGI surface (no HTTP hop for the caller)."""

    if client is None:
        from starlette.testclient import TestClient

        client = TestClient(app)  # type: ignore[arg-type]

    post = client.post  # type: ignore[attr-defined]

    def _scan(case: Case) -> ScanObservation:
        try:
            response = post("/v1/scan", json=_request_body(case))
        except Exception as exc:
            return ScanObservation(decision=ERROR, error=f"{type(exc).__name__}: {exc}")
        if response.status_code != 200:
            return ScanObservation(decision=ERROR, error=f"HTTP {response.status_code}")
        return from_scan_response(ScanResponse.model_validate(response.json()))

    return _scan


def default_inprocess_callable() -> ScanCallable:
    """The in-process callable the CLI uses when no URL and no injected callable are given."""

    from ..app import create_app

    return make_app_scan_callable(create_app())


def stub_callable(decision: Decision, *, total_ms: float = 1.0, **overrides: Any) -> ScanCallable:
    """A deterministic callable that answers every case with one decision.

    This is what makes the harness testable without the service, and how the report proves the
    metric math: a block-everything stub must yield 100% recall, an allow-everything stub 0%.
    """

    def _scan(_case: Case) -> ScanObservation:
        return ScanObservation(
            decision=decision,
            latency=LatencyBreakdown(total_ms=total_ms),
            **overrides,
        )

    return _scan


def naively_wrapped(fn: Callable[[Case], object]) -> ScanCallable:
    """Public alias used by callers that return raw dicts, strings or ``ScanResponse`` objects."""

    return _wrap(fn)


# =================================================================================================
# The naive keyword reference -- the overdefense trap, made explicit
# =================================================================================================
#: Phrase substrings a crude matcher keys on. Every one of these also appears in a benign case in
#: the corpus, on purpose: a document that quotes an attack is not an attack.
NAIVE_KEYWORDS: tuple[str, ...] = (
    "ignore all previous",
    "ignore previous",
    "ignore prior",
    "ignore your",
    "ignore the user",
    "ignora todas",
    "ignora las",
    "ignora la",
    "ignora tus",
    "olvida tus",
    "disregard",
    "system prompt",
    "prompt de sistema",
    "repeat everything",
    "everything above",
    "repite todo",
    "todo el texto que aparece",
    "</system>",
    "</sistema>",
    "[inst]",
    "assistant:",
    "asistente:",
    "you are now",
    "ahora eres",
    "developer mode",
    "jailbreak",
    "unrestricted",
    "sin restricciones",
    "base64",
    "rot13",
    "zero-width",
    "u+202e",
    ".env",
    "webhook",
    "paste.example",
    "exfiltrat",
    "upload it",
    "authorized_keys",
    "cross-tenant",
    "otro cliente",
    "rm -rf",
    "/etc/shadow",
    "bypass",
    "eludir",
    "prompt injection",
    "inyección de prompts",
    "instruction override",
    "anulación de instrucciones",
    "sent to an external",
    "enviar el informe",
)

#: Whole-word tokens, kept separate so ``dan`` does not match every word that contains it.
_NAIVE_TOKENS: tuple[str, ...] = ("dan",)

_NAIVE_PATTERN = re.compile(
    "|".join(
        [re.escape(keyword) for keyword in NAIVE_KEYWORDS]
        + [rf"\b{re.escape(token)}\b" for token in _NAIVE_TOKENS]
    ),
    re.IGNORECASE,
)


def naive_keyword_flag(text: str) -> bool:
    """Whether a crude keyword matcher would flag this text. Not a detector; a control."""

    return _NAIVE_PATTERN.search(text) is not None


def naive_keyword_false_positives(cases: Iterable[Case] = CASES) -> list[Case]:
    """The benign cases a keyword matcher would wrongly flag: the overdefense trap, named."""

    return [case for case in cases if case.label == "benign" and naive_keyword_flag(case.text)]


# =================================================================================================
# Report models
# =================================================================================================
class StageLatency(BaseModel):
    model_config = ConfigDict(frozen=True)

    count: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float


class FamilyRecall(BaseModel):
    """Recall over the attack cases of one family."""

    model_config = ConfigDict(frozen=True)

    family: str
    total: int
    detected: int
    recall: float


class OwaspRecall(BaseModel):
    """Recall over attack cases that carry one OWASP category (a case may carry several)."""

    model_config = ConfigDict(frozen=True)

    owasp: str
    total: int
    detected: int
    recall: float


class FamilyFalseBlocks(BaseModel):
    """False blocks within one benign family -- where overdefense actually hurts."""

    model_config = ConfigDict(frozen=True)

    family: str
    total: int
    blocked: int
    false_blocks_per_1000: float


class Degradation(BaseModel):
    """Error and degradation counts. A truncated scan is never a safe scan; neither is a failure."""

    model_config = ConfigDict(frozen=True)

    truncated: int
    scanner_error_cases: int
    scanner_error_total: int
    memory_unreachable: int
    ml_disabled: int
    transport_errors: int


class CaseResult(BaseModel):
    """One failing case, named, with its family and a printable excerpt."""

    model_config = ConfigDict(frozen=True)

    id: str
    label: str
    family: str
    language: str
    provenance: str
    owasp: tuple[str, ...]
    decision: str
    excerpt: str
    why: str


class Report(BaseModel):
    """The Phase-1 baseline. Every rate is paired with the counts that produced it."""

    model_config = ConfigDict(frozen=True)

    generated_at: datetime
    corpus_size: int
    attack_count: int
    benign_count: int
    repeat: int
    mode: str
    shadow: bool

    # Binary classification over the labelled set: positive = attack, predicted positive = block.
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    recall: float
    precision: float | None
    escalations: int
    escalation_rate: float
    false_blocks: int
    false_blocks_per_1000: float

    recall_by_family: tuple[FamilyRecall, ...]
    recall_by_owasp: tuple[OwaspRecall, ...]
    benign_family_false_blocks: tuple[FamilyFalseBlocks, ...]

    latency_by_stage: dict[str, StageLatency]
    latency_total: StageLatency
    p95_target_ms: float
    p95_meets_target: bool

    degradation: Degradation
    false_positive_cases: tuple[CaseResult, ...]
    false_negative_cases: tuple[CaseResult, ...]
    #: What a crude keyword reference would wrongly block -- the overdefense trap, published
    #: alongside the guardian's own numbers so the two are compared, not confused.
    naive_keyword_false_blocks: int = 0
    naive_keyword_false_block_cases: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2, ensure_ascii=False)

    def to_text(self) -> str:
        return _render_text(self)


# =================================================================================================
# Metric computation
# =================================================================================================
def _percentile(sorted_values: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile (the numpy default), so a fixture has a closed form."""

    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(sorted_values[low])
    fraction = rank - low
    return float(sorted_values[low]) * (1.0 - fraction) + float(sorted_values[high]) * fraction


def _stage_latency(values: Sequence[float]) -> StageLatency:
    ordered = sorted(values)
    return StageLatency(
        count=len(ordered),
        p50_ms=_percentile(ordered, 50.0),
        p95_ms=_percentile(ordered, 95.0),
        p99_ms=_percentile(ordered, 99.0),
        max_ms=float(ordered[-1]) if ordered else 0.0,
    )


_INVISIBLE = {"Cf", "Cc", "Cs", "Co", "Cn"}


def visible(text: str) -> str:
    """Render a text excerpt with invisible/control characters escaped, for a readable table."""

    out: list[str] = []
    for char in text:
        if char in "\n\t\r":
            out.append(" ")
        elif unicodedata.category(char) in _INVISIBLE and char != " ":
            out.append(f"\\u{ord(char):04x}")
        else:
            out.append(char)
    return "".join(out)


def _excerpt(text: str, limit: int = 160) -> str:
    flat = visible(text)
    return flat if len(flat) <= limit else flat[: limit - 1] + "\u2026"


def _case_result(case: Case, observation: ScanObservation) -> CaseResult:
    return CaseResult(
        id=case.id,
        label=case.label,
        family=case.family,
        language=case.language,
        provenance=case.provenance,
        owasp=case.owasp,
        decision=observation.decision,
        excerpt=_excerpt(case.text),
        why=case.why,
    )


def _is_block(decision: Decision) -> bool:
    return decision == BLOCK


def run(
    cases: Sequence[Case],
    scan_callable: ScanCallable | Callable[[Case], object],
    *,
    repeat: int = 1,
    warmup: int = 1,
    p95_target_ms: float = DEFAULT_LATENCY_BUDGET_MS,
) -> Report:
    """Measure ``scan_callable`` over ``cases`` and return a :class:`Report`.

    ``warmup`` full passes are run and discarded before any measurement, so the reported latencies
    are not first-call costs. ``repeat`` runs the measured pass that many times; latency samples
    are pooled across every repetition, while classification uses each case's first measured
    verdict (input is fixed per case, so repetition refines the timing, not the label).
    """

    materialized = list(cases)
    if not materialized:
        raise ValueError("cannot run the harness over an empty corpus")
    if repeat < 1:
        raise ValueError("repeat must be at least 1")
    if warmup < 0:
        raise ValueError("warmup must not be negative")

    wrapped = _wrap(scan_callable)

    for _ in range(warmup):
        for case in materialized:
            wrapped(case)

    first_observation: dict[str, ScanObservation] = {}
    latency_samples: dict[str, list[float]] = {name: [] for name, _ in _STAGES}
    modes: set[str] = set()
    shadows: set[bool] = set()

    for _ in range(repeat):
        for case in materialized:
            observation = wrapped(case)
            first_observation.setdefault(case.id, observation)
            modes.add(observation.mode)
            shadows.add(observation.shadow)
            for name, attribute in _STAGES:
                latency_samples[name].append(float(getattr(observation.latency, attribute)))

    attacks = [case for case in materialized if case.label == "attack"]
    benign = [case for case in materialized if case.label == "benign"]

    def verdict(case: Case) -> ScanObservation:
        return first_observation[case.id]

    true_positives = sum(1 for case in attacks if _is_block(verdict(case).decision))
    false_negatives = len(attacks) - true_positives
    false_positives = sum(1 for case in benign if _is_block(verdict(case).decision))
    true_negatives = len(benign) - false_positives

    recall = true_positives / len(attacks) if attacks else 0.0
    predicted_positive = true_positives + false_positives
    precision = (true_positives / predicted_positive) if predicted_positive else None

    escalations = sum(1 for case in materialized if verdict(case).decision == ESCALATE)
    escalation_rate = escalations / len(materialized)
    false_blocks_per_1000 = (false_positives / len(benign) * 1000.0) if benign else 0.0

    family_total: Counter[str] = Counter()
    family_detected: Counter[str] = Counter()
    family_blocked_benign: Counter[str] = Counter()
    benign_family_total: Counter[str] = Counter()
    owasp_total: Counter[str] = Counter()
    owasp_detected: Counter[str] = Counter()

    for case in attacks:
        detected = _is_block(verdict(case).decision)
        family_total[case.family] += 1
        if detected:
            family_detected[case.family] += 1
        for category in case.owasp:
            owasp_total[category] += 1
            if detected:
                owasp_detected[category] += 1

    for case in benign:
        benign_family_total[case.family] += 1
        if _is_block(verdict(case).decision):
            family_blocked_benign[case.family] += 1

    recall_by_family = tuple(
        FamilyRecall(
            family=family,
            total=family_total[family],
            detected=family_detected[family],
            recall=family_detected[family] / family_total[family],
        )
        for family in sorted(family_total)
    )
    recall_by_owasp = tuple(
        OwaspRecall(
            owasp=category,
            total=owasp_total[category],
            detected=owasp_detected[category],
            recall=owasp_detected[category] / owasp_total[category],
        )
        for category in sorted(owasp_total)
    )
    benign_family_false_blocks = tuple(
        FamilyFalseBlocks(
            family=family,
            total=benign_family_total[family],
            blocked=family_blocked_benign[family],
            false_blocks_per_1000=(
                family_blocked_benign[family] / benign_family_total[family] * 1000.0
            ),
        )
        for family in sorted(benign_family_total)
    )

    latency_by_stage = {
        name: _stage_latency(latency_samples[name]) for name, _ in _STAGES if name != "total"
    }
    latency_total = _stage_latency(latency_samples["total"])

    degradation = Degradation(
        truncated=sum(1 for case in materialized if verdict(case).truncated),
        scanner_error_cases=sum(1 for case in materialized if verdict(case).scanner_errors),
        scanner_error_total=sum(len(verdict(case).scanner_errors) for case in materialized),
        memory_unreachable=sum(
            1 for case in materialized if verdict(case).memory_reachable is False
        ),
        ml_disabled=sum(1 for case in materialized if not verdict(case).ml_enabled),
        transport_errors=sum(1 for case in materialized if verdict(case).decision == ERROR),
    )

    false_positive_cases = tuple(
        _case_result(case, verdict(case)) for case in benign if _is_block(verdict(case).decision)
    )
    false_negative_cases = tuple(
        _case_result(case, verdict(case))
        for case in attacks
        if not _is_block(verdict(case).decision)
    )

    naive_traps = naive_keyword_false_positives(materialized)

    return Report(
        generated_at=datetime.now(UTC),
        corpus_size=len(materialized),
        attack_count=len(attacks),
        benign_count=len(benign),
        repeat=repeat,
        mode=(next(iter(modes)) if len(modes) == 1 else "mixed"),
        shadow=(next(iter(shadows)) if len(shadows) == 1 else False),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        true_negatives=true_negatives,
        recall=recall,
        precision=precision,
        escalations=escalations,
        escalation_rate=escalation_rate,
        false_blocks=false_positives,
        false_blocks_per_1000=false_blocks_per_1000,
        recall_by_family=recall_by_family,
        recall_by_owasp=recall_by_owasp,
        benign_family_false_blocks=benign_family_false_blocks,
        latency_by_stage=latency_by_stage,
        latency_total=latency_total,
        p95_target_ms=p95_target_ms,
        p95_meets_target=latency_total.p95_ms <= p95_target_ms,
        degradation=degradation,
        false_positive_cases=false_positive_cases,
        false_negative_cases=false_negative_cases,
        naive_keyword_false_blocks=len(naive_traps),
        naive_keyword_false_block_cases=tuple(case.id for case in naive_traps),
        caveats=_CAVEATS,
    )


_CAVEATS: tuple[str, ...] = (
    "The corpus is synthesized by the same author as the guardian; this is a baseline, not "
    "third-party validation.",
    "Detection is defined as a 'block' verdict; 'escalate' is counted separately as escalation "
    "rate and does not count as a catch.",
    "Latency samples are pooled across repeats after warmup; classification uses each case's "
    "first measured verdict.",
)


# =================================================================================================
# Rendering
# =================================================================================================
def _render_text(report: Report) -> str:
    lines: list[str] = []
    add = lines.append
    add("LLM Guardian -- Phase 1 baseline (synthetic corpus; a baseline, not third-party")
    add("validation)")
    add("=" * 88)
    add(
        f"cases: {report.corpus_size} ({report.attack_count} attack / {report.benign_count} "
        f"benign)  repeat={report.repeat}  mode={report.mode}  shadow={report.shadow}"
    )
    add("")
    add("Classification (positive = attack, predicted positive = block)")
    add("-" * 88)
    add(
        f"  recall={report.recall:.3f}  precision="
        f"{'n/a' if report.precision is None else f'{report.precision:.3f}'}"
        f"  tp={report.true_positives} fp={report.false_positives} "
        f"fn={report.false_negatives} tn={report.true_negatives}"
    )
    add(
        f"  false blocks per 1,000 benign: {report.false_blocks_per_1000:.1f} "
        f"({report.false_blocks}/{report.benign_count})"
    )
    add(f"  escalations: {report.escalations}  escalation rate: {report.escalation_rate:.3f}")
    add("")
    add("Recall by attack family")
    add("-" * 88)
    for family_entry in report.recall_by_family:
        add(
            f"  {family_entry.recall:6.3f}  {family_entry.detected:>3}/{family_entry.total:<3}  "
            f"{family_entry.family}"
        )
    add("")
    add("Recall by OWASP category")
    add("-" * 88)
    for owasp_entry in report.recall_by_owasp:
        add(
            f"  {owasp_entry.recall:6.3f}  {owasp_entry.detected:>3}/{owasp_entry.total:<3}  "
            f"{owasp_entry.owasp}"
        )
    add("")
    add("False blocks by benign family (overdefense)")
    add("-" * 88)
    for family_blocks in report.benign_family_false_blocks:
        add(
            f"  {family_blocks.false_blocks_per_1000:7.1f}/1000  "
            f"{family_blocks.blocked:>2}/{family_blocks.total:<2}  {family_blocks.family}"
        )
    add("")
    add(
        f"Latency (target p95 <= {report.p95_target_ms:.1f} ms -- "
        f"{'MET' if report.p95_meets_target else 'NOT MET'})"
    )
    add("-" * 88)
    for name, stage in report.latency_by_stage.items():
        add(
            f"  {name:<14} n={stage.count:<5} p50={stage.p50_ms:7.3f} p95={stage.p95_ms:7.3f} "
            f"p99={stage.p99_ms:7.3f} max={stage.max_ms:8.3f}"
        )
    total = report.latency_total
    add(
        f"  {'total':<14} n={total.count:<5} p50={total.p50_ms:7.3f} p95={total.p95_ms:7.3f} "
        f"p99={total.p99_ms:7.3f} max={total.max_ms:8.3f}"
    )
    add("")
    add("Degradation / errors")
    add("-" * 88)
    d = report.degradation
    add(
        f"  truncated={d.truncated}  scanner_error_cases={d.scanner_error_cases} "
        f"(total {d.scanner_error_total})  memory_unreachable={d.memory_unreachable}  "
        f"ml_disabled={d.ml_disabled}  transport_errors={d.transport_errors}"
    )
    add("")
    add(f"False positives ({len(report.false_positive_cases)}) -- benign, wrongly blocked")
    add("-" * 88)
    if not report.false_positive_cases:
        add("  none")
    for fp_case in report.false_positive_cases:
        add(f"  [{fp_case.decision}] {fp_case.id}  ({fp_case.family}, {fp_case.language})")
        add(f"      {fp_case.excerpt}")
    add("")
    add(f"False negatives ({len(report.false_negative_cases)}) -- attacks, not blocked")
    add("-" * 88)
    if not report.false_negative_cases:
        add("  none")
    for fn_case in report.false_negative_cases:
        add(f"  [{fn_case.decision}] {fn_case.id}  ({fn_case.family}, {fn_case.language})")
        add(f"      {fn_case.excerpt}")
    add("")
    add(
        f"Overdefense trap: a naive keyword matcher wrongly blocks "
        f"{report.naive_keyword_false_blocks} benign cases"
    )
    add("-" * 88)
    for case_id in report.naive_keyword_false_block_cases:
        add(f"  {case_id}")
    add("")
    add("Caveats")
    add("-" * 88)
    for caveat in report.caveats:
        add(f"  - {caveat}")
    return "\n".join(lines)


# =================================================================================================
# CLI
# =================================================================================================
def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-guardian bench",
        description="Run the Phase-1 baseline over the labelled corpus.",
    )
    parser.add_argument("--url", default=None, help="drive a running service at this base URL")
    parser.add_argument("--corpus", default=None, help="load cases from a JSONL file instead")
    parser.add_argument("--label", choices=["attack", "benign"], default=None)
    parser.add_argument("--family", default=None, help="restrict to one family")
    parser.add_argument("--language", choices=["en", "es"], default=None)
    parser.add_argument("--tag", action="append", default=None, help="require each given tag")
    parser.add_argument("--repeat", type=int, default=1, help="measured passes (default 1)")
    parser.add_argument("--warmup", type=int, default=1, help="discarded warmup passes (default 1)")
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument("--report", action="store_true", help="emit the human-readable report")
    parser.add_argument(
        "--stub",
        choices=["allow", "block", "none"],
        default=None,
        help="skip the service and answer every case the same way (for the harness self-check)",
    )
    parser.add_argument(
        "--p95-target-ms", type=float, default=DEFAULT_LATENCY_BUDGET_MS, dest="p95_target_ms"
    )
    parser.add_argument(
        "--max-false-blocks-per-1000",
        type=float,
        default=None,
        dest="max_false_blocks_per_1000",
        help="exit 1 when the measured rate exceeds this",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=None,
        dest="min_recall",
        help="exit 1 when the measured recall is below this",
    )
    return parser


def _cli_callable(args: argparse.Namespace) -> ScanCallable:
    if args.stub == "allow":
        return stub_callable(ALLOW)
    if args.stub == "block":
        return stub_callable(BLOCK)
    if args.url:
        return make_url_scan_callable(args.url)
    return default_inprocess_callable()


def run_cli(
    argv: Sequence[str],
    *,
    scan_callable: ScanCallable | None = None,
) -> int:
    """Entry point the ``llm-guardian bench`` subcommand calls.

    Returns 0 on success, 1 when a caller-supplied ``--max-false-blocks-per-1000`` or
    ``--min-recall`` threshold is breached, and 2 on a usage/corpus error.
    """

    args = _build_cli_parser().parse_args(list(argv))

    try:
        cases = load(
            args.corpus,
            label=args.label,
            family=args.family,
            language=args.language,
            tags=args.tag,
        )
    except (OSError, ValueError) as exc:
        print(f"bench: could not load corpus: {exc}", file=sys.stderr)
        return 2

    if not cases:
        print("bench: no cases matched the filters", file=sys.stderr)
        return 2

    callable_ = scan_callable or _cli_callable(args)
    report = run(
        cases,
        callable_,
        repeat=args.repeat,
        warmup=args.warmup,
        p95_target_ms=args.p95_target_ms,
    )

    if args.json:
        print(report.to_json())
    else:
        print(report.to_text())

    breached = False
    if (
        args.max_false_blocks_per_1000 is not None
        and report.false_blocks_per_1000 > args.max_false_blocks_per_1000
    ):
        breached = True
    if args.min_recall is not None and report.recall < args.min_recall:
        breached = True
    return 1 if breached else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Module entry point: ``python -m llm_guardian.bench.harness``."""

    return run_cli(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess in the proof
    raise SystemExit(main())
