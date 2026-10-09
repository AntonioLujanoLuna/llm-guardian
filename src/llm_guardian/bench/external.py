"""Held-out external corpora: licence-manifested loaders over public prompt-injection datasets.

The in-repo corpus (``bench/corpus.py``) is saturated -- the rules were tuned on it, so its 1.000
recall is a regression result, not an estimate of generalisation. This module supplies the missing
measurement: it turns a small, licence-checked set of *public* datasets into the same
:class:`~llm_guardian.bench.corpus.Case` shape the harness already measures, so the frozen
detectors can be scored on traffic they were never written against.

Two properties are deliberate:

* **No new runtime dependency.** Rows are fetched over the Hugging Face *datasets-server* HTTP API
  with :mod:`urllib` only (no ``datasets`` or ``huggingface_hub``). Every response is cached as raw
  JSON under a gitignored path, so a re-run is offline and the exact bytes a number came from are
  recoverable.
* **The licence is part of the data.** Each declared source carries the licence tag quoted from
  its Hub card, and :func:`enforce_license` refuses anything not on the allow-list. A report that
  cannot say what it ingested under what terms is not a report.

Honesty guardrails, enforced in code rather than described:

* a per-dataset ``--limit`` (default 300 cases) bounds the run and the sampling is stated;
* every source carries a ``provenance_rationale`` (a single-turn chat prompt is not a tool result)
  and a ``label_basis`` (what the authors' label actually means -- JailbreakBench's ``benign`` is
  "benign behaviours", not "benign developer traffic"; MSCAP's ``level`` is attack progression);
* :func:`license_manifest` emits the ingest table with the run.

This module never edits the detectors. The frozen pipeline it measures is the same
``ScannerPipeline.from_settings(ScannerSettings())`` -> ``aggregate`` -> ``VerdictEngine`` path the
service uses, driven through :class:`~llm_guardian.service.ScanService` so segmentation and
truncation behave exactly as in production.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .corpus import PROVENANCE_VALUES, Case
from .harness import Report, ScanCallable, ScanObservation, run

__all__ = [
    "ALLOWED_LICENSES",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_LIMIT",
    "DEFAULT_SPECS",
    "EXCLUDED_DATASETS",
    "DatasetSpec",
    "ExternalDataError",
    "LicenseRefused",
    "Loaded",
    "ManifestEntry",
    "PageFetcher",
    "Skipped",
    "Stream",
    "build_frozen_callable",
    "enforce_license",
    "license_manifest",
    "license_refusal",
    "load_all",
    "load_dataset",
    "main",
    "render_markdown",
    "run_cli",
    "to_cases",
]

#: The datasets-server host. The ``rows`` endpoint is paginated with ``offset``/``length``.
DATASETS_SERVER = "https://datasets-server.huggingface.co"
USER_AGENT = "llm-guardian-bench/0.1 (+held-out evaluation)"

#: Where raw page JSON is cached. ``data/`` is gitignored, so a run leaves no committed artefacts.
DEFAULT_CACHE_DIR = Path("data/external-cache")

#: Default case budget per dataset. Bounds the run and forces the sampling to be declared.
DEFAULT_LIMIT = 300

#: The datasets-server caps ``length`` at 100 rows per request.
DEFAULT_PAGE_SIZE = 100

#: HTTP statuses worth retrying (the public datasets-server rate-limits with 429).
_RETRY_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
_MAX_ATTEMPTS = 6
_RETRY_CAP_SECONDS = 45.0

Label = Literal["attack", "benign"]
Language = Literal["en", "es"]

# -------------------------------------------------------------------------------------------------
# Licence policy
# -------------------------------------------------------------------------------------------------
#: The only licences this repository will ingest, as they appear in a Hub ``license:`` tag.
ALLOWED_LICENSES: frozenset[str] = frozenset({"apache-2.0", "mit", "cc-by-4.0"})

#: Datasets refused outright, with the reason a reader can check. Nothing here is fetched.
EXCLUDED_DATASETS: Mapping[str, str] = {
    "rogue-security/prompt-injections-benchmark": (
        "no licence tag on the Hub card (unlicensed), so it cannot be redistributed or cited"
    ),
    "xTRam1/safe-guard-prompt-injection": (
        "gated on the Hub (access requires accepting terms / an authenticated account)"
    ),
}

#: Datasets refused by prefix (families we will not ingest under any circumstance).
EXCLUDED_PREFIXES: Mapping[str, str] = {
    "ai4privacy/": "licence is not on the allow-list (gated / licence:other PII corpora)",
    "meta-llama/Llama-Prompt-Guard-2-": (
        "gated=manual under the Llama 4 Community Licence (license: other); would need a licence "
        "decision this repository cannot make"
    ),
}


class ExternalDataError(RuntimeError):
    """A network, cache or payload problem while materialising an external dataset."""


class LicenseRefused(ExternalDataError):
    """A dataset is excluded, unlicensed, or licensed under terms that are not allowed."""


def license_refusal(dataset: str, license_id: str | None) -> str | None:
    """The reason to refuse ``dataset``, or ``None`` when it is licensed and allowed.

    The check is deliberately ordered: an excluded name is refused regardless of its licence, then
    an absent licence, then anything not on :data:`ALLOWED_LICENSES`.
    """

    if dataset in EXCLUDED_DATASETS:
        return EXCLUDED_DATASETS[dataset]
    for prefix, reason in EXCLUDED_PREFIXES.items():
        if dataset.startswith(prefix):
            return reason
    if not license_id:
        return "no licence tag on the Hub card (unlicensed)"
    if license_id not in ALLOWED_LICENSES:
        allowed = ", ".join(sorted(ALLOWED_LICENSES))
        return f"licence {license_id!r} is not on the allow-list ({allowed})"
    return None


def enforce_license(dataset: str, license_id: str | None) -> None:
    """Raise :class:`LicenseRefused` unless ``dataset`` is licensed under allowed terms."""

    reason = license_refusal(dataset, license_id)
    if reason is not None:
        raise LicenseRefused(f"refusing {dataset!r}: {reason}")


# -------------------------------------------------------------------------------------------------
# The declared source table
# -------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Stream:
    """How one label is drawn from one ``(config, split)`` of a dataset.

    ``*_path`` fields are dotted paths into a row, so a nested record (``injection.injection_text``)
    is reachable without a bespoke loader. A ``label_path``/``label_value`` pair selects the rows
    that carry this stream's label; a stream with neither draws every row in the split.
    """

    label: Label
    config: str
    split: str
    text_path: tuple[str, ...]
    label_path: tuple[str, ...] | None = None
    label_value: str | int | None = None
    provenance: str | None = None
    provenance_path: tuple[str, ...] | None = None
    provenance_map: tuple[tuple[str, str], ...] = ()
    provenance_fallback: str = "retrieved_chunk"
    family_path: tuple[str, ...] | None = None
    family_fallback: str | None = None
    language_path: tuple[str, ...] | None = None
    sample_path: tuple[str, ...] | None = None
    per_level: bool = False


@dataclass(frozen=True)
class DatasetSpec:
    """A declared source: identity, licence, the reason for each provenance and label choice."""

    dataset: str
    license_id: str
    license_tag: str
    source_url: str
    provenance: str
    provenance_rationale: str
    label_basis: str
    sampling_note: str
    streams: tuple[Stream, ...]


# 3nesdeniz: the row states which boundary it exercises; that is the provenance.
_SC_PROVENANCE: tuple[tuple[str, str], ...] = (
    ("direct_user", "user"),
    ("chat_message", "user"),
    ("agent_message", "tool_result"),
    ("mcp_tool_response", "tool_result"),
    ("agent_tool_request", "tool_call"),
    ("conversation_memory", "retrieved_chunk"),
    ("retrieved_document", "retrieved_chunk"),
    ("support_ticket", "retrieved_chunk"),
    ("spreadsheet_cell", "retrieved_chunk"),
    ("web_page", "retrieved_chunk"),
    ("document", "retrieved_chunk"),
    ("pdf_document", "retrieved_chunk"),
)

# nvidia: the injection vector names where the text is planted in the environment.
_VEC_PROVENANCE: tuple[tuple[str, str], ...] = (
    ("message_body", "tool_result"),
    ("chart_notes", "retrieved_chunk"),
    ("appointment_reason", "retrieved_chunk"),
    ("lab_comments", "retrieved_chunk"),
    ("product_description", "retrieved_chunk"),
    ("order_notes", "retrieved_chunk"),
    ("resolution_notes", "retrieved_chunk"),
    ("ticket_description", "retrieved_chunk"),
    ("case_summary", "retrieved_chunk"),
    ("case_notes", "retrieved_chunk"),
    ("court_notes", "retrieved_chunk"),
    ("court_location", "retrieved_chunk"),
)

DEFAULT_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        dataset="rgeada/tool-response-injections",
        license_id="apache-2.0",
        license_tag="license:apache-2.0",
        source_url="https://huggingface.co/datasets/rgeada/tool-response-injections",
        provenance="tool_result",
        provenance_rationale=(
            "Every row is a serialized tool/function response payload, so the segment's boundary "
            "is a tool result -- the closest public analogue of our tool_call_injection family."
        ),
        label_basis=(
            "The publisher's `label` column: 'injection' vs 'safe'. The injection text is a "
            "synthetic harmful instruction planted inside the response body."
        ),
        sampling_note="first N accepted rows in train order, per label.",
        streams=(
            Stream(
                label="attack",
                config="default",
                split="train",
                text_path=("text",),
                label_path=("label",),
                label_value="injection",
                provenance="tool_result",
                family_fallback="tool_response_injection",
            ),
            Stream(
                label="benign",
                config="default",
                split="train",
                text_path=("text",),
                label_path=("label",),
                label_value="safe",
                provenance="tool_result",
                family_fallback="tool_response_injection",
            ),
        ),
    ),
    DatasetSpec(
        dataset="JailbreakBench/JBB-Behaviors",
        license_id="mit",
        license_tag="license:mit",
        source_url="https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors",
        provenance="user",
        provenance_rationale=(
            "JBB behaviours are single-turn user goals -- the user segment of a chat, not "
            "retrieved or tool content."
        ),
        label_basis=(
            "The `harmful` split is the attack set; the `benign` split is the authors' "
            "benign *behaviour* matched to each goal -- benign behaviours, not benign developer "
            "traffic."
        ),
        sampling_note="first N rows in dataset order, per split.",
        streams=(
            Stream(
                label="attack",
                config="behaviors",
                split="harmful",
                text_path=("Goal",),
                provenance="user",
                family_path=("Category",),
                family_fallback="jailbreak_benchmark",
            ),
            Stream(
                label="benign",
                config="behaviors",
                split="benign",
                text_path=("Goal",),
                provenance="user",
                family_path=("Category",),
                family_fallback="jailbreak_benchmark",
            ),
        ),
    ),
    DatasetSpec(
        dataset="TrustAIRLab/in-the-wild-jailbreak-prompts",
        license_id="mit",
        license_tag="license:mit",
        source_url="https://huggingface.co/datasets/TrustAIRLab/in-the-wild-jailbreak-prompts",
        provenance="user",
        provenance_rationale=(
            "Real platform traffic (Discord, Reddit, ...) rather than synthetic; collected prompts "
            "are user messages."
        ),
        label_basis=(
            "The `jailbreak_*` configs are prompts the authors labelled as jailbreaks; the "
            "`regular_*` configs are the non-jailbreak traffic from the same platforms."
        ),
        sampling_note="first N rows in dataset order (2023-12-25 releases).",
        streams=(
            Stream(
                label="attack",
                config="jailbreak_2023_12_25",
                split="train",
                text_path=("prompt",),
                provenance="user",
                family_fallback="in_the_wild_jailbreak",
            ),
            Stream(
                label="benign",
                config="regular_2023_12_25",
                split="train",
                text_path=("prompt",),
                provenance="user",
                family_fallback="in_the_wild_regular",
            ),
        ),
    ),
    DatasetSpec(
        dataset="3nesdeniz/agentic-prompt-injection-boundary-pairs",
        license_id="cc-by-4.0",
        license_tag="license:cc-by-4.0",
        source_url="https://huggingface.co/datasets/3nesdeniz/agentic-prompt-injection-boundary-pairs",
        provenance="retrieved_chunk",
        provenance_rationale=(
            "Provenance comes from each row's own `source_context` (agent message, tool request, "
            "memory, retrieved document, ...), falling back to retrieved_chunk when the value is "
            "unknown."
        ),
        label_basis=(
            "The authors' `label` column: 1 = the boundary should be blocked or reviewed, 0 = the "
            "matched benign control. Shipped as pairs, so it is an overdefense test."
        ),
        sampling_note="first N rows in dataset order over the test split (matched pairs).",
        streams=(
            Stream(
                label="attack",
                config="default",
                split="test",
                text_path=("text",),
                label_path=("label",),
                label_value=1,
                provenance_path=("source_context",),
                provenance_map=_SC_PROVENANCE,
                provenance_fallback="retrieved_chunk",
                family_path=("attack_family",),
                family_fallback="agentic_boundary",
                language_path=("language",),
            ),
            Stream(
                label="benign",
                config="default",
                split="test",
                text_path=("text",),
                label_path=("label",),
                label_value=0,
                provenance_path=("source_context",),
                provenance_map=_SC_PROVENANCE,
                provenance_fallback="retrieved_chunk",
                family_path=("pair_family",),
                family_fallback="agentic_boundary",
                language_path=("language",),
            ),
        ),
    ),
    DatasetSpec(
        dataset="Lakera/mosscap_prompt_injection",
        license_id="mit",
        license_tag="license:mit",
        source_url="https://huggingface.co/datasets/Lakera/mosscap_prompt_injection",
        provenance="user",
        provenance_rationale="MSCAP rows are user-turn jailbreak attempts.",
        label_basis=(
            "No benign split ships: every row is a jailbreak attempt and `level` is an "
            "attack-progression level (1-8), not an attack/benign label. Contributes attack cases "
            "only."
        ),
        sampling_note="first N per `level`, round-robin over levels (deterministic).",
        streams=(
            Stream(
                label="attack",
                config="default",
                split="train",
                text_path=("prompt",),
                provenance="user",
                family_path=("level",),
                family_fallback="jailbreak_progression",
                sample_path=("level",),
                per_level=True,
            ),
        ),
    ),
    DatasetSpec(
        dataset="nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1",
        license_id="cc-by-4.0",
        license_tag="license:cc-by-4.0",
        source_url=(
            "https://huggingface.co/datasets/"
            "nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1"
        ),
        provenance="retrieved_chunk",
        provenance_rationale=(
            "Each row's injection text is authored to be planted in a downstream record; "
            "provenance comes from `injection.vector` (chart notes, order notes, message body)."
        ),
        label_basis=(
            "RL environment: every row carries an indirect injection "
            "(`injection.injection_text`); there is no benign split, so it contributes attack "
            "cases only."
        ),
        sampling_note="first N rows in dataset order (train split).",
        streams=(
            Stream(
                label="attack",
                config="default",
                split="train",
                text_path=("injection", "injection_text"),
                provenance_path=("injection", "vector"),
                provenance_map=_VEC_PROVENANCE,
                provenance_fallback="retrieved_chunk",
                family_path=("attack_category",),
                family_fallback="nvidia_agentic_ipi",
            ),
        ),
    ),
    DatasetSpec(
        dataset="CaiZhiTech/Evaluation-Dataset-of-AI-Agent-Security-Guardrails",
        license_id="apache-2.0",
        license_tag="license:apache-2.0",
        source_url=(
            "https://huggingface.co/datasets/"
            "CaiZhiTech/Evaluation-Dataset-of-AI-Agent-Security-Guardrails"
        ),
        provenance="user",
        provenance_rationale=(
            "Rows are single-turn user instructions wrapped in an `### Instruction` / "
            "`### Response` frame, so the boundary is the user message."
        ),
        label_basis=(
            "The `action` column is the authors' guardrail verdict: 'blocked' -> attack, "
            "'allowed' -> benign. This allow/deny mapping is unambiguous in the data."
        ),
        sampling_note="first N rows in dataset order, per action value.",
        streams=(
            Stream(
                label="attack",
                config="default",
                split="train",
                text_path=("text",),
                label_path=("action",),
                label_value="blocked",
                provenance="user",
                family_fallback="guardrail_blocked",
            ),
            Stream(
                label="benign",
                config="default",
                split="train",
                text_path=("text",),
                label_path=("action",),
                label_value="allowed",
                provenance="user",
                family_fallback="guardrail_allowed",
            ),
        ),
    ),
)


# -------------------------------------------------------------------------------------------------
# Fetching + caching (stdlib only)
# -------------------------------------------------------------------------------------------------
def _info_url(dataset: str) -> str:
    return f"{DATASETS_SERVER}/info?dataset={urllib.parse.quote(dataset, safe='')}"


def _rows_url(dataset: str, config: str, split: str, offset: int, length: int) -> str:
    query = urllib.parse.urlencode(
        {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length}
    )
    return f"{DATASETS_SERVER}/rows?{query}"


def _cache_name(url: str) -> str:
    """A readable, collision-free cache filename for a request URL."""

    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    dataset = query.get("dataset", ["_info"])[0].replace("/", "__")
    config = query.get("config", ["-"])[0]
    split = query.get("split", ["-"])[0]
    offset = query.get("offset", ["-"])[0]
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    raw = f"{dataset}__{config}__{split}__{offset}__{digest}.json"
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in raw)


def _http_json(url: str, timeout: float, *, max_attempts: int = _MAX_ATTEMPTS) -> dict[str, Any]:
    """GET a URL as JSON, retrying the statuses the public datasets-server rate-limits on."""

    last_error: str | None = None
    for attempt in range(max_attempts):
        request = urllib.request.Request(url, headers={"user-agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            if exc.code in _RETRY_STATUSES and attempt < max_attempts - 1:
                last_error = f"HTTP {exc.code}"
                time.sleep(_retry_delay(attempt, exc))
                continue
            raise ExternalDataError(f"HTTP {exc.code} for {url}") from exc
        except urllib.error.URLError as exc:
            if attempt < max_attempts - 1:
                last_error = f"network error: {exc.reason}"
                time.sleep(_retry_delay(attempt, None))
                continue
            raise ExternalDataError(f"network error for {url}: {exc.reason}") from exc
        data: dict[str, Any] = json.loads(payload)
        return data
    raise ExternalDataError(f"giving up on {url} after {max_attempts} attempts ({last_error})")


def _retry_delay(attempt: int, error: urllib.error.HTTPError | None) -> float:
    """Exponential backoff (3, 6, 12, ...), honouring ``Retry-After``; capped at 45s."""

    delay = float(3 * 2**attempt)
    if error is not None:
        retry_after = error.headers.get("retry-after") if error.headers else None
        if retry_after and retry_after.strip().isdigit():
            delay = max(delay, float(retry_after.strip()))
    return min(delay, _RETRY_CAP_SECONDS)


class PageFetcher:
    """Fetch datasets-server JSON, caching each response verbatim under ``cache_dir``.

    With ``offline=True`` a cache miss raises instead of touching the network, which is how the
    tests prove they run without it.
    """

    def __init__(self, cache_dir: Path, *, timeout: float = 60.0, offline: bool = False) -> None:
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.offline = offline

    def __call__(self, url: str) -> dict[str, Any]:
        path = self.cache_dir / _cache_name(url)
        if path.is_file():
            cached: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            return cached
        if self.offline:
            raise ExternalDataError(f"offline: no cached page for {url} (expected {path})")
        payload = _http_json(url, self.timeout)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return payload


# -------------------------------------------------------------------------------------------------
# Loading a spec into Cases
# -------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Loaded:
    """A dataset that loaded, with its cases and the sampling actually applied."""

    dataset: str
    license_id: str
    license_tag: str
    source_url: str
    provenance: str
    provenance_rationale: str
    label_basis: str
    configs: tuple[str, ...]
    splits: tuple[str, ...]
    text_columns: tuple[str, ...]
    sampling: str
    cases: tuple[Case, ...]


@dataclass(frozen=True)
class Skipped:
    """A dataset that could not be loaded, and why -- never an exception the caller must catch."""

    dataset: str
    reason: str


@dataclass(frozen=True)
class ManifestEntry:
    """One row of the licence manifest emitted with a report."""

    dataset: str
    license_id: str
    license_tag: str
    source_url: str
    configs: tuple[str, ...]
    splits: tuple[str, ...]
    text_columns: tuple[str, ...]
    provenance: str
    provenance_rationale: str
    label_basis: str
    sampling: str
    cases: int
    attacks: int
    benign: int


def _dig(row: Mapping[str, Any], path: Sequence[str]) -> Any:
    """Walk a dotted path into a row, returning ``None`` when any link is missing."""

    value: Any = row
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _features_by_config(info: Mapping[str, Any]) -> dict[str, set[str]]:
    dataset_info = info.get("dataset_info")
    if not isinstance(dataset_info, Mapping):
        return {}
    out: dict[str, set[str]] = {}
    for config, cinfo in dataset_info.items():
        if isinstance(cinfo, Mapping):
            features = cinfo.get("features")
            if isinstance(features, Mapping):
                out[str(config)] = {str(name) for name in features}
    return out


def _num_examples(info: Mapping[str, Any], config: str, split: str) -> int | None:
    dataset_info = info.get("dataset_info")
    if not isinstance(dataset_info, Mapping):
        return None
    cinfo = dataset_info.get(config)
    if not isinstance(cinfo, Mapping):
        return None
    splits = cinfo.get("splits")
    if not isinstance(splits, Mapping):
        return None
    sinfo = splits.get(split)
    if not isinstance(sinfo, Mapping):
        return None
    num = sinfo.get("num_examples")
    return int(num) if isinstance(num, int) else None


def _slug(dataset: str) -> str:
    return dataset.replace("/", "-").replace("_", "-")


def _matches(stream: Stream, row: Mapping[str, Any]) -> bool:
    if stream.label_path is None:
        return True
    return bool(_dig(row, stream.label_path) == stream.label_value)


def _provenance(stream: Stream, row: Mapping[str, Any]) -> str:
    if stream.provenance is not None:
        candidate = stream.provenance
    elif stream.provenance_path is not None:
        value = _dig(row, stream.provenance_path)
        looked_up = dict(stream.provenance_map).get(str(value))
        candidate = looked_up if looked_up is not None else stream.provenance_fallback
    else:
        candidate = stream.provenance_fallback
    return candidate if candidate in PROVENANCE_VALUES else "retrieved_chunk"


def _family(spec: DatasetSpec, stream: Stream, row: Mapping[str, Any]) -> str:
    if stream.family_path is not None:
        value = _dig(row, stream.family_path)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if stream.family_fallback:
        return stream.family_fallback
    return _slug(spec.dataset)


def _language(stream: Stream, row: Mapping[str, Any]) -> Language:
    if stream.language_path is not None and _dig(row, stream.language_path) == "es":
        return "es"
    return "en"


@dataclass(frozen=True)
class _Row:
    index: int
    data: Mapping[str, Any]


def _fetch_rows(
    spec: DatasetSpec,
    stream: Stream,
    *,
    fetcher: Callable[[str], dict[str, Any]],
    page_size: int,
    target: int,
    num_examples: int | None,
) -> list[_Row]:
    """Page through a split until ``target`` accepted rows are collected or the split ends."""

    accepted: list[_Row] = []
    offset = 0
    while len(accepted) < target:
        if num_examples is not None and offset >= num_examples:
            break
        payload = fetcher(_rows_url(spec.dataset, stream.config, stream.split, offset, page_size))
        rows = payload.get("rows")
        if not isinstance(rows, list) or not rows:
            break
        for envelope in rows:
            if not isinstance(envelope, Mapping):
                continue
            row = envelope.get("row")
            if not isinstance(row, Mapping):
                continue
            if _matches(stream, row):
                index = envelope.get("row_idx")
                accepted.append(
                    _Row(index=int(index) if isinstance(index, int) else offset, data=row)
                )
        offset += len(rows)
        if len(rows) < page_size:
            break
    return accepted


def _select(stream: Stream, accepted: list[_Row], quota: int) -> list[_Row]:
    """Deterministic, bounded sampling. First-N, or first-N-per-``sample_path`` round-robin."""

    if not stream.per_level or stream.sample_path is None or len(accepted) <= quota:
        return accepted[:quota]

    groups: dict[str, list[_Row]] = {}
    for row in accepted:
        key = str(_dig(row.data, stream.sample_path))
        groups.setdefault(key, []).append(row)
    levels = sorted(groups)
    if not levels:
        return []
    base, extra = divmod(quota, len(levels))
    selected: list[_Row] = []
    for position, level in enumerate(levels):
        take = base + (1 if position < extra else 0)
        selected.extend(groups[level][:take])
    if len(selected) < quota:  # a level was short; fill from anything not already chosen
        chosen = {id(row) for row in selected}
        for row in accepted:
            if len(selected) >= quota:
                break
            if id(row) not in chosen:
                selected.append(row)
    return selected[:quota]


def _case(spec: DatasetSpec, stream: Stream, row: _Row) -> Case | None:
    text = _dig(row.data, stream.text_path)
    if not isinstance(text, str) or not text.strip():
        return None
    slug = _slug(spec.dataset)
    provenance = _provenance(stream, row.data)
    family = _family(spec, stream, row.data)
    language = _language(stream, row.data)
    why = (
        f"{spec.dataset} {stream.config}/{stream.split} row {row.index}: authors' label "
        f"{stream.label!r} ({spec.label_basis}); provenance {provenance!r} "
        f"({spec.provenance_rationale})"
    )
    return Case(
        id=f"ext:{slug}:{stream.config}:{stream.split}:{row.index}",
        text=text,
        provenance=provenance,
        label=stream.label,
        family=family,
        language=language,
        why=why,
        tags=(slug, "external", f"{stream.config}/{stream.split}"),
    )


def load_dataset(
    spec: DatasetSpec,
    *,
    fetcher: Callable[[str], dict[str, Any]],
    limit: int = DEFAULT_LIMIT,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> Loaded | Skipped:
    """Load one declared source into cases, or return a :class:`Skipped` with the reason.

    A missing config or a missing text column is a skip, never a raise -- the report must be able
    to say "this dataset was left out because its schema differs" rather than crash mid-run.
    """

    enforce_license(spec.dataset, spec.license_id)
    info = fetcher(_info_url(spec.dataset))
    features_by_config = _features_by_config(info)

    for stream in spec.streams:
        features = features_by_config.get(stream.config)
        if features is None:
            return Skipped(spec.dataset, f"config {stream.config!r} is absent from the Hub /info")
        head = stream.text_path[0]
        if head not in features:
            dotted = ".".join(stream.text_path)
            return Skipped(
                spec.dataset,
                f"text column {dotted!r} is absent from config {stream.config!r} "
                f"(features: {', '.join(sorted(features))})",
            )

    labels = sorted({stream.label for stream in spec.streams})
    base, extra = divmod(limit, len(labels))
    quota = {label: base + (1 if position < extra else 0) for position, label in enumerate(labels)}

    cases: list[Case] = []
    counts: dict[str, int] = {}
    for stream in spec.streams:
        wanted = quota[stream.label]
        num_examples = _num_examples(info, stream.config, stream.split)
        target = wanted
        if stream.per_level:
            target = max(wanted * 4, 40)
            if num_examples is not None:
                target = min(target, num_examples)
        accepted = _fetch_rows(
            spec,
            stream,
            fetcher=fetcher,
            page_size=page_size,
            target=target,
            num_examples=num_examples,
        )
        selected = _select(stream, accepted, wanted)
        built: list[Case] = []
        for row in selected:
            case = _case(spec, stream, row)
            if case is not None:
                built.append(case)
        cases.extend(built)
        counts[stream.label] = counts.get(stream.label, 0) + len(built)

    if not cases:
        return Skipped(spec.dataset, "no rows had a usable text value after sampling")

    sampling = "; ".join(
        f"{label}: {count} cases ({spec.sampling_note})" for label, count in sorted(counts.items())
    )
    return Loaded(
        dataset=spec.dataset,
        license_id=spec.license_id,
        license_tag=spec.license_tag,
        source_url=spec.source_url,
        provenance=spec.provenance,
        provenance_rationale=spec.provenance_rationale,
        label_basis=spec.label_basis,
        configs=tuple(sorted({stream.config for stream in spec.streams})),
        splits=tuple(sorted({stream.split for stream in spec.streams})),
        text_columns=tuple(sorted({".".join(stream.text_path) for stream in spec.streams})),
        sampling=sampling,
        cases=tuple(cases),
    )


def load_all(
    specs: Sequence[DatasetSpec] = DEFAULT_SPECS,
    *,
    fetcher: Callable[[str], dict[str, Any]],
    limit: int = DEFAULT_LIMIT,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> tuple[list[Loaded], list[Skipped]]:
    """Load every declared source, partitioning into the loaded and the skipped."""

    loaded: list[Loaded] = []
    skipped: list[Skipped] = []
    for spec in specs:
        try:
            result = load_dataset(spec, fetcher=fetcher, limit=limit, page_size=page_size)
        except ExternalDataError as exc:
            skipped.append(Skipped(spec.dataset, f"{type(exc).__name__}: {exc}"))
            continue
        if isinstance(result, Skipped):
            skipped.append(result)
        else:
            loaded.append(result)
    return loaded, skipped


def to_cases(loaded: Sequence[Loaded]) -> list[Case]:
    """Flatten every loaded dataset into one case list."""

    return [case for entry in loaded for case in entry.cases]


def license_manifest(loaded: Sequence[Loaded]) -> tuple[ManifestEntry, ...]:
    """The ingest table: what was read, under which licence, with which sampling."""

    entries: list[ManifestEntry] = []
    for entry in loaded:
        entries.append(
            ManifestEntry(
                dataset=entry.dataset,
                license_id=entry.license_id,
                license_tag=entry.license_tag,
                source_url=entry.source_url,
                configs=entry.configs,
                splits=entry.splits,
                text_columns=entry.text_columns,
                provenance=entry.provenance,
                provenance_rationale=entry.provenance_rationale,
                label_basis=entry.label_basis,
                sampling=entry.sampling,
                cases=len(entry.cases),
                attacks=sum(1 for case in entry.cases if case.label == "attack"),
                benign=sum(1 for case in entry.cases if case.label == "benign"),
            )
        )
    return tuple(entries)


# -------------------------------------------------------------------------------------------------
# The frozen pipeline
# -------------------------------------------------------------------------------------------------
def build_frozen_callable() -> ScanCallable:
    """The real service path over default settings: segment -> scan -> aggregate -> verdict.

    It is driven through :class:`~llm_guardian.service.ScanService` so segmentation, chunking and
    truncation behave exactly as in production. Nothing here reads or writes a detector.
    """

    from ..deps import create_service
    from ..models import ScanRequest, SegmentInput
    from ..provenance import Provenance
    from ..service import ScanService
    from ..settings import GuardianSettings

    settings = GuardianSettings()
    container = create_service(settings)
    service = ScanService(
        settings,
        container.clock,
        container.pipeline,
        container.cache,
        container.memory,
        container.incidents,
        container.metrics,
    )

    def _scan(case: Case) -> ScanObservation:
        request = ScanRequest(
            request_id=case.id,
            conversation_id=f"ext-{case.id}",
            segments=(SegmentInput(text=case.text, provenance=Provenance(case.provenance)),),
        )
        return _from_response(service.scan(request))

    return _scan


def _from_response(response: Any) -> ScanObservation:
    from .harness import from_scan_response

    return from_scan_response(response)


# -------------------------------------------------------------------------------------------------
# Measurement + rendering
# -------------------------------------------------------------------------------------------------
def _report_row(report: Report) -> str:
    return (
        f"| {report.corpus_size} | {report.attack_count} | {report.benign_count} | "
        f"{report.recall:.3f} ({report.true_positives}/{report.attack_count}) | "
        f"{report.false_blocks_per_1000:.1f} ({report.false_blocks}/{report.benign_count}) |"
    )


def render_markdown(
    results: Sequence[tuple[Loaded, Report]],
    skipped: Sequence[Skipped],
    pooled: Report,
    *,
    limit: int,
    repeat: int,
) -> str:
    """A markdown fragment: manifest, per-dataset and pooled numbers, and every failing case."""

    lines: list[str] = []
    add = lines.append
    add("## Licence manifest")
    add("")
    add("| dataset | licence (Hub tag) | config / split | cases | provenance |")
    add("| --- | --- | --- | --- | --- |")
    for loaded, report in results:
        tags = sorted({case.tags[2] for case in loaded.cases})
        add(
            f"| `{loaded.dataset}` | `{loaded.license_tag}` (id `{loaded.license_id}`) | "
            f"{', '.join(tags)} | {report.corpus_size} | `{loaded.provenance}` |"
        )
    for skip in skipped:
        add(f"| `{skip.dataset}` | -- | -- | -- | skipped: {skip.reason} |")
    add("")
    add(f"`--limit {limit}` per dataset, `repeat={repeat}`. Sampling per dataset is stated below.")
    add("")

    add("## Numbers")
    add("")
    add("| dataset | cases | attack | benign | recall | false blocks /1,000 benign |")
    add("| --- | --- | --- | --- | --- | --- |")
    for loaded, report in results:
        add(f"| `{loaded.dataset}` {_report_row(report)}")
    add(f"| **pooled** {_report_row(pooled)}")
    add("")

    add("### Sampling and label basis, per dataset")
    add("")
    for loaded, _report in results:
        add(f"- **`{loaded.dataset}`** — {loaded.sampling}")
        add(f"  - label basis: {loaded.label_basis}")
    add("")

    add("## Failing cases")
    add("")
    for loaded, report in results:
        add(f"### `{loaded.dataset}`")
        add("")
        add(
            f"false negatives (attacks not blocked): **{len(report.false_negative_cases)}**"
            f" of {report.attack_count}; false positives (benign blocked): "
            f"**{len(report.false_positive_cases)}** of {report.benign_count}."
        )
        add("")
        if not report.false_negative_cases:
            add("- no false negatives")
        for case in report.false_negative_cases:
            add(f"- `{case.id}` — {case.excerpt}")
        if report.false_positive_cases:
            add("")
            add("false positives:")
            for case in report.false_positive_cases:
                add(f"- `{case.id}` — {case.excerpt}")
        add("")
    return "\n".join(lines)


def run_cli(argv: Sequence[str]) -> int:
    """Load the declared sources, measure the frozen pipeline, print the report."""

    parser = argparse.ArgumentParser(prog="llm-guardian bench.external")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, dest="cache_dir")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE, dest="page_size")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--offline", action="store_true", help="never touch the network")
    parser.add_argument("--json", action="store_true", help="emit the manifest as JSON only")
    args = parser.parse_args(list(argv))

    fetcher = PageFetcher(args.cache_dir, offline=args.offline)
    loaded, skipped = load_all(fetcher=fetcher, limit=args.limit, page_size=args.page_size)
    if not loaded:
        print("bench.external: nothing loaded", file=sys.stderr)
        for skip in skipped:
            print(f"  skipped {skip.dataset}: {skip.reason}", file=sys.stderr)
        return 2

    for skip in skipped:
        print(f"bench.external: skipped {skip.dataset}: {skip.reason}", file=sys.stderr)

    scan = build_frozen_callable()
    results: list[tuple[Loaded, Report]] = []
    for entry in loaded:
        results.append((entry, run(list(entry.cases), scan, repeat=args.repeat, warmup=0)))
    pooled = run(to_cases(loaded), scan, repeat=args.repeat, warmup=0)

    if args.json:
        payload = {
            "manifest": [entry.__dict__ for entry in license_manifest(loaded)],
            "skipped": [skip.__dict__ for skip in skipped],
            "pooled": json.loads(pooled.to_json()),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    print(render_markdown(results, skipped, pooled, limit=args.limit, repeat=args.repeat))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_cli(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess in the run
    raise SystemExit(main())
