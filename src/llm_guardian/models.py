"""The wire and storage contracts. Frozen: every module codes against these names.

Two shapes are deliberately absent. There is no "safety score" for a request, because a single
number that mixes an instruction-override signal with an egress request cannot be compared to a
threshold meaningfully; the score is per OWASP category and the verdict is a state. There is also
no field for a client-supplied role: provenance is server-assigned (`provenance.py`).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .provenance import Provenance, TrustLevel


class OwaspCategory(StrEnum):
    """OWASP LLM Top 10 (2026) categories, as used by the coverage report."""

    LLM01 = "LLM01"  # Prompt injection
    LLM02 = "LLM02"  # Sensitive information disclosure
    LLM03 = "LLM03"  # Excessive agency
    LLM04 = "LLM04"  # Supply chain
    LLM05 = "LLM05"  # Data and model poisoning
    LLM06 = "LLM06"  # Unbounded consumption
    LLM07 = "LLM07"  # Misinformation
    LLM08 = "LLM08"  # Hidden context exposure
    LLM09 = "LLM09"  # Vector and embedding weaknesses
    LLM10 = "LLM10"  # Improper output handling


class GuardMode(StrEnum):
    """``shadow`` observes and reports; ``enforce`` is allowed to block."""

    SHADOW = "shadow"
    ENFORCE = "enforce"


class VerdictDecision(StrEnum):
    ALLOW = "allow"
    ESCALATE = "escalate"
    BLOCK = "block"


class Severity(StrEnum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ScannerErrorPolicy(StrEnum):
    """What a scanner raising an exception means, per scanner and per provenance."""

    WARN = "warn"
    BLOCK = "block"


class IncidentStatus(StrEnum):
    """Only ``confirmed_attack`` and ``benign`` are trusted labels."""

    PENDING = "pending"
    CONFIRMED_ATTACK = "confirmed_attack"
    BENIGN = "benign"
    DISMISSED = "dismissed"


class Finding(BaseModel):
    """One signal from one scanner. Never carries the matched text itself."""

    model_config = ConfigDict(frozen=True)

    scanner: str
    scanner_version: str
    kind: str = Field(description="Stable machine name, e.g. 'instruction_override'.")
    severity: Severity
    message: str = Field(description="Human sentence, safe to log; no secret material.")
    owasp: tuple[OwaspCategory, ...] = ()
    span: tuple[int, int] | None = Field(
        default=None, description="Offsets within the segment, for redaction in evidence."
    )
    attributes: dict[str, str] = Field(
        default_factory=dict, description="Scanner-specific, redacted metadata."
    )


class SegmentInput(BaseModel):
    """A segment as submitted by the gateway, which states which parts are new."""

    text: str
    provenance: Provenance
    char_start: int | None = Field(default=None, description="Offset in the parent message.")
    char_end: int | None = None
    truncated: bool = Field(
        default=False,
        description="The gateway clipped this segment. A truncated scan is never evidence of "
        "safety and is reported as its own warning.",
    )


class Segment(BaseModel):
    """A segment after digesting: what the pipeline actually walks."""

    model_config = ConfigDict(frozen=True)

    segment_id: str = Field(description="Stable id: conversation, index and content digest.")
    index: int
    digest: str = Field(description="SHA-256 of the exact bytes scanned.")
    text: str
    provenance: Provenance
    trust: TrustLevel
    char_start: int | None = None
    char_end: int | None = None
    truncated: bool = False
    source_index: int = Field(
        default=0, description="Which submitted input this chunk came from (seam detection)."
    )

    @property
    def chars(self) -> int:
        return len(self.text)


class ScannerOutcome(BaseModel):
    """What one scanner did on one segment, including the fact that it failed."""

    model_config = ConfigDict(frozen=True)

    scanner: str
    scanner_version: str
    findings: tuple[Finding, ...] = ()
    error: str | None = None
    duration_ms: float = 0.0
    cached: bool = Field(
        default=False, description="Findings came from the feature cache, not from a scan."
    )


class Signals(BaseModel):
    """Aggregated per-category signals. A category with no evidence stays at zero, not at 'safe'."""

    model_config = ConfigDict(frozen=True)

    score: float = Field(
        default=0.0,
        description="Aggregate of this segment's findings, in the configured scale.",
    )
    category_scores: dict[OwaspCategory, float] = Field(default_factory=dict)
    top_findings: tuple[Finding, ...] = ()
    matched_rules: tuple[str, ...] = ()
    scanner_errors: tuple[str, ...] = ()


class MemoryNeighbor(BaseModel):
    """A reviewed precedent, returned as evidence rather than as a verdict."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    label: IncidentStatus
    similarity: float
    owasp: tuple[OwaspCategory, ...] = ()
    family: str | None = None
    digest: str


class MemoryEvidence(BaseModel):
    """The risk-memory lookup, or the reason it did not happen.

    ``margin`` is a cosine difference, never a probability of malicious intent, and the verdict
    never allows content because of it.
    """

    model_config = ConfigDict(frozen=True)

    queried: bool
    reason_not_queried: str | None = None
    attack_sim: float | None = None
    benign_sim: float | None = None
    margin: float | None = None
    neighbors: tuple[MemoryNeighbor, ...] = ()
    query_ms: float = 0.0
    #: Reviewed precedents dropped because their retention deadline had passed. A stale benign
    #: precedent is a false-negative risk, so the drop is counted and surfaced, never silent.
    expired_examples: int = 0


class EvidenceRef(BaseModel):
    """A finding echoed from a prior ``/v1/scan``, so the policy layer can see provenance.

    A gateway that proposes an action reports *where the proposal came from*: the segment id, the
    finding kind that was observed, its severity and the server-assigned provenance of the segment.
    The policy layer never re-reads the text and never trusts a client-supplied role -- it only
    compares these declared coordinates against its own rules.
    """

    model_config = ConfigDict(frozen=True)

    segment_id: str
    finding_kind: str = Field(description="Stable scanner kind, e.g. 'tool_call_instruction'.")
    severity: Severity
    provenance: Provenance


class PolicyCheck(BaseModel):
    """A hard policy decision. Computed from declared scopes, never from a model output."""

    model_config = ConfigDict(frozen=True)

    decision: VerdictDecision
    allowed: bool
    reason_codes: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = Field(
        default=(), description="Tool capabilities the principal actually holds."
    )
    egress_hosts: tuple[str, ...] = ()
    budgets: dict[str, float] = Field(default_factory=dict)
    denied_by: str | None = Field(
        default=None,
        description="The first reason that denied the request, or None when it was allowed.",
    )
    argument_failures: tuple[str, ...] = Field(
        default=(),
        description="Stable 'argument_*' codes (with the offending argument) from schema checks.",
    )
    evidence_used: tuple[str, ...] = Field(
        default=(),
        description="Finding kinds from the supplied evidence that drove a denial.",
    )
    checked_at: datetime


class LatencyBreakdown(BaseModel):
    """Per-stage timings. p50/p95/p99 are separate on purpose: they answer different questions."""

    model_config = ConfigDict(frozen=True)

    preprocess_ms: float = 0.0
    deterministic_ms: float = 0.0
    ml_ms: float = 0.0
    retrieval_ms: float = 0.0
    total_ms: float = 0.0


class SegmentVerdict(BaseModel):
    """The full account of what the guardian did with one segment."""

    model_config = ConfigDict(frozen=True)

    segment_id: str
    index: int
    provenance: Provenance
    digest: str
    decision: VerdictDecision
    score: float
    threshold: float
    signals: Signals
    scanners: tuple[ScannerOutcome, ...] = ()
    memory: MemoryEvidence | None = None
    policy: PolicyCheck | None = None
    cached_features: bool = False
    notes: tuple[str, ...] = ()


class ScanRequest(BaseModel):
    request_id: str | None = None
    conversation_id: str | None = None
    tenant: str | None = Field(
        default=None, description="Used for memory isolation; never used for an auth decision."
    )
    segments: tuple[SegmentInput, ...]
    mode: GuardMode | None = Field(
        default=None, description="Per-request override of the configured mode, for shadow trials."
    )


class ScanResponse(BaseModel):
    request_id: str
    conversation_id: str | None
    mode: GuardMode
    shadow: bool = Field(description="True when the verdict is advisory only.")
    decision: VerdictDecision = Field(description="Worst decision across the segments.")
    segments: tuple[SegmentVerdict, ...]
    latency: LatencyBreakdown
    warnings: tuple[str, ...] = ()
    scanner_versions: dict[str, str] = Field(default_factory=dict)
    guardian_version: str
    session_risk: float | None = Field(
        default=None,
        description="Decayed risk accumulated across this conversation's requests, when tracked.",
    )

    @property
    def blocked(self) -> bool:
        return self.decision is VerdictDecision.BLOCK


class PolicyAction(StrEnum):
    TOOL_CALL = "tool_call"
    RETRIEVAL = "retrieval"
    EGRESS = "egress"
    BUDGET = "budget"


class PolicyCheckRequest(BaseModel):
    principal: str
    tenant: str | None = None
    action: PolicyAction
    tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    tool_arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Argument names/values of the proposed tool call, validated against the "
        "tool schema declared in the policy file.",
    )
    evidence: tuple[EvidenceRef, ...] = Field(
        default=(),
        description="Findings from a prior scan that support this proposed action.",
    )
    requested_capabilities: tuple[str, ...] = ()
    egress_hosts: tuple[str, ...] = ()
    requested_budget: dict[str, float] = Field(default_factory=dict)
    target_tenant: str | None = Field(
        default=None, description="For a cross-tenant read or write, which tenant is targeted."
    )


class PolicyCheckResponse(BaseModel):
    check: PolicyCheck
    guardian_version: str


class Authorization(BaseModel):
    """The tool-boundary decision a tool executor receives from ``/v1/agent/authorize``.

    It is the :class:`PolicyCheck` plus the detail the executor cares about at the point of
    execution: the decision, the single code that denied it (if any), the argument-schema failures
    and the finding kinds the evidence rule acted on.
    """

    model_config = ConfigDict(frozen=True)

    check: PolicyCheck
    decision: VerdictDecision
    denied_by: str | None = None
    argument_failures: tuple[str, ...] = ()
    evidence_used: tuple[str, ...] = ()
    guardian_version: str


#: The longest excerpt an incident keeps: one full default-sized segment.
MAX_EXCERPT_CHARS = 4000


class IncidentCreateRequest(BaseModel):
    conversation_id: str | None = None
    tenant: str | None = None
    provenance: Provenance
    content_digest: str
    redacted_excerpt: str = Field(
        max_length=MAX_EXCERPT_CHARS,
        description="The content with secrets masked (redacted by the caller, or by the guardian "
        "for scan-recorded incidents). On review it becomes the precedent's text.",
    )
    findings_summary: str = Field(
        default="",
        max_length=2000,
        description="What the detectors reported, safe to log; never carries secret material.",
    )
    owasp: tuple[OwaspCategory, ...] = ()
    detector_versions: dict[str, str] = Field(default_factory=dict)
    score: float = 0.0
    retention_days: int = Field(default=180, ge=1, le=3650)


class IncidentRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    created_at: datetime
    conversation_id: str | None = None
    tenant: str | None = None
    provenance: Provenance
    content_digest: str
    redacted_excerpt: str
    findings_summary: str = ""
    owasp: tuple[OwaspCategory, ...] = ()
    detector_versions: dict[str, str] = Field(default_factory=dict)
    score: float = 0.0
    #: Sightings of the same (tenant, provenance, content digest). A repeat is folded into the
    #: existing case instead of opening a new one, so a retry loop cannot flood the review queue.
    occurrences: int = Field(default=1, ge=1)
    last_seen_at: datetime | None = None
    status: IncidentStatus = IncidentStatus.PENDING
    family: str | None = None
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    review_note: str | None = None
    retention_deadline: datetime


class ReviewRequest(BaseModel):
    reviewer: str = Field(min_length=1)
    label: IncidentStatus = Field(
        description="confirmed_attack or benign promote the case; dismissed closes it untrusted."
    )
    note: str | None = None
    family: str | None = None
    owasp: tuple[OwaspCategory, ...] = ()


class MemoryQueryRequest(BaseModel):
    text: str
    provenance: Provenance = Provenance.USER
    k: int = Field(default=5, ge=1, le=50)


class MemoryQueryResponse(BaseModel):
    attack_sim: float
    benign_sim: float
    margin: float
    neighbors: tuple[MemoryNeighbor, ...]
    backend: str
    took_ms: float
    note: str = (
        "Cosine similarity is not a probability of malicious intent and never whitelists content."
    )


class MemoryStats(BaseModel):
    backend: str
    reachable: bool
    confirmed_attacks: int
    benign_hard_negatives: int
    pending_incidents: int
    detail: str | None = None


class CacheStats(BaseModel):
    entries: int
    hits: int
    misses: int
    evictions: int
    max_entries: int
    ttl_seconds: float


class RiskMemoryHealth(BaseModel):
    backend: str
    reachable: bool
    detail: str | None = None


class HealthReport(BaseModel):
    status: str
    guardian_version: str
    mode: GuardMode
    shadow: bool
    uptime_seconds: float
    scanner_versions: dict[str, str] = Field(default_factory=dict)
    risk_memory: RiskMemoryHealth
    cache: CacheStats
    reasons: tuple[str, ...] = ()


class StageLatency(BaseModel):
    count: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float


class MetricsReport(BaseModel):
    window_seconds: float
    scans: int
    segments: int
    verdicts: dict[VerdictDecision, int]
    decisions_by_category: dict[OwaspCategory, int] = Field(default_factory=dict)
    stages: dict[str, StageLatency] = Field(default_factory=dict)
    escalations: int = 0
    retrieval_queries: int = 0
    scanner_errors: int = 0
    cache: CacheStats


class OwaspCategoryCoverage(BaseModel):
    category: OwaspCategory
    title: str
    safeguards: tuple[str, ...]
    implemented: bool
    enforced_by: str = Field(description="'guardian' or the component that actually owns it.")
    tests: tuple[str, ...] = ()
    notes: str | None = None


class OwaspCoverageReport(BaseModel):
    generated_at: datetime
    categories: tuple[OwaspCategoryCoverage, ...]
    note: str = (
        "Per-category implementation and test status. There is deliberately no aggregate "
        "'OWASP safety score'."
    )
