"""Configuration. Every safeguard is a declared number, not a literal in a branch.

Environment variables use the ``GUARDIAN_`` prefix and ``__`` for nesting, e.g.
``GUARDIAN_MODE=enforce GUARDIAN_THRESHOLDS__BLOCK_AT=0.9 GUARDIAN_MEMORY__BACKEND=elasticsearch``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .models import GuardMode, ScannerErrorPolicy

#: The experimental SLO from the design: added gateway p95 on short requests. It is a target to
#: measure against and report, never a result this service claims on its own.
DEFAULT_LATENCY_BUDGET_MS = 30.0


class ThresholdSettings(BaseModel):
    """Score bands. ``escalate_at`` is where the risk memory becomes worth consulting."""

    escalate_at: float = Field(default=0.35, ge=0.0, le=1.0)
    block_at: float = Field(default=0.75, ge=0.0, le=1.0)
    #: A deterministic finding at or above this severity blocks even if the summed score is low.
    severity_floor: str = Field(default="critical", description="Severity or higher blocks alone.")
    #: Per-category overrides, by OWASP id: ``{"LLM01": {"block_at": 0.6}}``.
    category_overrides: dict[str, dict[str, float]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _ordered(self) -> ThresholdSettings:
        if self.escalate_at > self.block_at:
            raise ValueError("escalate_at must not exceed block_at")
        return self


class ScannerSettings(BaseModel):
    """Which deterministic checks run, and what a scanner's own failure means."""

    enable_rules: bool = True
    enable_secrets: bool = True
    enable_urls: bool = True
    enable_invisible_text: bool = True
    enable_deobfuscation: bool = True
    enable_exfil: bool = Field(
        default=True,
        description="The source+movement+destination correlation scanner (see scanners/exfil.py).",
    )
    enable_agent_directive: bool = Field(
        default=True,
        description="Directives addressed to the model from untrusted channels (see "
        "scanners/agent_directive.py); inert for operator provenance by construction.",
    )
    enable_ml: bool = Field(default=False, description="Optional local classifier; off by default.")
    ml_model_path: Path | None = None
    #: Weights per scanner when aggregating; unlisted scanners weigh 1.0.
    weights: dict[str, float] = Field(default_factory=dict)
    #: What an exception inside a scanner means. High-impact provenance always fails closed.
    error_policy: ScannerErrorPolicy = ScannerErrorPolicy.WARN
    #: Deobfuscation re-scans decoded content with the same deterministic checks.
    deobfuscation_max_depth: int = Field(default=2, ge=1, le=4)


class CacheSettings(BaseModel):
    """The exact-content feature cache. Features only: authorization is never cached."""

    enabled: bool = True
    max_entries: int = Field(default=4096, ge=1)
    ttl_seconds: float = Field(default=900.0, gt=0.0)


class ChunkSettings(BaseModel):
    max_segment_chars: int = Field(default=4000, ge=64)
    overlap_chars: int = Field(default=200, ge=0)
    #: Bounded inspection: a long segment contributes at most this many chunks per request.
    max_chunks_per_segment: int = Field(default=8, ge=1)


class MemorySettings(BaseModel):
    """Risk memory: reviewed attacks and benign lookalikes, kept separately permissioned."""

    backend: Literal["inmemory", "elasticsearch"] = "inmemory"
    enabled: bool = True
    #: Query the index only inside this band; the normal path never touches retrieval.
    query_only_level_below: float = Field(default=0.75, ge=0.0, le=1.0)
    retrieval_k: int = Field(default=5, ge=1, le=50)
    es_urls: tuple[str, ...] = ()
    es_index_prefix: str = "guardian"
    es_username: str | None = None
    es_password: SecretStr | None = None
    es_request_timeout_seconds: float = Field(default=2.0, gt=0.0)
    #: Advisory only: a review is a human decision and is never automated.
    auto_promote: bool = Field(default=False, frozen=True)


class PolicySettings(BaseModel):
    """Hard controls. These decide authorization; no model output is consulted here."""

    #: Declared principals, scopes, egress allow-list and budgets, as TOML or JSON.
    file: Path | None = None
    fail_closed: bool = Field(default=True, description="Unknown principal or tool is denied.")
    max_budget_defaults: dict[str, float] = Field(
        default_factory=lambda: {
            "tokens": 200_000.0,
            "concurrency": 4.0,
            "requests_per_minute": 60.0,
        }
    )


class ServerSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8090, ge=1, le=65535)
    #: Optional shared token. The service is gateway-local; a token keeps a stray local client out.
    service_token: SecretStr | None = None
    latency_budget_ms: float = Field(default=DEFAULT_LATENCY_BUDGET_MS, gt=0.0)


class GuardianSettings(BaseSettings):
    """The whole surface. ``GUARDIAN_`` prefix, ``__`` between nested sections."""

    model_config = SettingsConfigDict(
        env_prefix="GUARDIAN_",
        env_nested_delimiter="__",
        extra="forbid",
        frozen=True,
    )

    mode: GuardMode = GuardMode.SHADOW
    #: Whether ``ScanRequest.mode=shadow`` may soften a guardian configured to enforce. Off by
    #: default: a caller can always opt *into* enforcement, never silently out of it.
    allow_request_mode_downgrade: bool = False
    default_tenant: str | None = None
    retention_days_default: int = Field(default=180, ge=1, le=3650)
    thresholds: ThresholdSettings = Field(default_factory=ThresholdSettings)
    scanners: ScannerSettings = Field(default_factory=ScannerSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    chunking: ChunkSettings = Field(default_factory=ChunkSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    policy: PolicySettings = Field(default_factory=PolicySettings)
    server: ServerSettings = Field(default_factory=ServerSettings)

    @property
    def shadow(self) -> bool:
        """True when verdicts are advisory only. Enforce is an explicit opt-in."""
        return self.mode is GuardMode.SHADOW
