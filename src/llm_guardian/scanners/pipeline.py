"""The scanner pipeline: which scanners run, in what order, and what a scanner failure means.

A scanner that raises does not disappear: its :class:`~llm_guardian.models.ScannerOutcome` carries
the error, and the verdict layer decides the consequence from the declared error policy and the
provenance. The pipeline only reports; it never swallows a failure into silence.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence

from ..models import ScannerOutcome
from ..provenance import Provenance
from ..settings import ScannerSettings
from .agent_directive import AgentDirectiveScanner
from .base import Scanner
from .command import CommandRiskScanner
from .deobfuscate import DeobfuscatingScanner
from .exfil import ExfilChannelScanner
from .invisible import InvisibleTextScanner
from .ml import MlDependencyMissing, MlScanner, NullMlScanner, OnnxMlScanner
from .rules import RuleScanner
from .secrets import SecretScanner
from .urls import UrlScanner

#: Bumped whenever the text is transformed before scanning (normalisation, unicode folding...).
PREPROCESSING_VERSION = "pre-1"

ML_DISABLED_WARNING = (
    "ml_disabled: the optional local classifier is off; ML coverage is absent, not 'clean'."
)


class ScannerPipeline:
    """Runs the enabled scanners over one segment and reports each outcome, error or not."""

    def __init__(
        self,
        scanners: Sequence[Scanner],
        *,
        warnings: Sequence[str] = (),
    ) -> None:
        self._scanners: tuple[Scanner, ...] = tuple(scanners)
        self._warnings: tuple[str, ...] = tuple(warnings)

    @classmethod
    def from_settings(cls, settings: ScannerSettings) -> ScannerPipeline:
        scanners: list[Scanner] = []
        rules = RuleScanner()
        if settings.enable_deobfuscation:
            scanners.append(DeobfuscatingScanner(rules, settings.deobfuscation_max_depth))
        if settings.enable_rules:
            scanners.append(rules)
        if settings.enable_agent_directive:
            scanners.append(AgentDirectiveScanner())
        if settings.enable_exfil:
            scanners.append(ExfilChannelScanner())
        if settings.enable_command:
            scanners.append(CommandRiskScanner())
        if settings.enable_secrets:
            scanners.append(SecretScanner())
        if settings.enable_urls:
            scanners.append(UrlScanner())
        if settings.enable_invisible_text:
            scanners.append(InvisibleTextScanner())
        warnings: list[str] = []
        ml = _build_ml(settings, warnings)
        if ml.available:
            scanners.append(ml)
        return cls(scanners, warnings=warnings)

    @property
    def warnings(self) -> tuple[str, ...]:
        return self._warnings

    @property
    def scanners(self) -> tuple[Scanner, ...]:
        return self._scanners

    def versions(self) -> dict[str, str]:
        return {scanner.name: scanner.version for scanner in self._scanners}

    def fingerprint(self) -> str:
        """A digest of the active scanner names and versions, for the feature-cache key."""
        joined = "|".join(f"{s.name}:{s.version}" for s in self._scanners)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]

    def scan(self, text: str, provenance: Provenance) -> tuple[ScannerOutcome, ...]:
        outcomes: list[ScannerOutcome] = []
        for scanner in self._scanners:
            started = time.perf_counter()
            try:
                findings = scanner.scan(text, provenance)
            except Exception as exc:
                outcomes.append(
                    ScannerOutcome(
                        scanner=scanner.name,
                        scanner_version=scanner.version,
                        error=f"{type(exc).__name__}: {exc}",
                        duration_ms=(time.perf_counter() - started) * 1000.0,
                    )
                )
                continue
            outcomes.append(
                ScannerOutcome(
                    scanner=scanner.name,
                    scanner_version=scanner.version,
                    findings=tuple(findings),
                    duration_ms=(time.perf_counter() - started) * 1000.0,
                )
            )
        return tuple(outcomes)


def _build_ml(settings: ScannerSettings, warnings: list[str]) -> MlScanner:
    if not settings.enable_ml:
        warnings.append(ML_DISABLED_WARNING)
        return NullMlScanner()
    if settings.ml_model_path is None:
        warnings.append(
            "ml_enabled_without_model: enable_ml is true but no ml_model_path is set; "
            "ML stays disabled rather than reporting content as clean."
        )
        return NullMlScanner()
    try:
        return OnnxMlScanner(settings.ml_model_path)
    except MlDependencyMissing as exc:
        warnings.append(f"ml_dependency_missing: {exc}")
        return NullMlScanner()
