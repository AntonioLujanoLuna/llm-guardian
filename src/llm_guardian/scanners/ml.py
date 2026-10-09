"""The optional local ML classifier, behind a hard boundary.

The default is off. ``NullMlScanner`` reports ``available=False`` and the pipeline surfaces that as
an explicit ``ml_disabled`` warning -- ML coverage being absent is *not* the same as content being
clean, and the two must never be conflated in a response.

``OnnxMlScanner`` imports its heavy dependencies lazily and raises :class:`MlDependencyMissing`
with the exact install command when the ``ml`` extra is not present, so importing this module
never drags onnxruntime into a service that does not use it.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from ..models import Finding, OwaspCategory, Severity
from ..provenance import Provenance
from .base import build_finding

VERSION = "ml-1"


class MlDependencyMissing(RuntimeError):
    """Raised when the optional ``ml`` extra is required but not installed."""


class MlScanner(Protocol):
    """A classifier that scores a segment. Advisory: it produces findings, never a verdict."""

    name: str
    version: str
    available: bool

    def scan(self, text: str, provenance: Provenance) -> Sequence[Finding]: ...


class NullMlScanner:
    """The default. Reports nothing and advertises itself as unavailable."""

    name = "ml"
    version = "null-ml-0"
    available = False

    def scan(self, text: str, provenance: Provenance) -> Sequence[Finding]:
        return []


class OnnxMlScanner:
    """A local ONNX/transformers classifier. Off unless explicitly enabled and installed."""

    name = "ml"
    version = VERSION
    available = True

    def __init__(
        self,
        model_path: Path,
        *,
        threshold: float = 0.5,
        predict: Callable[[str], float] | None = None,
    ) -> None:
        self._threshold = threshold
        self._predict: Callable[[str], float] = predict or self._load_predictor(model_path)

    @staticmethod
    def _load_predictor(model_path: Path) -> Callable[[str], float]:
        """Load the model behind a lazy import. Never reached when the extra is absent."""
        # pragma: no cover - requires the optional ``ml`` extra and a real model file.
        try:
            ort = importlib.import_module("onnxruntime")
            transformers = importlib.import_module("transformers")
        except ImportError as exc:
            raise MlDependencyMissing(
                "The ML scanner needs the optional 'ml' extra. Install it with "
                "`uv sync --extra ml` (adds onnxruntime and transformers), or leave "
                "GUARDIAN_SCANNERS__ENABLE_ML=false and rely on the deterministic scanners."
            ) from exc
        if not model_path.exists():
            raise MlDependencyMissing(
                f"ML model path does not exist: {model_path}. Point "
                "GUARDIAN_SCANNERS__ML_MODEL_PATH at a local ONNX model or disable ML."
            )
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_path))

        def _predict(text: str) -> float:
            encoded = tokenizer(text, return_tensors="np", truncation=True, max_length=512)
            logits = session.run(None, dict(encoded))[0]
            raw = float(logits[0][-1])
            value = max(-30.0, min(30.0, raw))
            return 1.0 / (1.0 + math.exp(-value))

        return _predict

    def scan(self, text: str, provenance: Provenance) -> Sequence[Finding]:
        score = self._predict(text)
        if score < self._threshold:
            return []
        severity = Severity.HIGH if score >= 0.85 else Severity.MEDIUM
        return [
            build_finding(
                scanner=self.name,
                scanner_version=self.version,
                kind="ml_prompt_injection",
                severity=severity,
                message="The local classifier scored this segment as prompt-injection-like.",
                owasp=(OwaspCategory.LLM01,),
                attributes={"score": f"{score:.3f}"},
            )
        ]
