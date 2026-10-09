"""The optional local ML classifier, behind a hard boundary.

The default is off. ``NullMlScanner`` reports ``available=False`` and the pipeline surfaces that as
an explicit ``ml_disabled`` warning -- ML coverage being absent is *not* the same as content being
clean, and the two must never be conflated in a response.

``OnnxMlScanner`` imports its heavy dependencies lazily and raises :class:`MlDependencyMissing`
with the exact install command when the ``ml`` extra is not present, so importing this module
never drags onnxruntime into a service that does not use it.

The model is a two-class DeBERTa-family prompt-injection classifier exported to ONNX (the measured
winner is PIGuard, ``leolee99/PIGuard``; ProtectAI's ``deberta-v3-base-prompt-injection-v2`` uses
the same interface). The scanner reads ``config.json`` for the label names so the *positive* class
is found by name -- ``injection`` for PIGuard, ``INJECTION`` for ProtectAI -- rather than assumed
to be index 1, and it maps the logits through the softmax (or the sigmoid for a single-logit head).
The classifier is advisory: it emits a :class:`~llm_guardian.models.Finding` and never a verdict.
"""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol

from ..models import Finding, OwaspCategory, Severity
from ..provenance import Provenance
from .base import build_finding

#: Bumped when the scoring semantics change: the version participates in the feature-cache key, so
#: a different predictor is a cache miss rather than a stale hit.
VERSION = "ml-2"

#: Label substrings that mark the *positive* (injection) class in a checkpoint's ``id2label``.
_POSITIVE_LABEL_TOKENS: tuple[str, ...] = (
    "inject",
    "malicious",
    "unsafe",
    "attack",
    "jailbreak",
)

#: Tokenizer window. Both measured checkpoints are trained at 512; longer input is truncated and
#: the loss of the tail is the caller's problem, not silently treated as a clean scan.
DEFAULT_MAX_LENGTH = 512


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


def _sigmoid(value: float) -> float:
    """Numerically stable logistic function, clamped so an extreme logit cannot overflow."""
    clamped = max(-30.0, min(30.0, value))
    return 1.0 / (1.0 + math.exp(-clamped))


def _softmax(row: Sequence[float]) -> list[float]:
    """Softmax over one logits row, shifted by the max for numerical stability."""
    if not row:
        return []
    peak = max(row)
    exps = [math.exp(value - peak) for value in row]
    total = sum(exps)
    return [value / total for value in exps]


def class_probability(logits: Sequence[float], positive_index: int, num_labels: int) -> float:
    """Probability of the positive class from one row of logits.

    ``num_labels <= 1`` is a single-logit head (sigmoid). Otherwise the row is softmaxed and the
    ``positive_index`` entry is returned, falling back to the last column if the index is out of
    range. A scored input always yields a number in ``[0, 1]`` so the caller can threshold it.
    """
    values = [float(value) for value in logits]
    if num_labels <= 1:
        return _sigmoid(values[0]) if values else 0.0
    probabilities = _softmax(values)
    if not probabilities:
        return 0.0
    index = positive_index if 0 <= positive_index < len(probabilities) else len(probabilities) - 1
    return probabilities[index]


def injection_label_index(id2label: Mapping[str, str] | None) -> int:
    """Find the index of the positive/injection class in a checkpoint's ``id2label``.

    Matches by name (case-insensitive substring) so the mapping is read, not guessed. Falls back
    to the highest-numbered label when no name matches -- for a two-class head that is index 1.
    """
    if not id2label:
        return 1
    numeric: dict[int, str] = {}
    for key, name in id2label.items():
        try:
            numeric[int(key)] = str(name)
        except (TypeError, ValueError):
            continue
    if not numeric:
        return 1
    for index in sorted(numeric):
        lowered = numeric[index].lower()
        if any(token in lowered for token in _POSITIVE_LABEL_TOKENS):
            return index
    return max(numeric)


class OnnxMlScanner:
    """A local ONNX prompt-injection classifier. Off unless explicitly enabled and installed.

    ``model_path`` points either at a directory that holds ``model.onnx`` plus the tokenizer files
    (the layout ``transformers`` writes and the checkpoint publishes), or directly at an
    ``*.onnx`` file whose parent directory holds the tokenizer. Loading is lazy: constructing the
    scanner without the ``ml`` extra raises :class:`MlDependencyMissing`.
    """

    name = "ml"
    version = VERSION
    available = True

    def __init__(
        self,
        model_path: Path,
        *,
        threshold: float = 0.5,
        max_length: int = DEFAULT_MAX_LENGTH,
        predict: Callable[[str], float] | None = None,
    ) -> None:
        self._threshold = threshold
        self._max_length = max_length
        self._positive_index = 1
        self._num_labels = 2
        self._positive_label: str | None = None
        if predict is not None:
            # Test seam: a caller-supplied scorer bypasses the heavy load entirely.
            self._predict: Callable[[str], float] = predict
        else:
            self._predict = self._load_predictor(Path(model_path))

    def _load_predictor(  # pragma: no cover - needs the optional 'ml' extra and a model file
        self, model_path: Path
    ) -> Callable[[str], float]:
        """Load the model behind a lazy import. Never reached when the extra is absent."""
        try:
            ort = importlib.import_module("onnxruntime")
            transformers = importlib.import_module("transformers")
        except ImportError as exc:
            raise MlDependencyMissing(
                "The ML scanner needs the optional 'ml' extra. Install it with "
                "`uv sync --extra ml` (adds onnxruntime and transformers), or leave "
                "GUARDIAN_SCANNERS__ENABLE_ML=false and rely on the deterministic scanners."
            ) from exc

        model_dir, model_file = self._resolve_paths(model_path)
        if not model_file.exists():
            raise MlDependencyMissing(
                f"ML model file does not exist: {model_file}. Point "
                "GUARDIAN_SCANNERS__ML_MODEL_PATH at a directory holding model.onnx and the "
                "tokenizer files, or disable ML."
            )

        id2label, num_labels = self._read_labels(model_dir)
        self._positive_index = injection_label_index(id2label)
        self._num_labels = num_labels
        self._positive_label = id2label.get(str(self._positive_index)) if id2label else None

        session = ort.InferenceSession(str(model_file), providers=["CPUExecutionProvider"])
        input_names = {spec.name for spec in session.get_inputs()}
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_dir))

        def _predict(text: str) -> float:
            encoded = tokenizer(
                text, return_tensors="np", truncation=True, max_length=self._max_length
            )
            feeds = {
                key: value.astype("int64") for key, value in encoded.items() if key in input_names
            }
            logits = session.run(None, feeds)[0][0]
            return class_probability(list(logits), self._positive_index, self._num_labels)

        return _predict

    @staticmethod
    def _resolve_paths(  # pragma: no cover - only reached by the real loader
        model_path: Path,
    ) -> tuple[Path, Path]:
        """Return ``(model_dir, model_file)`` for a directory or a direct ``.onnx`` path."""
        if model_path.is_dir():
            return model_path, model_path / "model.onnx"
        return model_path.parent, model_path

    @staticmethod
    def _read_labels(  # pragma: no cover - only reached by the real loader
        model_dir: Path,
    ) -> tuple[dict[str, str], int]:
        """Read ``id2label`` and the label count from ``config.json``; default to a 2-class head."""
        config_path = model_dir / "config.json"
        if not config_path.exists():
            return {}, 2
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}, 2
        id2label = raw.get("id2label") if isinstance(raw, dict) else None
        if not isinstance(id2label, dict) or not id2label:
            return {}, 2
        labels = {str(key): str(value) for key, value in id2label.items()}
        return labels, len(labels)

    def scan(self, text: str, provenance: Provenance) -> Sequence[Finding]:
        # A raised predictor is surfaced as a scanner error by the pipeline, never as "clean".
        score = self._predict(text)
        if score < self._threshold:
            return []
        severity = Severity.HIGH if score >= 0.85 else Severity.MEDIUM
        attributes = {"score": f"{score:.3f}", "threshold": f"{self._threshold:.2f}"}
        if self._positive_label:
            attributes["label"] = self._positive_label
        return [
            build_finding(
                scanner=self.name,
                scanner_version=self.version,
                kind="ml_prompt_injection",
                severity=severity,
                message="The local classifier scored this segment as prompt-injection-like.",
                owasp=(OwaspCategory.LLM01,),
                attributes=attributes,
            )
        ]
