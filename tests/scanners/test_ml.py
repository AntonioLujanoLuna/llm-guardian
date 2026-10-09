"""ML scanner: the null default and the lazy-import boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_guardian.provenance import Provenance
from llm_guardian.scanners.ml import (
    MlDependencyMissing,
    NullMlScanner,
    OnnxMlScanner,
)

PROV = Provenance.USER


def test_null_scanner_is_off() -> None:
    scanner = NullMlScanner()
    assert scanner.available is False
    assert scanner.scan("anything at all", PROV) == []


def test_onnx_scanner_high_score() -> None:
    scanner = OnnxMlScanner(Path("model.onnx"), predict=lambda text: 0.95)
    assert scanner.available is True
    found = scanner.scan("suspicious", PROV)
    assert found[0].kind == "ml_prompt_injection"
    assert found[0].severity.value == "high"
    assert found[0].attributes["score"] == "0.950"


def test_onnx_scanner_medium_score() -> None:
    scanner = OnnxMlScanner(Path("model.onnx"), predict=lambda text: 0.7)
    assert scanner.scan("x", PROV)[0].severity.value == "medium"


def test_onnx_scanner_below_threshold() -> None:
    scanner = OnnxMlScanner(Path("model.onnx"), predict=lambda text: 0.1)
    assert scanner.scan("x", PROV) == []


def test_missing_extra_raises_actionable_error() -> None:
    # The ml extra is not installed in the dev environment, so the lazy load must raise.
    with pytest.raises(MlDependencyMissing) as error:
        OnnxMlScanner(Path("model.onnx"))
    assert "uv sync --extra ml" in str(error.value)
