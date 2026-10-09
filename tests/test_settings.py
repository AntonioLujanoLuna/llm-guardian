"""Settings: defaults, the ordering validator and the frozen auto-promote flag."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_guardian.models import GuardMode
from llm_guardian.settings import (
    DEFAULT_LATENCY_BUDGET_MS,
    GuardianSettings,
    ThresholdSettings,
)


def test_defaults() -> None:
    settings = GuardianSettings()
    assert settings.mode is GuardMode.SHADOW
    assert settings.shadow is True
    assert settings.thresholds.escalate_at < settings.thresholds.block_at
    assert settings.memory.auto_promote is False
    assert DEFAULT_LATENCY_BUDGET_MS == 30.0


def test_enforce_is_not_shadow() -> None:
    assert GuardianSettings(mode=GuardMode.ENFORCE).shadow is False


def test_escalate_above_block_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ThresholdSettings(escalate_at=0.9, block_at=0.5)


def test_category_overrides_default_empty() -> None:
    assert ThresholdSettings().category_overrides == {}
