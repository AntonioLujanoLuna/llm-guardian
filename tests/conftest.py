"""Shared fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llm_guardian.models import GuardMode
from llm_guardian.settings import GuardianSettings, PolicySettings
from tests.support import POLICY_TOML, build_app


@pytest.fixture
def policy_file(tmp_path: Path) -> Path:
    path = tmp_path / "policy.toml"
    path.write_text(POLICY_TOML, encoding="utf-8")
    return path


@pytest.fixture
def settings(policy_file: Path) -> GuardianSettings:
    return GuardianSettings(mode=GuardMode.ENFORCE, policy=PolicySettings(file=policy_file))


@pytest.fixture
def shadow_settings(policy_file: Path) -> GuardianSettings:
    return GuardianSettings(mode=GuardMode.SHADOW, policy=PolicySettings(file=policy_file))


@pytest.fixture
def client(settings: GuardianSettings) -> Iterator[TestClient]:
    with build_app(settings) as test_client:
        yield test_client


@pytest.fixture
def shadow_client(shadow_settings: GuardianSettings) -> Iterator[TestClient]:
    with build_app(shadow_settings) as test_client:
        yield test_client
