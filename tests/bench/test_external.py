"""External holdout loaders, offline.

Every network call is replaced by a recorded fixture (``fixtures/external/*.json``, captured from
the datasets-server) so these tests never touch the network: :class:`PageFetcher` runs with
``offline=True`` and raises on a cache miss. What is asserted is the honesty machinery -- a licence
gate that refuses, deterministic sampling, valid ``Case`` objects with unique ids, and a skip (not a
crash) when a declared text column is absent.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from llm_guardian.bench import external, harness
from llm_guardian.bench.corpus import PROVENANCE_VALUES

FIXTURES = Path(__file__).parent / "fixtures" / "external"
LIMIT = 6
PAGE_SIZE = 10


def _load() -> tuple[list[external.Loaded], list[external.Skipped]]:
    fetcher = external.PageFetcher(FIXTURES, offline=True)
    return external.load_all(fetcher=fetcher, limit=LIMIT, page_size=PAGE_SIZE)


@pytest.fixture(scope="module")
def datasets() -> tuple[list[external.Loaded], list[external.Skipped]]:
    return _load()


# -------------------------------------------------------------------------------------------------
# The licence gate
# -------------------------------------------------------------------------------------------------
def test_refuses_an_excluded_dataset() -> None:
    with pytest.raises(external.LicenseRefused, match="rogue-security"):
        external.enforce_license("rogue-security/prompt-injections-benchmark", "apache-2.0")
    with pytest.raises(external.LicenseRefused, match="Llama-Prompt-Guard-2"):
        external.enforce_license("meta-llama/Llama-Prompt-Guard-2-22M", "mit")


def test_refuses_an_unlicensed_dataset() -> None:
    assert external.license_refusal("someone/untagged", None) is not None
    with pytest.raises(external.LicenseRefused, match="no licence tag"):
        external.enforce_license("someone/untagged", "")


def test_refuses_a_disallowed_licence() -> None:
    with pytest.raises(external.LicenseRefused, match="allow-list"):
        external.enforce_license("someone/noncommercial", "cc-by-nc-4.0")


def test_every_declared_source_is_licensed() -> None:
    for spec in external.DEFAULT_SPECS:
        assert external.license_refusal(spec.dataset, spec.license_id) is None
        assert spec.license_tag.startswith("license:")


def test_load_all_partitions_a_refused_spec_into_skipped() -> None:
    bad = external.DatasetSpec(
        dataset="rogue-security/prompt-injections-benchmark",
        license_id="apache-2.0",
        license_tag="license:apache-2.0",
        source_url="https://huggingface.co/datasets/rogue-security/prompt-injections-benchmark",
        provenance="user",
        provenance_rationale="n/a",
        label_basis="n/a",
        sampling_note="n/a",
        streams=(
            external.Stream(label="attack", config="default", split="train", text_path=("x",)),
        ),
    )
    loaded, skipped = external.load_all(
        [bad], fetcher=external.PageFetcher(FIXTURES, offline=True), limit=1, page_size=10
    )
    assert loaded == []
    assert len(skipped) == 1
    assert "rogue-security" in skipped[0].reason


# -------------------------------------------------------------------------------------------------
# Schema handling: skip, never raise
# -------------------------------------------------------------------------------------------------
def _spec(text_path: tuple[str, ...], config: str = "default") -> external.DatasetSpec:
    return external.DatasetSpec(
        dataset="example/allowed",
        license_id="mit",
        license_tag="license:mit",
        source_url="https://huggingface.co/datasets/example/allowed",
        provenance="user",
        provenance_rationale="n/a",
        label_basis="n/a",
        sampling_note="n/a",
        streams=(
            external.Stream(label="attack", config=config, split="train", text_path=text_path),
        ),
    )


def _info(features: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset_info": {
            "default": {
                "features": dict(features),
                "splits": {"train": {"num_examples": 5}},
            }
        }
    }


def test_missing_text_column_is_skipped_with_a_reason() -> None:
    def fetcher(url: str) -> dict[str, Any]:
        assert "/info?" in url, "rows must not be requested when the schema is wrong"
        return _info({"present": {"_type": "Value"}})

    result = external.load_dataset(_spec(("absent",)), fetcher=fetcher, limit=6, page_size=10)
    assert isinstance(result, external.Skipped)
    assert "absent" in result.reason


def test_missing_config_is_skipped_with_a_reason() -> None:
    def fetcher(url: str) -> dict[str, Any]:
        return _info({"present": {"_type": "Value"}})

    result = external.load_dataset(
        _spec(("present",), config="nope"), fetcher=fetcher, limit=6, page_size=10
    )
    assert isinstance(result, external.Skipped)
    assert "nope" in result.reason


def test_nested_text_path_is_resolved() -> None:
    payload = {
        "dataset_info": {
            "default": {
                "features": {"injection": {"_type": "Value"}},
                "splits": {"train": {"num_examples": 1}},
            }
        }
    }
    rows_page = {"rows": [{"row_idx": 0, "row": {"injection": {"injection_text": "hello there"}}}]}
    calls: list[str] = []

    def fetcher(url: str) -> dict[str, Any]:
        calls.append(url)
        return payload if "/info?" in url else rows_page

    spec = _spec(("injection", "injection_text"))
    result = external.load_dataset(spec, fetcher=fetcher, limit=6, page_size=10)
    assert isinstance(result, external.Loaded)
    assert [case.text for case in result.cases] == ["hello there"]


# -------------------------------------------------------------------------------------------------
# Offline
# -------------------------------------------------------------------------------------------------
def test_offline_fetcher_raises_on_a_cache_miss(tmp_path: Path) -> None:
    fetcher = external.PageFetcher(tmp_path, offline=True)
    with pytest.raises(external.ExternalDataError, match="offline"):
        fetcher("https://datasets-server.huggingface.co/info?dataset=nope/nope")


# -------------------------------------------------------------------------------------------------
# The loaded corpus
# -------------------------------------------------------------------------------------------------
def test_all_declared_datasets_load_offline(
    datasets: tuple[list[external.Loaded], list[external.Skipped]],
) -> None:
    loaded, skipped = datasets
    assert skipped == []
    declared = {spec.dataset for spec in external.DEFAULT_SPECS}
    assert {entry.dataset for entry in loaded} == declared


def test_every_case_is_a_valid_case(
    datasets: tuple[list[external.Loaded], list[external.Skipped]],
) -> None:
    loaded, _ = datasets
    for entry in loaded:
        assert entry.cases
        for case in entry.cases:
            assert case.text.strip(), case.id
            assert case.why.strip(), case.id
            assert case.family.strip(), case.id
            assert case.label in {"attack", "benign"}, case.id
            assert case.language in {"en", "es"}, case.id
            assert case.provenance in PROVENANCE_VALUES, case.id


def test_ids_are_unique_and_prefixed(
    datasets: tuple[list[external.Loaded], list[external.Skipped]],
) -> None:
    loaded, _ = datasets
    ids = [case.id for entry in loaded for case in entry.cases]
    assert ids
    assert len(ids) == len(set(ids))
    assert all(case_id.startswith("ext:") for case_id in ids)


def test_sampling_is_deterministic_across_runs() -> None:
    first = [case.id for entry in _load()[0] for case in entry.cases]
    second = [case.id for entry in _load()[0] for case in entry.cases]
    assert first == second


def test_pooled_corpus_has_attacks_and_benign(
    datasets: tuple[list[external.Loaded], list[external.Skipped]],
) -> None:
    loaded, _ = datasets
    cases = external.to_cases(loaded)
    labels = {case.label for case in cases}
    assert labels == {"attack", "benign"}


def test_provenance_is_taken_from_the_dataset_scenario(
    datasets: tuple[list[external.Loaded], list[external.Skipped]],
) -> None:
    loaded, _ = datasets
    pairs = next(e for e in loaded if e.dataset.startswith("3nesdeniz/"))
    assert {case.provenance for case in pairs.cases} & {"tool_call", "tool_result", "user"}
    nvidia = next(e for e in loaded if e.dataset.startswith("nvidia/"))
    assert {case.provenance for case in nvidia.cases} == {"retrieved_chunk"}


def test_family_comes_from_the_datasets_own_column(
    datasets: tuple[list[external.Loaded], list[external.Skipped]],
) -> None:
    loaded, _ = datasets
    by_dataset = {entry.dataset: entry for entry in loaded}
    mscap = by_dataset["Lakera/mosscap_prompt_injection"]
    assert all(case.family.startswith("Level ") for case in mscap.cases)
    nvidia = by_dataset["nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1"]
    assert all(case.family != "nvidia_agentic_ipi" for case in nvidia.cases)


def test_manifest_carries_the_hub_licence_tags(
    datasets: tuple[list[external.Loaded], list[external.Skipped]],
) -> None:
    loaded, _ = datasets
    manifest = external.license_manifest(loaded)
    assert len(manifest) == len(external.DEFAULT_SPECS)
    by_dataset = {entry.dataset: entry for entry in manifest}
    assert by_dataset["JailbreakBench/JBB-Behaviors"].license_tag == "license:mit"
    assert by_dataset["rgeada/tool-response-injections"].license_tag == "license:apache-2.0"
    assert (
        by_dataset["3nesdeniz/agentic-prompt-injection-boundary-pairs"].license_tag
        == "license:cc-by-4.0"
    )
    for entry in manifest:
        assert entry.cases == entry.attacks + entry.benign
        assert entry.label_basis.strip()
        assert entry.provenance_rationale.strip()
        assert entry.sampling.strip()


# -------------------------------------------------------------------------------------------------
# The measurement path the report documents -- frozen pipeline + offline CLI
# -------------------------------------------------------------------------------------------------
def test_frozen_pipeline_measures_the_loaded_cases(
    datasets: tuple[list[external.Loaded], list[external.Skipped]],
) -> None:
    loaded, skipped = datasets
    cases = external.to_cases(loaded)
    scan = external.build_frozen_callable()
    pooled = harness.run(cases, scan, warmup=0)
    assert pooled.corpus_size == len(cases)
    assert pooled.attack_count + pooled.benign_count == pooled.corpus_size

    first = loaded[0]
    report = harness.run(list(first.cases), scan, warmup=0)
    text = external.render_markdown([(first, report)], skipped, pooled, limit=LIMIT, repeat=1)
    assert "Licence manifest" in text
    assert "## Numbers" in text
    assert first.dataset in text
    assert "pooled" in text


def test_cli_runs_offline_and_emits_json(capsys: pytest.CaptureFixture[str]) -> None:
    code = external.run_cli(
        [
            "--offline",
            "--cache-dir",
            str(FIXTURES),
            "--limit",
            str(LIMIT),
            "--page-size",
            str(PAGE_SIZE),
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["manifest"]) == len(external.DEFAULT_SPECS)
    assert payload["skipped"] == []
