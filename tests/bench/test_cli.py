"""CLI contract: exit codes for the recall / false-block thresholds, JSON output, and filters.

``run_cli`` is the function ``llm-guardian bench`` calls; these tests pin the parts of its contract
another module depends on -- the signature, the ``--json`` shape and the exit codes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from llm_guardian.bench import corpus, harness


def _out(capsys: pytest.CaptureFixture[str]) -> str:
    return capsys.readouterr().out


def test_stub_allow_returns_zero_and_emits_json(capsys: pytest.CaptureFixture[str]) -> None:
    code = harness.run_cli(["--stub", "allow", "--json", "--warmup", "0"])
    assert code == 0
    report = json.loads(_out(capsys))
    assert report["recall"] == pytest.approx(0.0)
    assert report["false_blocks"] == 0
    assert report["caveats"]


def test_stub_block_json_reflects_full_recall(capsys: pytest.CaptureFixture[str]) -> None:
    code = harness.run_cli(["--stub", "block", "--json", "--warmup", "0"])
    assert code == 0
    report = json.loads(_out(capsys))
    assert report["recall"] == pytest.approx(1.0)
    assert report["false_blocks_per_1000"] == pytest.approx(1000.0)


def test_max_false_blocks_threshold_breaches(capsys: pytest.CaptureFixture[str]) -> None:
    code = harness.run_cli(
        ["--stub", "block", "--json", "--warmup", "0", "--max-false-blocks-per-1000", "0"]
    )
    assert code == 1
    assert json.loads(_out(capsys))["false_blocks_per_1000"] == pytest.approx(1000.0)


def test_min_recall_threshold_breaches(capsys: pytest.CaptureFixture[str]) -> None:
    code = harness.run_cli(["--stub", "allow", "--json", "--warmup", "0", "--min-recall", "0.9"])
    assert code == 1
    _out(capsys)


def test_thresholds_within_bounds_return_zero(capsys: pytest.CaptureFixture[str]) -> None:
    code = harness.run_cli(
        [
            "--stub",
            "block",
            "--json",
            "--warmup",
            "0",
            "--max-false-blocks-per-1000",
            "2000",
            "--min-recall",
            "0.5",
        ]
    )
    assert code == 0
    _out(capsys)


def test_min_recall_zero_passes_for_allow_stub(capsys: pytest.CaptureFixture[str]) -> None:
    code = harness.run_cli(["--stub", "allow", "--warmup", "0", "--min-recall", "0.0"])
    assert code == 0
    assert "Phase 1 baseline" in _out(capsys)


def test_label_filter_selects_only_benign(capsys: pytest.CaptureFixture[str]) -> None:
    code = harness.run_cli(["--stub", "block", "--json", "--warmup", "0", "--label", "benign"])
    assert code == 0
    report = json.loads(_out(capsys))
    assert report["attack_count"] == 0
    assert report["benign_count"] == sum(1 for case in corpus.CASES if case.label == "benign")


def test_family_filter_narrows_the_corpus(capsys: pytest.CaptureFixture[str]) -> None:
    expected = sum(1 for case in corpus.CASES if case.family == "instruction_override")
    code = harness.run_cli(
        ["--stub", "allow", "--json", "--warmup", "0", "--family", "instruction_override"]
    )
    assert code == 0
    assert json.loads(_out(capsys))["corpus_size"] == expected


def test_language_filter(capsys: pytest.CaptureFixture[str]) -> None:
    expected = sum(1 for case in corpus.CASES if case.language == "es")
    code = harness.run_cli(["--stub", "allow", "--json", "--warmup", "0", "--language", "es"])
    assert code == 0
    assert json.loads(_out(capsys))["corpus_size"] == expected


def test_tag_filter(capsys: pytest.CaptureFixture[str]) -> None:
    expected = sum(1 for case in corpus.CASES if "overdefense" in case.tags)
    code = harness.run_cli(["--stub", "allow", "--json", "--warmup", "0", "--tag", "overdefense"])
    assert code == 0
    assert json.loads(_out(capsys))["corpus_size"] == expected


def test_repeat_pools_latency_samples(capsys: pytest.CaptureFixture[str]) -> None:
    code = harness.run_cli(
        [
            "--stub",
            "allow",
            "--json",
            "--warmup",
            "0",
            "--repeat",
            "2",
            "--family",
            "instruction_override",
        ]
    )
    assert code == 0
    report = json.loads(_out(capsys))
    assert report["latency_total"]["count"] == report["corpus_size"] * 2


def test_external_corpus_jsonl(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    subset = [case for case in corpus.CASES if case.label == "benign"][:3]
    path = corpus.export_jsonl(tmp_path / "subset.jsonl", subset)
    code = harness.run_cli(["--corpus", str(path), "--stub", "allow", "--json", "--warmup", "0"])
    assert code == 0
    assert json.loads(_out(capsys))["corpus_size"] == 3


def test_missing_corpus_file_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = harness.run_cli(["--corpus", str(tmp_path / "nope.jsonl"), "--stub", "allow"])
    assert code == 2
    assert "could not load corpus" in capsys.readouterr().err


def test_unknown_family_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    code = harness.run_cli(["--stub", "allow", "--family", "does-not-exist"])
    assert code == 2
    assert "no cases matched" in capsys.readouterr().err


def test_injected_scan_callable_is_used(capsys: pytest.CaptureFixture[str]) -> None:
    def scan(case: corpus.Case) -> harness.ScanObservation:
        return harness.ScanObservation(decision="block" if case.label == "attack" else "allow")

    code = harness.run_cli(["--json", "--warmup", "0"], scan_callable=scan)
    assert code == 0
    report = json.loads(_out(capsys))
    assert report["corpus_size"] == len(corpus.CASES)
    assert report["recall"] == pytest.approx(1.0)
    assert report["false_blocks"] == 0


def test_url_branch_drives_a_dead_endpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    subset = corpus.CASES[:1]
    path = corpus.export_jsonl(tmp_path / "one.jsonl", subset)
    # An unused port: the request fails and the harness records a transport error rather than
    # crashing, so no threshold is breached and the exit code is 0.
    code = harness.run_cli(
        ["--corpus", str(path), "--url", "http://127.0.0.1:1", "--json", "--warmup", "0"]
    )
    assert code == 0
    report = json.loads(_out(capsys))
    assert report["degradation"]["transport_errors"] == 1


def test_default_inprocess_branch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        harness, "default_inprocess_callable", lambda: harness.stub_callable("allow")
    )
    code = harness.run_cli(["--json", "--warmup", "0", "--family", "instruction_override"])
    assert code == 0
    assert json.loads(_out(capsys))["corpus_size"] >= 1


def test_report_flag_prints_text(capsys: pytest.CaptureFixture[str]) -> None:
    code = harness.run_cli(
        ["--stub", "allow", "--report", "--warmup", "0", "--family", "instruction_override"]
    )
    assert code == 0
    text = _out(capsys)
    assert "Recall by attack family" in text
    assert "instruction_override" in text


def test_main_entry_point(capsys: pytest.CaptureFixture[str]) -> None:
    code = harness.main(
        ["--stub", "allow", "--json", "--warmup", "0", "--family", "instruction_override"]
    )
    assert code == 0
    assert json.loads(_out(capsys))["corpus_size"] >= 1


def test_run_cli_signature_accepts_positional_argv_only(capsys: pytest.CaptureFixture[str]) -> None:
    # cli.py calls run_cli(argv); argv must be the only required argument.
    argv: list[str] = [
        "--stub",
        "allow",
        "--json",
        "--warmup",
        "0",
        "--family",
        "instruction_override",
    ]
    assert harness.run_cli(argv) == 0
    _out(capsys)


def test_json_report_is_serializable_round_trip(capsys: pytest.CaptureFixture[str]) -> None:
    harness.run_cli(
        ["--stub", "allow", "--json", "--warmup", "0", "--family", "instruction_override"]
    )
    payload: dict[str, Any] = json.loads(_out(capsys))
    assert "naive_keyword_false_block_cases" in payload
    assert "latency_by_stage" in payload
