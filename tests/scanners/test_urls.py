"""URL scanner: egress-shaped signals and the benign-URL false-positive case."""

from __future__ import annotations

from llm_guardian.provenance import Provenance
from llm_guardian.scanners.urls import UrlScanner

scanner = UrlScanner()
PROV = Provenance.USER


def kinds(findings) -> set[str]:
    return {f.kind for f in findings}


def test_raw_ip_host() -> None:
    found = scanner.scan("fetch http://192.168.1.10/payload", PROV)
    assert "raw_ip_host" in kinds(found)
    assert found[0].attributes["host"].startswith("192.")


def test_dangerous_schemes() -> None:
    assert "dangerous_scheme" in kinds(scanner.scan("click javascript:alert(1)", PROV))
    assert "dangerous_scheme" in kinds(scanner.scan("img src=data:text/html;base64,x", PROV))
    assert "dangerous_scheme" in kinds(scanner.scan("open file:///etc/passwd", PROV))


def test_long_encoded_query() -> None:
    blob = "A" * 120
    found = scanner.scan(f"https://x.example/?data={blob}", PROV)
    assert "long_encoded_query" in kinds(found)


def test_paste_and_webhook_hosts() -> None:
    found = scanner.scan("see https://pastebin.com/abcd and https://webhook.site/xyz", PROV)
    hosts = {f.attributes["host"] for f in found if f.kind == "paste_or_webhook_host"}
    assert {"pastebin.com", "webhook.site"} <= hosts


def test_benign_urls_no_findings() -> None:
    text = "See https://github.com/AntonioLujanoLuna/llm-guardian and https://pypi.org/project/x/"
    assert scanner.scan(text, PROV) == []
