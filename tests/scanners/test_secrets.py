"""Secret scanner: detection, redaction and false-positive cases."""

from __future__ import annotations

from llm_guardian.provenance import Provenance
from llm_guardian.scanners.secrets import SecretScanner

scanner = SecretScanner()
PROV = Provenance.REPOSITORY_FILE


def kinds(findings) -> set[str]:
    return {f.kind for f in findings}


def test_pem_private_key() -> None:
    text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEabc\n-----END RSA PRIVATE KEY-----"
    found = scanner.scan(text, PROV)
    assert "private_key" in kinds(found)
    assert found[0].severity.value == "critical"


def test_provider_api_keys() -> None:
    text = "key AKIAIOSFODNN7EXAMPLE and sk-abcdef0123456789ABCDEF and gh" + "p_" + "x" * 36
    found = scanner.scan(text, PROV)
    assert sum(1 for f in found if f.kind == "api_key") >= 3


def test_jwt_and_connection_string() -> None:
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abc123DEF456ghi789jkl"
    conn = "postgres://user:pass@db.internal:5432/app"
    found = scanner.scan(f"token {jwt} url {conn}", PROV)
    assert {"jwt", "connection_string"} <= kinds(found)


def test_env_assignment_secret_name() -> None:
    found = scanner.scan("SESSION_SECRET=aBcD1234xyz9", PROV)
    assert "env_assignment" in kinds(found)


def test_env_assignment_looks_secret_value() -> None:
    found = scanner.scan("FOO=aBcD1234efgh", PROV)
    assert "env_assignment" in kinds(found)


def test_benign_env_assignment_not_flagged() -> None:
    assert scanner.scan("DEBUG=true\nPATH=/usr/bin\nFOO=hello", PROV) == []


def test_empty_and_plain_text() -> None:
    assert scanner.scan("", PROV) == []
    assert scanner.scan("The API key is stored securely in the vault.", PROV) == []


def test_secret_value_is_never_in_the_finding() -> None:
    secret = "AKIAIOSFODNN7EXAMPLE"
    found = scanner.scan(f"aws {secret}", PROV)
    assert found
    for finding in found:
        assert secret not in finding.message
        assert secret not in str(finding.attributes)
    hint = found[0].attributes["redacted"]
    assert hint.startswith("AKI") or hint.startswith("***")
    assert hint.endswith("***") or "sha" in hint
