# OWASP LLM Top 10 (2026) coverage

Generated from `owasp.coverage_table()`, the same source as `GET /v1/owasp`. `implemented` means this service implements a control for the category; it is not a claim that the category is fully addressed here. There is deliberately no aggregate score.

| Category | Title | Implemented | Enforced by | Tests |
| --- | --- | --- | --- | --- |
| LLM01 | Prompt Injection | yes | guardian | `tests/scanners/test_rules.py`, `tests/scanners/test_deobfuscate.py`, `tests/scanners/test_invisible.py` |
| LLM02 | Sensitive Information Disclosure | yes | guardian | `tests/scanners/test_secrets.py`, `tests/scanners/test_urls.py` |
| LLM03 | Excessive Agency | yes | policy.py + gateway tool executor | `tests/test_policy.py` |
| LLM04 | Supply Chain | no | build pipeline / gateway | - |
| LLM05 | Data and Model Poisoning | no | gateway ingestion + incident review | `tests/test_incidents.py` |
| LLM06 | Unbounded Consumption | yes | policy.py + gateway rate limiting | `tests/test_policy.py`, `tests/test_segments.py` |
| LLM07 | Misinformation | no | gateway / application | - |
| LLM08 | Hidden Context Exposure | yes | guardian | `tests/scanners/test_invisible.py`, `tests/test_memory.py`, `tests/scanners/test_rules.py` |
| LLM09 | Vector and Embedding Weaknesses | yes | memory/ + gateway retrieval ACLs | `tests/test_memory.py` |
| LLM10 | Improper Output Handling | yes | guardian (input) + gateway output escaping | `tests/scanners/test_urls.py`, `tests/scanners/test_rules.py` |
