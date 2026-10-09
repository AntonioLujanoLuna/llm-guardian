"""``python -m llm_guardian.bench`` — the benchmark without a runpy double-import warning.

Passing ``--tool-boundary`` runs the tool-boundary policy measurement (``bench.policy_report``)
instead of the detector harness, so both live behind the same entry point. The token is distinct
from ``policy_report``'s own ``--policy <file>`` flag so the two do not collide. ``--risk-memory``
runs the risk-memory measurement (``bench.risk_memory``) instead.
"""

from __future__ import annotations

import sys

__all__ = ["main"]


def main(argv: list[str]) -> int:
    if "--tool-boundary" in argv:
        from .policy_report import main as policy_main

        return policy_main([arg for arg in argv if arg != "--tool-boundary"])

    if "--risk-memory" in argv:
        from .risk_memory import main as risk_memory_main

        return risk_memory_main([arg for arg in argv if arg != "--risk-memory"])

    from .harness import main as harness_main

    return harness_main(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
