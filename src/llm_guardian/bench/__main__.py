"""``python -m llm_guardian.bench`` — the benchmark without a runpy double-import warning."""

from __future__ import annotations

import sys

from .harness import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
