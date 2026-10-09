"""Command line entry point: ``serve``, ``scan`` and ``bench``."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-guardian", description="LLM Guardian service.")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the guardian service")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--reload", action="store_true")

    scan = sub.add_parser("scan", help="scan one text segment and print the verdict")
    scan.add_argument("text", nargs="?", default=None)
    scan.add_argument("--file", default=None)
    scan.add_argument("--provenance", default="user")

    bench = sub.add_parser("bench", help="run the synthetic evaluation corpus")
    bench.add_argument("--report", action="store_true", help="print the metric report")
    bench.add_argument("--url", default=None, help="drive a running service instead of in-process")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        return _serve(args)
    raise SystemExit(f"'{args.command}' is not implemented yet")


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .app import create_app
    from .settings import GuardianSettings

    settings = GuardianSettings()
    host = args.host or settings.server.host
    port = args.port or settings.server.port
    uvicorn.run(create_app(settings), host=host, port=port)
    return 0
