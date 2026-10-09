"""Command line entry point: ``serve``, ``scan`` and ``bench``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .provenance import Provenance

_PROVENANCE_CHOICES = [item.value for item in Provenance]


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
    scan.add_argument("--provenance", default="user", choices=_PROVENANCE_CHOICES)

    bench = sub.add_parser("bench", help="run the synthetic evaluation corpus")
    bench.add_argument("--report", action="store_true", help="print the metric report")
    bench.add_argument("--url", default=None, help="drive a running service instead of in-process")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        return _serve(args)
    if args.command == "scan":
        return _scan(args)
    if args.command == "bench":
        return _bench(args, argv)
    raise SystemExit(f"'{args.command}' is not implemented")


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .app import create_app
    from .settings import GuardianSettings

    settings = GuardianSettings()
    host = args.host or settings.server.host
    port = args.port or settings.server.port
    uvicorn.run(create_app(settings), host=host, port=port)
    return 0


def _scan(args: argparse.Namespace) -> int:
    """Scan one text in-process and print the verdict JSON."""
    from .deps import create_service
    from .models import ScanRequest, SegmentInput
    from .service import ScanService
    from .settings import GuardianSettings

    text = args.text
    if text is None and args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    if text is None:
        print("scan needs a text argument or --file", file=sys.stderr)
        return 2

    settings = GuardianSettings()
    container = create_service(settings)
    service = ScanService(
        settings,
        container.clock,
        container.pipeline,
        container.cache,
        container.memory,
        container.incidents,
        container.metrics,
    )
    provenance = Provenance(args.provenance)
    response = service.scan(ScanRequest(segments=(SegmentInput(text=text, provenance=provenance),)))
    print(json.dumps(json.loads(response.model_dump_json()), indent=2))
    return 0


def _bench(args: argparse.Namespace, argv: Sequence[str] | None) -> int:
    """Delegate to the benchmark harness if this build ships one."""
    import importlib

    try:
        module = importlib.import_module("llm_guardian.bench.harness")
    except ImportError:
        print(
            "bench is unavailable: the llm_guardian.bench.harness module is not part of this "
            "build. Install a build that includes the benchmark harness to use `bench`.",
            file=sys.stderr,
        )
        return 2
    run_cli = getattr(module, "run_cli", None)
    if run_cli is None:
        print(
            "bench is unavailable: llm_guardian.bench.harness has no run_cli entry point.",
            file=sys.stderr,
        )
        return 2
    forwarded = list(argv) if argv is not None else sys.argv[1:]
    return int(run_cli(forwarded))
