"""Command line entry point: ``serve``, ``scan``, ``bench`` and ``export-training``."""

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

    export = sub.add_parser(
        "export-training",
        help="write reviewed incidents (confirmed_attack / benign) as JSONL training data",
    )
    export.add_argument("--db", default=None, help="SQLite incident store (default: settings)")
    export.add_argument("--out", default="-", help="output file, or - for stdout")

    # Every bench option belongs to the harness's own parser; they are forwarded untouched.
    sub.add_parser(
        "bench", help="run the synthetic evaluation corpus (see `llm-guardian bench --help`)"
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser()
    args, extra = parser.parse_known_args(raw)
    if args.command == "bench":
        return _bench(raw[raw.index("bench") + 1 :])
    if extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    if args.command == "serve":
        return _serve(args)
    if args.command == "scan":
        return _scan(args)
    if args.command == "export-training":
        return _export_training(args)
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


def _export_training(args: argparse.Namespace) -> int:
    """Reviewed cases as ``{"text", "label", ...}`` lines, for fine-tuning a local classifier.

    Only human-reviewed labels are exported, and only cases whose excerpt is content: a case
    recorded with ``incidents.excerpt=findings`` holds a detector summary, which would teach a
    classifier to recognise the guardian's own messages.
    """
    from .incidents_sqlite import SqliteIncidentRepository
    from .models import IncidentStatus
    from .settings import GuardianSettings

    if args.db is not None:
        path = Path(args.db)
    else:
        settings = GuardianSettings()
        if settings.incidents.backend != "sqlite":
            print(
                "export-training needs a persistent store: pass --db or set "
                "GUARDIAN_INCIDENTS__BACKEND=sqlite (the in-memory store is empty in a new "
                "process).",
                file=sys.stderr,
            )
            return 2
        path = settings.incidents.sqlite_path
    if not path.exists():
        print(f"export-training: no incident store at {path}", file=sys.stderr)
        return 2

    labels = {IncidentStatus.CONFIRMED_ATTACK: "attack", IncidentStatus.BENIGN: "benign"}
    repo = SqliteIncidentRepository(path)
    written = skipped = 0
    lines: list[str] = []
    for record in repo.records():
        label = labels.get(record.status)
        if label is None:
            continue
        if not record.redacted_excerpt or record.redacted_excerpt == record.findings_summary:
            skipped += 1
            continue
        lines.append(
            json.dumps(
                {
                    "text": record.redacted_excerpt,
                    "label": label,
                    "provenance": record.provenance.value,
                    "owasp": [category.value for category in record.owasp],
                    "family": record.family,
                    "tenant": record.tenant,
                    "case_id": record.case_id,
                    "reviewed_by": record.reviewed_by,
                },
                ensure_ascii=False,
            )
        )
        written += 1
    repo.close()

    payload = "".join(line + "\n" for line in lines)
    if args.out == "-":
        sys.stdout.write(payload)
    else:
        Path(args.out).write_text(payload, encoding="utf-8")
    print(f"export-training: {written} written, {skipped} skipped (no content)", file=sys.stderr)
    return 0


def _bench(forwarded: list[str]) -> int:
    """Delegate to the benchmark harness (with the arguments after ``bench``), if shipped."""
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
    return int(run_cli(forwarded))
