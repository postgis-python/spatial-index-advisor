"""Command line interface.

Two subcommands:

``analyze``
    Read one or more workload files plus a catalog snapshot and print ranked
    recommendations.
``collect``
    Connect to a live PostGIS database and write the catalog snapshot that
    ``analyze`` consumes. This is the only command that needs a database.

Exit codes are designed for CI: 0 means nothing at or above the failure
threshold was found, 2 means something was, and 1 means the tool itself could not
run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final, Sequence

from rich.console import Console

from . import __version__
from .catalog import dump_catalog, load_catalog
from .collector import DEFAULT_SCHEMAS, collect_snapshot, connect
from .engine import analyse
from .errors import AdvisorError
from .models import Severity
from .report import FORMAT_CHOICES, render_json, render_sql, render_terminal
from .workload import FORMAT_CHOICES as WORKLOAD_FORMAT_CHOICES
from .workload import load_workload

EXIT_OK: Final[int] = 0
EXIT_ERROR: Final[int] = 1
EXIT_FINDINGS: Final[int] = 2

_FAIL_LEVELS: Final[dict[str, Severity | None]] = {
    "critical": Severity.CRITICAL,
    "high": Severity.HIGH,
    "medium": Severity.MEDIUM,
    "low": Severity.LOW,
    "never": None,
}


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for both subcommands."""
    parser = argparse.ArgumentParser(
        prog="python -m spatial_index_advisor",
        description=(
            "Analyse a PostgreSQL/PostGIS query workload and recommend spatial indexes, "
            "partial and composite indexes, clustering and query rewrites."
        ),
    )
    parser.add_argument("--version", action="version", version=f"spatial-index-advisor {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser(
        "analyze",
        aliases=["analyse"],
        help="recommend indexes from a workload and a catalog snapshot",
    )
    analyze.add_argument(
        "-w",
        "--workload",
        action="append",
        required=True,
        metavar="PATH",
        help="workload file; repeat for several. pg_stat_statements CSV/JSON, "
        "PostgreSQL csvlog, or a plain .sql file.",
    )
    analyze.add_argument(
        "-c",
        "--catalog",
        required=True,
        metavar="PATH",
        help="JSON catalog snapshot describing the tables (see 'collect')",
    )
    analyze.add_argument(
        "--workload-format",
        choices=WORKLOAD_FORMAT_CHOICES,
        default="auto",
        help="force a workload parser instead of sniffing each file (default: auto)",
    )
    analyze.add_argument(
        "-f",
        "--format",
        choices=FORMAT_CHOICES,
        default="rich",
        help="output format (default: rich)",
    )
    analyze.add_argument(
        "-n",
        "--top",
        type=int,
        default=None,
        metavar="N",
        help="show only the N highest-ranked recommendations",
    )
    analyze.add_argument(
        "--fail-on",
        choices=tuple(_FAIL_LEVELS),
        default="critical",
        help="lowest severity that makes the command exit with status 2 (default: critical)",
    )
    analyze.add_argument(
        "--no-color", action="store_true", help="disable colour and styling in rich output"
    )
    analyze.set_defaults(handler=_run_analyze)

    collect = subparsers.add_parser(
        "collect", help="dump a catalog snapshot from a live PostGIS database"
    )
    collect.add_argument(
        "-d",
        "--dsn",
        required=True,
        help="libpq connection string, e.g. 'postgresql://user@host/db'",
    )
    collect.add_argument(
        "-s",
        "--schema",
        action="append",
        metavar="NAME",
        help=f"schema to inspect; repeat for several (default: {', '.join(DEFAULT_SCHEMAS)})",
    )
    collect.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help="write the snapshot here instead of standard output",
    )
    collect.add_argument(
        "--no-sample",
        action="store_true",
        help="skip the TABLESAMPLE queries that measure average feature size",
    )
    collect.set_defaults(handler=_run_collect)
    return parser


def _existing_paths(values: Sequence[str]) -> list[Path]:
    return [Path(value).expanduser() for value in values]


def _run_analyze(args: argparse.Namespace, stdout: Console, stderr: Console) -> int:
    if args.top is not None and args.top < 1:
        raise AdvisorError("--top must be at least 1")

    workload = load_workload(_existing_paths(args.workload), args.workload_format)
    catalog = load_catalog(Path(args.catalog).expanduser())
    report = analyse(workload, catalog)

    if args.format == "json":
        stdout.file.write(render_json(report, args.top) + "\n")
    elif args.format == "sql":
        stdout.file.write(render_sql(report, args.top))
    else:
        render_terminal(report, stdout, args.top)

    threshold = _FAIL_LEVELS[args.fail_on]
    if threshold is None:
        return EXIT_OK
    triggered = [
        recommendation
        for recommendation in report.recommendations
        if recommendation.severity.rank <= threshold.rank
    ]
    if triggered:
        stderr.print(
            f"{len(triggered)} recommendation(s) at or above severity "
            f"'{threshold.value}'.",
            style="yellow",
        )
        return EXIT_FINDINGS
    return EXIT_OK


def _run_collect(args: argparse.Namespace, stdout: Console, stderr: Console) -> int:
    schemas = args.schema or list(DEFAULT_SCHEMAS)
    connection = connect(args.dsn)
    try:
        snapshot = collect_snapshot(
            connection, schemas=schemas, sample_geometry=not args.no_sample
        )
    finally:
        close = getattr(connection, "close", None)
        if callable(close):
            close()

    document = json.dumps(dump_catalog(snapshot), indent=2)
    if args.output:
        path = Path(args.output).expanduser()
        try:
            path.write_text(document + "\n", encoding="utf-8")
        except OSError as error:
            raise AdvisorError(f"cannot write {path}: {error}") from error
        stderr.print(f"Wrote snapshot for {len(snapshot.tables)} table(s) to {path}")
    else:
        stdout.file.write(document + "\n")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns the process exit code rather than calling ``sys.exit``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    stdout = Console(no_color=getattr(args, "no_color", False), soft_wrap=False)
    stderr = Console(stderr=True, no_color=getattr(args, "no_color", False), soft_wrap=True)
    try:
        return int(args.handler(args, stdout, stderr))
    except AdvisorError as error:
        stderr.print(f"error: {error}", style="bold red")
        return EXIT_ERROR
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        stderr.print("interrupted", style="yellow")
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
