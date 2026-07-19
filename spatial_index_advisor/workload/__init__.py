"""Workload ingestion: one parser per source format behind a common interface.

Use :func:`load_workload` to read one or more files; pass ``fmt="auto"`` to let
each file be sniffed independently, which is what the CLI does by default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from ..errors import WorkloadParseError
from ..models import Workload
from .base import RawStatement, WorkloadSourceParser, aggregate
from .csvlog import CsvLogParser
from .pg_stat_statements import PgStatStatementsParser
from .sqlfile import SqlFileParser

#: Registry in sniffing priority order: the most specific formats come first, and
#: the permissive plain-SQL parser is the last resort.
PARSERS: Final[tuple[WorkloadSourceParser, ...]] = (
    PgStatStatementsParser(),
    CsvLogParser(),
    SqlFileParser(),
)

PARSERS_BY_NAME: Final[dict[str, WorkloadSourceParser]] = {p.name: p for p in PARSERS}

#: Values accepted by ``--workload-format``.
FORMAT_CHOICES: Final[tuple[str, ...]] = ("auto",) + tuple(PARSERS_BY_NAME)


def detect_parser(text: str, path: Path) -> WorkloadSourceParser:
    """Return the parser that recognises ``text``.

    Raises:
        WorkloadParseError: when no registered parser claims the file.
    """
    for parser in PARSERS:
        if parser.sniff(text, path):
            return parser
    raise WorkloadParseError(
        str(path),
        "unrecognised workload format; pass --workload-format to choose one of "
        + ", ".join(PARSERS_BY_NAME),
    )


def load_workload(paths: list[Path], fmt: str = "auto") -> Workload:
    """Read every path and fold the result into a single :class:`Workload`.

    Args:
        paths: files to read; each may be a different format when ``fmt='auto'``.
        fmt: ``'auto'`` or a key of :data:`PARSERS_BY_NAME`.

    Raises:
        WorkloadParseError: when a file is missing, empty or unparseable.
    """
    if not paths:
        raise WorkloadParseError("<workload>", "no workload files given")
    if fmt not in FORMAT_CHOICES:
        raise WorkloadParseError(
            "<workload>", f"unknown workload format {fmt!r}; expected one of {FORMAT_CHOICES}"
        )

    collected: list[RawStatement] = []
    sources: list[str] = []
    for path in paths:
        if not path.is_file():
            raise WorkloadParseError(str(path), "file does not exist")
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            raise WorkloadParseError(str(path), f"cannot read file: {error}") from error
        if not text.strip():
            raise WorkloadParseError(str(path), "file is empty")
        parser = PARSERS_BY_NAME[fmt] if fmt != "auto" else detect_parser(text, path)
        collected.extend(parser.parse(text, path))
        sources.append(f"{path}({parser.name})")
    return aggregate(collected, sources)


__all__ = [
    "FORMAT_CHOICES",
    "PARSERS",
    "PARSERS_BY_NAME",
    "RawStatement",
    "WorkloadSourceParser",
    "aggregate",
    "detect_parser",
    "load_workload",
]
