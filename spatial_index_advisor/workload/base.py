"""Common interface shared by every workload source parser.

A parser's only job is to turn a file into :class:`RawStatement` records. All
normalization, fingerprinting and aggregation happens once, in :func:`aggregate`,
so that every source produces identically shaped output.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path

from ..errors import WorkloadParseError
from ..models import Workload, WorkloadStatement
from ..normalize import normalize_and_fingerprint


@dataclass(frozen=True)
class RawStatement:
    """One statement occurrence as read from a workload file.

    ``calls`` is 1 for sources that record individual executions (logs, SQL
    files) and the server-side counter for ``pg_stat_statements``.
    """

    sql: str
    calls: int = 1
    total_exec_time_ms: float = 0.0
    rows: int = 0


class WorkloadSourceParser(abc.ABC):
    """Base class for workload parsers."""

    #: Short identifier used by ``--workload-format`` and in reports.
    name: str = ""

    @abc.abstractmethod
    def sniff(self, text: str, path: Path) -> bool:
        """Return True when this parser recognises ``text`` as its own format."""

    @abc.abstractmethod
    def parse(self, text: str, path: Path) -> list[RawStatement]:
        """Parse the file contents into raw statement records.

        Raises:
            WorkloadParseError: if the file is structurally invalid.
        """

    def read(self, path: Path) -> list[RawStatement]:
        """Read and parse a file from disk."""
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            raise WorkloadParseError(str(path), f"cannot read file: {error}") from error
        return self.parse(text, path)


def aggregate(statements: list[RawStatement], sources: list[str]) -> Workload:
    """Normalize, fingerprint and fold raw statements into a :class:`Workload`.

    Statements that normalize to an empty string (blank input, comment-only
    chunks) are dropped. Results are ordered by total execution time descending,
    falling back to call count when no timing data is available.
    """
    folded: dict[str, WorkloadStatement] = {}
    for raw in statements:
        normalized, digest = normalize_and_fingerprint(raw.sql)
        if not normalized:
            continue
        source = sources[0] if len(sources) == 1 else "multiple"
        statement = WorkloadStatement(
            fingerprint=digest,
            normalized_sql=normalized,
            sample_sql=raw.sql.strip().rstrip(";").strip(),
            calls=max(raw.calls, 0),
            total_exec_time_ms=max(raw.total_exec_time_ms, 0.0),
            rows=max(raw.rows, 0),
            source=source,
        )
        existing = folded.get(digest)
        folded[digest] = existing.merged_with(statement) if existing else statement
    ordered = sorted(
        folded.values(),
        key=lambda s: (s.total_exec_time_ms, s.calls),
        reverse=True,
    )
    return Workload(statements=tuple(ordered), sources=tuple(sources))
