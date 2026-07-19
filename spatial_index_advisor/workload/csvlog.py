"""Parser for the PostgreSQL CSV log format (``log_destination = 'csvlog'``).

Statements land here when ``log_min_duration_statement`` is set, as messages of
the form::

    duration: 42.113 ms  statement: SELECT ...
    duration: 12.004 ms  execute <unnamed>: SELECT ...

Each line is one execution, so calls are counted by occurrence and the reported
duration is summed per fingerprint. The column layout of csvlog differs between
server versions (23 fields in PostgreSQL 12, 26 in 15+), but the ``message``
field has been the fourteenth column since 8.4, which is what this parser relies
on; a fallback locates it by content when a row is shorter than expected.
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Final

from ..errors import WorkloadParseError
from .base import RawStatement, WorkloadSourceParser

#: Zero-based index of the ``message`` column in PostgreSQL's csvlog layout.
MESSAGE_COLUMN: Final[int] = 13

_DURATION_RE: Final[re.Pattern[str]] = re.compile(
    r"^duration:\s*([0-9]+(?:\.[0-9]+)?)\s*ms\s+"
    r"(?:statement|execute[^:]*|parse[^:]*|bind[^:]*):\s*(?P<sql>.+)$",
    re.IGNORECASE | re.DOTALL,
)
_STATEMENT_RE: Final[re.Pattern[str]] = re.compile(
    r"^statement:\s*(?P<sql>.+)$", re.IGNORECASE | re.DOTALL
)
_SQL_START_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*(SELECT|INSERT|UPDATE|DELETE|WITH)\b", re.IGNORECASE
)


class CsvLogParser(WorkloadSourceParser):
    """Reads statements and durations out of a PostgreSQL csvlog file."""

    name = "csvlog"

    def sniff(self, text: str, path: Path) -> bool:
        """Recognise csvlog by its leading timestamp column and duration messages."""
        head = text[:8192]
        if not re.match(r'^"?\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', head):
            return False
        return "duration:" in head or "statement:" in head

    def parse(self, text: str, path: Path) -> list[RawStatement]:
        """Parse the log into one raw statement per logged execution."""
        try:
            rows = list(csv.reader(io.StringIO(text)))
        except csv.Error as error:
            raise WorkloadParseError(str(path), f"malformed CSV: {error}") from error

        statements: list[RawStatement] = []
        for line_number, row in enumerate(rows, start=1):
            if not row or not any(field.strip() for field in row):
                continue
            message = self._message_field(row, path, line_number)
            parsed = self._parse_message(message)
            if parsed is not None:
                statements.append(parsed)
        if not statements:
            raise WorkloadParseError(
                str(path),
                "no 'duration: ... statement:' messages found; is "
                "log_min_duration_statement enabled?",
            )
        return statements

    @staticmethod
    def _message_field(row: list[str], path: Path, line_number: int) -> str:
        if len(row) > MESSAGE_COLUMN:
            candidate = row[MESSAGE_COLUMN]
            if candidate.strip():
                return candidate
        for field in row:
            stripped = field.strip()
            if stripped.lower().startswith(("duration:", "statement:")):
                return field
        if len(row) < 10:
            raise WorkloadParseError(
                str(path),
                f"row has {len(row)} fields; this does not look like PostgreSQL csvlog",
                line_number,
            )
        return ""

    @staticmethod
    def _parse_message(message: str) -> RawStatement | None:
        text = message.strip()
        if not text:
            return None
        match = _DURATION_RE.match(text)
        if match is not None:
            sql = match.group("sql").strip()
            if not _SQL_START_RE.match(sql):
                return None
            return RawStatement(sql=sql, calls=1, total_exec_time_ms=float(match.group(1)))
        match = _STATEMENT_RE.match(text)
        if match is not None:
            sql = match.group("sql").strip()
            if not _SQL_START_RE.match(sql):
                return None
            return RawStatement(sql=sql, calls=1, total_exec_time_ms=0.0)
        return None
