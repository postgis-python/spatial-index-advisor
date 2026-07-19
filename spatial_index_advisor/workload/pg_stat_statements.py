"""Parser for ``pg_stat_statements`` exports in CSV or JSON form.

Accepts the output of, for example::

    \\copy (SELECT query, calls, total_exec_time, rows FROM pg_stat_statements
           ORDER BY total_exec_time DESC LIMIT 200) TO 'workload.csv' CSV HEADER

and the equivalent ``row_to_json`` array. Column naming differs across server
versions (``total_time`` before PostgreSQL 13, ``total_exec_time`` after), and
some tools export ``mean_exec_time`` instead of a total, so all of those spellings
are accepted and reconciled.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Final, Mapping

from ..errors import WorkloadParseError
from .base import RawStatement, WorkloadSourceParser

_QUERY_COLUMNS: Final[tuple[str, ...]] = ("query", "statement", "normalized_query", "sql")
_CALLS_COLUMNS: Final[tuple[str, ...]] = ("calls", "call_count", "executions")
_TOTAL_TIME_COLUMNS: Final[tuple[str, ...]] = (
    "total_exec_time",
    "total_time",
    "total_exec_time_ms",
    "total_ms",
)
_MEAN_TIME_COLUMNS: Final[tuple[str, ...]] = ("mean_exec_time", "mean_time", "avg_time")
_ROWS_COLUMNS: Final[tuple[str, ...]] = ("rows", "row_count", "returned_rows")


def _pick(row: Mapping[str, Any], candidates: tuple[str, ...]) -> Any | None:
    """First present, non-empty value among ``candidates`` (case-insensitive)."""
    lowered = {str(key).strip().lower(): value for key, value in row.items()}
    for candidate in candidates:
        value = lowered.get(candidate)
        if value not in (None, ""):
            return value
    return None


def _as_int(value: Any, default: int = 0) -> int:
    """Coerce a CSV/JSON scalar to int, tolerating floats and thousands separators."""
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(float(str(value).replace(",", "").strip()))
    except ValueError:
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    """Coerce a CSV/JSON scalar to float, tolerating thousands separators."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return default


class PgStatStatementsParser(WorkloadSourceParser):
    """Reads a ``pg_stat_statements`` snapshot from CSV or JSON."""

    name = "pgss"

    def sniff(self, text: str, path: Path) -> bool:
        """Recognise the format by its column names.

        The check is deliberately strict — it looks at CSV header *fields* and
        JSON *keys*, not at substrings anywhere in the file — so that a SQL file
        containing the word "statement" in a comment is not misclassified.
        """
        stripped = text.lstrip()
        if not stripped:
            return False
        if stripped.startswith(("[", "{")):
            head = stripped[:8192].lower()
            has_query = any(f'"{column}"' in head for column in _QUERY_COLUMNS)
            has_counter = any(
                f'"{column}"' in head for column in _CALLS_COLUMNS + _TOTAL_TIME_COLUMNS
            )
            return has_query and has_counter
        header = stripped.splitlines()[0].lower()
        fields = {field.strip().strip('"').strip() for field in header.split(",")}
        return bool(fields & set(_QUERY_COLUMNS)) and bool(
            fields & set(_CALLS_COLUMNS + _TOTAL_TIME_COLUMNS + _MEAN_TIME_COLUMNS)
        )

    def parse(self, text: str, path: Path) -> list[RawStatement]:
        """Parse the export into raw statements."""
        stripped = text.lstrip()
        if stripped.startswith(("[", "{")):
            rows = self._parse_json(stripped, path)
        else:
            rows = self._parse_csv(text, path)
        statements = [statement for row in rows if (statement := self._to_statement(row))]
        if not statements:
            raise WorkloadParseError(str(path), "no usable rows found in pg_stat_statements export")
        return statements

    @staticmethod
    def _parse_json(text: str, path: Path) -> list[Mapping[str, Any]]:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise WorkloadParseError(str(path), f"invalid JSON: {error.msg}", error.lineno) from error
        if isinstance(payload, Mapping):
            candidate = payload.get("statements") or payload.get("rows") or payload.get("data")
            if candidate is None:
                raise WorkloadParseError(
                    str(path),
                    "JSON object must contain a 'statements', 'rows' or 'data' array",
                )
            payload = candidate
        if not isinstance(payload, list):
            raise WorkloadParseError(str(path), "expected a JSON array of row objects")
        rows: list[Mapping[str, Any]] = []
        for index, item in enumerate(payload, start=1):
            if not isinstance(item, Mapping):
                raise WorkloadParseError(str(path), f"row {index} is not an object")
            rows.append(item)
        return rows

    @staticmethod
    def _parse_csv(text: str, path: Path) -> list[Mapping[str, Any]]:
        reader = csv.DictReader(io.StringIO(text))
        if reader.fieldnames is None:
            raise WorkloadParseError(str(path), "CSV export has no header row")
        if not any(
            (name or "").strip().lower() in _QUERY_COLUMNS for name in reader.fieldnames
        ):
            raise WorkloadParseError(
                str(path),
                f"CSV export has no query column (looked for {', '.join(_QUERY_COLUMNS)})",
            )
        return [row for row in reader if any(value for value in row.values())]

    @staticmethod
    def _to_statement(row: Mapping[str, Any]) -> RawStatement | None:
        query = _pick(row, _QUERY_COLUMNS)
        if not query or not str(query).strip():
            return None
        calls = max(1, _as_int(_pick(row, _CALLS_COLUMNS), default=1))
        total = _pick(row, _TOTAL_TIME_COLUMNS)
        if total is not None:
            total_ms = _as_float(total)
        else:
            total_ms = _as_float(_pick(row, _MEAN_TIME_COLUMNS)) * calls
        return RawStatement(
            sql=str(query),
            calls=calls,
            total_exec_time_ms=total_ms,
            rows=_as_int(_pick(row, _ROWS_COLUMNS)),
        )
