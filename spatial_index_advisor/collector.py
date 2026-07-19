"""Optional live-database collector.

The recommendation engine never talks to a database. This module exists solely to
*produce* the JSON snapshot the engine consumes, so that a real deployment does
not have to assemble one by hand.

:func:`collect_snapshot` takes any DB-API 2.0 connection, which keeps it testable
with a fake; :func:`connect` is the thin wrapper that imports ``psycopg`` and is
the only part of the package that requires a driver at all.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Any, Final, Iterable, Protocol, Sequence

from .costmodel import BLOCK_SIZE
from .errors import CollectorError
from .models import CatalogSnapshot, ExistingIndex, GeometryColumn, TableStats

DEFAULT_SCHEMAS: Final[tuple[str, ...]] = ("public",)

_NUMBER: Final[str] = r"-?[\d.eE+-]+"

_BOX_RE: Final[re.Pattern[str]] = re.compile(
    rf"BOX\(\s*({_NUMBER})\s+({_NUMBER})\s*,\s*({_NUMBER})\s+({_NUMBER})\s*\)",
    re.IGNORECASE,
)

#: A box2d rendered through ``ST_AsText`` comes back as a closed five-vertex
#: polygon rather than ``BOX(...)``. Accepting it too means an extent is never
#: silently dropped just because a caller wrapped the value differently.
_POLYGON_RE: Final[re.Pattern[str]] = re.compile(
    rf"POLYGON\(\(\s*(?:{_NUMBER}\s+{_NUMBER}\s*,\s*){{2}}({_NUMBER})\s+({_NUMBER})",
    re.IGNORECASE,
)
_POLYGON_ORIGIN_RE: Final[re.Pattern[str]] = re.compile(
    rf"POLYGON\(\(\s*({_NUMBER})\s+({_NUMBER})", re.IGNORECASE
)


class Cursor(Protocol):
    """The subset of the DB-API cursor interface this module uses."""

    def execute(self, query: str, params: Sequence[Any] | None = ...) -> Any: ...

    def fetchall(self) -> list[Any]: ...

    def fetchone(self) -> Any: ...

    def close(self) -> None: ...


class Connection(Protocol):
    """The subset of the DB-API connection interface this module uses."""

    def cursor(self) -> Cursor: ...


VERSION_QUERY: Final[str] = "SELECT current_database(), postgis_lib_version()"

GEOMETRY_COLUMNS_QUERY: Final[str] = """
SELECT gc.f_table_schema,
       gc.f_table_name,
       gc.f_geometry_column,
       gc.type,
       gc.srid,
       c.reltuples::bigint,
       -- pg_relation_size, not pg_table_size: the cost model turns this into a
       -- heap page count for sequential-scan costing, and a seq scan reads only
       -- the main fork. pg_table_size adds TOAST, which for a table of large
       -- polygons can be most of the bytes and none of the scanned pages.
       pg_relation_size(c.oid)
FROM geometry_columns gc
JOIN pg_namespace n ON n.nspname = gc.f_table_schema
JOIN pg_class c ON c.relname = gc.f_table_name AND c.relnamespace = n.oid
WHERE gc.f_table_schema = ANY(%s)
ORDER BY 1, 2, 3
"""

TABLE_ACTIVITY_QUERY: Final[str] = """
SELECT schemaname, relname, n_tup_ins, n_tup_upd, n_tup_del
FROM pg_stat_user_tables
WHERE schemaname = ANY(%s)
"""

INDEX_QUERY: Final[str] = """
SELECT n.nspname,
       t.relname,
       i.relname,
       am.amname,
       idx.indisunique,
       pg_relation_size(i.oid),
       pg_get_expr(idx.indpred, idx.indrelid),
       pg_get_indexdef(i.oid),
       -- indnkeyatts, not indnatts: on PostgreSQL 11+ indnatts also counts
       -- INCLUDE payload columns, which are not part of the index key. Counting
       -- them would make "(a) INCLUDE (b)" look like a two-column index and so
       -- break both duplicate and leading-prefix redundancy detection.
       ARRAY(
         SELECT pg_get_indexdef(idx.indexrelid, k + 1, true)
         FROM generate_series(0, idx.indnkeyatts - 1) AS k
       )
FROM pg_index idx
JOIN pg_class i ON i.oid = idx.indexrelid
JOIN pg_class t ON t.oid = idx.indrelid
JOIN pg_namespace n ON n.oid = t.relnamespace
JOIN pg_am am ON am.oid = i.relam
WHERE n.nspname = ANY(%s)
ORDER BY 1, 2, 3
"""

CORRELATION_QUERY: Final[str] = """
SELECT schemaname, tablename, attname, correlation
FROM pg_stats
WHERE schemaname = ANY(%s) AND correlation IS NOT NULL
"""

#: ``ST_EstimatedExtent`` returns a ``box2d``. Casting it to text yields the
#: ``BOX(x1 y1,x2 y2)`` form :func:`parse_box2d` reads. Do not wrap this in
#: ``ST_AsText``: that renders the box as a five-vertex ``POLYGON`` instead, which
#: parses to nothing and silently discards the extent.
EXTENT_QUERY: Final[str] = "SELECT ST_EstimatedExtent(%s, %s, %s)::text"

AVERAGE_BBOX_QUERY: Final[str] = """
SELECT avg(ST_XMax({column}) - ST_XMin({column})),
       avg(ST_YMax({column}) - ST_YMin({column}))
FROM (SELECT {column} FROM {table} TABLESAMPLE SYSTEM (%s) WHERE {column} IS NOT NULL) AS s
"""

#: Percentage of pages sampled when measuring average feature size.
SAMPLE_PERCENT: Final[float] = 1.0

#: Fewest heap pages a sample should be expected to touch. ``TABLESAMPLE SYSTEM``
#: accepts each page independently, so on a small table a flat 1% often selects
#: nothing at all -- a 250-page table comes back empty roughly 8% of the time,
#: silently dropping the feature-size statistic. The rate is raised on small
#: tables so the expected page count clears this floor; large tables, where 1%
#: is already plenty, are unaffected.
MIN_SAMPLE_PAGES: Final[int] = 32

#: Fraction of writes that may be updates or deletes for a table to count as
#: append-mostly.
APPEND_ONLY_WRITE_RATIO: Final[float] = 0.01


def parse_box2d(text: str | None) -> tuple[float, float] | None:
    """Extract ``(width, height)`` from an extent string.

    Accepts the ``BOX(x1 y1,x2 y2)`` form a ``box2d`` casts to, and also the
    ``POLYGON((...))`` form ``ST_AsText`` produces from the same value, whose
    third vertex is the opposite corner.
    """
    if not text:
        return None
    match = _BOX_RE.search(text)
    if match is not None:
        x_min, y_min, x_max, y_max = (float(value) for value in match.groups())
        return abs(x_max - x_min), abs(y_max - y_min)

    origin = _POLYGON_ORIGIN_RE.search(text)
    opposite = _POLYGON_RE.search(text)
    if origin is None or opposite is None:
        return None
    x_min, y_min = (float(value) for value in origin.groups())
    x_max, y_max = (float(value) for value in opposite.groups())
    return abs(x_max - x_min), abs(y_max - y_min)


def _query(cursor: Cursor, sql: str, params: Sequence[Any] | None = None) -> list[Any]:
    try:
        cursor.execute(sql, params)
        return cursor.fetchall()
    except Exception as error:  # noqa: BLE001 - driver exceptions are not a fixed type
        raise CollectorError(f"query failed: {error}") from error


def _query_one(cursor: Cursor, sql: str, params: Sequence[Any] | None = None) -> Any:
    try:
        cursor.execute(sql, params)
        return cursor.fetchone()
    except Exception as error:  # noqa: BLE001 - driver exceptions are not a fixed type
        raise CollectorError(f"query failed: {error}") from error


def _safe_optional_row(cursor: Cursor, sql: str, params: Sequence[Any]) -> Any:
    """Run a query whose failure is not fatal (missing stats, permissions)."""
    try:
        cursor.execute(sql, params)
        return cursor.fetchone()
    except Exception:  # noqa: BLE001 - degrade to "statistic unavailable"
        return None


def collect_snapshot(
    connection: Connection,
    schemas: Iterable[str] = DEFAULT_SCHEMAS,
    sample_geometry: bool = True,
) -> CatalogSnapshot:
    """Build a :class:`CatalogSnapshot` from a live connection.

    Args:
        connection: any DB-API 2.0 connection to a PostGIS-enabled database.
        schemas: schemas to inspect.
        sample_geometry: when true, run a ``TABLESAMPLE`` query per geometry
            column to measure average feature size. This reads data, so it can be
            disabled on very large or heavily loaded systems.

    Raises:
        CollectorError: if a required catalog query fails.
    """
    schema_list = list(schemas)
    if not schema_list:
        raise CollectorError("no schemas given to collect")

    cursor = connection.cursor()
    try:
        version_row = _query_one(cursor, VERSION_QUERY)
        database = version_row[0] if version_row else None
        postgis_version = version_row[1] if version_row and len(version_row) > 1 else None

        geometry_rows = _query(cursor, GEOMETRY_COLUMNS_QUERY, [schema_list])
        if not geometry_rows:
            raise CollectorError(
                f"no geometry columns found in schema(s) {', '.join(schema_list)}"
            )

        activity = {
            (row[0], row[1]): (int(row[2] or 0), int(row[3] or 0), int(row[4] or 0))
            for row in _query(cursor, TABLE_ACTIVITY_QUERY, [schema_list])
        }
        correlations: dict[tuple[str, str], dict[str, float]] = {}
        for schema, table, column, correlation in _query(
            cursor, CORRELATION_QUERY, [schema_list]
        ):
            if correlation is None:
                continue
            correlations.setdefault((schema, table), {})[column] = float(correlation)

        indexes: dict[tuple[str, str], list[ExistingIndex]] = {}
        for row in _query(cursor, INDEX_QUERY, [schema_list]):
            schema, table, name, method, unique, size, predicate, definition, columns = row
            indexes.setdefault((schema, table), []).append(
                ExistingIndex(
                    name=name,
                    method=method,
                    columns=tuple(str(column) for column in (columns or ())),
                    predicate=predicate,
                    is_unique=bool(unique),
                    size_bytes=int(size) if size is not None else None,
                    definition=definition,
                )
            )

        geometry_by_table: dict[tuple[str, str], list[GeometryColumn]] = {}
        sizes: dict[tuple[str, str], tuple[int, int]] = {}
        for row in geometry_rows:
            schema, table, column, geometry_type, srid, reltuples, table_bytes = row
            key = (schema, table)
            sizes[key] = (max(0, int(reltuples or 0)), int(table_bytes or 0))
            extent = parse_box2d(
                _first(_safe_optional_row(cursor, EXTENT_QUERY, [schema, table, column]))
            )
            average = (
                _average_bbox(cursor, schema, table, column, sizes[key][1])
                if sample_geometry
                else None
            )
            table_correlations = correlations.get(key, {})
            geometry_by_table.setdefault(key, []).append(
                GeometryColumn(
                    name=column,
                    geometry_type=str(geometry_type or "GEOMETRY"),
                    srid=int(srid or 0),
                    avg_bbox_width=None if average is None else average[0],
                    avg_bbox_height=None if average is None else average[1],
                    extent_width=None if extent is None else extent[0],
                    extent_height=None if extent is None else extent[1],
                    correlation=table_correlations.get(column),
                )
            )

        tables: dict[str, TableStats] = {}
        for key, columns in geometry_by_table.items():
            schema, table = key
            inserts, updates, deletes = activity.get(key, (0, 0, 0))
            total_writes = inserts + updates + deletes
            row_count, table_bytes = sizes[key]
            qualified = f"{schema}.{table}"
            tables[qualified] = TableStats(
                name=qualified,
                row_count=row_count,
                table_bytes=table_bytes,
                geometry_columns=tuple(columns),
                indexes=tuple(indexes.get(key, ())),
                column_correlation=correlations.get(key, {}),
                append_only=(
                    total_writes > 0
                    and (updates + deletes) / total_writes < APPEND_ONLY_WRITE_RATIO
                ),
                inserts=inserts,
                updates=updates,
                deletes=deletes,
            )

        return CatalogSnapshot(
            tables=tables,
            database=database,
            collected_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            postgis_version=postgis_version,
        )
    finally:
        cursor.close()


def _first(row: Any) -> Any:
    """First element of a fetched row, or None."""
    if row is None:
        return None
    return row[0] if isinstance(row, (list, tuple)) else row


def sample_percent(table_bytes: int) -> float:
    """Sampling rate for a table of ``table_bytes``, floored so small tables hit rows."""
    pages = max(1, table_bytes // BLOCK_SIZE)
    return min(100.0, max(SAMPLE_PERCENT, 100.0 * MIN_SAMPLE_PAGES / pages))


def _average_bbox(
    cursor: Cursor, schema: str, table: str, column: str, table_bytes: int = 0
) -> tuple[float, float] | None:
    """Measure the mean feature bounding box from a small table sample."""
    sql = AVERAGE_BBOX_QUERY.format(column=f'"{column}"', table=f'"{schema}"."{table}"')
    row = _safe_optional_row(cursor, sql, [sample_percent(table_bytes)])
    if not row or row[0] is None or row[1] is None:
        return None
    return float(row[0]), float(row[1])


def connect(dsn: str) -> Connection:
    """Open a connection using ``psycopg``.

    Raises:
        CollectorError: if psycopg is not installed or the connection fails.
    """
    try:
        import psycopg
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise CollectorError(
            "psycopg is required for --collect; install it with "
            "'pip install -r requirements.txt'"
        ) from error
    try:
        return psycopg.connect(dsn)
    except psycopg.Error as error:
        raise CollectorError(f"cannot connect to the database: {error}") from error
