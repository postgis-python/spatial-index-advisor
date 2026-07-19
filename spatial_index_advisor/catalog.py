"""Loading and validation of the JSON catalog snapshot.

The snapshot is the only source of table statistics the recommendation engine
reads. Keeping it a plain file — rather than a live connection — is what makes
the engine deterministic and testable offline; ``--collect`` (see
:mod:`spatial_index_advisor.collector`) produces one from a real database.

Snapshot shape::

    {
      "database": "fleet",
      "collected_at": "2026-07-14T09:12:03Z",
      "postgis_version": "3.4.2",
      "tables": [
        {
          "name": "public.vehicle_positions",
          "row_count": 412000000,
          "table_bytes": 61000000000,
          "inserts": 412000000, "updates": 0, "deletes": 0,
          "append_only": true,
          "column_correlation": {"recorded_at": 0.999},
          "geometry_columns": [
            {"name": "geom", "geometry_type": "POINT", "srid": 4326,
             "avg_bbox_width": 0.0, "avg_bbox_height": 0.0,
             "extent_width": 0.62, "extent_height": 0.41, "correlation": 0.97}
          ],
          "indexes": [
            {"name": "vehicle_positions_pkey", "method": "btree",
             "columns": ["id"], "unique": true, "size_bytes": 9100000000}
          ]
        }
      ]
    }

Every field except ``name``, ``row_count`` and ``table_bytes`` is optional; the
engine degrades to a lower confidence level rather than failing when a statistic
is missing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .errors import CatalogError
from .models import CatalogSnapshot, ExistingIndex, GeometryColumn, TableStats


def _require(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise CatalogError(f"{where}: missing required key {key!r}")
    return mapping[key]


def _as_int(value: Any, where: str, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CatalogError(f"{where}: {key!r} must be a number, got {type(value).__name__}")
    if value < 0:
        raise CatalogError(f"{where}: {key!r} must not be negative")
    return int(value)


def _as_optional_float(value: Any, where: str, key: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CatalogError(f"{where}: {key!r} must be a number or null")
    return float(value)


def _parse_geometry_column(payload: Mapping[str, Any], where: str) -> GeometryColumn:
    name = str(_require(payload, "name", where))
    correlation = _as_optional_float(payload.get("correlation"), where, "correlation")
    if correlation is not None and not -1.0 <= correlation <= 1.0:
        raise CatalogError(f"{where}: 'correlation' must be between -1 and 1")
    return GeometryColumn(
        name=name,
        geometry_type=str(payload.get("geometry_type") or "GEOMETRY"),
        srid=_as_int(payload.get("srid", 0), where, "srid"),
        avg_bbox_width=_as_optional_float(payload.get("avg_bbox_width"), where, "avg_bbox_width"),
        avg_bbox_height=_as_optional_float(
            payload.get("avg_bbox_height"), where, "avg_bbox_height"
        ),
        extent_width=_as_optional_float(payload.get("extent_width"), where, "extent_width"),
        extent_height=_as_optional_float(payload.get("extent_height"), where, "extent_height"),
        correlation=correlation,
    )


def _parse_index(payload: Mapping[str, Any], where: str) -> ExistingIndex:
    columns = payload.get("columns")
    if not isinstance(columns, list) or not columns:
        raise CatalogError(f"{where}: 'columns' must be a non-empty array")
    predicate = payload.get("predicate")
    if predicate is not None and not isinstance(predicate, str):
        raise CatalogError(f"{where}: 'predicate' must be a string or null")
    return ExistingIndex(
        name=str(_require(payload, "name", where)),
        method=str(payload.get("method") or "btree"),
        columns=tuple(str(column) for column in columns),
        predicate=predicate,
        is_unique=bool(payload.get("unique", False)),
        size_bytes=(
            None
            if payload.get("size_bytes") is None
            else _as_int(payload["size_bytes"], where, "size_bytes")
        ),
        definition=payload.get("definition"),
    )


def _parse_table(payload: Mapping[str, Any], where: str) -> TableStats:
    name = str(_require(payload, "name", where))
    correlation_payload = payload.get("column_correlation")
    correlation_payload = {} if correlation_payload is None else correlation_payload
    if not isinstance(correlation_payload, Mapping):
        raise CatalogError(f"{where}: 'column_correlation' must be an object")
    geometry_payload = payload.get("geometry_columns")
    geometry_payload = [] if geometry_payload is None else geometry_payload
    if not isinstance(geometry_payload, list):
        raise CatalogError(f"{where}: 'geometry_columns' must be an array")
    index_payload = payload.get("indexes")
    index_payload = [] if index_payload is None else index_payload
    if not isinstance(index_payload, list):
        raise CatalogError(f"{where}: 'indexes' must be an array")

    inserts = _as_int(payload.get("inserts", 0), where, "inserts")
    updates = _as_int(payload.get("updates", 0), where, "updates")
    deletes = _as_int(payload.get("deletes", 0), where, "deletes")
    declared_append_only = payload.get("append_only")
    append_only = (
        bool(declared_append_only)
        if declared_append_only is not None
        else (inserts > 0 and (updates + deletes) / (inserts + updates + deletes) < 0.01)
    )

    return TableStats(
        name=name,
        row_count=_as_int(_require(payload, "row_count", where), where, "row_count"),
        table_bytes=_as_int(_require(payload, "table_bytes", where), where, "table_bytes"),
        geometry_columns=tuple(
            _parse_geometry_column(column, f"{where}.geometry_columns[{i}]")
            for i, column in enumerate(_as_mappings(geometry_payload, where, "geometry_columns"))
        ),
        indexes=tuple(
            _parse_index(index, f"{where}.indexes[{i}]")
            for i, index in enumerate(_as_mappings(index_payload, where, "indexes"))
        ),
        column_correlation={
            str(key): float(value)
            for key, value in correlation_payload.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        },
        append_only=append_only,
        inserts=inserts,
        updates=updates,
        deletes=deletes,
    )


def _as_mappings(items: list[Any], where: str, key: str) -> list[Mapping[str, Any]]:
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise CatalogError(f"{where}.{key}[{index}]: expected an object")
    return items


def parse_catalog(payload: Any) -> CatalogSnapshot:
    """Validate a decoded snapshot document and build a :class:`CatalogSnapshot`.

    Raises:
        CatalogError: on any structural problem, naming the offending path.
    """
    if not isinstance(payload, Mapping):
        raise CatalogError("catalog: top level must be a JSON object")
    tables_payload = _require(payload, "tables", "catalog")

    entries: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(tables_payload, Mapping):
        for key, value in tables_payload.items():
            if not isinstance(value, Mapping):
                raise CatalogError(f"catalog.tables[{key!r}]: expected an object")
            entries.append((f"catalog.tables[{key!r}]", {"name": key, **value}))
    elif isinstance(tables_payload, list):
        for index, value in enumerate(tables_payload):
            if not isinstance(value, Mapping):
                raise CatalogError(f"catalog.tables[{index}]: expected an object")
            entries.append((f"catalog.tables[{index}]", value))
    else:
        raise CatalogError("catalog: 'tables' must be an array or an object")

    if not entries:
        raise CatalogError("catalog: 'tables' is empty; nothing to advise on")

    tables: dict[str, TableStats] = {}
    for where, entry in entries:
        table = _parse_table(entry, where)
        if table.name in tables:
            raise CatalogError(f"{where}: duplicate table name {table.name!r}")
        tables[table.name] = table

    return CatalogSnapshot(
        tables=tables,
        database=payload.get("database"),
        collected_at=payload.get("collected_at"),
        postgis_version=payload.get("postgis_version"),
    )


def load_catalog(path: Path) -> CatalogSnapshot:
    """Read and validate a catalog snapshot from disk.

    Raises:
        CatalogError: if the file is missing, is not JSON, or fails validation.
    """
    if not path.is_file():
        raise CatalogError(f"{path}: catalog snapshot does not exist")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise CatalogError(f"{path}: cannot read catalog snapshot: {error}") from error
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise CatalogError(f"{path}:{error.lineno}: invalid JSON: {error.msg}") from error
    try:
        return parse_catalog(payload)
    except CatalogError as error:
        raise CatalogError(f"{path}: {error}") from error


def dump_catalog(snapshot: CatalogSnapshot) -> dict[str, Any]:
    """Serialise a snapshot back to the on-disk document shape."""
    return {
        "database": snapshot.database,
        "collected_at": snapshot.collected_at,
        "postgis_version": snapshot.postgis_version,
        "tables": [
            {
                "name": table.name,
                "row_count": table.row_count,
                "table_bytes": table.table_bytes,
                "inserts": table.inserts,
                "updates": table.updates,
                "deletes": table.deletes,
                "append_only": table.append_only,
                "column_correlation": table.column_correlation,
                "geometry_columns": [
                    {
                        "name": column.name,
                        "geometry_type": column.geometry_type,
                        "srid": column.srid,
                        "avg_bbox_width": column.avg_bbox_width,
                        "avg_bbox_height": column.avg_bbox_height,
                        "extent_width": column.extent_width,
                        "extent_height": column.extent_height,
                        "correlation": column.correlation,
                    }
                    for column in table.geometry_columns
                ],
                "indexes": [
                    {
                        "name": index.name,
                        "method": index.method,
                        "columns": list(index.columns),
                        "unique": index.is_unique,
                        "predicate": index.predicate,
                        "size_bytes": index.size_bytes,
                        "definition": index.definition,
                    }
                    for index in table.indexes
                ],
            }
            for table in snapshot.tables.values()
        ],
    }
