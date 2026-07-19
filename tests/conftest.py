"""Shared fixtures and builders for the test suite.

Nothing here touches a database: catalogs are built in memory or read from
``examples/``, and workloads are assembled from SQL strings.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spatial_index_advisor.catalog import load_catalog
from spatial_index_advisor.models import (
    CatalogSnapshot,
    ExistingIndex,
    GeometryColumn,
    TableStats,
    Workload,
    WorkloadStatement,
)
from spatial_index_advisor.normalize import normalize_and_fingerprint

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"


@pytest.fixture(scope="session")
def examples_dir() -> Path:
    """Directory holding the shipped sample inputs."""
    return EXAMPLES


@pytest.fixture(scope="session")
def example_catalog() -> CatalogSnapshot:
    """The fleet-tracking catalog snapshot shipped in ``examples/``."""
    return load_catalog(EXAMPLES / "catalog.json")


def make_geometry(
    name: str = "geom",
    geometry_type: str = "POLYGON",
    correlation: float | None = 0.1,
    avg_bbox: tuple[float, float] | None = (900.0, 760.0),
    extent: tuple[float, float] | None = (62000.0, 41000.0),
) -> GeometryColumn:
    """Build a geometry column description for tests."""
    return GeometryColumn(
        name=name,
        geometry_type=geometry_type,
        srid=3857,
        avg_bbox_width=None if avg_bbox is None else avg_bbox[0],
        avg_bbox_height=None if avg_bbox is None else avg_bbox[1],
        extent_width=None if extent is None else extent[0],
        extent_height=None if extent is None else extent[1],
        correlation=correlation,
    )


def make_table(
    name: str = "public.things",
    row_count: int = 20_000_000,
    table_bytes: int = 4_000_000_000,
    geometry_columns: tuple[GeometryColumn, ...] | None = None,
    indexes: tuple[ExistingIndex, ...] = (),
    append_only: bool = False,
    inserts: int = 0,
    updates: int = 0,
    deletes: int = 0,
    column_correlation: dict[str, float] | None = None,
) -> TableStats:
    """Build a table description for tests."""
    return TableStats(
        name=name,
        row_count=row_count,
        table_bytes=table_bytes,
        geometry_columns=geometry_columns if geometry_columns is not None else (make_geometry(),),
        indexes=indexes,
        column_correlation=column_correlation or {},
        append_only=append_only,
        inserts=inserts,
        updates=updates,
        deletes=deletes,
    )


def make_catalog(*tables: TableStats) -> CatalogSnapshot:
    """Build a catalog snapshot from table descriptions."""
    return CatalogSnapshot(tables={table.name: table for table in tables}, database="test")


def make_workload(*statements: tuple[str, int]) -> Workload:
    """Build a workload from ``(sql, calls)`` pairs, normalizing as the parsers do."""
    built: list[WorkloadStatement] = []
    for sql, calls in statements:
        normalized, digest = normalize_and_fingerprint(sql)
        built.append(
            WorkloadStatement(
                fingerprint=digest,
                normalized_sql=normalized,
                sample_sql=sql,
                calls=calls,
                total_exec_time_ms=float(calls),
                rows=calls,
                source="test",
            )
        )
    return Workload(statements=tuple(built), sources=("test",))
