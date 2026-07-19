"""Tests for the live-database collector, driven by a fake DB-API connection.

No PostgreSQL server is involved: the fake matches queries by a distinctive
fragment and replays recorded rows.
"""

from __future__ import annotations

from typing import Any, Sequence

import pytest

from spatial_index_advisor.collector import (
    AVERAGE_BBOX_QUERY,
    CORRELATION_QUERY,
    EXTENT_QUERY,
    GEOMETRY_COLUMNS_QUERY,
    INDEX_QUERY,
    TABLE_ACTIVITY_QUERY,
    VERSION_QUERY,
    collect_snapshot,
    parse_box2d,
)
from spatial_index_advisor.errors import CollectorError


class FakeCursor:
    """A DB-API cursor that answers from a fragment-keyed table of rows."""

    def __init__(self, responses: dict[str, list[Any]], failures: set[str] | None = None):
        self._responses = responses
        self._failures = failures or set()
        self._rows: list[Any] = []
        self.closed = False
        self.executed: list[str] = []

    def execute(self, query: str, params: Sequence[Any] | None = None) -> None:
        self.executed.append(query)
        for fragment, rows in self._responses.items():
            if fragment in query:
                if fragment in self._failures:
                    raise RuntimeError(f"simulated failure for {fragment}")
                self._rows = list(rows)
                return
        self._rows = []

    def fetchall(self) -> list[Any]:
        return self._rows

    def fetchone(self) -> Any:
        return self._rows[0] if self._rows else None

    def close(self) -> None:
        self.closed = True


class FakeConnection:
    """A DB-API connection handing out :class:`FakeCursor` instances."""

    def __init__(self, responses: dict[str, list[Any]], failures: set[str] | None = None):
        self._responses = responses
        self._failures = failures
        self.cursors: list[FakeCursor] = []

    def cursor(self) -> FakeCursor:
        cursor = FakeCursor(self._responses, self._failures)
        self.cursors.append(cursor)
        return cursor

    def close(self) -> None:
        return None


def catalog_rows() -> dict[str, list[Any]]:
    """A minimal but complete set of catalog query responses."""
    return {
        "postgis_lib_version": [("fleet", "3.4.2")],
        "FROM geometry_columns": [
            ("public", "pings", "geom", "POINT", 3857, 88_000_000, 14_000_000_000)
        ],
        "FROM pg_stat_user_tables": [("public", "pings", 88_000_000, 210_000, 0)],
        "FROM pg_stats": [
            ("public", "pings", "geom", 0.91),
            ("public", "pings", "created_at", 0.998),
        ],
        "FROM pg_index": [
            (
                "public",
                "pings",
                "pings_pkey",
                "btree",
                True,
                1_980_000_000,
                None,
                "CREATE UNIQUE INDEX pings_pkey ON public.pings USING btree (id)",
                ["id"],
            )
        ],
        "ST_EstimatedExtent": [("BOX(520000 160000,582000 201000)",)],
        "ST_XMax": [(12.5, 9.25)],
    }


def test_collect_builds_a_complete_snapshot() -> None:
    connection = FakeConnection(catalog_rows())
    snapshot = collect_snapshot(connection)

    assert snapshot.database == "fleet"
    assert snapshot.postgis_version == "3.4.2"
    assert snapshot.collected_at is not None
    table = snapshot.tables["public.pings"]
    assert table.row_count == 88_000_000
    assert table.append_only is True
    assert table.inserts == 88_000_000

    geometry = table.geometry_column("geom")
    assert geometry.is_point
    assert geometry.srid == 3857
    assert geometry.correlation == pytest.approx(0.91)
    assert geometry.extent_width == pytest.approx(62000.0)
    assert geometry.extent_height == pytest.approx(41000.0)
    assert geometry.avg_bbox_width == pytest.approx(12.5)

    (index,) = table.indexes
    assert index.name == "pings_pkey"
    assert index.is_unique
    assert index.columns == ("id",)
    assert table.column_correlation["created_at"] == pytest.approx(0.998)


def test_append_only_is_false_when_updates_are_significant() -> None:
    rows = catalog_rows()
    rows["FROM pg_stat_user_tables"] = [("public", "pings", 1_000_000, 500_000, 0)]
    snapshot = collect_snapshot(FakeConnection(rows))
    assert not snapshot.tables["public.pings"].append_only


def test_sampling_can_be_disabled() -> None:
    connection = FakeConnection(catalog_rows())
    snapshot = collect_snapshot(connection, sample_geometry=False)
    geometry = snapshot.tables["public.pings"].geometry_column("geom")
    assert geometry.avg_bbox_width is None
    assert not any("ST_XMax" in query for query in connection.cursors[0].executed)


def test_optional_statistics_degrade_instead_of_failing() -> None:
    connection = FakeConnection(catalog_rows(), failures={"ST_EstimatedExtent", "ST_XMax"})
    geometry = collect_snapshot(connection).tables["public.pings"].geometry_column("geom")
    assert geometry.extent_width is None
    assert geometry.avg_bbox_width is None


def test_a_failing_required_query_raises() -> None:
    connection = FakeConnection(catalog_rows(), failures={"FROM pg_index"})
    with pytest.raises(CollectorError, match="query failed"):
        collect_snapshot(connection)


def test_a_database_without_geometry_columns_is_an_error() -> None:
    rows = catalog_rows()
    rows["FROM geometry_columns"] = []
    with pytest.raises(CollectorError, match="no geometry columns"):
        collect_snapshot(FakeConnection(rows))


def test_no_schemas_is_an_error() -> None:
    with pytest.raises(CollectorError, match="no schemas"):
        collect_snapshot(FakeConnection(catalog_rows()), schemas=[])


def test_the_cursor_is_always_closed() -> None:
    connection = FakeConnection(catalog_rows())
    collect_snapshot(connection)
    assert connection.cursors[0].closed

    failing = FakeConnection(catalog_rows(), failures={"FROM pg_index"})
    with pytest.raises(CollectorError):
        collect_snapshot(failing)
    assert failing.cursors[0].closed


def test_the_snapshot_round_trips_through_the_catalog_loader() -> None:
    import json

    from spatial_index_advisor.catalog import dump_catalog, parse_catalog

    snapshot = collect_snapshot(FakeConnection(catalog_rows()))
    reloaded = parse_catalog(json.loads(json.dumps(dump_catalog(snapshot))))
    assert reloaded.tables["public.pings"] == snapshot.tables["public.pings"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("BOX(0 0,10 20)", (10.0, 20.0)),
        ("BOX(-1.5 -2.5,1.5 2.5)", (3.0, 5.0)),
        ("BOX(1e2 0,2e2 0)", (100.0, 0.0)),
        (None, None),
        ("", None),
        ("not a box", None),
    ],
)
def test_parse_box2d(text, expected) -> None:
    assert parse_box2d(text) == expected


def test_every_query_constant_is_exercised() -> None:
    connection = FakeConnection(catalog_rows())
    collect_snapshot(connection)
    executed = "\n".join(connection.cursors[0].executed)
    for query in (
        VERSION_QUERY,
        GEOMETRY_COLUMNS_QUERY,
        TABLE_ACTIVITY_QUERY,
        INDEX_QUERY,
        CORRELATION_QUERY,
        EXTENT_QUERY,
    ):
        assert query.strip().splitlines()[0].strip() in executed
    assert "ST_XMax" in executed and "TABLESAMPLE" in AVERAGE_BBOX_QUERY
