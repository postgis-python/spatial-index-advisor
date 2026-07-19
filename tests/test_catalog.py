"""Tests for catalog snapshot loading and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spatial_index_advisor.catalog import dump_catalog, load_catalog, parse_catalog
from spatial_index_advisor.errors import CatalogError

MINIMAL = {
    "tables": [
        {"name": "public.t", "row_count": 100, "table_bytes": 8192},
    ]
}


def test_example_catalog_loads_completely(example_catalog) -> None:
    assert set(example_catalog.tables) == {
        "public.vehicle_positions",
        "public.trips",
        "public.geofences",
        "public.driver_pings",
        "public.zone_visits",
    }
    positions = example_catalog.tables["public.vehicle_positions"]
    assert positions.append_only
    assert positions.geometry_column("geom").is_point
    assert positions.geometry_column("GEOM") is not None
    assert example_catalog.postgis_version == "3.4.2"


def test_minimal_table_gets_sensible_defaults() -> None:
    snapshot = parse_catalog(MINIMAL)
    table = snapshot.tables["public.t"]
    assert table.geometry_columns == ()
    assert table.indexes == ()
    assert table.append_only is False
    assert table.pages == 1


def test_tables_may_be_given_as_an_object() -> None:
    snapshot = parse_catalog({"tables": {"public.t": {"row_count": 5, "table_bytes": 8192}}})
    assert snapshot.tables["public.t"].row_count == 5


def test_append_only_is_derived_from_write_counters() -> None:
    snapshot = parse_catalog(
        {
            "tables": [
                {
                    "name": "a",
                    "row_count": 10,
                    "table_bytes": 8192,
                    "inserts": 1000,
                    "updates": 1,
                    "deletes": 0,
                },
                {
                    "name": "b",
                    "row_count": 10,
                    "table_bytes": 8192,
                    "inserts": 1000,
                    "updates": 500,
                    "deletes": 0,
                },
            ]
        }
    )
    assert snapshot.tables["a"].append_only
    assert not snapshot.tables["b"].append_only
    assert snapshot.tables["b"].write_ratio == pytest.approx(500 / 1500)


def test_explicit_append_only_flag_wins() -> None:
    snapshot = parse_catalog(
        {
            "tables": [
                {
                    "name": "a",
                    "row_count": 10,
                    "table_bytes": 8192,
                    "inserts": 10,
                    "updates": 10,
                    "append_only": True,
                }
            ]
        }
    )
    assert snapshot.tables["a"].append_only


def test_resolve_matches_qualified_and_bare_names(example_catalog) -> None:
    assert example_catalog.resolve("trips").name == "public.trips"
    assert example_catalog.resolve("PUBLIC.TRIPS").name == "public.trips"
    assert example_catalog.resolve("nope") is None


def test_resolve_is_ambiguous_across_schemas() -> None:
    snapshot = parse_catalog(
        {
            "tables": [
                {"name": "a.t", "row_count": 1, "table_bytes": 1},
                {"name": "b.t", "row_count": 1, "table_bytes": 1},
            ]
        }
    )
    assert snapshot.resolve("t") is None
    assert snapshot.resolve("a.t").name == "a.t"


def test_indexes_on_matches_leading_column(example_catalog) -> None:
    trips = example_catalog.tables["public.trips"]
    assert [i.name for i in trips.indexes_on("fleet_id", "btree")] == [
        "idx_trips_fleet_started",
        "idx_trips_fleet",
    ]
    assert trips.indexes_on("fleet_id", "gist") == []


def test_average_area_fraction_needs_both_extent_and_bbox(example_catalog) -> None:
    zone_geom = example_catalog.tables["public.zone_visits"].geometry_column("geom")
    assert zone_geom.average_area_fraction == pytest.approx((8000 * 6200) / (62000 * 41000))
    point_geom = example_catalog.tables["public.driver_pings"].geometry_column("geom")
    assert point_geom.average_area_fraction == 0.0


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "top level must be a JSON object"),
        ({}, "missing required key 'tables'"),
        ({"tables": []}, "is empty"),
        ({"tables": "x"}, "must be an array or an object"),
        ({"tables": [1]}, "expected an object"),
        ({"tables": {"a": 1}}, "expected an object"),
        ({"tables": [{"row_count": 1, "table_bytes": 1}]}, "missing required key 'name'"),
        ({"tables": [{"name": "a", "table_bytes": 1}]}, "missing required key 'row_count'"),
        ({"tables": [{"name": "a", "row_count": -1, "table_bytes": 1}]}, "not be negative"),
        ({"tables": [{"name": "a", "row_count": "x", "table_bytes": 1}]}, "must be a number"),
        (
            {"tables": [{"name": "a", "row_count": 1, "table_bytes": 1, "indexes": {}}]},
            "'indexes' must be an array",
        ),
        (
            {
                "tables": [
                    {
                        "name": "a",
                        "row_count": 1,
                        "table_bytes": 1,
                        "indexes": [{"name": "i", "columns": []}],
                    }
                ]
            },
            "non-empty array",
        ),
        (
            {
                "tables": [
                    {
                        "name": "a",
                        "row_count": 1,
                        "table_bytes": 1,
                        "geometry_columns": [{"name": "g", "correlation": 5}],
                    }
                ]
            },
            "between -1 and 1",
        ),
        (
            {
                "tables": [
                    {"name": "a", "row_count": 1, "table_bytes": 1, "column_correlation": []}
                ]
            },
            "must be an object",
        ),
        (
            {
                "tables": [
                    {"name": "a", "row_count": 1, "table_bytes": 1},
                    {"name": "a", "row_count": 1, "table_bytes": 1},
                ]
            },
            "duplicate table name",
        ),
    ],
)
def test_invalid_catalogs_are_rejected_with_a_useful_message(payload, message: str) -> None:
    with pytest.raises(CatalogError, match=message):
        parse_catalog(payload)


def test_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(CatalogError, match="does not exist"):
        load_catalog(tmp_path / "nope.json")


def test_invalid_json_reports_the_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{\n  'tables': []\n}", encoding="utf-8")
    with pytest.raises(CatalogError, match="invalid JSON"):
        load_catalog(path)


def test_validation_errors_are_prefixed_with_the_path(tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"tables": []}), encoding="utf-8")
    with pytest.raises(CatalogError, match=str(path)):
        load_catalog(path)


def test_dump_round_trips(example_catalog) -> None:
    reloaded = parse_catalog(json.loads(json.dumps(dump_catalog(example_catalog))))
    assert reloaded.tables.keys() == example_catalog.tables.keys()
    for name, table in example_catalog.tables.items():
        assert reloaded.tables[name] == table
