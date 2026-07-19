"""Tests for spatial predicate extraction."""

from __future__ import annotations

import pytest

from spatial_index_advisor.analysis import analyze_statement


def analyse(sql: str):
    """Analyse a statement under a fixed fingerprint."""
    return analyze_statement(sql, "fp")


def test_resolves_aliases_to_table_names() -> None:
    analysis = analyse(
        "SELECT v.id FROM public.vehicles v JOIN zones z ON v.geom && z.geom WHERE z.kind = 1"
    )
    assert analysis.tables == ("public.vehicles", "zones")
    assert analysis.aliases["v"] == "public.vehicles"
    assert analysis.aliases["z"] == "zones"


def test_bbox_operator_is_sargable_on_both_sides() -> None:
    analysis = analyse("SELECT 1 FROM a JOIN b ON a.geom && b.geom")
    (predicate,) = analysis.spatial_predicates
    assert predicate.name == "&&"
    assert predicate.kind == "operator"
    assert predicate.sargable
    assert {column.qualified for column in predicate.columns} == {"a.geom", "b.geom"}


def test_st_dwithin_captures_radius() -> None:
    analysis = analyse(
        "SELECT 1 FROM t WHERE ST_DWithin(geom, ST_SetSRID(ST_MakePoint(1, 2), 3857), 250)"
    )
    (predicate,) = analysis.spatial_predicates
    assert predicate.name == "ST_DWithin"
    assert predicate.sargable
    assert predicate.radius == 250.0
    assert [column.qualified for column in predicate.columns] == ["t.geom"]


def test_constructor_arguments_are_not_treated_as_indexable_columns() -> None:
    analysis = analyse(
        "SELECT 1 FROM t WHERE ST_Intersects(t.geom, ST_SetSRID(ST_MakePoint(t.lon, t.lat), 4326))"
    )
    (predicate,) = analysis.spatial_predicates
    assert [column.column for column in predicate.columns] == ["geom"]


def test_st_distance_comparison_is_not_sargable_and_suggests_dwithin() -> None:
    analysis = analyse("SELECT 1 FROM t WHERE ST_Distance(geom, $1) < 100")
    (predicate,) = analysis.spatial_predicates
    assert not predicate.sargable
    assert predicate.rewrite_hint is not None
    assert "ST_DWithin" in predicate.rewrite_hint


def test_distance_operator_in_where_is_not_sargable() -> None:
    analysis = analyse("SELECT 1 FROM t WHERE geom <-> $1 < 100")
    (predicate,) = analysis.spatial_predicates
    assert not predicate.sargable
    assert "ORDER BY" in predicate.reason


def test_transform_on_the_column_defeats_the_index() -> None:
    analysis = analyse("SELECT 1 FROM t WHERE ST_Intersects(ST_Transform(geom, 3857), $1)")
    (predicate,) = analysis.spatial_predicates
    assert not predicate.sargable
    assert predicate.wrapping_functions == ("ST_Transform",)
    assert predicate.transform_srid == 3857


def test_transform_srid_is_none_when_parameterised() -> None:
    analysis = analyse("SELECT 1 FROM t WHERE ST_Intersects(ST_Transform(geom, $1), $2)")
    (predicate,) = analysis.spatial_predicates
    assert predicate.transform_srid is None


def test_negated_predicate_is_not_sargable() -> None:
    analysis = analyse("SELECT 1 FROM t WHERE NOT ST_Intersects(geom, $1)")
    (predicate,) = analysis.spatial_predicates
    assert predicate.negated
    assert not predicate.sargable


@pytest.mark.parametrize("operator", ["<->", "<#>"])
def test_knn_order_by_is_detected_with_its_limit(operator: str) -> None:
    analysis = analyse(f"SELECT id FROM pings ORDER BY geom {operator} $1 LIMIT 7")
    (knn,) = analysis.knn
    assert knn.column.qualified == "pings.geom"
    assert knn.operator == operator
    assert knn.limit == 7
    assert analysis.limit == 7
    assert not analysis.spatial_predicates


def test_scalar_filters_record_constants_and_parameters_differently() -> None:
    analysis = analyse(
        "SELECT 1 FROM t WHERE ST_Intersects(geom, $1) AND status = 'active' AND fleet_id = $2"
    )
    filters = {f.column.column: f for f in analysis.scalar_filters}
    assert filters["status"].is_constant
    assert filters["status"].predicate_sql == "status = 'active'"
    assert not filters["fleet_id"].is_constant


def test_bare_boolean_column_becomes_an_equality_filter() -> None:
    analysis = analyse("SELECT 1 FROM t WHERE ST_Intersects(geom, $1) AND active")
    filters = {f.column.column: f for f in analysis.scalar_filters}
    assert filters["active"].predicate_sql == "active = TRUE"


def test_is_null_filters_are_captured() -> None:
    analysis = analyse(
        "SELECT 1 FROM t WHERE ST_Intersects(geom, $1) AND retired_at IS NOT NULL"
    )
    filters = {f.column.column: f for f in analysis.scalar_filters}
    assert filters["retired_at"].predicate_sql == "retired_at IS NOT NULL"


def test_filters_under_or_are_ignored() -> None:
    analysis = analyse(
        "SELECT 1 FROM t WHERE ST_Intersects(geom, $1) AND (status = 'a' OR status = 'b')"
    )
    assert [f.column.column for f in analysis.scalar_filters] == []


def test_geometry_columns_are_not_reported_as_scalar_filters() -> None:
    analysis = analyse("SELECT 1 FROM t WHERE geom && $1 AND geom IS NOT NULL")
    assert analysis.scalar_filters == ()


def test_join_conditions_are_searched_for_predicates() -> None:
    analysis = analyse(
        "SELECT 1 FROM geofences g JOIN pings p ON ST_Contains(g.geom, p.geom) WHERE g.active"
    )
    (predicate,) = analysis.spatial_predicates
    assert predicate.sargable
    assert {c.qualified for c in predicate.columns} == {"geofences.geom", "pings.geom"}


def test_unqualified_column_in_multi_table_statement_is_unresolved() -> None:
    analysis = analyse("SELECT 1 FROM a, b WHERE ST_Intersects(geom, $1)")
    (predicate,) = analysis.spatial_predicates
    assert predicate.columns[0].table is None


def test_geometry_columns_for_filters_by_table() -> None:
    analysis = analyse("SELECT 1 FROM a JOIN b ON a.geom && b.shape")
    assert analysis.geometry_columns_for("a") == {"geom"}
    assert analysis.geometry_columns_for("b") == {"shape"}


def test_parse_failure_is_reported_not_raised() -> None:
    analysis = analyse("SELECT FROM WHERE ((((")
    assert analysis.parse_error is not None
    assert not analysis.is_spatial


def test_empty_statement_is_reported() -> None:
    assert analyse("   ").parse_error == "empty statement"


def test_non_spatial_statement_produces_no_predicates() -> None:
    analysis = analyse("SELECT id FROM vehicles WHERE fleet_id = 3 ORDER BY plate")
    assert not analysis.is_spatial
    assert analysis.parse_error is None
