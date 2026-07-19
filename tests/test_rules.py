"""Tests for the individual recommendation rules.

Each test drives one rule from a hand-built catalog and workload so that the
trigger conditions and the guards against firing are both exercised.
"""

from __future__ import annotations

import pytest

from spatial_index_advisor import rules
from spatial_index_advisor.engine import analyse, analyse_statements
from spatial_index_advisor.models import ExistingIndex, Severity
from spatial_index_advisor.rules import build_context

from .conftest import make_catalog, make_geometry, make_table, make_workload

GIST_ON_GEOM = ExistingIndex(name="t_geom_gist", method="gist", columns=("geom",))


def context_for(catalog, workload):
    """Build a rule context for a catalog and workload pair."""
    return build_context(catalog, workload, analyse_statements(workload))


def kinds(recommendations) -> set[str]:
    return {recommendation.kind for recommendation in recommendations}


# --------------------------------------------------------------------------- #
# missing GiST
# --------------------------------------------------------------------------- #


def test_missing_gist_fires_on_an_unindexed_geometry_column() -> None:
    catalog = make_catalog(make_table(name="public.things"))
    workload = make_workload(("SELECT 1 FROM things WHERE ST_Intersects(geom, $1)", 5000))
    (recommendation,) = rules.rule_missing_gist(context_for(catalog, workload))
    assert recommendation.index_type == "GiST"
    assert recommendation.ddl == (
        "CREATE INDEX CONCURRENTLY idx_things_geom_gist "
        "ON public.things USING GIST (geom);"
    )
    assert recommendation.benefit is not None
    assert recommendation.benefit.calls == 5000
    assert recommendation.fingerprints == (workload.statements[0].fingerprint,)


def test_missing_gist_is_silent_when_a_gist_index_exists() -> None:
    catalog = make_catalog(make_table(indexes=(GIST_ON_GEOM,)))
    workload = make_workload(("SELECT 1 FROM things WHERE ST_Intersects(geom, $1)", 5000))
    assert rules.rule_missing_gist(context_for(catalog, workload)) == []


@pytest.mark.parametrize("method", ["gist", "spgist", "brin"])
def test_any_spatial_index_method_suppresses_the_finding(method: str) -> None:
    index = ExistingIndex(name="i", method=method, columns=("geom",))
    catalog = make_catalog(make_table(indexes=(index,)))
    workload = make_workload(("SELECT 1 FROM things WHERE ST_Intersects(geom, $1)", 5000))
    assert rules.rule_missing_gist(context_for(catalog, workload)) == []


def test_a_partial_index_does_not_suppress_the_finding() -> None:
    partial = ExistingIndex(
        name="i", method="gist", columns=("geom",), predicate="active = TRUE"
    )
    catalog = make_catalog(make_table(indexes=(partial,)))
    workload = make_workload(("SELECT 1 FROM things WHERE ST_Intersects(geom, $1)", 5000))
    assert len(rules.rule_missing_gist(context_for(catalog, workload))) == 1


def test_missing_gist_ignores_non_sargable_predicates() -> None:
    catalog = make_catalog(make_table())
    workload = make_workload(("SELECT 1 FROM things WHERE ST_Distance(geom, $1) < 5", 5000))
    assert rules.rule_missing_gist(context_for(catalog, workload)) == []


def test_missing_gist_ignores_small_tables() -> None:
    catalog = make_catalog(make_table(row_count=500, table_bytes=100_000))
    workload = make_workload(("SELECT 1 FROM things WHERE ST_Intersects(geom, $1)", 5000))
    assert rules.rule_missing_gist(context_for(catalog, workload)) == []


def test_missing_gist_ignores_tables_absent_from_the_catalog() -> None:
    catalog = make_catalog(make_table(name="public.other"))
    workload = make_workload(("SELECT 1 FROM elsewhere WHERE ST_Intersects(geom, $1)", 5000))
    assert rules.rule_missing_gist(context_for(catalog, workload)) == []


def test_missing_gist_ignores_columns_that_are_not_geometry() -> None:
    table = make_table(geometry_columns=(make_geometry(name="shape"),))
    catalog = make_catalog(table)
    workload = make_workload(("SELECT 1 FROM things WHERE ST_Intersects(geom, $1)", 5000))
    assert rules.rule_missing_gist(context_for(catalog, workload)) == []


# --------------------------------------------------------------------------- #
# KNN
#
# Only GiST registers <-> as an ordering operator, so KNN traffic must be sent
# to GiST. Recommending SP-GiST here — as this rule once did — produces an index
# the planner cannot use for ordering at all.
# --------------------------------------------------------------------------- #


def test_knn_traffic_is_sent_to_gist() -> None:
    table = make_table(
        geometry_columns=(make_geometry(geometry_type="POINT", avg_bbox=(0.0, 0.0)),),
    )
    workload = make_workload(("SELECT 1 FROM things ORDER BY geom <-> $1 LIMIT 5", 90_000))
    (recommendation,) = rules.rule_knn_index(context_for(make_catalog(table), workload))
    assert recommendation.index_type == "GiST"
    assert "USING GIST" in recommendation.ddl
    assert "SPGIST" not in recommendation.ddl.upper()


def test_knn_is_recommended_for_non_point_geometries_too() -> None:
    """GiST KNN is not restricted to points, unlike the SP-GiST rule it replaced."""
    table = make_table(geometry_columns=(make_geometry(geometry_type="POLYGON"),))
    workload = make_workload(("SELECT 1 FROM things ORDER BY geom <-> $1 LIMIT 5", 90_000))
    (recommendation,) = rules.rule_knn_index(context_for(make_catalog(table), workload))
    assert recommendation.index_type == "GiST"


def test_knn_is_silent_when_a_gist_index_already_exists() -> None:
    table = make_table(
        geometry_columns=(make_geometry(geometry_type="POINT"),), indexes=(GIST_ON_GEOM,)
    )
    workload = make_workload(("SELECT 1 FROM things ORDER BY geom <-> $1 LIMIT 5", 90_000))
    assert rules.rule_knn_index(context_for(make_catalog(table), workload)) == []


def test_an_spgist_index_does_not_suppress_the_knn_finding() -> None:
    """SP-GiST cannot answer KNN, so its presence is no reason to stay quiet."""
    index = ExistingIndex(name="i", method="spgist", columns=("geom",))
    table = make_table(
        geometry_columns=(make_geometry(geometry_type="POINT"),), indexes=(index,)
    )
    workload = make_workload(("SELECT 1 FROM things ORDER BY geom <-> $1 LIMIT 5", 90_000))
    (recommendation,) = rules.rule_knn_index(context_for(make_catalog(table), workload))
    assert recommendation.index_type == "GiST"


def test_knn_defers_to_the_missing_gist_rule_when_the_column_is_also_filtered() -> None:
    """Both rules would emit the same CREATE INDEX; only one should."""
    table = make_table(geometry_columns=(make_geometry(geometry_type="POINT"),))
    workload = make_workload(
        ("SELECT 1 FROM things ORDER BY geom <-> $1 LIMIT 5", 90_000),
        ("SELECT 1 FROM things WHERE ST_Intersects(geom, $1)", 10),
    )
    context = context_for(make_catalog(table), workload)
    assert rules.rule_knn_index(context) == []
    (gist,) = rules.rule_missing_gist(context)
    assert "order by distance" in gist.rationale


# --------------------------------------------------------------------------- #
# BRIN
# --------------------------------------------------------------------------- #


def brin_table(correlation: float = 0.97, append_only: bool = True):
    return make_table(
        name="public.events",
        row_count=200_000_000,
        table_bytes=40_000_000_000,
        geometry_columns=(make_geometry(correlation=correlation),),
        append_only=append_only,
        inserts=200_000_000,
    )


def test_brin_fires_for_large_append_only_correlated_tables() -> None:
    workload = make_workload(("SELECT 1 FROM events WHERE ST_Intersects(geom, $1)", 4000))
    (recommendation,) = rules.rule_brin_for_append_only(
        context_for(make_catalog(brin_table()), workload)
    )
    assert recommendation.index_type == "BRIN"
    assert "USING BRIN" in recommendation.ddl
    assert any("bad idea" in caveat for caveat in recommendation.caveats)
    assert recommendation.estimated_size_bytes < 50_000_000


def test_brin_is_skipped_when_the_table_is_updated_in_place() -> None:
    workload = make_workload(("SELECT 1 FROM events WHERE ST_Intersects(geom, $1)", 4000))
    catalog = make_catalog(brin_table(append_only=False))
    assert rules.rule_brin_for_append_only(context_for(catalog, workload)) == []


def test_brin_is_skipped_when_correlation_is_weak() -> None:
    workload = make_workload(("SELECT 1 FROM events WHERE ST_Intersects(geom, $1)", 4000))
    catalog = make_catalog(brin_table(correlation=0.3))
    assert rules.rule_brin_for_append_only(context_for(catalog, workload)) == []


def test_brin_is_skipped_for_small_tables() -> None:
    table = make_table(
        name="public.events",
        row_count=100_000,
        table_bytes=20_000_000,
        geometry_columns=(make_geometry(correlation=0.99),),
        append_only=True,
    )
    workload = make_workload(("SELECT 1 FROM events WHERE ST_Intersects(geom, $1)", 4000))
    assert rules.rule_brin_for_append_only(context_for(make_catalog(table), workload)) == []


# --------------------------------------------------------------------------- #
# partial and composite
# --------------------------------------------------------------------------- #


def test_partial_index_uses_a_dominant_constant_filter() -> None:
    catalog = make_catalog(make_table())
    workload = make_workload(
        ("SELECT 1 FROM things WHERE ST_Intersects(geom, $1) AND status = 'active'", 9000),
        ("SELECT 1 FROM things WHERE ST_Intersects(geom, $1)", 100),
    )
    (recommendation,) = rules.rule_partial_index(context_for(catalog, workload))
    assert recommendation.ddl.endswith("WHERE status = 'active';")
    assert recommendation.estimated_size_bytes is not None


def test_partial_index_ignores_parameterised_filters() -> None:
    catalog = make_catalog(make_table())
    workload = make_workload(
        ("SELECT 1 FROM things WHERE ST_Intersects(geom, $1) AND status = $2", 9000)
    )
    assert rules.rule_partial_index(context_for(catalog, workload)) == []


def test_partial_index_ignores_filters_that_are_not_dominant() -> None:
    catalog = make_catalog(make_table())
    workload = make_workload(
        ("SELECT 1 FROM things WHERE ST_Intersects(geom, $1) AND status = 'active'", 100),
        ("SELECT 1 FROM things WHERE ST_Intersects(geom, $1)", 9000),
    )
    assert rules.rule_partial_index(context_for(catalog, workload)) == []


def test_partial_index_is_not_repeated_when_it_already_exists() -> None:
    existing = ExistingIndex(
        name="i", method="gist", columns=("geom",), predicate="status = 'active'"
    )
    catalog = make_catalog(make_table(indexes=(existing,)))
    workload = make_workload(
        ("SELECT 1 FROM things WHERE ST_Intersects(geom, $1) AND status = 'active'", 9000)
    )
    assert rules.rule_partial_index(context_for(catalog, workload)) == []


def test_composite_index_uses_a_dominant_parameterised_equality() -> None:
    catalog = make_catalog(make_table())
    workload = make_workload(
        ("SELECT 1 FROM things WHERE ST_Intersects(geom, $1) AND fleet_id = $2", 9000)
    )
    (recommendation,) = rules.rule_composite_index(context_for(catalog, workload))
    assert "USING GIST (fleet_id, geom)" in recommendation.ddl
    assert "btree_gist" in recommendation.ddl
    assert any("btree_gist" in caveat for caveat in recommendation.caveats)


def test_composite_index_defers_to_a_partial_index_for_constant_filters() -> None:
    catalog = make_catalog(make_table())
    workload = make_workload(
        ("SELECT 1 FROM things WHERE ST_Intersects(geom, $1) AND status = 'active'", 9000)
    )
    assert rules.rule_composite_index(context_for(catalog, workload)) == []


def test_composite_index_ignores_range_filters() -> None:
    catalog = make_catalog(make_table())
    workload = make_workload(
        ("SELECT 1 FROM things WHERE ST_Intersects(geom, $1) AND created_at > $2", 9000)
    )
    assert rules.rule_composite_index(context_for(catalog, workload)) == []


def test_composite_index_is_not_repeated_when_it_already_exists() -> None:
    existing = ExistingIndex(name="i", method="gist", columns=("fleet_id", "geom"))
    catalog = make_catalog(make_table(indexes=(existing,)))
    workload = make_workload(
        ("SELECT 1 FROM things WHERE ST_Intersects(geom, $1) AND fleet_id = $2", 9000)
    )
    assert rules.rule_composite_index(context_for(catalog, workload)) == []


# --------------------------------------------------------------------------- #
# CLUSTER
# --------------------------------------------------------------------------- #


def cluster_table(correlation: float = 0.05):
    return make_table(
        row_count=40_000_000,
        table_bytes=8_000_000_000,
        geometry_columns=(
            make_geometry(correlation=correlation, avg_bbox=(8000.0, 6200.0)),
        ),
        indexes=(GIST_ON_GEOM,),
        inserts=40_000_000,
        updates=100,
    )


def test_cluster_fires_for_range_scans_on_an_unordered_heap() -> None:
    workload = make_workload(("SELECT 1 FROM things WHERE geom && $1", 20_000))
    (recommendation,) = rules.rule_cluster(context_for(make_catalog(cluster_table()), workload))
    assert recommendation.ddl.startswith("CLUSTER public.things USING t_geom_gist;")
    assert recommendation.index_type is None
    assert any("ACCESS EXCLUSIVE" in caveat for caveat in recommendation.caveats)


def test_cluster_is_skipped_when_the_heap_is_already_ordered() -> None:
    workload = make_workload(("SELECT 1 FROM things WHERE geom && $1", 20_000))
    catalog = make_catalog(cluster_table(correlation=0.95))
    assert rules.rule_cluster(context_for(catalog, workload)) == []


def test_cluster_is_skipped_without_an_index_to_cluster_on() -> None:
    table = make_table(
        row_count=40_000_000,
        geometry_columns=(make_geometry(correlation=0.05, avg_bbox=(8000.0, 6200.0)),),
    )
    workload = make_workload(("SELECT 1 FROM things WHERE geom && $1", 20_000))
    assert rules.rule_cluster(context_for(make_catalog(table), workload)) == []


def test_cluster_is_skipped_for_highly_selective_scans() -> None:
    table = make_table(
        row_count=40_000_000,
        geometry_columns=(make_geometry(correlation=0.05, avg_bbox=(1.0, 1.0)),),
        indexes=(GIST_ON_GEOM,),
    )
    workload = make_workload(("SELECT 1 FROM things WHERE geom && $1", 20_000))
    assert rules.rule_cluster(context_for(make_catalog(table), workload)) == []


def test_cluster_is_skipped_on_update_heavy_tables() -> None:
    table = make_table(
        row_count=40_000_000,
        geometry_columns=(make_geometry(correlation=0.05, avg_bbox=(8000.0, 6200.0)),),
        indexes=(GIST_ON_GEOM,),
        inserts=1_000_000,
        updates=9_000_000,
    )
    workload = make_workload(("SELECT 1 FROM things WHERE geom && $1", 20_000))
    assert rules.rule_cluster(context_for(make_catalog(table), workload)) == []


# --------------------------------------------------------------------------- #
# redundant indexes
# --------------------------------------------------------------------------- #


def test_exact_duplicate_indexes_are_reported() -> None:
    table = make_table(
        indexes=(
            ExistingIndex(name="a_gist", method="gist", columns=("geom",), size_bytes=1000),
            ExistingIndex(name="b_gist", method="gist", columns=("geom",), size_bytes=1000),
        )
    )
    workload = make_workload(("SELECT 1 FROM things WHERE geom && $1", 10))
    (recommendation,) = rules.rule_redundant_indexes(context_for(make_catalog(table), workload))
    assert recommendation.ddl == "DROP INDEX CONCURRENTLY public.b_gist;"
    assert recommendation.estimated_size_bytes == 1000


def test_a_unique_duplicate_is_kept_and_the_other_dropped() -> None:
    table = make_table(
        indexes=(
            ExistingIndex(name="plain", method="btree", columns=("id",)),
            ExistingIndex(name="uniq", method="btree", columns=("id",), is_unique=True),
        )
    )
    workload = make_workload(("SELECT 1 FROM things WHERE geom && $1", 10))
    (recommendation,) = rules.rule_redundant_indexes(context_for(make_catalog(table), workload))
    assert recommendation.ddl == "DROP INDEX CONCURRENTLY public.plain;"


def test_prefix_indexes_are_reported() -> None:
    table = make_table(
        indexes=(
            ExistingIndex(name="wide", method="btree", columns=("fleet_id", "started_at")),
            ExistingIndex(name="narrow", method="btree", columns=("fleet_id",)),
        )
    )
    workload = make_workload(("SELECT 1 FROM things WHERE geom && $1", 10))
    (recommendation,) = rules.rule_redundant_indexes(context_for(make_catalog(table), workload))
    assert recommendation.ddl == "DROP INDEX CONCURRENTLY public.narrow;"
    assert "prefix" in recommendation.rationale


def test_a_unique_prefix_index_is_never_dropped() -> None:
    table = make_table(
        indexes=(
            ExistingIndex(name="wide", method="btree", columns=("id", "kind")),
            ExistingIndex(name="pk", method="btree", columns=("id",), is_unique=True),
        )
    )
    workload = make_workload(("SELECT 1 FROM things WHERE geom && $1", 10))
    assert rules.rule_redundant_indexes(context_for(make_catalog(table), workload)) == []


def test_indexes_of_different_methods_are_not_redundant() -> None:
    table = make_table(
        indexes=(
            ExistingIndex(name="a", method="btree", columns=("kind",)),
            ExistingIndex(name="b", method="hash", columns=("kind",)),
        )
    )
    workload = make_workload(("SELECT 1 FROM things WHERE geom && $1", 10))
    assert rules.rule_redundant_indexes(context_for(make_catalog(table), workload)) == []


def test_partial_indexes_are_never_treated_as_redundant_prefixes() -> None:
    table = make_table(
        indexes=(
            ExistingIndex(name="wide", method="btree", columns=("a", "b")),
            ExistingIndex(name="narrow", method="btree", columns=("a",), predicate="b > 0"),
        )
    )
    workload = make_workload(("SELECT 1 FROM things WHERE geom && $1", 10))
    assert rules.rule_redundant_indexes(context_for(make_catalog(table), workload)) == []


# --------------------------------------------------------------------------- #
# rewrite advisories
# --------------------------------------------------------------------------- #


def test_distance_comparison_produces_a_rewrite_advisory_with_no_ddl() -> None:
    catalog = make_catalog(make_table())
    workload = make_workload(("SELECT 1 FROM things WHERE ST_Distance(geom, $1) < 500", 4000))
    (recommendation,) = rules.rule_rewrite_advisories(context_for(catalog, workload))
    assert recommendation.ddl is None
    assert "ST_DWithin" in recommendation.caveats[0]


def test_transform_produces_an_expression_index_suggestion() -> None:
    catalog = make_catalog(make_table())
    workload = make_workload(
        ("SELECT 1 FROM things WHERE ST_Intersects(ST_Transform(geom, 4326), $1)", 4000)
    )
    (recommendation,) = rules.rule_rewrite_advisories(context_for(catalog, workload))
    assert "ST_Transform(geom, 4326)" in recommendation.ddl
    assert recommendation.index_type == "expression GiST"


def test_transform_with_a_parameterised_srid_leaves_a_placeholder() -> None:
    catalog = make_catalog(make_table())
    workload = make_workload(
        ("SELECT 1 FROM things WHERE ST_Intersects(ST_Transform(geom, $1), $2)", 4000)
    )
    (recommendation,) = rules.rule_rewrite_advisories(context_for(catalog, workload))
    assert "<target_srid>" in recommendation.ddl


def test_sargable_predicates_produce_no_rewrite_advisory() -> None:
    catalog = make_catalog(make_table())
    workload = make_workload(("SELECT 1 FROM things WHERE ST_Intersects(geom, $1)", 4000))
    assert rules.rule_rewrite_advisories(context_for(catalog, workload)) == []


# --------------------------------------------------------------------------- #
# severity and helpers
# --------------------------------------------------------------------------- #


def test_severity_is_capped_by_table_size() -> None:
    small = make_table(row_count=20_000)
    large = make_table(row_count=50_000_000)
    assert rules.table_severity_cap(small) is Severity.LOW
    assert rules.table_severity_cap(large) is Severity.CRITICAL


def test_severity_requires_both_a_large_saving_and_a_large_speedup() -> None:
    from spatial_index_advisor.models import BenefitEstimate

    table = make_table(row_count=50_000_000)
    slow_but_huge = BenefitEstimate(100.0, 99.0, 10_000_000_000, "test")
    assert rules.severity_for(slow_but_huge, table) is Severity.HIGH
    real = BenefitEstimate(10_000.0, 10.0, 1_000_000, "test")
    assert rules.severity_for(real, table) is Severity.CRITICAL


def test_rule_cap_lowers_severity_regardless_of_saving() -> None:
    from spatial_index_advisor.models import BenefitEstimate

    table = make_table(row_count=50_000_000)
    benefit = BenefitEstimate(10_000.0, 10.0, 1_000_000, "test")
    assert rules.severity_for(benefit, table, cap=Severity.MEDIUM) is Severity.MEDIUM


def test_index_names_are_valid_identifiers() -> None:
    name = rules.index_name("public.some-very-long-table-name" * 4, "geom", "gist")
    assert len(name) <= 63
    assert all(character.isalnum() or character == "_" for character in name)


def test_context_records_usage_per_geometry_column() -> None:
    table = make_table(
        geometry_columns=(make_geometry(name="geom"), make_geometry(name="shape"))
    )
    workload = make_workload(
        ("SELECT 1 FROM things WHERE ST_Intersects(geom, $1) AND ST_Intersects(shape, $2)", 7)
    )
    context = context_for(make_catalog(table), workload)
    assert set(context.usages) == {("public.things", "geom"), ("public.things", "shape")}
    assert len(context.usages_for("public.things")) == 2


def test_context_skips_unparseable_statements() -> None:
    catalog = make_catalog(make_table())
    workload = make_workload(("SELECT FROM WHERE ((((", 10))
    assert context_for(catalog, workload).usages == {}


def test_every_rule_fires_somewhere_across_the_shipped_examples(
    example_catalog, examples_dir
) -> None:
    from spatial_index_advisor.workload import load_workload

    seen: set[str] = set()
    for name in (
        "pg_stat_statements.csv",
        "queries.sql",
        "postgresql-2026-07-14.csv",
    ):
        report = analyse(load_workload([examples_dir / name]), example_catalog)
        seen |= kinds(report.recommendations)
    assert seen >= {
        "missing_gist",
        "brin",
        "knn_gist",
        "partial_index",
        "composite_index",
        "cluster",
        "redundant_index",
        "rewrite",
    }


def test_the_drop_statement_is_schema_qualified() -> None:
    """A bare index name in DROP INDEX resolves through search_path.

    Against a schema that is not on the path that fails outright, and if an
    index of the same name exists in a schema earlier on the path it drops that
    one instead — silently, and irreversibly.
    """
    table = make_table(
        name="tenant_7.things",
        indexes=(
            ExistingIndex(name="a_gist", method="gist", columns=("geom",), size_bytes=1000),
            ExistingIndex(name="b_gist", method="gist", columns=("geom",), size_bytes=1000),
        ),
    )
    workload = make_workload(("SELECT 1 FROM things WHERE geom && $1", 10))
    (recommendation,) = rules.rule_redundant_indexes(context_for(make_catalog(table), workload))
    assert recommendation.ddl == "DROP INDEX CONCURRENTLY tenant_7.b_gist;"


def test_an_unqualified_table_leaves_the_index_name_bare() -> None:
    """A snapshot without a schema must not gain a stray leading dot."""
    table = make_table(
        name="things",
        indexes=(
            ExistingIndex(name="a_gist", method="gist", columns=("geom",), size_bytes=1000),
            ExistingIndex(name="b_gist", method="gist", columns=("geom",), size_bytes=1000),
        ),
    )
    workload = make_workload(("SELECT 1 FROM things WHERE geom && $1", 10))
    (recommendation,) = rules.rule_redundant_indexes(context_for(make_catalog(table), workload))
    assert recommendation.ddl == "DROP INDEX CONCURRENTLY b_gist;"
