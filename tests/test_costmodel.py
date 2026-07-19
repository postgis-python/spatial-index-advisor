"""Tests for the heuristic cost and size model.

These assert the *properties* the model must have — monotonicity, ordering,
sane bounds — rather than pinning exact numbers, which would just restate the
constants.
"""

from __future__ import annotations

import pytest

from spatial_index_advisor import costmodel
from spatial_index_advisor.models import GeometryColumn

from .conftest import make_geometry, make_table


def test_selectivity_from_radius_uses_the_search_window() -> None:
    geometry = make_geometry(extent=(1000.0, 1000.0))
    selectivity, basis = costmodel.estimate_selectivity(geometry, radius=50.0)
    assert selectivity == pytest.approx((100 * 100) / (1000 * 1000))
    assert "search window" in basis


def test_larger_radius_is_less_selective() -> None:
    geometry = make_geometry(extent=(1000.0, 1000.0))
    small, _ = costmodel.estimate_selectivity(geometry, 10.0)
    large, _ = costmodel.estimate_selectivity(geometry, 100.0)
    assert small < large


def test_selectivity_is_clamped_to_one() -> None:
    geometry = make_geometry(extent=(10.0, 10.0))
    selectivity, _ = costmodel.estimate_selectivity(geometry, radius=1_000_000.0)
    assert selectivity == costmodel.MAX_SELECTIVITY


def test_selectivity_falls_back_to_feature_size_without_a_radius() -> None:
    geometry = make_geometry(avg_bbox=(10_000.0, 10_000.0), extent=(100_000.0, 100_000.0))
    selectivity, basis = costmodel.estimate_selectivity(geometry, radius=None)
    assert selectivity == pytest.approx(0.01)
    assert "mean feature bbox" in basis


def test_selectivity_falls_back_to_the_default_without_statistics() -> None:
    selectivity, basis = costmodel.estimate_selectivity(None, None)
    assert selectivity == costmodel.DEFAULT_SPATIAL_SELECTIVITY
    assert "default assumption" in basis


def test_selectivity_never_drops_below_the_default_from_feature_size() -> None:
    geometry = GeometryColumn(
        name="g", avg_bbox_width=1.0, avg_bbox_height=1.0,
        extent_width=1e9, extent_height=1e9,
    )
    selectivity, _ = costmodel.estimate_selectivity(geometry, None)
    assert selectivity == costmodel.DEFAULT_SPATIAL_SELECTIVITY


def test_gist_beats_a_sequential_scan_at_low_selectivity() -> None:
    table = make_table(row_count=10_000_000, table_bytes=2_000_000_000)
    assert costmodel.gist_scan_cost(table, 0.001) < costmodel.sequential_scan_cost(table)


def test_gist_loses_to_a_sequential_scan_when_everything_matches() -> None:
    table = make_table(row_count=10_000_000, table_bytes=2_000_000_000)
    assert costmodel.gist_scan_cost(table, 1.0) > costmodel.sequential_scan_cost(table)


def test_gist_cost_rises_with_selectivity() -> None:
    table = make_table()
    costs = [costmodel.gist_scan_cost(table, s) for s in (0.0001, 0.001, 0.01, 0.1)]
    assert costs == sorted(costs)


def test_correlated_heap_fetches_are_cheaper() -> None:
    table = make_table()
    assert costmodel.gist_scan_cost(table, 0.02, correlation=1.0) < costmodel.gist_scan_cost(
        table, 0.02, correlation=0.0
    )


def test_spgist_descent_is_cheaper_than_gist() -> None:
    table = make_table(row_count=100_000_000)
    assert costmodel.spgist_scan_cost(table, 0.001) <= costmodel.gist_scan_cost(table, 0.001)


def test_brin_reads_everything_when_correlation_is_zero() -> None:
    assert costmodel.brin_effective_fraction(0.001, 0.0) == pytest.approx(1.0)


def test_brin_reads_only_the_matching_fraction_at_perfect_correlation() -> None:
    assert costmodel.brin_effective_fraction(0.02, 1.0) == pytest.approx(0.02)


def test_brin_beats_a_sequential_scan_only_when_correlated() -> None:
    table = make_table(row_count=200_000_000, table_bytes=40_000_000_000)
    sequential = costmodel.sequential_scan_cost(table)
    assert costmodel.brin_scan_cost(table, 0.001, 0.99) < sequential
    assert costmodel.brin_scan_cost(table, 0.001, 0.0) > sequential


def test_brin_is_orders_of_magnitude_smaller_than_gist() -> None:
    table = make_table(row_count=400_000_000, table_bytes=60_000_000_000)
    assert costmodel.brin_index_size(table) * 1000 < costmodel.gist_index_size(table.row_count)


def test_knn_index_scan_beats_a_full_sort() -> None:
    table = make_table(row_count=50_000_000)
    assert costmodel.knn_index_cost(table, 10, costmodel.GIST_ENTRY_BYTES) < (
        costmodel.knn_sort_cost(table, 10)
    )


def test_knn_index_cost_grows_with_the_limit() -> None:
    table = make_table(row_count=50_000_000)
    small = costmodel.knn_index_cost(table, 5, costmodel.GIST_ENTRY_BYTES)
    large = costmodel.knn_index_cost(table, 500, costmodel.GIST_ENTRY_BYTES)
    assert small < large


def test_knn_without_a_limit_is_treated_as_the_whole_table() -> None:
    table = make_table(row_count=100_000)
    assert costmodel.knn_index_cost(table, None, costmodel.GIST_ENTRY_BYTES) > (
        costmodel.knn_index_cost(table, 10, costmodel.GIST_ENTRY_BYTES)
    )


def test_index_sizes_grow_with_row_count_and_key_width() -> None:
    assert costmodel.gist_index_size(1_000_000) > costmodel.gist_index_size(100_000)
    assert costmodel.spgist_index_size(1_000_000) < costmodel.gist_index_size(1_000_000)
    assert costmodel.composite_index_size(1_000_000) > costmodel.gist_index_size(1_000_000)
    assert costmodel.btree_index_size(1_000_000, 32) > costmodel.btree_index_size(1_000_000, 8)


def test_partial_index_is_smaller_than_the_full_one() -> None:
    assert costmodel.partial_index_size(1_000_000, 0.1) < costmodel.gist_index_size(1_000_000)


def test_index_pages_and_tree_height_have_sane_floors() -> None:
    assert costmodel.index_pages(0, 40) == 1
    assert costmodel.tree_height(1) == 2
    assert costmodel.tree_height(1_000_000) > costmodel.tree_height(100)


def test_empty_table_costs_do_not_explode() -> None:
    table = make_table(row_count=0, table_bytes=0)
    assert costmodel.gist_scan_cost(table, 0.5) > 0
    assert costmodel.sequential_scan_cost(table) > 0


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "unknown"),
        (512, "512 bytes"),
        (2048, "2.0 kB"),
        (5 * 1024**2, "5.0 MB"),
        (3 * 1024**3, "3.0 GB"),
        (2 * 1024**4, "2.0 TB"),
    ],
)
def test_format_bytes(value, expected: str) -> None:
    assert costmodel.format_bytes(value) == expected
