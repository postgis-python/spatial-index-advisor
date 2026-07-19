"""Heuristic cost and size model.

Everything in this module is an **estimate**, not a measurement. The advisor has
no access to a planner, no ``EXPLAIN`` output and no runtime feedback; it works
from row counts, table sizes and geometry statistics in a JSON snapshot. The
numbers exist to rank recommendations against one another and to give an order of
magnitude, and they should never be quoted as expected speedups. Verify with
``EXPLAIN (ANALYZE, BUFFERS)`` before and after applying anything.

Model outline
-------------

Costs are expressed in PostgreSQL's arbitrary cost units, using the planner's
default parameters, so a figure here is broadly comparable to the ``cost=`` value
in an ``EXPLAIN`` plan.

*Sequential scan*::

    pages * seq_page_cost
      + rows * (cpu_tuple_cost + cpu_operator_cost * SPATIAL_OPERATOR_FACTOR * n_predicates)

The spatial factor models the fact that PostGIS predicate functions are one to
two orders of magnitude more expensive than an integer comparison; PostGIS itself
declares them with elevated planner costs for the same reason.

*GiST index scan*::

    height * random_page_cost                                   # tree descent
      + matched * (cpu_index_tuple_cost + cpu_operator_cost * SPATIAL_OPERATOR_FACTOR)
      + heap_fetch_cost(matched)                                # bounding-box hits
      + matched * cpu_operator_cost * SPATIAL_OPERATOR_FACTOR   # exact recheck

*BRIN index scan* reads whole page ranges, so its cost is governed by
correlation rather than selectivity::

    effective_fraction = selectivity + (1 - |correlation|) * (1 - selectivity)

At correlation 1.0 a BRIN scan reads only the matching fraction of the heap; at
correlation 0 it reads the whole table and is strictly worse than a sequential
scan because of the extra index read.

*Selectivity* comes from geometry statistics when they are present: for a
``ST_DWithin(geom, point, r)`` predicate the search window area ``(2r)^2`` is
compared against the layer extent area. Without an extent or a literal radius the
model falls back to :data:`DEFAULT_SPATIAL_SELECTIVITY`, and the recommendation
that uses it is reported at reduced confidence.
"""

from __future__ import annotations

import math
from typing import Final

from .models import GeometryColumn, TableStats

# --------------------------------------------------------------------------- #
# Planner constants (PostgreSQL defaults)
# --------------------------------------------------------------------------- #

SEQ_PAGE_COST: Final[float] = 1.0
RANDOM_PAGE_COST: Final[float] = 4.0
CPU_TUPLE_COST: Final[float] = 0.01
CPU_INDEX_TUPLE_COST: Final[float] = 0.005
CPU_OPERATOR_COST: Final[float] = 0.0025
BLOCK_SIZE: Final[int] = 8192

# --------------------------------------------------------------------------- #
# Model constants
# --------------------------------------------------------------------------- #

#: How much more expensive a PostGIS predicate is than a scalar comparison.
SPATIAL_OPERATOR_FACTOR: Final[float] = 100.0

#: Selectivity assumed for a spatial predicate with no usable statistics. 0.5% is
#: deliberately conservative: real geofence and viewport queries are usually more
#: selective, so this understates rather than overstates the benefit.
DEFAULT_SPATIAL_SELECTIVITY: Final[float] = 0.005

#: Bounds applied to any computed selectivity.
MIN_SELECTIVITY: Final[float] = 1e-6
MAX_SELECTIVITY: Final[float] = 1.0

#: Bytes per index entry. A 2D GiST entry stores a float4 box (32 bytes) plus the
#: item pointer and per-tuple overhead; SP-GiST quadtree leaves are more compact.
GIST_ENTRY_BYTES: Final[int] = 40
SPGIST_ENTRY_BYTES: Final[int] = 24
BTREE_ENTRY_OVERHEAD_BYTES: Final[int] = 16

#: Default index fill factor for GiST/SP-GiST builds.
INDEX_FILLFACTOR: Final[float] = 0.9

#: Entries per internal GiST page, used to derive tree height.
GIST_FANOUT: Final[int] = 100

#: BRIN default ``pages_per_range`` and the bytes a summarised range occupies.
BRIN_PAGES_PER_RANGE: Final[int] = 128
BRIN_BYTES_PER_RANGE: Final[int] = 64

#: A GiST KNN scan visits more entries than it returns because it must pop
#: internal nodes off the priority queue before it can prove the next winner.
KNN_VISIT_FACTOR: Final[float] = 3.0

#: Correlation above which physical ordering is considered strong enough for BRIN
#: or for the sequential-heap-fetch approximation.
STRONG_CORRELATION: Final[float] = 0.9


def clamp_selectivity(value: float) -> float:
    """Clip a selectivity into the modelled range."""
    return max(MIN_SELECTIVITY, min(MAX_SELECTIVITY, value))


def estimate_selectivity(
    geometry: GeometryColumn | None, radius: float | None
) -> tuple[float, str]:
    """Estimate the fraction of rows a spatial predicate returns.

    Returns the selectivity and a short string describing where it came from,
    which is surfaced in the report so the reader can judge it.
    """
    if radius is not None and radius > 0 and geometry is not None:
        if geometry.extent_width and geometry.extent_height:
            extent_area = geometry.extent_width * geometry.extent_height
            if extent_area > 0:
                window_area = (2.0 * radius) ** 2
                return (
                    clamp_selectivity(window_area / extent_area),
                    f"search window {2 * radius:g} x {2 * radius:g} against layer extent",
                )
    if geometry is not None:
        fraction = geometry.average_area_fraction
        if fraction is not None and fraction > 0:
            return (
                clamp_selectivity(max(fraction, DEFAULT_SPATIAL_SELECTIVITY)),
                "mean feature bbox area against layer extent",
            )
    return DEFAULT_SPATIAL_SELECTIVITY, "default assumption, no geometry statistics available"


def sequential_scan_cost(table: TableStats, spatial_predicates: int = 1) -> float:
    """Modelled cost of scanning ``table`` end to end evaluating spatial filters."""
    predicates = max(1, spatial_predicates)
    return table.pages * SEQ_PAGE_COST + table.row_count * (
        CPU_TUPLE_COST + CPU_OPERATOR_COST * SPATIAL_OPERATOR_FACTOR * predicates
    )


def index_pages(entries: int, entry_bytes: int) -> int:
    """Number of 8 kB index pages needed for ``entries`` entries."""
    if entries <= 0:
        return 1
    per_page = max(1, int((BLOCK_SIZE * INDEX_FILLFACTOR) // entry_bytes))
    return max(1, math.ceil(entries / per_page))


def tree_height(pages: int) -> int:
    """Height of a balanced tree over ``pages`` leaf pages, at least 2."""
    if pages <= 1:
        return 2
    return max(2, math.ceil(math.log(pages, GIST_FANOUT)) + 1)


def _heap_fetch_cost(table: TableStats, matched_rows: float, correlation: float) -> float:
    """Cost of fetching ``matched_rows`` heap tuples given physical correlation.

    Well-correlated matches land on consecutive pages and are read sequentially;
    uncorrelated matches cost a random page each, capped at the table size.
    """
    if table.row_count <= 0:
        return 0.0
    fraction = min(1.0, matched_rows / table.row_count)
    sequential_pages = fraction * table.pages
    random_pages = min(matched_rows, float(table.pages))
    weight = min(1.0, abs(correlation))
    return (
        weight * sequential_pages * SEQ_PAGE_COST
        + (1.0 - weight) * random_pages * RANDOM_PAGE_COST
    )


def gist_scan_cost(
    table: TableStats, selectivity: float, correlation: float = 0.0
) -> float:
    """Modelled cost of answering a spatial predicate from a GiST index."""
    matched = max(1.0, table.row_count * clamp_selectivity(selectivity))
    pages = index_pages(table.row_count, GIST_ENTRY_BYTES)
    descent = tree_height(pages) * RANDOM_PAGE_COST
    index_scan = matched * (CPU_INDEX_TUPLE_COST + CPU_OPERATOR_COST * SPATIAL_OPERATOR_FACTOR)
    recheck = matched * (CPU_TUPLE_COST + CPU_OPERATOR_COST * SPATIAL_OPERATOR_FACTOR)
    return descent + index_scan + _heap_fetch_cost(table, matched, correlation) + recheck


def spgist_scan_cost(table: TableStats, selectivity: float, correlation: float = 0.0) -> float:
    """Modelled cost of a SP-GiST scan.

    SP-GiST quadtree entries are smaller than GiST boxes, so the tree is shallower
    and the descent cheaper; the heap side of the scan is identical.
    """
    matched = max(1.0, table.row_count * clamp_selectivity(selectivity))
    pages = index_pages(table.row_count, SPGIST_ENTRY_BYTES)
    descent = tree_height(pages) * RANDOM_PAGE_COST
    index_scan = matched * (CPU_INDEX_TUPLE_COST + CPU_OPERATOR_COST * SPATIAL_OPERATOR_FACTOR)
    recheck = matched * (CPU_TUPLE_COST + CPU_OPERATOR_COST * SPATIAL_OPERATOR_FACTOR)
    return descent + index_scan + _heap_fetch_cost(table, matched, correlation) + recheck


def brin_effective_fraction(selectivity: float, correlation: float) -> float:
    """Fraction of the heap a BRIN scan must read.

    A BRIN scan cannot skip a page range unless the range summary excludes it, so
    poor correlation degrades directly into reading the whole table.
    """
    weight = min(1.0, abs(correlation))
    selectivity = clamp_selectivity(selectivity)
    return min(1.0, selectivity + (1.0 - weight) * (1.0 - selectivity))


def brin_scan_cost(table: TableStats, selectivity: float, correlation: float) -> float:
    """Modelled cost of answering a predicate from a BRIN index."""
    fraction = brin_effective_fraction(selectivity, correlation)
    ranges = max(1, table.pages // BRIN_PAGES_PER_RANGE)
    index_read = max(1, index_pages(ranges, BRIN_BYTES_PER_RANGE)) * SEQ_PAGE_COST
    scanned_pages = max(1.0, fraction * table.pages)
    scanned_rows = fraction * table.row_count
    return (
        index_read
        + scanned_pages * SEQ_PAGE_COST
        + scanned_rows * (CPU_TUPLE_COST + CPU_OPERATOR_COST * SPATIAL_OPERATOR_FACTOR)
    )


def knn_sort_cost(table: TableStats, limit: int | None) -> float:
    """Modelled cost of a nearest-neighbour query with no usable index.

    The planner must compute the distance for every row and keep a bounded heap of
    the best ``limit`` candidates.
    """
    rows = max(1, table.row_count)
    keep = max(1, limit or rows)
    distance_cost = rows * CPU_OPERATOR_COST * SPATIAL_OPERATOR_FACTOR
    heap_cost = rows * math.log2(max(2, keep)) * CPU_OPERATOR_COST
    return sequential_scan_cost(table, spatial_predicates=0) + distance_cost + heap_cost


def knn_index_cost(table: TableStats, limit: int | None, entry_bytes: int) -> float:
    """Modelled cost of an index-ordered nearest-neighbour scan."""
    rows = max(1, table.row_count)
    keep = max(1, limit or rows)
    pages = index_pages(rows, entry_bytes)
    descent = tree_height(pages) * RANDOM_PAGE_COST
    visited = keep * KNN_VISIT_FACTOR
    return descent + visited * (
        RANDOM_PAGE_COST
        + CPU_TUPLE_COST
        + CPU_INDEX_TUPLE_COST
        + CPU_OPERATOR_COST * SPATIAL_OPERATOR_FACTOR
    )


# --------------------------------------------------------------------------- #
# Size estimates
# --------------------------------------------------------------------------- #


def gist_index_size(rows: int) -> int:
    """Estimated on-disk size of a 2D GiST index over ``rows`` geometries."""
    return index_pages(rows, GIST_ENTRY_BYTES) * BLOCK_SIZE


def spgist_index_size(rows: int) -> int:
    """Estimated on-disk size of an SP-GiST index over ``rows`` points."""
    return index_pages(rows, SPGIST_ENTRY_BYTES) * BLOCK_SIZE


def brin_index_size(table: TableStats) -> int:
    """Estimated on-disk size of a BRIN index over ``table``.

    BRIN stores one summary per ``pages_per_range`` heap pages, which is why it is
    typically four to five orders of magnitude smaller than the table.
    """
    ranges = max(1, table.pages // BRIN_PAGES_PER_RANGE)
    return max(2, index_pages(ranges, BRIN_BYTES_PER_RANGE) + 1) * BLOCK_SIZE


def composite_index_size(rows: int, scalar_width_bytes: int = 8) -> int:
    """Estimated size of a GiST index on a geometry plus one scalar column."""
    return index_pages(rows, GIST_ENTRY_BYTES + scalar_width_bytes) * BLOCK_SIZE


def partial_index_size(rows: int, matching_fraction: float) -> int:
    """Estimated size of a GiST index restricted to a fraction of the table."""
    covered = max(1, int(rows * clamp_selectivity(matching_fraction)))
    return gist_index_size(covered)


def btree_index_size(rows: int, key_width_bytes: int = 8) -> int:
    """Estimated size of a B-tree index over a scalar column."""
    return index_pages(rows, key_width_bytes + BTREE_ENTRY_OVERHEAD_BYTES) * BLOCK_SIZE


def format_bytes(value: int | None) -> str:
    """Human-readable byte size, matching ``pg_size_pretty`` conventions."""
    if value is None:
        return "unknown"
    size = float(value)
    for unit in ("bytes", "kB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "bytes" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"
