"""Recommendation rules.

Each rule is a pure function from a :class:`RuleContext` to a list of
:class:`~spatial_index_advisor.models.Recommendation` objects. Rules never touch
a database, never read files and never print; that is what makes the whole set
testable from a fixture. :data:`RULES` is the ordered registry the engine runs.

Every rule follows the same contract:

* it only fires when the catalog actually contains the table it is talking about;
* it never recommends an index that the snapshot says already exists;
* it attaches the fingerprints of the statements that motivated it;
* it produces a benefit estimate from :mod:`spatial_index_advisor.costmodel`,
  clearly labelled as a model output rather than a measurement.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Final, Iterable, Sequence

from . import costmodel, docs
from .models import (
    BenefitEstimate,
    CatalogSnapshot,
    Confidence,
    ExistingIndex,
    GeometryColumn,
    KnnUsage,
    Recommendation,
    ScalarFilter,
    Severity,
    SpatialPredicate,
    StatementAnalysis,
    TableStats,
    Workload,
    WorkloadStatement,
)

# --------------------------------------------------------------------------- #
# Thresholds. These are judgement calls, documented in the README.
# --------------------------------------------------------------------------- #

#: Modelled cost saved across the workload at which a finding reaches each band.
CRITICAL_COST_THRESHOLD: Final[float] = 1e7
HIGH_COST_THRESHOLD: Final[float] = 1e5
MEDIUM_COST_THRESHOLD: Final[float] = 1e3

#: A finding is only CRITICAL if the change is also a large per-call win. A huge
#: aggregate saving spread over millions of already-fast calls is not an
#: emergency.
CRITICAL_SPEEDUP_THRESHOLD: Final[float] = 10.0

#: Table sizes at which a finding is allowed to reach each severity band. An
#: index problem on a small table cannot be critical however often it is hit.
TABLE_SEVERITY_CAPS: Final[tuple[tuple[int, Severity], ...]] = (
    (5_000_000, Severity.CRITICAL),
    (500_000, Severity.HIGH),
    (50_000, Severity.MEDIUM),
)

#: Minimum rows before a table is worth indexing at all; below this a sequential
#: scan is cheap enough that an index mostly costs write throughput.
MIN_ROWS_FOR_INDEX: Final[int] = 10_000

#: A table must be at least this large before BRIN is considered. Below it the
#: space saving is irrelevant and GiST is simply better.
BRIN_MIN_ROWS: Final[int] = 5_000_000
BRIN_MIN_BYTES: Final[int] = 1 << 30

#: Share of a geometry column's calls that must carry the same filter before a
#: partial or composite index is suggested for it.
FILTER_DOMINANCE: Final[float] = 0.8

#: Fraction of rows assumed to satisfy a partial index predicate when the
#: snapshot carries no statistics for that column.
ASSUMED_PARTIAL_FRACTION: Final[float] = 0.2

#: Selectivity assumed for an equality filter on an unindexed scalar column.
ASSUMED_SCALAR_SELECTIVITY: Final[float] = 0.1

#: Selectivity above which a spatial scan is "range heavy" and physical
#: clustering starts to pay off.
CLUSTER_MIN_SELECTIVITY: Final[float] = 0.01

#: Correlation below which the heap is considered unordered with respect to a
#: column, making CLUSTER worthwhile.
CLUSTER_MAX_CORRELATION: Final[float] = 0.5

#: Index methods that can serve a spatial predicate.
SPATIAL_METHODS: Final[frozenset[str]] = frozenset({"gist", "spgist", "brin"})

_IDENTIFIER_RE: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9_]+")
_MAX_IDENTIFIER_LENGTH: Final[int] = 63


# --------------------------------------------------------------------------- #
# Context
# --------------------------------------------------------------------------- #


@dataclass
class ColumnUsage:
    """How one geometry column is used across the whole workload."""

    table: TableStats
    column: str
    geometry: GeometryColumn | None
    sargable: list[tuple[WorkloadStatement, SpatialPredicate]] = field(default_factory=list)
    non_sargable: list[tuple[WorkloadStatement, SpatialPredicate]] = field(default_factory=list)
    knn: list[tuple[WorkloadStatement, KnnUsage]] = field(default_factory=list)
    filters: dict[str, list[tuple[WorkloadStatement, ScalarFilter]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    @property
    def sargable_calls(self) -> int:
        """Total executions of statements with an indexable predicate here."""
        return sum(statement.calls for statement, _ in self.sargable)

    @property
    def knn_calls(self) -> int:
        """Total executions of statements performing a KNN sort on this column."""
        return sum(statement.calls for statement, _ in self.knn)

    @property
    def min_radius(self) -> float | None:
        """Smallest literal distance seen, used to size the search window."""
        radii = [
            predicate.radius
            for _, predicate in self.sargable
            if predicate.radius is not None and predicate.radius > 0
        ]
        return min(radii) if radii else None

    @property
    def knn_limit(self) -> int | None:
        """Largest LIMIT seen on a KNN query, the worst case for the index scan."""
        limits = [usage.limit for _, usage in self.knn if usage.limit]
        return max(limits) if limits else None

    def selectivity(self) -> tuple[float, str]:
        """Estimated selectivity of the predicates against this column."""
        return costmodel.estimate_selectivity(self.geometry, self.min_radius)

    def fingerprints(self, *groups: Iterable[tuple[WorkloadStatement, object]]) -> tuple[str, ...]:
        """Deduplicated fingerprints of the statements in the given groups."""
        seen: dict[str, None] = {}
        for group in groups:
            for statement, _ in group:
                seen.setdefault(statement.fingerprint, None)
        return tuple(seen)

    def correlation(self) -> float:
        """Physical correlation of this geometry column, 0.0 when unknown."""
        if self.geometry is not None and self.geometry.correlation is not None:
            return self.geometry.correlation
        return self.table.column_correlation.get(self.column, 0.0)


@dataclass
class RuleContext:
    """Everything the rules need, precomputed once."""

    catalog: CatalogSnapshot
    workload: Workload
    analyses: dict[str, StatementAnalysis]
    usages: dict[tuple[str, str], ColumnUsage]

    def usages_for(self, table_name: str) -> list[ColumnUsage]:
        """Every geometry-column usage belonging to one table."""
        return [usage for (name, _), usage in self.usages.items() if name == table_name]


def _resolve_tables(
    analysis: StatementAnalysis, catalog: CatalogSnapshot
) -> dict[str, TableStats]:
    """Map the table names written in a statement to catalog entries."""
    resolved: dict[str, TableStats] = {}
    for name in analysis.tables:
        table = catalog.resolve(name)
        if table is not None:
            resolved[name] = table
    return resolved


def build_context(
    catalog: CatalogSnapshot, workload: Workload, analyses: dict[str, StatementAnalysis]
) -> RuleContext:
    """Fold the workload and its analyses into per-column usage records."""
    usages: dict[tuple[str, str], ColumnUsage] = {}

    def usage_for(table: TableStats, column: str) -> ColumnUsage:
        key = (table.name, column.lower())
        if key not in usages:
            usages[key] = ColumnUsage(
                table=table, column=column, geometry=table.geometry_column(column)
            )
        return usages[key]

    for statement in workload.statements:
        analysis = analyses.get(statement.fingerprint)
        if analysis is None or analysis.parse_error is not None:
            continue
        tables = _resolve_tables(analysis, catalog)
        if not tables:
            continue

        touched: set[tuple[str, str]] = set()
        for predicate in analysis.spatial_predicates:
            for column in predicate.columns:
                table = tables.get(column.table or "")
                if table is None or table.geometry_column(column.column) is None:
                    continue
                usage = usage_for(table, column.column)
                target = usage.sargable if predicate.sargable else usage.non_sargable
                target.append((statement, predicate))
                touched.add((table.name, column.column.lower()))

        for knn in analysis.knn:
            table = tables.get(knn.column.table or "")
            if table is None or table.geometry_column(knn.column.column) is None:
                continue
            usage = usage_for(table, knn.column.column)
            usage.knn.append((statement, knn))
            touched.add((table.name, knn.column.column.lower()))

        for table_name, column_name in touched:
            usage = usages[(table_name, column_name)]
            for scalar in analysis.scalar_filters:
                scalar_table = tables.get(scalar.column.table or "")
                if scalar_table is None or scalar_table.name != table_name:
                    continue
                if scalar_table.geometry_column(scalar.column.column) is not None:
                    continue
                usage.filters[_filter_key(scalar)].append((statement, scalar))

    return RuleContext(catalog=catalog, workload=workload, analyses=analyses, usages=usages)


def _filter_key(scalar: ScalarFilter) -> str:
    """Grouping key for a scalar filter: same column, operator and constant."""
    return f"{scalar.column.column.lower()}|{scalar.operator}|{scalar.literal or '?'}"


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def table_severity_cap(table: TableStats) -> Severity:
    """Highest severity a finding about ``table`` may reach, given its size."""
    for rows, severity in TABLE_SEVERITY_CAPS:
        if table.row_count >= rows:
            return severity
    return Severity.LOW


def severity_for(
    benefit: BenefitEstimate, table: TableStats, cap: Severity = Severity.CRITICAL
) -> Severity:
    """Map a modelled benefit onto a severity band, applying both caps.

    ``cap`` is the rule's own ceiling: "there is no index at all" can be critical,
    "a better index exists" cannot. The table-size ceiling is applied on top.
    """
    saved = benefit.total_cost_saved
    if saved >= CRITICAL_COST_THRESHOLD and benefit.speedup >= CRITICAL_SPEEDUP_THRESHOLD:
        level = Severity.CRITICAL
    elif saved >= HIGH_COST_THRESHOLD:
        level = Severity.HIGH
    elif saved >= MEDIUM_COST_THRESHOLD:
        level = Severity.MEDIUM
    else:
        level = Severity.LOW
    ceiling = max(cap.rank, table_severity_cap(table).rank)
    return level if level.rank >= ceiling else _BY_RANK[ceiling]


_BY_RANK: Final[dict[int, Severity]] = {severity.rank: severity for severity in Severity}


def confidence_for(usage: ColumnUsage, basis: str) -> Confidence:
    """Grade confidence by how much of the estimate rests on defaults."""
    if usage.geometry is None:
        return Confidence.LOW
    if "default assumption" in basis:
        return Confidence.LOW if usage.sargable_calls < 100 else Confidence.MEDIUM
    if usage.table.row_count <= 0:
        return Confidence.LOW
    return Confidence.HIGH


def index_name(table: str, column: str, suffix: str) -> str:
    """Build a valid, deterministic index name."""
    bare = table.rsplit(".", 1)[-1]
    raw = f"idx_{bare}_{column}_{suffix}".lower()
    cleaned = _IDENTIFIER_RE.sub("_", raw).strip("_")
    return cleaned[:_MAX_IDENTIFIER_LENGTH]


def has_spatial_index(table: TableStats, column: str, methods: Iterable[str]) -> bool:
    """True when a usable index of one of ``methods`` already covers ``column``."""
    wanted = {method.lower() for method in methods}
    lowered = column.lower()
    for index in table.indexes:
        if index.method.lower() not in wanted:
            continue
        if index.predicate:
            continue
        if lowered in {c.lower() for c in index.columns}:
            return True
    return False


def existing_index_on(table: TableStats, column: str, method: str) -> ExistingIndex | None:
    """First unconditional index of ``method`` covering ``column``."""
    lowered = column.lower()
    for index in table.indexes:
        if index.method.lower() != method.lower() or index.predicate:
            continue
        if lowered in {c.lower() for c in index.columns}:
            return index
    return None


def _scaled_table(table: TableStats, fraction: float) -> TableStats:
    """A copy of ``table`` shrunk to a fraction of its rows and bytes.

    Used to model a partial index, which behaves like a GiST index over a smaller
    table.
    """
    fraction = costmodel.clamp_selectivity(fraction)
    return TableStats(
        name=table.name,
        row_count=max(1, int(table.row_count * fraction)),
        table_bytes=max(costmodel.BLOCK_SIZE, int(table.table_bytes * fraction)),
        geometry_columns=table.geometry_columns,
        indexes=table.indexes,
        column_correlation=table.column_correlation,
        append_only=table.append_only,
    )


def _dominant_filters(
    usage: ColumnUsage, constant_only: bool
) -> list[tuple[ScalarFilter, int]]:
    """Filters carried by a dominant share of the calls against this column.

    Returns ``(filter, calls)`` pairs sorted by call count descending.
    """
    total = usage.sargable_calls
    if total <= 0:
        return []
    results: list[tuple[ScalarFilter, int]] = []
    for occurrences in usage.filters.values():
        scalar = occurrences[0][1]
        if constant_only and not scalar.is_constant:
            continue
        if scalar.operator in {"<>", "IS NULL"}:
            continue
        calls = sum(statement.calls for statement, _ in occurrences)
        if calls >= total * FILTER_DOMINANCE:
            results.append((scalar, calls))
    results.sort(key=lambda item: item[1], reverse=True)
    return results


def _geometry_type(geometry: GeometryColumn | None) -> str:
    return geometry.geometry_type if geometry is not None else "GEOMETRY"


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #


def rule_missing_gist(context: RuleContext) -> list[Recommendation]:
    """Recommend a GiST index on geometry columns filtered without one."""
    recommendations: list[Recommendation] = []
    for usage in context.usages.values():
        if not usage.sargable or usage.geometry is None:
            continue
        if has_spatial_index(usage.table, usage.column, SPATIAL_METHODS):
            continue
        if usage.table.row_count < MIN_ROWS_FOR_INDEX:
            continue

        selectivity, basis = usage.selectivity()
        correlation = usage.correlation()
        current = costmodel.sequential_scan_cost(usage.table, spatial_predicates=1)
        projected = costmodel.gist_scan_cost(usage.table, selectivity, correlation)
        benefit = BenefitEstimate(
            current_cost_per_call=current,
            projected_cost_per_call=projected,
            calls=usage.sargable_calls,
            basis=f"sequential scan vs GiST scan at {selectivity:.4%} selectivity ({basis})",
        )
        predicates = sorted({predicate.name for _, predicate in usage.sargable})
        recommendations.append(
            Recommendation(
                kind="missing_gist",
                title=f"Add a GiST index on {usage.table.name}.{usage.column}",
                table=usage.table.name,
                severity=severity_for(benefit, usage.table),
                confidence=confidence_for(usage, basis),
                rationale=(
                    f"{usage.sargable_calls:,} executions filter "
                    f"{usage.table.name}.{usage.column} with {', '.join(predicates)}, "
                    f"but the column has no spatial index. Every one of those calls scans "
                    f"all {usage.table.row_count:,} rows and evaluates the predicate per row."
                ),
                index_type="GiST",
                type_rationale=(
                    "GiST is the general-purpose PostGIS index: it handles any geometry type, "
                    "supports the bounding-box operators these predicates expand to, and is "
                    "orderable for KNN. SP-GiST is only competitive for point data and BRIN "
                    "only for physically clustered tables."
                ),
                ddl=(
                    f"CREATE INDEX CONCURRENTLY {index_name(usage.table.name, usage.column, 'gist')} "
                    f"ON {usage.table.name} USING GIST ({usage.column});"
                ),
                estimated_size_bytes=costmodel.gist_index_size(usage.table.row_count),
                benefit=benefit,
                caveats=(
                    "CONCURRENTLY avoids an ACCESS EXCLUSIVE lock but cannot run inside a "
                    "transaction block, and leaves an INVALID index behind if it fails.",
                    "Building a GiST index on a large table is I/O heavy; consider raising "
                    "maintenance_work_mem for the session.",
                ),
                fingerprints=usage.fingerprints(usage.sargable),
                docs_url=docs.GIST_OPTIMIZATION,
            )
        )
    return recommendations


def rule_spgist_for_knn(context: RuleContext) -> list[Recommendation]:
    """Recommend SP-GiST for point tables dominated by nearest-neighbour queries."""
    recommendations: list[Recommendation] = []
    for usage in context.usages.values():
        geometry = usage.geometry
        if geometry is None or not geometry.is_point or not usage.knn:
            continue
        if usage.table.row_count < MIN_ROWS_FOR_INDEX:
            continue
        if has_spatial_index(usage.table, usage.column, {"spgist"}):
            continue
        if usage.knn_calls < usage.sargable_calls:
            continue

        limit = usage.knn_limit
        gist = existing_index_on(usage.table, usage.column, "gist")
        if gist is not None:
            current = costmodel.knn_index_cost(usage.table, limit, costmodel.GIST_ENTRY_BYTES)
            baseline = f"existing GiST index {gist.name}"
        else:
            current = costmodel.knn_sort_cost(usage.table, limit)
            baseline = "sequential scan with a top-N sort"
        projected = costmodel.knn_index_cost(usage.table, limit, costmodel.SPGIST_ENTRY_BYTES)
        benefit = BenefitEstimate(
            current_cost_per_call=current,
            projected_cost_per_call=projected,
            calls=usage.knn_calls,
            basis=f"{baseline} vs SP-GiST KNN scan returning {limit or 'all'} rows",
        )
        operators = sorted({knn.operator for _, knn in usage.knn})
        recommendations.append(
            Recommendation(
                kind="spgist_knn",
                title=f"Consider SP-GiST on {usage.table.name}.{usage.column} for KNN ordering",
                table=usage.table.name,
                severity=(
                    severity_for(benefit, usage.table, cap=Severity.HIGH)
                    if gist is None
                    else Severity.MEDIUM
                ),
                confidence=Confidence.MEDIUM,
                rationale=(
                    f"{usage.knn_calls:,} executions order by "
                    f"{', '.join(operators)} on a "
                    f"{_geometry_type(geometry)} column, and KNN is the dominant access "
                    f"pattern for it."
                ),
                index_type="SP-GiST",
                type_rationale=(
                    "SP-GiST indexes points with a quadtree rather than a bounding-box R-tree. "
                    "Entries are smaller and the tree is shallower, which makes ordered KNN "
                    "scans cheaper. It only applies to point geometries; for anything with "
                    "extent, stay on GiST."
                ),
                ddl=(
                    f"CREATE INDEX CONCURRENTLY {index_name(usage.table.name, usage.column, 'spgist')} "
                    f"ON {usage.table.name} USING SPGIST ({usage.column});"
                ),
                estimated_size_bytes=costmodel.spgist_index_size(usage.table.row_count),
                benefit=benefit,
                caveats=(
                    "SP-GiST supports fewer operators than GiST; keep the GiST index if other "
                    "statements use predicates SP-GiST cannot serve.",
                    "Benchmark both before dropping either — the advantage is real but modest, "
                    "and it depends on the point distribution.",
                ),
                fingerprints=usage.fingerprints(usage.knn),
                docs_url=docs.GIST_OPTIMIZATION,
            )
        )
    return recommendations


def rule_brin_for_append_only(context: RuleContext) -> list[Recommendation]:
    """Recommend BRIN on very large, append-mostly, physically ordered tables."""
    recommendations: list[Recommendation] = []
    for usage in context.usages.values():
        table = usage.table
        if not usage.sargable or usage.geometry is None:
            continue
        if table.row_count < BRIN_MIN_ROWS and table.table_bytes < BRIN_MIN_BYTES:
            continue
        if not table.append_only:
            continue
        correlation = usage.correlation()
        if abs(correlation) < costmodel.STRONG_CORRELATION:
            continue
        if has_spatial_index(table, usage.column, {"brin"}):
            continue

        selectivity, basis = usage.selectivity()
        current = costmodel.sequential_scan_cost(table, spatial_predicates=1)
        projected = costmodel.brin_scan_cost(table, selectivity, correlation)
        gist_size = costmodel.gist_index_size(table.row_count)
        brin_size = costmodel.brin_index_size(table)
        benefit = BenefitEstimate(
            current_cost_per_call=current,
            projected_cost_per_call=projected,
            calls=usage.sargable_calls,
            basis=(
                f"sequential scan vs BRIN scan reading "
                f"{costmodel.brin_effective_fraction(selectivity, correlation):.2%} of the heap "
                f"at correlation {correlation:.2f} ({basis})"
            ),
        )
        recommendations.append(
            Recommendation(
                kind="brin",
                title=f"Consider BRIN on {table.name}.{usage.column} as a low-cost alternative",
                table=table.name,
                severity=Severity.MEDIUM,
                confidence=Confidence.MEDIUM,
                rationale=(
                    f"{table.name} holds {table.row_count:,} rows "
                    f"({costmodel.format_bytes(table.table_bytes)}), is append-mostly "
                    f"({table.write_ratio:.1%} of writes are updates or deletes), and "
                    f"{usage.column} has physical correlation {correlation:.2f}. A BRIN index "
                    f"would occupy about {costmodel.format_bytes(brin_size)} against "
                    f"{costmodel.format_bytes(gist_size)} for GiST."
                ),
                index_type="BRIN",
                type_rationale=(
                    "BRIN stores one bounding box per "
                    f"{costmodel.BRIN_PAGES_PER_RANGE}-page range instead of one entry per row. "
                    "That is only useful when rows that are close in space are also close on "
                    "disk, which is what the high correlation on this column indicates."
                ),
                ddl=(
                    f"CREATE INDEX CONCURRENTLY {index_name(table.name, usage.column, 'brin')} "
                    f"ON {table.name} USING BRIN ({usage.column});"
                ),
                estimated_size_bytes=brin_size,
                benefit=benefit,
                caveats=(
                    "BRIN is a bad idea as soon as the physical ordering breaks down: heavy "
                    "UPDATEs, out-of-order backfills, or a VACUUM FULL with a different "
                    "ordering will silently turn every scan into a full heap read.",
                    "BRIN cannot answer highly selective single-row lookups efficiently, and it "
                    "is not orderable, so it cannot serve KNN queries.",
                    "New page ranges are only summarised by autovacuum or "
                    "brin_summarize_new_values(); until then they are always scanned.",
                    "Keep a GiST index as well if any statement needs precise, selective access.",
                ),
                fingerprints=usage.fingerprints(usage.sargable),
                docs_url=docs.GIST_OPTIMIZATION,
            )
        )
    return recommendations


def rule_partial_index(context: RuleContext) -> list[Recommendation]:
    """Recommend a partial GiST index when a constant filter dominates the traffic."""
    recommendations: list[Recommendation] = []
    for usage in context.usages.values():
        if not usage.sargable or usage.geometry is None:
            continue
        if usage.table.row_count < MIN_ROWS_FOR_INDEX:
            continue
        dominant = _dominant_filters(usage, constant_only=True)
        if not dominant:
            continue
        scalar, calls = dominant[0]
        predicate_sql = scalar.predicate_sql
        if any(
            index.predicate and index.predicate.strip().lower() == predicate_sql.lower()
            for index in usage.table.indexes
        ):
            continue

        fraction = ASSUMED_PARTIAL_FRACTION
        selectivity, basis = usage.selectivity()
        correlation = usage.correlation()
        full_index_cost = costmodel.gist_scan_cost(usage.table, selectivity, correlation)
        partial_cost = costmodel.gist_scan_cost(
            _scaled_table(usage.table, fraction), selectivity, correlation
        )
        has_index = has_spatial_index(usage.table, usage.column, SPATIAL_METHODS)
        current = (
            full_index_cost
            if has_index
            else costmodel.sequential_scan_cost(usage.table, spatial_predicates=1)
        )
        benefit = BenefitEstimate(
            current_cost_per_call=current,
            projected_cost_per_call=partial_cost,
            calls=calls,
            basis=(
                f"{'full GiST index' if has_index else 'sequential scan'} vs a partial index "
                f"covering an assumed {fraction:.0%} of rows ({basis})"
            ),
        )
        recommendations.append(
            Recommendation(
                kind="partial_index",
                title=(
                    f"Add a partial GiST index on {usage.table.name}.{usage.column} "
                    f"WHERE {predicate_sql}"
                ),
                table=usage.table.name,
                severity=severity_for(benefit, usage.table, cap=Severity.HIGH),
                confidence=Confidence.MEDIUM,
                rationale=(
                    f"{calls:,} of {usage.sargable_calls:,} executions against "
                    f"{usage.column} always carry the constant filter `{predicate_sql}`. "
                    f"Restricting the index to those rows shrinks it and keeps the hot part "
                    f"of the tree in cache."
                ),
                index_type="GiST (partial)",
                type_rationale=(
                    "A partial index stores only the rows the workload actually queries. The "
                    "planner will use it only when it can prove the query predicate implies "
                    "the index predicate, so the filter must appear literally in the statement."
                ),
                ddl=(
                    f"CREATE INDEX CONCURRENTLY "
                    f"{index_name(usage.table.name, usage.column, 'gist_partial')} "
                    f"ON {usage.table.name} USING GIST ({usage.column}) "
                    f"WHERE {predicate_sql};"
                ),
                estimated_size_bytes=costmodel.partial_index_size(usage.table.row_count, fraction),
                benefit=benefit,
                caveats=(
                    f"The {fraction:.0%} coverage figure is an assumption; check the real "
                    f"selectivity of `{predicate_sql}` before relying on the size estimate.",
                    "Statements that pass the filter value as a bind parameter will not match "
                    "the index predicate and will fall back to the full index or a seq scan.",
                ),
                fingerprints=usage.fingerprints(usage.sargable),
                docs_url=docs.GIST_OPTIMIZATION,
            )
        )
    return recommendations


def rule_composite_index(context: RuleContext) -> list[Recommendation]:
    """Recommend a composite GiST index when a scalar equality always co-occurs."""
    recommendations: list[Recommendation] = []
    for usage in context.usages.values():
        if not usage.sargable or usage.geometry is None:
            continue
        if usage.table.row_count < MIN_ROWS_FOR_INDEX:
            continue
        # A filter with a literal constant is better served by a partial index,
        # which is smaller and needs no btree_gist; only parameterised equality
        # filters justify carrying the scalar inside the index key.
        equality = [
            (scalar, calls)
            for scalar, calls in _dominant_filters(usage, constant_only=False)
            if scalar.operator == "=" and not scalar.is_constant
        ]
        if not equality:
            continue
        scalar, calls = equality[0]
        scalar_column = scalar.column.column
        if any(
            {c.lower() for c in index.columns} >= {scalar_column.lower(), usage.column.lower()}
            for index in usage.table.indexes
            if index.method.lower() == "gist"
        ):
            continue

        selectivity, basis = usage.selectivity()
        correlation = usage.correlation()
        has_index = has_spatial_index(usage.table, usage.column, SPATIAL_METHODS)
        current = (
            costmodel.gist_scan_cost(usage.table, selectivity, correlation)
            if has_index
            else costmodel.sequential_scan_cost(usage.table, spatial_predicates=1)
        )
        combined = selectivity * ASSUMED_SCALAR_SELECTIVITY
        projected = costmodel.gist_scan_cost(usage.table, combined, correlation)
        benefit = BenefitEstimate(
            current_cost_per_call=current,
            projected_cost_per_call=projected,
            calls=calls,
            basis=(
                f"{'GiST on the geometry alone' if has_index else 'sequential scan'} vs a "
                f"composite index at {combined:.4%} combined selectivity, assuming "
                f"{ASSUMED_SCALAR_SELECTIVITY:.0%} for {scalar_column} ({basis})"
            ),
        )
        recommendations.append(
            Recommendation(
                kind="composite_index",
                title=(
                    f"Add a composite GiST index on "
                    f"{usage.table.name} ({scalar_column}, {usage.column})"
                ),
                table=usage.table.name,
                severity=severity_for(benefit, usage.table, cap=Severity.HIGH),
                confidence=Confidence.MEDIUM,
                rationale=(
                    f"{calls:,} executions filter {usage.column} spatially and "
                    f"{scalar_column} by equality in the same statement. A single index on "
                    f"both lets one scan apply both restrictions instead of rechecking "
                    f"{scalar_column} against the heap."
                ),
                index_type="GiST (composite)",
                type_rationale=(
                    "Putting the equality column first narrows the tree before the geometry "
                    "comparison runs, which is the cheaper order for a highly selective scalar."
                ),
                ddl=(
                    "CREATE EXTENSION IF NOT EXISTS btree_gist;\n"
                    f"CREATE INDEX CONCURRENTLY "
                    f"{index_name(usage.table.name, usage.column, 'gist_' + scalar_column)} "
                    f"ON {usage.table.name} USING GIST ({scalar_column}, {usage.column});"
                ),
                estimated_size_bytes=costmodel.composite_index_size(usage.table.row_count),
                benefit=benefit,
                caveats=(
                    "GiST has no built-in operator class for scalar types: btree_gist must be "
                    "installed, and it is a superuser-or-trusted-extension operation.",
                    "A btree_gist column is less efficient than a real B-tree for the scalar "
                    "part; if the scalar alone is highly selective, a separate B-tree index "
                    "combined via bitmap AND may beat this.",
                    f"The {ASSUMED_SCALAR_SELECTIVITY:.0%} selectivity assumed for "
                    f"{scalar_column} is a placeholder; check pg_stats.n_distinct.",
                ),
                fingerprints=usage.fingerprints(usage.sargable),
                docs_url=docs.GIST_OPTIMIZATION,
            )
        )
    return recommendations


def rule_cluster(context: RuleContext) -> list[Recommendation]:
    """Recommend physical clustering for range-heavy scans on an unordered heap."""
    recommendations: list[Recommendation] = []
    for usage in context.usages.values():
        if not usage.sargable or usage.geometry is None:
            continue
        if usage.table.row_count < MIN_ROWS_FOR_INDEX:
            continue
        index = existing_index_on(usage.table, usage.column, "gist")
        if index is None:
            continue
        correlation = usage.correlation()
        if abs(correlation) >= CLUSTER_MAX_CORRELATION:
            continue
        selectivity, basis = usage.selectivity()
        if selectivity < CLUSTER_MIN_SELECTIVITY:
            continue
        if usage.table.write_ratio > 0.5:
            continue

        current = costmodel.gist_scan_cost(usage.table, selectivity, correlation)
        projected = costmodel.gist_scan_cost(usage.table, selectivity, correlation=1.0)
        benefit = BenefitEstimate(
            current_cost_per_call=current,
            projected_cost_per_call=projected,
            calls=usage.sargable_calls,
            basis=(
                f"random heap fetches at correlation {correlation:.2f} vs sequential fetches "
                f"after clustering, at {selectivity:.4%} selectivity ({basis})"
            ),
        )
        recommendations.append(
            Recommendation(
                kind="cluster",
                title=f"CLUSTER {usage.table.name} on {index.name}",
                table=usage.table.name,
                severity=severity_for(benefit, usage.table, cap=Severity.HIGH),
                confidence=Confidence.MEDIUM,
                rationale=(
                    f"Spatial predicates on {usage.column} match roughly "
                    f"{selectivity:.2%} of {usage.table.row_count:,} rows, so each scan pulls "
                    f"many heap tuples, but physical correlation is only {correlation:.2f}: "
                    f"those tuples are scattered across the whole table and cost a random page "
                    f"fetch each."
                ),
                index_type=None,
                type_rationale="",
                ddl=(
                    f"CLUSTER {usage.table.name} USING {index.name};\n"
                    f"ANALYZE {usage.table.name};"
                ),
                estimated_size_bytes=None,
                benefit=benefit,
                caveats=(
                    "CLUSTER takes an ACCESS EXCLUSIVE lock and rewrites the entire table; it "
                    "needs free space equal to the table plus its indexes.",
                    "The ordering is not maintained: it decays as rows are inserted and "
                    "updated, so this is a periodic maintenance job, not a one-off fix.",
                    "pg_repack achieves the same result without the long exclusive lock.",
                ),
                fingerprints=usage.fingerprints(usage.sargable),
                docs_url=docs.SCHEMA_MIGRATIONS,
            )
        )
    return recommendations


def rule_redundant_indexes(context: RuleContext) -> list[Recommendation]:
    """Report duplicate and prefix-redundant indexes that can be dropped."""
    recommendations: list[Recommendation] = []
    for table in context.catalog.tables.values():
        seen: dict[tuple[str, tuple[str, ...], str | None], ExistingIndex] = {}
        for index in table.indexes:
            keeper = seen.get(index.signature)
            if keeper is None:
                seen[index.signature] = index
                continue
            drop, keep = _choose_drop(keeper, index)
            if drop is None or keep is None:
                continue
            seen[index.signature] = keep
            recommendations.append(_drop_recommendation(table, drop, keep, "an exact duplicate of"))

        for index in table.indexes:
            if index.is_unique or index.predicate:
                continue
            for other in table.indexes:
                if other is index or other.method.lower() != index.method.lower():
                    continue
                if other.predicate or index.signature == other.signature:
                    continue
                if _is_prefix(index.columns, other.columns):
                    recommendations.append(
                        _drop_recommendation(table, index, other, "a leading-column prefix of")
                    )
                    break
    return recommendations


def _choose_drop(
    first: ExistingIndex, second: ExistingIndex
) -> tuple[ExistingIndex | None, ExistingIndex | None]:
    """Decide which of two identical indexes to drop, preferring to keep unique ones."""
    if first.is_unique and not second.is_unique:
        return second, first
    if second.is_unique and not first.is_unique:
        return first, second
    return second, first


def _is_prefix(shorter: Sequence[str], longer: Sequence[str]) -> bool:
    """True when ``shorter`` is a strict leading-column prefix of ``longer``."""
    if len(shorter) >= len(longer):
        return False
    return [c.lower() for c in shorter] == [c.lower() for c in longer[: len(shorter)]]


def _drop_recommendation(
    table: TableStats, drop: ExistingIndex, keep: ExistingIndex, relation: str
) -> Recommendation:
    return Recommendation(
        kind="redundant_index",
        title=f"Drop redundant index {drop.name} on {table.name}",
        table=table.name,
        severity=Severity.LOW,
        confidence=Confidence.HIGH,
        rationale=(
            f"{drop.name} ({drop.method} on {', '.join(drop.columns)}) is {relation} "
            f"{keep.name} ({keep.method} on {', '.join(keep.columns)}). It cannot serve any "
            f"query the other cannot, but it is maintained on every write and vacuumed on "
            f"every cleanup pass."
        ),
        index_type=drop.method,
        type_rationale="",
        ddl=f"DROP INDEX CONCURRENTLY {drop.name};",
        estimated_size_bytes=drop.size_bytes,
        benefit=None,
        caveats=(
            "Confirm with pg_stat_user_indexes.idx_scan that the index really is unused "
            "before dropping it; a constraint or a rarely-run report may depend on it.",
            "DROP INDEX CONCURRENTLY cannot run inside a transaction block.",
        ),
        fingerprints=(),
        docs_url=docs.PERFORMANCE_MONITORING,
    )


def rule_rewrite_advisories(context: RuleContext) -> list[Recommendation]:
    """Report statements that no index can help until they are rewritten."""
    grouped: dict[tuple[str, str, str], list[tuple[WorkloadStatement, SpatialPredicate]]] = (
        defaultdict(list)
    )
    for usage in context.usages.values():
        for statement, predicate in usage.non_sargable:
            if predicate.rewrite_hint is None:
                continue
            key = (usage.table.name, usage.column, predicate.rewrite_hint)
            grouped[key].append((statement, predicate))

    recommendations: list[Recommendation] = []
    for (table_name, column, hint), occurrences in grouped.items():
        table = context.catalog.resolve(table_name)
        if table is None:
            continue
        usage = context.usages[(table.name, column.lower())]
        calls = sum(statement.calls for statement, _ in occurrences)
        selectivity, basis = usage.selectivity()
        current = costmodel.sequential_scan_cost(table, spatial_predicates=1)
        projected = costmodel.gist_scan_cost(table, selectivity, usage.correlation())
        names = sorted({predicate.name for _, predicate in occurrences})
        is_transform = "transform" in hint.lower()
        srids = sorted(
            {
                predicate.transform_srid
                for _, predicate in occurrences
                if predicate.transform_srid is not None
            }
        )
        target_srid = str(srids[0]) if len(srids) == 1 else "<target_srid>"
        ddl = (
            "CREATE INDEX CONCURRENTLY "
            f"{index_name(table.name, column, 'gist_srid_' + target_srid.strip('<>'))} "
            f"ON {table.name} USING GIST (ST_Transform({column}, {target_srid}));"
            if is_transform
            else None
        )
        benefit = BenefitEstimate(
            current_cost_per_call=current,
            projected_cost_per_call=projected,
            calls=calls,
            basis="sequential scan today vs an indexed scan after the rewrite",
        )
        recommendations.append(
            Recommendation(
                kind="rewrite",
                title=f"Rewrite the predicate on {table.name}.{column}: {', '.join(names)}",
                table=table.name,
                severity=severity_for(benefit, table),
                confidence=Confidence.HIGH,
                rationale=(
                    f"{calls:,} executions use {', '.join(names)} against {column} in a form "
                    f"that cannot use any index: "
                    f"{occurrences[0][1].reason or 'the predicate is not sargable'}. "
                    f"No index will help until the statement changes."
                ),
                index_type="expression GiST" if is_transform else None,
                type_rationale=(
                    "An expression index stores the transformed geometry, so the planner can "
                    "match it against the transformed predicate. It costs a second copy of the "
                    "geometry data and must be rebuilt if the target SRID changes."
                    if is_transform
                    else ""
                ),
                ddl=ddl,
                estimated_size_bytes=(
                    costmodel.gist_index_size(table.row_count) if is_transform else None
                ),
                benefit=benefit,
                caveats=(hint,),
                fingerprints=tuple(
                    dict.fromkeys(statement.fingerprint for statement, _ in occurrences)
                ),
                docs_url=docs.CORE_QUERY_PATTERNS,
            )
        )
    return recommendations


#: The rule registry, executed in order. Ordering does not affect ranking, which
#: is done on the modelled saving, but it keeps equal-scoring output stable.
RULES: Final[tuple[Callable[[RuleContext], list[Recommendation]], ...]] = (
    rule_missing_gist,
    rule_rewrite_advisories,
    rule_partial_index,
    rule_composite_index,
    rule_spgist_for_knn,
    rule_brin_for_append_only,
    rule_cluster,
    rule_redundant_indexes,
)

__all__ = [
    "ColumnUsage",
    "RULES",
    "RuleContext",
    "build_context",
    "confidence_for",
    "has_spatial_index",
    "index_name",
    "severity_for",
]
