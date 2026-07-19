"""Core data structures shared by the parsing, analysis and rendering layers.

Everything here is a plain frozen/immutable-ish dataclass with no behaviour that
depends on a database connection, which is what makes the whole engine runnable
from a static JSON snapshot.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from typing import Any, Final

_BIND_PARAMETER_RE: Final[re.Pattern[str]] = re.compile(r"\$\d+")

# --------------------------------------------------------------------------- #
# Workload
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class WorkloadStatement:
    """One normalized statement class with aggregated execution statistics.

    ``fingerprint`` identifies the statement shape; every raw variant that
    normalizes to the same shape is folded into a single instance whose counters
    are the sum of its variants.
    """

    fingerprint: str
    normalized_sql: str
    sample_sql: str
    calls: int
    total_exec_time_ms: float
    rows: int
    source: str

    @property
    def mean_exec_time_ms(self) -> float:
        """Average wall-clock time per execution, or 0.0 when call count is unknown."""
        return self.total_exec_time_ms / self.calls if self.calls else 0.0

    @property
    def bind_parameter_count(self) -> int:
        """Number of ``$n`` bind parameters in the representative sample.

        A sample with real literals carries strictly more information — a radius,
        a filter value a partial index could use — so the variant with the fewest
        parameters wins when two variants of the same fingerprint are merged.
        """
        return len(_BIND_PARAMETER_RE.findall(self.sample_sql))

    def merged_with(self, other: WorkloadStatement) -> WorkloadStatement:
        """Return the sum of two statements sharing a fingerprint."""
        if other.fingerprint != self.fingerprint:
            raise ValueError(
                f"cannot merge different fingerprints: {self.fingerprint} != {other.fingerprint}"
            )
        best = other if other.bind_parameter_count < self.bind_parameter_count else self
        return WorkloadStatement(
            fingerprint=self.fingerprint,
            normalized_sql=self.normalized_sql,
            sample_sql=best.sample_sql,
            calls=self.calls + other.calls,
            total_exec_time_ms=self.total_exec_time_ms + other.total_exec_time_ms,
            rows=self.rows + other.rows,
            source=self.source if self.source == other.source else "multiple",
        )


@dataclass(frozen=True)
class Workload:
    """A parsed workload: statements plus the sources they came from."""

    statements: tuple[WorkloadStatement, ...]
    sources: tuple[str, ...]

    @property
    def total_calls(self) -> int:
        """Total number of executions across every statement."""
        return sum(s.calls for s in self.statements)

    @property
    def total_exec_time_ms(self) -> float:
        """Total execution time across every statement."""
        return sum(s.total_exec_time_ms for s in self.statements)


# --------------------------------------------------------------------------- #
# SQL analysis
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ColumnRef:
    """A column reference resolved (where possible) to a real table name."""

    column: str
    table: str | None = None
    alias: str | None = None

    @property
    def qualified(self) -> str:
        """``table.column`` when the table is known, otherwise just the column."""
        return f"{self.table}.{self.column}" if self.table else self.column

    def __str__(self) -> str:
        return self.qualified


@dataclass(frozen=True)
class SpatialPredicate:
    """A single spatial operator or function occurrence in a statement.

    ``sargable`` records whether an index on ``columns`` could actually be used
    for this predicate. ``wrapping_functions`` lists functions applied to the
    column itself (``ST_Transform(geom, 3857)``), which is the usual reason a
    predicate stops being sargable.
    """

    name: str
    kind: str  # "operator" | "function"
    columns: tuple[ColumnRef, ...]
    sargable: bool
    reason: str = ""
    radius: float | None = None
    wrapping_functions: tuple[str, ...] = ()
    negated: bool = False
    rewrite_hint: str | None = None
    transform_srid: int | None = None


@dataclass(frozen=True)
class ScalarFilter:
    """A non-spatial filter that co-occurs with a spatial predicate.

    ``literal`` is the rendered right-hand side when it is a constant, which is
    what makes a partial index possible; parameterised filters have
    ``is_constant=False`` and can only support a composite index.
    """

    column: ColumnRef
    operator: str
    literal: str | None = None
    is_constant: bool = False

    @property
    def predicate_sql(self) -> str:
        """The filter rendered back as a SQL boolean expression."""
        if self.operator == "IS NULL":
            return f"{self.column.column} IS NULL"
        if self.operator == "IS NOT NULL":
            return f"{self.column.column} IS NOT NULL"
        return f"{self.column.column} {self.operator} {self.literal}"


@dataclass(frozen=True)
class KnnUsage:
    """An ``ORDER BY geom <-> ...`` nearest-neighbour sort."""

    column: ColumnRef
    operator: str
    limit: int | None


@dataclass(frozen=True)
class StatementAnalysis:
    """Structural facts extracted from one statement.

    When ``parse_error`` is set the statement could not be parsed and every other
    field is empty; such statements are reported but never drive a recommendation.
    """

    fingerprint: str
    tables: tuple[str, ...] = ()
    aliases: dict[str, str] = field(default_factory=dict)
    spatial_predicates: tuple[SpatialPredicate, ...] = ()
    scalar_filters: tuple[ScalarFilter, ...] = ()
    knn: tuple[KnnUsage, ...] = ()
    limit: int | None = None
    operators_seen: frozenset[str] = frozenset()
    parse_error: str | None = None

    @property
    def is_spatial(self) -> bool:
        """True when the statement contains at least one spatial predicate or KNN sort."""
        return bool(self.spatial_predicates or self.knn)

    def geometry_columns_for(self, table: str) -> set[str]:
        """Names of geometry columns of ``table`` touched by this statement."""
        found: set[str] = set()
        for predicate in self.spatial_predicates:
            found.update(c.column for c in predicate.columns if c.table == table)
        found.update(k.column.column for k in self.knn if k.column.table == table)
        return found


# --------------------------------------------------------------------------- #
# Catalog
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GeometryColumn:
    """A geometry column and the statistics the cost model needs about it."""

    name: str
    geometry_type: str = "GEOMETRY"
    srid: int = 0
    avg_bbox_width: float | None = None
    avg_bbox_height: float | None = None
    extent_width: float | None = None
    extent_height: float | None = None
    correlation: float | None = None

    @property
    def is_point(self) -> bool:
        """True for point-typed columns, for which SP-GiST is a candidate."""
        return self.geometry_type.upper().replace("MULTI", "").startswith("POINT")

    @property
    def average_area_fraction(self) -> float | None:
        """Mean feature bbox area as a fraction of the layer extent, when known."""
        if None in (self.avg_bbox_width, self.avg_bbox_height):
            return None
        if not self.extent_width or not self.extent_height:
            return None
        assert self.avg_bbox_width is not None and self.avg_bbox_height is not None
        extent_area = self.extent_width * self.extent_height
        if extent_area <= 0:
            return None
        return (self.avg_bbox_width * self.avg_bbox_height) / extent_area


@dataclass(frozen=True)
class ExistingIndex:
    """An index that already exists on a table."""

    name: str
    method: str
    columns: tuple[str, ...]
    predicate: str | None = None
    is_unique: bool = False
    size_bytes: int | None = None
    definition: str | None = None

    @property
    def signature(self) -> tuple[str, tuple[str, ...], str | None]:
        """Identity used for duplicate detection: method, column list, predicate."""
        return (self.method.lower(), tuple(c.lower() for c in self.columns), self.predicate)


@dataclass(frozen=True)
class TableStats:
    """Everything the engine knows about one table."""

    name: str
    row_count: int
    table_bytes: int
    geometry_columns: tuple[GeometryColumn, ...] = ()
    indexes: tuple[ExistingIndex, ...] = ()
    column_correlation: dict[str, float] = field(default_factory=dict)
    append_only: bool = False
    updates: int = 0
    deletes: int = 0
    inserts: int = 0

    @property
    def pages(self) -> int:
        """Table size in 8 kB heap pages (minimum 1)."""
        return max(1, self.table_bytes // 8192)

    def geometry_column(self, name: str) -> GeometryColumn | None:
        """Look up a geometry column by name, case-insensitively."""
        lowered = name.lower()
        for column in self.geometry_columns:
            if column.name.lower() == lowered:
                return column
        return None

    def indexes_on(self, column: str, method: str | None = None) -> list[ExistingIndex]:
        """Existing indexes whose leading column is ``column``."""
        lowered = column.lower()
        return [
            index
            for index in self.indexes
            if index.columns
            and index.columns[0].lower() == lowered
            and (method is None or index.method.lower() == method.lower())
        ]

    @property
    def write_ratio(self) -> float:
        """Fraction of writes that are updates or deletes rather than inserts."""
        total = self.inserts + self.updates + self.deletes
        if total == 0:
            return 0.0
        return (self.updates + self.deletes) / total


@dataclass(frozen=True)
class CatalogSnapshot:
    """A point-in-time description of the tables the workload touches."""

    tables: dict[str, TableStats]
    database: str | None = None
    collected_at: str | None = None
    postgis_version: str | None = None

    def resolve(self, name: str) -> TableStats | None:
        """Find a table by qualified or bare name, case-insensitively.

        ``public.vehicles``, ``vehicles`` and ``VEHICLES`` all resolve to the
        same entry when it is unambiguous.
        """
        lowered = name.lower()
        for key, table in self.tables.items():
            if key.lower() == lowered:
                return table
        bare_matches = [
            table
            for key, table in self.tables.items()
            if key.lower().rsplit(".", 1)[-1] == lowered.rsplit(".", 1)[-1]
        ]
        if len(bare_matches) == 1:
            return bare_matches[0]
        return None


# --------------------------------------------------------------------------- #
# Recommendations
# --------------------------------------------------------------------------- #


class Severity(enum.Enum):
    """How urgently a recommendation should be acted on."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def rank(self) -> int:
        """Sort key, lower is more urgent."""
        return _SEVERITY_ORDER[self]


_SEVERITY_ORDER: dict[Severity, int] = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
    Severity.LOW: 3,
}


class Confidence(enum.Enum):
    """How much the advisor trusts its own estimate for a recommendation."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class BenefitEstimate:
    """A heuristic before/after cost comparison for one recommendation.

    All figures are model output in arbitrary PostgreSQL cost units, not
    measurements. They exist to rank recommendations against each other.
    """

    current_cost_per_call: float
    projected_cost_per_call: float
    calls: int
    basis: str

    @property
    def cost_saved_per_call(self) -> float:
        """Modelled cost units saved by a single execution."""
        return max(0.0, self.current_cost_per_call - self.projected_cost_per_call)

    @property
    def total_cost_saved(self) -> float:
        """Modelled cost units saved across the whole observed workload."""
        return self.cost_saved_per_call * self.calls

    @property
    def speedup(self) -> float:
        """Modelled ratio of current cost to projected cost."""
        if self.projected_cost_per_call <= 0:
            return float("inf")
        return self.current_cost_per_call / self.projected_cost_per_call


@dataclass(frozen=True)
class Recommendation:
    """One ranked, actionable finding."""

    kind: str
    title: str
    table: str
    severity: Severity
    confidence: Confidence
    rationale: str
    docs_url: str
    index_type: str | None = None
    type_rationale: str = ""
    ddl: str | None = None
    estimated_size_bytes: int | None = None
    benefit: BenefitEstimate | None = None
    caveats: tuple[str, ...] = ()
    fingerprints: tuple[str, ...] = ()

    @property
    def score(self) -> float:
        """Ranking score: modelled total cost saved, 0 when there is no estimate."""
        return self.benefit.total_cost_saved if self.benefit else 0.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form used by ``--format json``."""
        payload: dict[str, Any] = {
            "kind": self.kind,
            "title": self.title,
            "table": self.table,
            "severity": self.severity.value,
            "confidence": self.confidence.value,
            "rationale": self.rationale,
            "index_type": self.index_type,
            "type_rationale": self.type_rationale or None,
            "ddl": self.ddl,
            "estimated_size_bytes": self.estimated_size_bytes,
            "caveats": list(self.caveats),
            "fingerprints": list(self.fingerprints),
            "docs_url": self.docs_url,
            "estimate_is_heuristic": True,
        }
        if self.benefit:
            payload["benefit"] = {
                "current_cost_per_call": round(self.benefit.current_cost_per_call, 2),
                "projected_cost_per_call": round(self.benefit.projected_cost_per_call, 2),
                "calls": self.benefit.calls,
                "total_cost_saved": round(self.benefit.total_cost_saved, 2),
                "speedup": (
                    None
                    if self.benefit.speedup == float("inf")
                    else round(self.benefit.speedup, 1)
                ),
                "basis": self.benefit.basis,
            }
        else:
            payload["benefit"] = None
        return payload


@dataclass(frozen=True)
class AdvisorReport:
    """The complete result of a run."""

    recommendations: tuple[Recommendation, ...]
    workload: Workload
    analyses: dict[str, StatementAnalysis]
    unparsed: tuple[str, ...] = ()
    unknown_tables: tuple[str, ...] = ()

    @property
    def has_critical(self) -> bool:
        """True when at least one recommendation is CRITICAL."""
        return any(r.severity is Severity.CRITICAL for r in self.recommendations)

    def top(self, n: int | None) -> tuple[Recommendation, ...]:
        """The first ``n`` recommendations, or all of them when ``n`` is None."""
        return self.recommendations if n is None else self.recommendations[:n]
