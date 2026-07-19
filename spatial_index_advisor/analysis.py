"""Extraction of spatial predicates and index-relevant structure from SQL.

The parser is sqlglot (see :mod:`spatial_index_advisor.dialect`); this module only
walks the resulting AST. For each statement it answers the questions the
recommendation rules need:

* which tables and aliases are referenced;
* which geometry columns appear in spatial predicates, and wrapped in what;
* which spatial operators and functions are used, and whether PostGIS can serve
  each of them from a GiST index (sargability);
* which non-spatial equality/range filters co-occur, and whether their right hand
  side is a constant (partial index) or a bind parameter (composite index only);
* whether the statement performs a KNN ``ORDER BY`` and with what ``LIMIT``.

Known limitations are documented in the README: correlated subqueries and CTEs
are flattened into a single alias namespace, and an unqualified column in a
multi-table statement is left unresolved rather than guessed.
"""

from __future__ import annotations

from typing import Final, Iterable

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from .dialect import KNN_OPERATORS, PostGIS, operators_in_text
from .models import ColumnRef, KnnUsage, ScalarFilter, SpatialPredicate, StatementAnalysis

# --------------------------------------------------------------------------- #
# Function classification
# --------------------------------------------------------------------------- #

#: PostGIS relationship functions that internally emit a ``&&`` bounding-box
#: check and are therefore served by a GiST index on the geometry column.
SARGABLE_FUNCTIONS: Final[frozenset[str]] = frozenset(
    {
        "ST_INTERSECTS",
        "ST_CONTAINS",
        "ST_CONTAINSPROPERLY",
        "ST_WITHIN",
        "ST_COVERS",
        "ST_COVEREDBY",
        "ST_OVERLAPS",
        "ST_CROSSES",
        "ST_TOUCHES",
        "ST_EQUALS",
        "ST_DWITHIN",
        "ST_DFULLYWITHIN",
        "ST_3DDWITHIN",
        "ST_3DINTERSECTS",
    }
)

#: Spatial functions that never use an index on their own. Seeing one in a
#: predicate is a signal that the statement needs rewriting, not indexing.
NON_SARGABLE_FUNCTIONS: Final[frozenset[str]] = frozenset(
    {
        "ST_DISTANCE",
        "ST_3DDISTANCE",
        "ST_DISTANCESPHERE",
        "ST_DISTANCESPHEROID",
        "ST_MAXDISTANCE",
        "ST_HAUSDORFFDISTANCE",
        "ST_DISJOINT",
        "ST_RELATE",
        "ST_LENGTH",
        "ST_AREA",
        "ST_PERIMETER",
    }
)

#: Functions that, when wrapped around the column side of a predicate, defeat a
#: plain index on that column because the index stores the untransformed value.
DEFEATING_WRAPPERS: Final[frozenset[str]] = frozenset(
    {
        "ST_TRANSFORM",
        "ST_BUFFER",
        "ST_CENTROID",
        "ST_SIMPLIFY",
        "ST_SIMPLIFYPRESERVETOPOLOGY",
        "ST_MAKEVALID",
        "ST_FORCE2D",
        "ST_FORCE3D",
        "ST_ENVELOPE",
        "ST_COLLECTIONEXTRACT",
        "ST_POINTONSURFACE",
        "ST_REVERSE",
        "ST_SNAPTOGRID",
    }
)

#: Functions that build a constant search geometry. They wrap the *literal* side
#: of a predicate, never the column side, so they are harmless.
_CONSTRUCTORS: Final[frozenset[str]] = frozenset(
    {
        "ST_POINT",
        "ST_MAKEPOINT",
        "ST_SETSRID",
        "ST_GEOMFROMTEXT",
        "ST_GEOGFROMTEXT",
        "ST_GEOMFROMEWKT",
        "ST_MAKEENVELOPE",
        "ST_TILEENVELOPE",
        "ST_GEOMFROMGEOJSON",
    }
)

#: Canonical display spelling for the function names we recognise.
_DISPLAY_NAMES: Final[dict[str, str]] = {
    name.upper(): name
    for name in (
        "ST_Intersects",
        "ST_Contains",
        "ST_ContainsProperly",
        "ST_Within",
        "ST_Covers",
        "ST_CoveredBy",
        "ST_Overlaps",
        "ST_Crosses",
        "ST_Touches",
        "ST_Equals",
        "ST_DWithin",
        "ST_DFullyWithin",
        "ST_3DDWithin",
        "ST_3DIntersects",
        "ST_Distance",
        "ST_3DDistance",
        "ST_DistanceSphere",
        "ST_DistanceSpheroid",
        "ST_MaxDistance",
        "ST_HausdorffDistance",
        "ST_Disjoint",
        "ST_Relate",
        "ST_Length",
        "ST_Area",
        "ST_Perimeter",
        "ST_Transform",
        "ST_Buffer",
        "ST_Centroid",
        "ST_Simplify",
        "ST_MakeValid",
        "ST_Envelope",
    )
}

#: Index of the radius argument for the distance-bounded functions.
_RADIUS_ARGUMENT: Final[dict[str, int]] = {
    "ST_DWITHIN": 2,
    "ST_DFULLYWITHIN": 2,
    "ST_3DDWITHIN": 2,
}

_COMPARISONS: Final[dict[type[exp.Expression], str]] = {
    exp.EQ: "=",
    exp.NEQ: "<>",
    exp.GT: ">",
    exp.GTE: ">=",
    exp.LT: "<",
    exp.LTE: "<=",
}

_DISTANCE_REWRITE: Final[str] = (
    "Replace the distance comparison with ST_DWithin(geom, search_geom, radius); "
    "only ST_DWithin can be answered from a GiST index."
)
_TRANSFORM_REWRITE: Final[str] = (
    "The indexed column is wrapped in a projection/transform function. Either index "
    "the expression itself or store the reprojected geometry in a second column."
)


def _display_name(upper_name: str) -> str:
    """Canonical mixed-case spelling of a PostGIS function name."""
    return _DISPLAY_NAMES.get(upper_name, upper_name)


def _function_name(node: exp.Expression) -> str | None:
    """Upper-case SQL name of a function node, or None if it is not a function."""
    if isinstance(node, exp.Anonymous):
        name = node.name or node.this
        return str(name).upper() if name else None
    if isinstance(node, exp.Func):
        names = type(node).sql_names()
        return names[0].upper() if names else None
    return None


# --------------------------------------------------------------------------- #
# AST helpers
# --------------------------------------------------------------------------- #


def _table_key(table: exp.Table) -> str:
    """Schema-qualified name of a table node, e.g. ``public.vehicles``."""
    parts = [part for part in (table.db, table.name) if part]
    return ".".join(parts)


def _collect_aliases(tree: exp.Expression) -> dict[str, str]:
    """Map every alias and bare table name in the tree to its qualified table name."""
    aliases: dict[str, str] = {}
    for table in tree.find_all(exp.Table):
        key = _table_key(table)
        if not key:
            continue
        aliases.setdefault(table.name, key)
        alias = table.alias
        if alias:
            aliases[alias] = key
    return aliases


def _resolve(column: exp.Column, aliases: dict[str, str], tables: tuple[str, ...]) -> ColumnRef:
    """Turn a sqlglot column node into a :class:`ColumnRef` with a resolved table."""
    qualifier = column.table
    if qualifier:
        return ColumnRef(column=column.name, table=aliases.get(qualifier), alias=qualifier)
    if len(tables) == 1:
        return ColumnRef(column=column.name, table=tables[0])
    return ColumnRef(column=column.name)


def _columns_with_wrappers(
    node: exp.Expression,
) -> list[tuple[exp.Column, tuple[str, ...]]]:
    """Find every column below ``node`` with the function names wrapping it.

    ``ST_Transform(v.geom, 3857)`` yields ``(v.geom, ("ST_Transform",))``; a bare
    column yields an empty wrapper tuple.
    """
    results: list[tuple[exp.Column, tuple[str, ...]]] = []
    for column in node.find_all(exp.Column):
        wrappers: list[str] = []
        parent = column.parent
        while parent is not None and parent is not node.parent:
            name = _function_name(parent)
            if name:
                wrappers.append(_display_name(name))
            if parent is node:
                break
            parent = parent.parent
        results.append((column, tuple(reversed(wrappers))))
    return results


def _is_negated(node: exp.Expression, root: exp.Expression) -> bool:
    """True when ``node`` sits under a ``NOT`` somewhere below ``root``."""
    parent = node.parent
    while parent is not None:
        if isinstance(parent, exp.Not):
            return True
        if parent is root:
            break
        parent = parent.parent
    return False


def _literal_value(node: exp.Expression | None) -> float | None:
    """Numeric value of a literal node, or None when it is not a plain number."""
    if isinstance(node, exp.Literal) and not node.is_string:
        try:
            return float(node.this)
        except (TypeError, ValueError):
            return None
    if isinstance(node, exp.Neg):
        inner = _literal_value(node.this)
        return None if inner is None else -inner
    return None


def _render_constant(node: exp.Expression | None) -> str | None:
    """Render a constant right-hand side back to SQL, or None if it is not constant."""
    if node is None:
        return None
    if isinstance(node, exp.Boolean):
        return "TRUE" if node.this else "FALSE"
    if isinstance(node, exp.Null):
        return "NULL"
    if isinstance(node, exp.Literal):
        return node.sql(dialect=PostGIS)
    if isinstance(node, exp.Neg) and _literal_value(node) is not None:
        return node.sql(dialect=PostGIS)
    if isinstance(node, exp.Cast) and _render_constant(node.this) is not None:
        return node.sql(dialect=PostGIS)
    return None


def _conjuncts(node: exp.Expression | None) -> list[exp.Expression]:
    """Flatten a boolean expression into its top-level AND operands.

    Only these are safe to reuse as a partial-index predicate: anything under an
    ``OR`` is not guaranteed to hold for every matching row.
    """
    if node is None:
        return []
    if isinstance(node, exp.And):
        return _conjuncts(node.this) + _conjuncts(node.expression)
    if isinstance(node, exp.Paren):
        return _conjuncts(node.this)
    return [node]


def _predicate_roots(tree: exp.Expression) -> list[exp.Expression]:
    """Every boolean context that a spatial predicate can usefully appear in."""
    roots: list[exp.Expression] = []
    for where in tree.find_all(exp.Where):
        if where.this is not None:
            roots.append(where.this)
    for join in tree.find_all(exp.Join):
        on = join.args.get("on")
        if on is not None:
            roots.append(on)
    return roots


# --------------------------------------------------------------------------- #
# Predicate extraction
# --------------------------------------------------------------------------- #


def _overlap_operator_name(operators_seen: frozenset[str]) -> str:
    """Pick the display name for a bounding-box overlap node."""
    if "&&" in operators_seen:
        return "&&"
    return "&&&" if "&&&" in operators_seen else "&&"


def _knn_operator_name(operators_seen: frozenset[str]) -> str:
    """Pick the display name for a distance node.

    ``<->`` and ``<#>`` parse to the same AST node, so when a statement uses
    exactly one of them we can name it precisely; a statement mixing them is
    reported under ``<->`` and the full operator set stays available on the
    analysis object.
    """
    present = sorted(operators_seen & KNN_OPERATORS)
    return present[0] if len(present) == 1 else "<->"


class _StatementWalker:
    """Single-use walker turning one parsed statement into a StatementAnalysis."""

    def __init__(self, tree: exp.Expression, sql: str, statement_fingerprint: str) -> None:
        self._tree = tree
        self._fingerprint = statement_fingerprint
        self._operators = operators_in_text(sql)
        self._aliases = _collect_aliases(tree)
        self._tables = tuple(sorted(set(self._aliases.values())))
        self._spatial: list[SpatialPredicate] = []
        self._spatial_columns: set[tuple[str | None, str]] = set()

    def run(self) -> StatementAnalysis:
        """Perform the walk and return the finished analysis."""
        roots = _predicate_roots(self._tree)
        for root in roots:
            self._walk_predicates(root)
        knn = self._extract_knn()
        scalars = self._extract_scalar_filters(roots)
        return StatementAnalysis(
            fingerprint=self._fingerprint,
            tables=self._tables,
            aliases=dict(self._aliases),
            spatial_predicates=tuple(self._spatial),
            scalar_filters=tuple(scalars),
            knn=tuple(knn),
            limit=self._extract_limit(),
            operators_seen=self._operators,
        )

    # -- spatial predicates ------------------------------------------------- #

    def _walk_predicates(self, root: exp.Expression) -> None:
        for node in root.find_all(exp.ArrayOverlaps):
            self._add_overlap(node, root)
        for node in root.find_all(exp.Func):
            self._add_function(node, root)
        for node in root.find_all(exp.Distance):
            self._add_distance_operator(node, root)

    def _column_refs(
        self, node: exp.Expression
    ) -> list[tuple[ColumnRef, tuple[str, ...]]]:
        return [
            (_resolve(column, self._aliases, self._tables), wrappers)
            for column, wrappers in _columns_with_wrappers(node)
        ]

    def _record(self, predicate: SpatialPredicate) -> None:
        self._spatial.append(predicate)
        for column in predicate.columns:
            self._spatial_columns.add((column.table, column.column.lower()))

    def _add_overlap(self, node: exp.ArrayOverlaps, root: exp.Expression) -> None:
        refs = self._column_refs(node.this) + self._column_refs(node.expression)
        if not refs:
            return
        negated = _is_negated(node, root)
        defeating = self._defeating(refs)
        self._record(
            SpatialPredicate(
                name=_overlap_operator_name(self._operators),
                kind="operator",
                columns=tuple(ref for ref, _ in refs),
                sargable=not negated and not defeating,
                reason=self._reason(negated, defeating),
                wrapping_functions=defeating,
                negated=negated,
                rewrite_hint=_TRANSFORM_REWRITE if defeating else None,
            )
        )

    def _add_function(self, node: exp.Func, root: exp.Expression) -> None:
        name = _function_name(node)
        if name is None:
            return
        if name not in SARGABLE_FUNCTIONS and name not in NON_SARGABLE_FUNCTIONS:
            return
        arguments = _function_arguments(node)
        refs: list[tuple[ColumnRef, tuple[str, ...]]] = []
        for argument in arguments:
            refs.extend(
                (ref, wrappers)
                for ref, wrappers in self._column_refs(argument)
                if not _is_constructor_only(wrappers)
            )
        if not refs:
            return
        negated = _is_negated(node, root)
        defeating = self._defeating(refs)
        sargable = name in SARGABLE_FUNCTIONS and not negated and not defeating
        rewrite: str | None = None
        if name in NON_SARGABLE_FUNCTIONS and _is_distance_comparison(node):
            rewrite = _DISTANCE_REWRITE
        elif defeating:
            rewrite = _TRANSFORM_REWRITE
        radius_index = _RADIUS_ARGUMENT.get(name)
        radius = (
            _literal_value(arguments[radius_index])
            if radius_index is not None and len(arguments) > radius_index
            else None
        )
        self._record(
            SpatialPredicate(
                name=_display_name(name),
                kind="function",
                columns=tuple(ref for ref, _ in refs),
                sargable=sargable,
                reason=self._function_reason(name, negated, defeating),
                radius=radius,
                wrapping_functions=defeating,
                negated=negated,
                rewrite_hint=rewrite,
                transform_srid=_transform_srid(node) if defeating else None,
            )
        )

    def _add_distance_operator(self, node: exp.Distance, root: exp.Expression) -> None:
        """Record ``<->`` used as a filter rather than as an ORDER BY key."""
        if _in_order_by(node):
            return
        parent = node.parent
        if not isinstance(parent, (exp.LT, exp.LTE, exp.GT, exp.GTE)):
            return
        refs = self._column_refs(node.this) + self._column_refs(node.expression)
        if not refs:
            return
        self._record(
            SpatialPredicate(
                name=_knn_operator_name(self._operators),
                kind="operator",
                columns=tuple(ref for ref, _ in refs),
                sargable=False,
                reason=(
                    "a distance operator in WHERE is not index-accelerated; it is only "
                    "index-orderable in ORDER BY"
                ),
                negated=_is_negated(node, root),
                rewrite_hint=_DISTANCE_REWRITE,
            )
        )

    def _defeating(
        self, refs: Iterable[tuple[ColumnRef, tuple[str, ...]]]
    ) -> tuple[str, ...]:
        found: list[str] = []
        for _, wrappers in refs:
            found.extend(w for w in wrappers if w.upper() in DEFEATING_WRAPPERS)
        return tuple(dict.fromkeys(found))

    @staticmethod
    def _reason(negated: bool, defeating: tuple[str, ...]) -> str:
        if negated:
            return "predicate is negated, so no index can restrict the scan"
        if defeating:
            return f"column is wrapped in {', '.join(defeating)}, which the index does not store"
        return ""

    def _function_reason(
        self, name: str, negated: bool, defeating: tuple[str, ...]
    ) -> str:
        base = self._reason(negated, defeating)
        if base:
            return base
        if name in NON_SARGABLE_FUNCTIONS:
            return f"{_display_name(name)} does not emit a bounding-box check, so no index applies"
        return ""

    # -- KNN, scalars, limit ------------------------------------------------ #

    def _extract_knn(self) -> list[KnnUsage]:
        limit = self._extract_limit()
        usages: list[KnnUsage] = []
        for order in self._tree.find_all(exp.Order):
            for ordered in order.find_all(exp.Ordered):
                for node in ordered.find_all(exp.Distance):
                    refs = self._column_refs(node.this) + self._column_refs(node.expression)
                    if not refs:
                        continue
                    column, _ = refs[0]
                    usages.append(
                        KnnUsage(
                            column=column,
                            operator=_knn_operator_name(self._operators),
                            limit=limit,
                        )
                    )
        return usages

    def _extract_scalar_filters(self, roots: list[exp.Expression]) -> list[ScalarFilter]:
        filters: list[ScalarFilter] = []
        for root in roots:
            for conjunct in _conjuncts(root):
                filters.extend(self._scalar_filter(conjunct))
        return filters

    def _scalar_filter(self, node: exp.Expression) -> list[ScalarFilter]:
        if isinstance(node, exp.Is):
            column = node.this
            if isinstance(column, exp.Column) and isinstance(node.expression, exp.Null):
                return self._maybe_scalar(column, "IS NULL", None, True)
            return []
        if isinstance(node, exp.Not) and isinstance(node.this, exp.Is):
            inner = node.this
            if isinstance(inner.this, exp.Column) and isinstance(inner.expression, exp.Null):
                return self._maybe_scalar(inner.this, "IS NOT NULL", None, True)
            return []
        if isinstance(node, exp.Column):
            # A bare boolean column used as a predicate: ``WHERE active``.
            return self._maybe_scalar(node, "=", "TRUE", True)
        operator = _COMPARISONS.get(type(node))
        if operator is None or not isinstance(node, exp.Binary):
            return []
        if not isinstance(node.this, exp.Column):
            return []
        rendered = _render_constant(node.expression)
        return self._maybe_scalar(node.this, operator, rendered, rendered is not None)

    def _maybe_scalar(
        self, column: exp.Column, operator: str, literal: str | None, is_constant: bool
    ) -> list[ScalarFilter]:
        ref = _resolve(column, self._aliases, self._tables)
        if (ref.table, ref.column.lower()) in self._spatial_columns:
            return []
        return [
            ScalarFilter(
                column=ref, operator=operator, literal=literal, is_constant=is_constant
            )
        ]

    def _extract_limit(self) -> int | None:
        limit = self._tree.args.get("limit")
        if isinstance(limit, exp.Limit):
            value = _literal_value(limit.expression)
            if value is not None:
                return int(value)
        return None


def _function_arguments(node: exp.Func) -> list[exp.Expression]:
    """Positional arguments of a function node, typed or anonymous."""
    if isinstance(node, exp.Anonymous):
        return list(node.expressions)
    arguments: list[exp.Expression] = []
    for key in node.arg_types:
        value = node.args.get(key)
        if isinstance(value, list):
            arguments.extend(v for v in value if isinstance(v, exp.Expression))
        elif isinstance(value, exp.Expression):
            arguments.append(value)
    return arguments


def _transform_srid(node: exp.Expression) -> int | None:
    """Target SRID of an ``ST_Transform`` call below ``node``, if it is a literal.

    Knowing it lets the advisor emit a runnable expression-index DDL instead of a
    placeholder.
    """
    for candidate in node.find_all(exp.Func):
        if _function_name(candidate) != "ST_TRANSFORM":
            continue
        arguments = _function_arguments(candidate)
        if len(arguments) < 2:
            continue
        value = _literal_value(arguments[1])
        if value is not None:
            return int(value)
    return None


def _is_constructor_only(wrappers: tuple[str, ...]) -> bool:
    """True when a column sits inside geometry-construction functions only.

    Such a column is part of a search geometry expression, not the indexable side
    of the predicate.
    """
    return bool(wrappers) and all(w.upper() in _CONSTRUCTORS for w in wrappers)


def _is_distance_comparison(node: exp.Expression) -> bool:
    """True when a distance function is compared against a threshold."""
    parent = node.parent
    return isinstance(parent, (exp.LT, exp.LTE, exp.GT, exp.GTE))


def _in_order_by(node: exp.Expression) -> bool:
    """True when ``node`` appears inside an ORDER BY clause."""
    parent = node.parent
    while parent is not None:
        if isinstance(parent, exp.Order):
            return True
        parent = parent.parent
    return False


def analyze_statement(sql: str, statement_fingerprint: str) -> StatementAnalysis:
    """Analyse one SQL statement.

    Returns a :class:`StatementAnalysis` whose ``parse_error`` is set when the
    statement cannot be parsed; callers should skip such statements rather than
    treat the empty fields as meaningful.
    """
    text = sql.strip().rstrip(";").strip()
    if not text:
        return StatementAnalysis(fingerprint=statement_fingerprint, parse_error="empty statement")
    try:
        tree = sqlglot.parse_one(text, dialect=PostGIS)
    except SqlglotError as error:
        return StatementAnalysis(
            fingerprint=statement_fingerprint, parse_error=str(error).splitlines()[0]
        )
    if tree is None:
        return StatementAnalysis(fingerprint=statement_fingerprint, parse_error="no statement found")
    if isinstance(tree, exp.Command):
        # sqlglot could not model the statement and kept it as opaque text.
        return StatementAnalysis(
            fingerprint=statement_fingerprint, parse_error="unsupported statement type"
        )
    return _StatementWalker(tree, text, statement_fingerprint).run()
