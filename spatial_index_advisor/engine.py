"""The recommendation engine: workload + catalog in, ranked findings out.

This module is the public entry point for library use::

    from pathlib import Path
    from spatial_index_advisor import analyse, load_catalog, load_workload

    report = analyse(
        load_workload([Path("examples/pg_stat_statements.csv")]),
        load_catalog(Path("examples/catalog.json")),
    )
    for recommendation in report.top(5):
        print(recommendation.title, recommendation.ddl)

It performs no I/O of its own, which keeps it deterministic and trivially
testable.
"""

from __future__ import annotations

from .analysis import analyze_statement
from .models import AdvisorReport, CatalogSnapshot, Recommendation, StatementAnalysis, Workload
from .rules import RULES, build_context


def analyse_statements(workload: Workload) -> dict[str, StatementAnalysis]:
    """Run the SQL analyser over every statement in the workload."""
    return {
        statement.fingerprint: analyze_statement(statement.sample_sql, statement.fingerprint)
        for statement in workload.statements
    }


def _rank(recommendations: list[Recommendation]) -> tuple[Recommendation, ...]:
    """Order findings by severity, then by modelled saving, then by table name."""
    return tuple(
        sorted(
            recommendations,
            key=lambda r: (r.severity.rank, -r.score, r.table, r.kind),
        )
    )


def analyse(workload: Workload, catalog: CatalogSnapshot) -> AdvisorReport:
    """Produce a ranked report for ``workload`` against ``catalog``.

    Statements that fail to parse and tables absent from the snapshot are
    collected on the report rather than raising, so that one bad line in a log
    file cannot suppress the rest of the analysis.
    """
    analyses = analyse_statements(workload)

    unparsed = tuple(
        fingerprint
        for fingerprint, analysis in analyses.items()
        if analysis.parse_error is not None
    )
    referenced: dict[str, None] = {}
    for analysis in analyses.values():
        if analysis.parse_error is not None:
            continue
        for name in analysis.tables:
            if catalog.resolve(name) is None:
                referenced.setdefault(name, None)

    context = build_context(catalog, workload, analyses)
    findings: list[Recommendation] = []
    for rule in RULES:
        findings.extend(rule(context))

    return AdvisorReport(
        recommendations=_rank(findings),
        workload=workload,
        analyses=analyses,
        unparsed=unparsed,
        unknown_tables=tuple(referenced),
    )
