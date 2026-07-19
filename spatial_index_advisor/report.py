"""Rendering of an :class:`~spatial_index_advisor.models.AdvisorReport`.

Three output formats share one report object:

``rich``
    A terminal report with a summary table and one panel per recommendation.
``json``
    A machine-readable document for pipelines and diffing between runs.
``sql``
    Nothing but the DDL, in rank order, with the reasoning as SQL comments, so it
    can be reviewed and applied by hand.

All three restate that the benefit figures are model output.
"""

from __future__ import annotations

import json
from typing import Final

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .costmodel import format_bytes
from .models import AdvisorReport, Confidence, Recommendation, Severity

#: Shown in every output format so the numbers are never mistaken for measurements.
ESTIMATE_DISCLAIMER: Final[str] = (
    "Benefit and size figures are heuristic estimates from a static cost model, "
    "not measurements. Verify each change with EXPLAIN (ANALYZE, BUFFERS)."
)

FORMAT_CHOICES: Final[tuple[str, ...]] = ("rich", "json", "sql")

_SEVERITY_STYLE: Final[dict[Severity, str]] = {
    Severity.CRITICAL: "bold red",
    Severity.HIGH: "bold yellow",
    Severity.MEDIUM: "cyan",
    Severity.LOW: "dim",
}

_CONFIDENCE_STYLE: Final[dict[Confidence, str]] = {
    Confidence.HIGH: "green",
    Confidence.MEDIUM: "yellow",
    Confidence.LOW: "red",
}


def _severity_text(recommendation: Recommendation) -> Text:
    return Text(
        recommendation.severity.value.upper(),
        style=_SEVERITY_STYLE[recommendation.severity],
    )


def _benefit_summary(recommendation: Recommendation) -> str:
    benefit = recommendation.benefit
    if benefit is None:
        return "no estimate"
    speedup = "unbounded" if benefit.speedup == float("inf") else f"{benefit.speedup:.0f}x"
    return (
        f"{benefit.total_cost_saved:,.0f} cost units over {benefit.calls:,} calls "
        f"(~{speedup} per call, estimated)"
    )


def _compact_number(value: float) -> str:
    """Abbreviate a large figure so it fits in a table cell."""
    for limit, suffix in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= limit:
            return f"{value / limit:.1f}{suffix}"
    return f"{value:.0f}"


def _benefit_compact(recommendation: Recommendation) -> str:
    """Short benefit cell: modelled saving and per-call speedup."""
    benefit = recommendation.benefit
    if benefit is None:
        return "-"
    speedup = "inf" if benefit.speedup == float("inf") else f"{benefit.speedup:.0f}x"
    return f"{_compact_number(benefit.total_cost_saved)} / {speedup}"


def render_summary_table(report: AdvisorReport, top: int | None) -> Table:
    """Build the one-line-per-recommendation overview table."""
    table = Table(title="Spatial index recommendations", header_style="bold")
    table.add_column("#", justify="right", width=3)
    table.add_column("Severity", width=8)
    table.add_column("Recommendation", overflow="fold", min_width=32)
    table.add_column("Type", width=14, overflow="fold")
    table.add_column("Saving/speedup", justify="right", width=14)
    table.add_column("Conf.", width=6)

    for position, recommendation in enumerate(report.top(top), start=1):
        table.add_row(
            str(position),
            _severity_text(recommendation),
            recommendation.title,
            recommendation.index_type or "-",
            _benefit_compact(recommendation),
            Text(
                recommendation.confidence.value,
                style=_CONFIDENCE_STYLE[recommendation.confidence],
            ),
        )
    table.caption = "Saving is modelled cost units across the workload; speedup is per call."
    return table


def render_detail_panel(position: int, recommendation: Recommendation) -> Panel:
    """Build the detail panel for a single recommendation."""
    body = Text()
    body.append(recommendation.rationale + "\n")
    if recommendation.type_rationale:
        body.append("\nWhy this index type: ", style="bold")
        body.append(recommendation.type_rationale + "\n")
    if recommendation.ddl:
        body.append("\n")
        body.append(recommendation.ddl + "\n", style="bold green")
    details: list[str] = []
    if recommendation.estimated_size_bytes is not None:
        details.append(f"estimated size {format_bytes(recommendation.estimated_size_bytes)}")
    if recommendation.benefit is not None:
        details.append(f"basis: {recommendation.benefit.basis}")
    details.append(f"confidence: {recommendation.confidence.value}")
    if details:
        body.append("\n" + "; ".join(details) + "\n", style="dim")
    if recommendation.caveats:
        body.append("\nCaveats:\n", style="bold")
        for caveat in recommendation.caveats:
            body.append(f"  - {caveat}\n")
    if recommendation.fingerprints:
        shown = ", ".join(recommendation.fingerprints[:6])
        extra = (
            f" (+{len(recommendation.fingerprints) - 6} more)"
            if len(recommendation.fingerprints) > 6
            else ""
        )
        body.append(f"\nStatements: {shown}{extra}\n", style="dim")
    body.append(f"\nFurther reading: {recommendation.docs_url}", style="blue")

    title = Text(f"{position}. ")
    title.append(_severity_text(recommendation))
    title.append(f"  {recommendation.title}")
    return Panel(body, title=title, border_style=_SEVERITY_STYLE[recommendation.severity])


def render_terminal(report: AdvisorReport, console: Console, top: int | None = None) -> None:
    """Write the full terminal report to ``console``."""
    workload = report.workload
    console.print(
        Text.assemble(
            ("Workload: ", "bold"),
            (
                f"{len(workload.statements):,} statement fingerprints, "
                f"{workload.total_calls:,} calls, "
                f"{workload.total_exec_time_ms:,.0f} ms total execution time",
                "",
            ),
        )
    )
    console.print(Text(f"Sources: {', '.join(workload.sources)}", style="dim"))

    if not report.recommendations:
        console.print(
            Panel(
                "No recommendations. Every spatial predicate in this workload is already "
                "served by an index, or the tables involved are too small to benefit.",
                border_style="green",
            )
        )
    else:
        console.print(render_summary_table(report, top))
        for position, recommendation in enumerate(report.top(top), start=1):
            console.print(render_detail_panel(position, recommendation))

    for warning in _warnings(report):
        console.print(Text(warning, style="yellow"))
    console.print(Text(ESTIMATE_DISCLAIMER, style="dim italic"))


def _warnings(report: AdvisorReport) -> list[str]:
    """Non-fatal problems worth telling the user about."""
    messages: list[str] = []
    if report.unparsed:
        messages.append(
            f"{len(report.unparsed)} statement(s) could not be parsed and were skipped."
        )
    if report.unknown_tables:
        shown = ", ".join(sorted(report.unknown_tables)[:8])
        messages.append(
            f"{len(report.unknown_tables)} referenced table(s) are absent from the catalog "
            f"snapshot and were ignored: {shown}."
        )
    if report.workload.total_exec_time_ms <= 0:
        messages.append(
            "The workload carries no timing data, so ranking is structural only; a "
            "pg_stat_statements export gives much better ordering."
        )
    return messages


def render_json(report: AdvisorReport, top: int | None = None) -> str:
    """Serialise the report as an indented JSON document."""
    payload = {
        "disclaimer": ESTIMATE_DISCLAIMER,
        "workload": {
            "sources": list(report.workload.sources),
            "statement_count": len(report.workload.statements),
            "total_calls": report.workload.total_calls,
            "total_exec_time_ms": round(report.workload.total_exec_time_ms, 3),
        },
        "warnings": _warnings(report),
        "unparsed_fingerprints": list(report.unparsed),
        "unknown_tables": list(report.unknown_tables),
        "recommendation_count": len(report.recommendations),
        "has_critical": report.has_critical,
        "recommendations": [
            recommendation.to_dict() for recommendation in report.top(top)
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=False)


def render_sql(report: AdvisorReport, top: int | None = None) -> str:
    """Emit only the DDL, in rank order, annotated with SQL comments."""
    lines: list[str] = [
        "-- Generated by spatial-index-advisor.",
        f"-- {ESTIMATE_DISCLAIMER}",
        "-- Review every statement before running it. CREATE/DROP INDEX CONCURRENTLY",
        "-- cannot run inside a transaction block, so apply these one at a time.",
        "",
    ]
    emitted = 0
    for position, recommendation in enumerate(report.top(top), start=1):
        if not recommendation.ddl:
            continue
        emitted += 1
        lines.append(
            f"-- [{position}] {recommendation.severity.value.upper()} "
            f"({recommendation.confidence.value} confidence): {recommendation.title}"
        )
        lines.append(f"--     estimated benefit: {_benefit_summary(recommendation)}")
        if recommendation.estimated_size_bytes is not None:
            lines.append(
                f"--     estimated size: {format_bytes(recommendation.estimated_size_bytes)}"
            )
        for caveat in recommendation.caveats:
            lines.append(f"--     caveat: {caveat}")
        lines.append(recommendation.ddl)
        lines.append("")
    if emitted == 0:
        lines.append("-- No DDL to apply.")
        lines.append("")
    return "\n".join(lines)
