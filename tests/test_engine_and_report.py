"""Tests for the engine's orchestration and the three output renderers."""

from __future__ import annotations

import io
import json

import pytest
from rich.console import Console

from spatial_index_advisor.engine import analyse
from spatial_index_advisor.models import Severity
from spatial_index_advisor.report import (
    ESTIMATE_DISCLAIMER,
    render_json,
    render_sql,
    render_terminal,
)
from spatial_index_advisor.workload import load_workload

from .conftest import make_catalog, make_table, make_workload


@pytest.fixture()
def example_report(example_catalog, examples_dir):
    return analyse(
        load_workload([examples_dir / "pg_stat_statements.csv"]), example_catalog
    )


# --------------------------------------------------------------------------- #
# engine
# --------------------------------------------------------------------------- #


def test_recommendations_are_sorted_by_severity_then_saving(example_report) -> None:
    ranks = [r.severity.rank for r in example_report.recommendations]
    assert ranks == sorted(ranks)
    criticals = [r for r in example_report.recommendations if r.severity is Severity.CRITICAL]
    scores = [r.score for r in criticals]
    assert scores == sorted(scores, reverse=True)


def test_unparseable_statements_are_recorded_not_raised() -> None:
    catalog = make_catalog(make_table())
    workload = make_workload(
        ("SELECT 1 FROM things WHERE ST_Intersects(geom, $1)", 5000),
        ("SELECT FROM WHERE ((((", 5),
    )
    report = analyse(workload, catalog)
    assert len(report.unparsed) == 1
    assert report.recommendations


def test_tables_missing_from_the_catalog_are_reported() -> None:
    catalog = make_catalog(make_table(name="public.things"))
    workload = make_workload(("SELECT 1 FROM absent WHERE ST_Intersects(geom, $1)", 5000))
    report = analyse(workload, catalog)
    assert report.unknown_tables == ("absent",)
    assert report.recommendations == ()


def test_has_critical_and_top_behave(example_report) -> None:
    assert example_report.has_critical
    assert len(example_report.top(3)) == 3
    assert example_report.top(None) == example_report.recommendations


def test_a_fully_indexed_workload_produces_nothing() -> None:
    from spatial_index_advisor.models import ExistingIndex

    table = make_table(
        row_count=100_000,
        indexes=(ExistingIndex(name="g", method="gist", columns=("geom",)),),
    )
    workload = make_workload(("SELECT 1 FROM things WHERE ST_Intersects(geom, $1)", 10))
    assert analyse(workload, make_catalog(table)).recommendations == ()


# --------------------------------------------------------------------------- #
# JSON output
# --------------------------------------------------------------------------- #


def test_json_output_is_valid_and_complete(example_report) -> None:
    payload = json.loads(render_json(example_report))
    assert payload["disclaimer"] == ESTIMATE_DISCLAIMER
    assert payload["has_critical"] is True
    assert payload["recommendation_count"] == len(example_report.recommendations)
    first = payload["recommendations"][0]
    assert first["estimate_is_heuristic"] is True
    assert first["docs_url"].startswith("https://www.postgis-python.com/")
    assert set(first["benefit"]) == {
        "current_cost_per_call",
        "projected_cost_per_call",
        "calls",
        "total_cost_saved",
        "speedup",
        "basis",
    }


def test_json_output_honours_top(example_report) -> None:
    payload = json.loads(render_json(example_report, top=2))
    assert len(payload["recommendations"]) == 2
    assert payload["recommendation_count"] == len(example_report.recommendations)


def test_json_encodes_a_missing_benefit_as_null() -> None:
    from spatial_index_advisor.models import ExistingIndex

    table = make_table(
        indexes=(
            ExistingIndex(name="a", method="gist", columns=("geom",)),
            ExistingIndex(name="b", method="gist", columns=("geom",)),
        )
    )
    workload = make_workload(("SELECT 1 FROM things WHERE ST_Intersects(geom, $1)", 10))
    payload = json.loads(render_json(analyse(workload, make_catalog(table))))
    drops = [r for r in payload["recommendations"] if r["kind"] == "redundant_index"]
    assert drops and drops[0]["benefit"] is None


def test_json_reports_a_workload_without_timing() -> None:
    catalog = make_catalog(make_table())
    workload = make_workload(("SELECT 1 FROM things WHERE ST_Intersects(geom, $1)", 5000))
    stripped = workload.__class__(
        statements=tuple(
            statement.__class__(**{**statement.__dict__, "total_exec_time_ms": 0.0})
            for statement in workload.statements
        ),
        sources=workload.sources,
    )
    payload = json.loads(render_json(analyse(stripped, catalog)))
    assert any("structural only" in warning for warning in payload["warnings"])


# --------------------------------------------------------------------------- #
# SQL output
# --------------------------------------------------------------------------- #


def test_sql_output_contains_only_comments_and_ddl(example_report) -> None:
    text = render_sql(example_report)
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        assert stripped.startswith(("CREATE ", "DROP ", "CLUSTER ", "ANALYZE "))
    assert ESTIMATE_DISCLAIMER in text


def test_sql_output_emits_exactly_the_recommendations_that_have_ddl(example_report) -> None:
    with_ddl = [r for r in example_report.recommendations if r.ddl]
    text = render_sql(example_report)
    for recommendation in example_report.recommendations:
        assert (recommendation.title in text) is bool(recommendation.ddl)
    assert len(with_ddl) == sum(
        1 for line in text.splitlines() if line.startswith("-- [")
    )


def test_sql_output_says_so_when_there_is_nothing_to_apply() -> None:
    catalog = make_catalog(make_table(row_count=100))
    workload = make_workload(("SELECT 1 FROM things WHERE ST_Intersects(geom, $1)", 10))
    assert "No DDL to apply." in render_sql(analyse(workload, catalog))


def test_sql_output_honours_top(example_report) -> None:
    assert render_sql(example_report, top=1).count("CREATE INDEX CONCURRENTLY") == 1


# --------------------------------------------------------------------------- #
# terminal output
# --------------------------------------------------------------------------- #


def render(report, top=None, width=120) -> str:
    """Render the terminal report into a string instead of a terminal."""
    console = Console(width=width, no_color=True, file=io.StringIO())
    render_terminal(report, console, top)
    return console.file.getvalue()


def test_terminal_output_includes_summary_details_and_disclaimer(example_report) -> None:
    text = render(example_report, top=2)
    assert "Spatial index recommendations" in text
    assert "CREATE INDEX CONCURRENTLY" in text
    assert "Caveats:" in text
    assert "postgis-python.com" in text
    assert "heuristic estimates" in text


def test_terminal_output_warns_about_unknown_tables(example_report) -> None:
    assert "absent from the catalog snapshot" in render(example_report)


def test_terminal_output_reports_a_clean_bill_of_health() -> None:
    catalog = make_catalog(make_table(row_count=100))
    workload = make_workload(("SELECT 1 FROM things WHERE ST_Intersects(geom, $1)", 10))
    assert "No recommendations" in render(analyse(workload, catalog))
