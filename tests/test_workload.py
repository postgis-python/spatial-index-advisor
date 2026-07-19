"""Tests for the workload parsers and their aggregation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from spatial_index_advisor.errors import WorkloadParseError
from spatial_index_advisor.workload import detect_parser, load_workload
from spatial_index_advisor.workload.csvlog import MESSAGE_COLUMN, CsvLogParser
from spatial_index_advisor.workload.pg_stat_statements import PgStatStatementsParser
from spatial_index_advisor.workload.sqlfile import SqlFileParser, split_statements

# --------------------------------------------------------------------------- #
# pg_stat_statements
# --------------------------------------------------------------------------- #


def write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_pgss_csv_is_parsed_with_counters(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        "pgss.csv",
        "query,calls,total_exec_time,rows\n"
        '"SELECT 1 FROM t WHERE geom && $1",100,2500.5,900\n',
    )
    workload = load_workload([path])
    (statement,) = workload.statements
    assert statement.calls == 100
    assert statement.total_exec_time_ms == pytest.approx(2500.5)
    assert statement.rows == 900
    assert statement.mean_exec_time_ms == pytest.approx(25.005)


def test_pgss_accepts_legacy_total_time_column(tmp_path: Path) -> None:
    path = write(tmp_path, "old.csv", "query,calls,total_time\nSELECT 1,4,80\n")
    (statement,) = load_workload([path]).statements
    assert statement.total_exec_time_ms == pytest.approx(80.0)


def test_pgss_derives_total_from_mean_when_only_mean_is_present(tmp_path: Path) -> None:
    path = write(tmp_path, "mean.csv", "query,calls,mean_exec_time\nSELECT 1,10,3.5\n")
    (statement,) = load_workload([path]).statements
    assert statement.total_exec_time_ms == pytest.approx(35.0)


def test_pgss_json_array_is_parsed(tmp_path: Path) -> None:
    payload = [{"query": "SELECT 1 FROM t", "calls": 7, "total_exec_time": 21.0, "rows": 7}]
    path = write(tmp_path, "pgss.json", json.dumps(payload))
    (statement,) = load_workload([path]).statements
    assert statement.calls == 7


def test_pgss_json_object_wrapper_is_unwrapped(tmp_path: Path) -> None:
    payload = {"statements": [{"query": "SELECT 1 FROM t", "calls": 2, "total_exec_time": 1.0}]}
    path = write(tmp_path, "wrapped.json", json.dumps(payload))
    assert load_workload([path], "pgss").statements[0].calls == 2


def test_pgss_json_object_without_known_key_is_rejected(tmp_path: Path) -> None:
    path = write(tmp_path, "bad.json", json.dumps({"calls": 1, "query": "SELECT 1"}))
    with pytest.raises(WorkloadParseError, match="statements"):
        load_workload([path], "pgss")


def test_pgss_invalid_json_reports_line(tmp_path: Path) -> None:
    path = write(tmp_path, "broken.json", '[{"query": "SELECT 1", "calls": ]')
    with pytest.raises(WorkloadParseError, match="invalid JSON"):
        load_workload([path], "pgss")


def test_pgss_csv_without_query_column_is_rejected(tmp_path: Path) -> None:
    path = write(tmp_path, "nope.csv", "foo,bar\n1,2\n")
    with pytest.raises(WorkloadParseError, match="no query column"):
        load_workload([path], "pgss")


def test_pgss_rows_with_blank_queries_are_skipped(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        "mixed.csv",
        "query,calls,total_exec_time\n,5,1.0\nSELECT 1 FROM t,5,1.0\n",
    )
    assert len(load_workload([path], "pgss").statements) == 1


def test_pgss_export_with_no_usable_rows_is_rejected(tmp_path: Path) -> None:
    path = write(tmp_path, "empty.csv", "query,calls,total_exec_time\n,,\n")
    with pytest.raises(WorkloadParseError, match="no usable rows"):
        load_workload([path], "pgss")


def test_pgss_tolerates_thousands_separators(tmp_path: Path) -> None:
    path = write(
        tmp_path, "sep.csv", 'query,calls,total_exec_time\nSELECT 1,"1,200","3,400.5"\n'
    )
    (statement,) = load_workload([path], "pgss").statements
    assert statement.calls == 1200
    assert statement.total_exec_time_ms == pytest.approx(3400.5)


def test_pgss_ignores_unparseable_counters(tmp_path: Path) -> None:
    path = write(tmp_path, "junk.csv", "query,calls,total_exec_time\nSELECT 1,abc,xyz\n")
    (statement,) = load_workload([path], "pgss").statements
    assert statement.calls == 1
    assert statement.total_exec_time_ms == 0.0


# --------------------------------------------------------------------------- #
# csvlog
# --------------------------------------------------------------------------- #


def csvlog_row(message: str) -> list[str]:
    row = [""] * 26
    row[0] = "2026-07-14 09:12:03.001 UTC"
    row[11] = "LOG"
    row[MESSAGE_COLUMN] = message
    return row


def write_csvlog(tmp_path: Path, messages: list[str]) -> Path:
    path = tmp_path / "postgresql.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, quoting=csv.QUOTE_ALL).writerows(
            csvlog_row(message) for message in messages
        )
    return path


def test_csvlog_groups_identical_statements_and_sums_duration(tmp_path: Path) -> None:
    path = write_csvlog(
        tmp_path,
        [
            "duration: 10.500 ms  statement: SELECT id FROM t WHERE geom && ST_MakeEnvelope(1,2,3,4,3857)",
            "duration: 20.500 ms  statement: SELECT id FROM t WHERE geom && ST_MakeEnvelope(9,9,9,9,3857)",
        ],
    )
    (statement,) = load_workload([path]).statements
    assert statement.calls == 2
    assert statement.total_exec_time_ms == pytest.approx(31.0)


def test_csvlog_handles_extended_protocol_execute_messages(tmp_path: Path) -> None:
    path = write_csvlog(
        tmp_path, ["duration: 4.000 ms  execute <unnamed>: SELECT 1 FROM t WHERE a = 1"]
    )
    assert load_workload([path], "csvlog").statements[0].calls == 1


def test_csvlog_ignores_non_statement_messages(tmp_path: Path) -> None:
    path = write_csvlog(
        tmp_path,
        [
            "connection received: host=10.0.0.1 port=5432",
            "duration: 1.000 ms  statement: SELECT 1 FROM t",
            "checkpoint starting: time",
            "duration: 2.000 ms  statement: COMMIT",
        ],
    )
    statements = load_workload([path], "csvlog").statements
    assert len(statements) == 1


def test_csvlog_without_duration_prefix_still_captures_statements(tmp_path: Path) -> None:
    path = write_csvlog(tmp_path, ["statement: SELECT 1 FROM t WHERE a = 1"])
    (statement,) = load_workload([path], "csvlog").statements
    assert statement.total_exec_time_ms == 0.0


def test_csvlog_with_no_statements_is_rejected(tmp_path: Path) -> None:
    path = write_csvlog(tmp_path, ["checkpoint complete"])
    with pytest.raises(WorkloadParseError, match="log_min_duration_statement"):
        load_workload([path], "csvlog")


def test_csvlog_short_rows_are_rejected_with_a_line_number(tmp_path: Path) -> None:
    path = tmp_path / "short.csv"
    path.write_text("a,b,c\nd,e,f\n", encoding="utf-8")
    with pytest.raises(WorkloadParseError, match="csvlog"):
        CsvLogParser().read(path)


def test_csvlog_finds_the_message_when_the_layout_is_shifted(tmp_path: Path) -> None:
    row = [""] * 20
    row[0] = "2026-07-14 09:12:03.001 UTC"
    row[9] = "duration: 3.000 ms  statement: SELECT 1 FROM t"
    path = tmp_path / "shifted.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(row)
    assert load_workload([path], "csvlog").statements[0].calls == 1


def test_example_csvlog_parses(examples_dir: Path) -> None:
    workload = load_workload([examples_dir / "postgresql-2026-07-14.csv"])
    assert workload.total_calls == 36
    assert workload.total_exec_time_ms > 0
    assert len(workload.statements) == 5


# --------------------------------------------------------------------------- #
# plain SQL files
# --------------------------------------------------------------------------- #


def test_sql_file_splits_on_semicolons_and_reads_call_hints(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        "queries.sql",
        "-- calls: 500\nSELECT 1 FROM t WHERE geom && $1;\n\nSELECT 2 FROM u;\n",
    )
    workload = load_workload([path])
    calls = {statement.calls for statement in workload.statements}
    assert calls == {500, 1}


def test_semicolons_inside_string_literals_do_not_split() -> None:
    assert len(split_statements("SELECT ';' FROM t; SELECT 2;")) == 2


def test_escaped_quotes_inside_literals_are_handled() -> None:
    assert len(split_statements("SELECT 'it''s; fine' FROM t;")) == 1


def test_semicolons_inside_comments_do_not_split() -> None:
    assert len(split_statements("SELECT 1 FROM t -- ; not a split\n;")) == 1
    assert len(split_statements("SELECT 1 /* ; */ FROM t;")) == 1


def test_comment_only_file_is_rejected(tmp_path: Path) -> None:
    path = write(tmp_path, "comments.sql", "-- nothing here\n-- really\n")
    with pytest.raises(WorkloadParseError, match="no SQL statements"):
        SqlFileParser().read(path)


def test_example_sql_file_parses(examples_dir: Path) -> None:
    workload = load_workload([examples_dir / "queries.sql"])
    assert len(workload.statements) == 7
    assert workload.total_calls == 2075700


# --------------------------------------------------------------------------- #
# dispatch and aggregation
# --------------------------------------------------------------------------- #


def test_format_detection_picks_the_right_parser(examples_dir: Path) -> None:
    for name, expected in (
        ("pg_stat_statements.csv", "pgss"),
        ("postgresql-2026-07-14.csv", "csvlog"),
        ("queries.sql", "sql"),
    ):
        path = examples_dir / name
        assert detect_parser(path.read_text(encoding="utf-8"), path).name == expected


def test_unrecognised_file_is_rejected(tmp_path: Path) -> None:
    path = write(tmp_path, "notes.txt", "this is prose, not a workload")
    with pytest.raises(WorkloadParseError, match="unrecognised workload format"):
        load_workload([path])


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(WorkloadParseError, match="does not exist"):
        load_workload([tmp_path / "absent.csv"])


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    path = write(tmp_path, "empty.sql", "   \n")
    with pytest.raises(WorkloadParseError, match="is empty"):
        load_workload([path])


def test_unknown_format_name_is_rejected(tmp_path: Path) -> None:
    path = write(tmp_path, "q.sql", "SELECT 1 FROM t;")
    with pytest.raises(WorkloadParseError, match="unknown workload format"):
        load_workload([path], "yaml")


def test_no_paths_is_rejected() -> None:
    with pytest.raises(WorkloadParseError, match="no workload files"):
        load_workload([])


def test_statements_from_several_sources_merge_and_prefer_literal_samples(
    tmp_path: Path,
) -> None:
    parameterised = write(
        tmp_path,
        "pgss.csv",
        "query,calls,total_exec_time\n"
        '"SELECT id FROM t WHERE ST_DWithin(geom, $1, $2)",100,500.0\n',
    )
    literal = write(
        tmp_path,
        "queries.sql",
        "-- calls: 5\nSELECT id FROM t WHERE ST_DWithin(geom, $1, 250);\n",
    )
    workload = load_workload([parameterised, literal])
    (statement,) = workload.statements
    assert statement.calls == 105
    assert "250" in statement.sample_sql
    assert statement.source == "multiple"
    assert workload.sources == (f"{parameterised}(pgss)", f"{literal}(sql)")


def test_merging_different_fingerprints_is_a_programming_error() -> None:
    from spatial_index_advisor.models import WorkloadStatement

    first = WorkloadStatement("a", "SELECT 1", "SELECT 1", 1, 1.0, 1, "test")
    second = WorkloadStatement("b", "SELECT 2", "SELECT 2", 1, 1.0, 1, "test")
    with pytest.raises(ValueError, match="different fingerprints"):
        first.merged_with(second)


def test_parser_read_reports_unreadable_files(tmp_path: Path) -> None:
    with pytest.raises(WorkloadParseError, match="cannot read file"):
        PgStatStatementsParser().read(tmp_path)
