#!/usr/bin/env python3
"""Verify ``collect`` and the emitted DDL against a real PostgreSQL/PostGIS server.

The offline suite in ``tests/`` runs the whole engine from a JSON snapshot and a
fake DB-API connection, which is what makes it fast and database-free. That
leaves exactly one thing untested: whether the catalog queries in
:mod:`spatial_index_advisor.collector` mean what we think they mean on a real
server, and whether the DDL the rules emit is actually valid SQL. Both depend on
the PostgreSQL major version, and both were silently wrong the first time this
tool met a live database.

This script builds a fixture schema with the shapes the rules care about, runs
``collect`` against it, checks the snapshot field by field against the same facts
read directly from the catalog, then runs ``analyze`` and executes every emitted
statement to prove it parses and runs on this server.

SCOPE: correctness only -- snapshot fidelity and DDL validity. This script
deliberately makes no assertion about the cost model's numeric estimates. Those
are heuristic figures in arbitrary PostgreSQL cost units, and the wall-clock
timings that would be needed to calibrate them are far too noisy on a shared CI
runner to assert on without the job becoming flaky. Do not add timing or
"speedup" assertions here; this is not a calibration harness.

Usage::

    python tests/live/verify_against_server.py --dsn postgresql://user@host/db

Exits 0 when every check passes, 1 when any check fails (naming each failure),
and 2 when the fixture itself could not be built.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from spatial_index_advisor.collector import (  # noqa: E402
    MIN_SAMPLE_PAGES,
    SAMPLE_PERCENT,
    sample_percent,
)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_SETUP = 2

SCHEMA = "public"

# Row counts. MIN_ROWS_FOR_INDEX in rules.py is 10 000, so any table a rule must
# fire on has to clear that. Everything here is sized to build in a few seconds.
SENSOR_ROWS = 120_000
PARCEL_ROWS = 12_000
POI_ROWS = 15_000
LAND_COVER_ROWS = 300
DISTRICT_ROWS = 60

#: Search point and radius used by both the workload statements and the rewrite
#: equivalence check. It has to sit inside the poi layer -- poi spans roughly
#: 500000..504000 by 170000..170090 -- or the equivalence check compares two
#: empty sets and proves nothing.
PROBE_POINT = "ST_SetSRID(ST_MakePoint(502000, 170050), 3857)"
PROBE_RADIUS = 500


# --------------------------------------------------------------------------- #
# Fixture
# --------------------------------------------------------------------------- #

#: Every shape the rules read from the catalog, in one schema:
#:
#: * ``sensor_readings`` -- geometry column *without* a GiST index, append-only,
#:   with a timestamp correlated to physical order, a boolean a partial index can
#:   target and an integer a composite index can carry.
#: * ``parcels`` -- geometry column *with* a GiST index, deliberately shuffled on
#:   load so physical correlation is low; carries the duplicate index pair and
#:   the expression index.
#: * ``poi`` -- points queried only by KNN and by a non-sargable distance
#:   comparison, so the KNN and rewrite rules fire without the missing-GiST rule
#:   masking them.
#: * ``land_cover`` -- a few very large polygons, so TOAST holds most of the
#:   bytes and pg_relation_size and pg_table_size disagree by a wide margin.
#: * ``districts`` -- carries the covering index that uses INCLUDE.
FIXTURE_SQL: tuple[str, ...] = (
    "CREATE EXTENSION IF NOT EXISTS postgis",
    """
    DROP TABLE IF EXISTS sensor_readings, parcels, poi, land_cover, districts CASCADE
    """,
    # ---- sensor_readings: no spatial index, append-only, correlated timestamp
    """
    CREATE TABLE sensor_readings (
        id          bigserial PRIMARY KEY,
        sensor_id   integer     NOT NULL,
        recorded_at timestamptz NOT NULL,
        is_active   boolean     NOT NULL,
        geom        geometry(Point, 3857) NOT NULL
    )
    """,
    # Loaded in timestamp order and never updated, so pg_stats.correlation on
    # recorded_at is ~1.0 and pg_stat_user_tables sees inserts only.
    f"""
    INSERT INTO sensor_readings (sensor_id, recorded_at, is_active, geom)
    SELECT g % 500,
           timestamptz '2026-01-01 00:00:00+00' + (g * interval '1 second'),
           g % 7 <> 0,
           ST_SetSRID(
               ST_MakePoint(500000 + (g % 3000) * 1.7, 170000 + (g / 3000) * 2.3),
               3857)
    FROM generate_series(1, {SENSOR_ROWS}) AS g
    """,
    # ---- parcels: GiST-indexed polygons, shuffled on load
    """
    CREATE TABLE parcels (
        id          bigserial PRIMARY KEY,
        parcel_code text NOT NULL,
        owner_name  text NOT NULL,
        status      text NOT NULL,
        geom        geometry(Polygon, 3857) NOT NULL
    )
    """,
    f"""
    INSERT INTO parcels (parcel_code, owner_name, status, geom)
    SELECT 'P-' || lpad(g::text, 8, '0'),
           'Owner ' || (g % 997),
           CASE WHEN g % 5 = 0 THEN 'pending' ELSE 'registered' END,
           ST_Buffer(
               ST_SetSRID(
                   ST_MakePoint(500000 + (g % 200) * 100.0, 170000 + (g / 200) * 100.0),
                   3857),
               40, 2)
    FROM generate_series(1, {PARCEL_ROWS}) AS g
    ORDER BY random()
    """,
    "CREATE INDEX parcels_geom_gist ON parcels USING GIST (geom)",
    # A duplicate index pair: byte-for-byte the same definition under two names.
    # rule_redundant_index must see both and emit a DROP for one of them.
    "CREATE INDEX parcels_code_idx ON parcels (parcel_code)",
    "CREATE INDEX parcels_code_dup_idx ON parcels (parcel_code)",
    # An expression index. The collector renders index keys with
    # pg_get_indexdef(indexrelid, k, true), which must yield the expression text
    # rather than an empty string.
    "CREATE INDEX parcels_owner_lower_idx ON parcels (lower(owner_name))",
    # ---- poi: KNN and non-sargable distance traffic only
    """
    CREATE TABLE poi (
        id       bigserial PRIMARY KEY,
        category text NOT NULL,
        geom     geometry(Point, 3857) NOT NULL
    )
    """,
    f"""
    INSERT INTO poi (category, geom)
    SELECT CASE WHEN g % 3 = 0 THEN 'retail' ELSE 'civic' END,
           ST_SetSRID(
               ST_MakePoint(500000 + (g % 1000) * 4.0, 170000 + (g / 1000) * 6.0),
               3857)
    FROM generate_series(1, {POI_ROWS}) AS g
    """,
    # ---- land_cover: few rows, very large geometries, so most bytes are TOAST
    """
    CREATE TABLE land_cover (
        id       bigserial PRIMARY KEY,
        cover    text NOT NULL,
        geom     geometry(Polygon, 3857) NOT NULL
    )
    """,
    f"""
    INSERT INTO land_cover (cover, geom)
    SELECT 'class-' || (g % 12),
           ST_Buffer(
               ST_SetSRID(
                   ST_MakePoint(500000 + g * 250.0, 170000 + g * 130.0), 3857),
               900, 512)
    FROM generate_series(1, {LAND_COVER_ROWS}) AS g
    """,
    "CREATE INDEX land_cover_geom_gist ON land_cover USING GIST (geom)",
    # ---- districts: the covering index
    """
    CREATE TABLE districts (
        id            bigserial PRIMARY KEY,
        district_code text    NOT NULL,
        population    integer NOT NULL,
        geom          geometry(MultiPolygon, 3857) NOT NULL
    )
    """,
    f"""
    INSERT INTO districts (district_code, population, geom)
    SELECT 'D-' || lpad(g::text, 4, '0'),
           1000 * g,
           ST_Multi(ST_Buffer(
               ST_SetSRID(
                   ST_MakePoint(500000 + g * 900.0, 170000 + g * 700.0), 3857),
               400, 4))
    FROM generate_series(1, {DISTRICT_ROWS}) AS g
    """,
    "CREATE INDEX districts_geom_gist ON districts USING GIST (geom)",
    # One key column, one INCLUDE payload column. pg_index.indnatts is 2 here and
    # indnkeyatts is 1; the collector must report 1, or "(a) INCLUDE (b)" looks
    # like a two-column index and redundancy detection breaks.
    """
    CREATE INDEX districts_code_covering_idx
        ON districts (district_code) INCLUDE (population)
    """,
    "ANALYZE sensor_readings, parcels, poi, land_cover, districts",
)

FIXTURE_TABLES: tuple[str, ...] = (
    "sensor_readings",
    "parcels",
    "poi",
    "land_cover",
    "districts",
)

#: The workload the fixture is designed to provoke recommendations from. Written
#: to a temporary .sql file and fed to ``analyze`` unchanged.
WORKLOAD_SQL = f"""\
-- Workload for the live fixture schema built by verify_against_server.py.
-- The call counts are shaped so that each rule under test clears its dominance
-- and row-count thresholds; they are not measurements of anything.

-- calls: 900000
-- Live map: active sensors near the operator's cursor. sensor_readings.geom has
-- no spatial index, `is_active = true` is a constant filter and `sensor_id` is
-- parameterised, so this one statement feeds the missing-GiST, partial-index and
-- composite-index rules at once.
SELECT id, sensor_id
FROM sensor_readings
WHERE ST_DWithin(geom, {PROBE_POINT}, 250)
  AND is_active = true
  AND sensor_id = $1;

-- calls: 300000
-- Same column and same two filters through a different spatial operator, so both
-- filters stay above the 80% dominance threshold for the column.
SELECT id
FROM sensor_readings
WHERE ST_Intersects(geom, ST_MakeEnvelope(500000, 170000, 505000, 175000, 3857))
  AND is_active = true
  AND sensor_id = $1;

-- calls: 250000
-- Nearest points of interest. poi.geom carries no sargable predicate anywhere in
-- this workload, so the KNN rule fires in its own right.
SELECT id
FROM poi
ORDER BY geom <-> {PROBE_POINT}
LIMIT 10;

-- calls: 180000
-- A distance comparison no index can answer. The advisory says to rewrite it as
-- ST_DWithin; check_rewrite_equivalence runs both forms against the server and
-- compares the row sets.
SELECT id
FROM poi
WHERE ST_Distance(geom, {PROBE_POINT}) < {PROBE_RADIUS};

-- calls: 90000
-- The indexed column wrapped in a projection, which defeats parcels_geom_gist.
-- This advisory carries an expression-index DDL, so it is one of the statements
-- executed against the server.
SELECT id
FROM parcels
WHERE ST_Intersects(ST_Transform(geom, 4326),
                    ST_MakeEnvelope(4.5, 51.9, 4.6, 52.0, 4326));

-- calls: 140000
-- Wide viewport scan over a GiST-indexed but physically shuffled table.
SELECT id
FROM parcels
WHERE ST_Intersects(geom, ST_MakeEnvelope(500000, 170000, 520000, 190000, 3857));
"""


# --------------------------------------------------------------------------- #
# Result collection
# --------------------------------------------------------------------------- #


class Checks:
    """Accumulates pass/fail results so one run reports every problem it found."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.passed = 0

    def ok(self, name: str, detail: str = "") -> None:
        self.passed += 1
        suffix = f" -- {detail}" if detail else ""
        print(f"  PASS  {name}{suffix}")

    def fail(self, name: str, detail: str) -> None:
        self.failures.append(f"{name}: {detail}")
        print(f"  FAIL  {name} -- {detail}")

    def check(self, condition: bool, name: str, detail: str) -> bool:
        if condition:
            self.ok(name, detail)
        else:
            self.fail(name, detail)
        return condition


def query(connection: Any, sql: str, params: Sequence[Any] | None = None) -> list[Any]:
    with connection.cursor() as cursor:
        cursor.execute(sql, params)
        return cursor.fetchall()


def query_one(connection: Any, sql: str, params: Sequence[Any] | None = None) -> Any:
    rows = query(connection, sql, params)
    return rows[0] if rows else None


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #


def build_fixture(connection: Any) -> None:
    """Create and populate the fixture schema, failing loudly on the first error."""
    for statement in FIXTURE_SQL:
        try:
            with connection.cursor() as cursor:
                cursor.execute(statement)
        except Exception as error:  # noqa: BLE001 - report which statement broke
            first_line = " ".join(statement.split())[:120]
            raise RuntimeError(f"fixture statement failed: {first_line}\n  {error}") from error


def wait_for_statistics(connection: Any, timeout: float = 30.0) -> bool:
    """Block until pg_stat_user_tables reports the fixture inserts.

    Cumulative statistics are not written synchronously. Through PostgreSQL 14 a
    separate collector process receives them over UDP and flushes on its own
    schedule; from 15 they live in shared memory and are flushed at transaction
    end but still with a delay. Reading pg_stat_user_tables too early makes every
    table look like it has never been written to, which silently turns off the
    append-only classification.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        row = query_one(
            connection,
            """
            SELECT coalesce(sum(n_tup_ins), 0)
            FROM pg_stat_user_tables
            WHERE schemaname = %s AND relname = ANY(%s)
            """,
            [SCHEMA, list(FIXTURE_TABLES)],
        )
        if row and int(row[0]) >= SENSOR_ROWS:
            return True
        time.sleep(0.5)
    return False


def run_collect(dsn: str, output: Path) -> None:
    """Invoke the real ``collect`` subcommand, the way an operator would."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "spatial_index_advisor",
            "collect",
            "--dsn",
            dsn,
            "--schema",
            SCHEMA,
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"collect exited {result.returncode}\n"
            f"stdout: {result.stdout.strip()}\nstderr: {result.stderr.strip()}"
        )


def run_analyze(workload: Path, catalog: Path) -> dict[str, Any]:
    """Invoke ``analyze`` and return the parsed JSON report.

    ``--fail-on never`` because the fixture is built to produce findings on
    purpose: exit status 2 would be the correct behaviour and a useless CI
    signal. What matters here is that the statements it emits are valid SQL, not
    what it concludes about them.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "spatial_index_advisor",
            "analyze",
            "--workload",
            str(workload),
            "--catalog",
            str(catalog),
            "--format",
            "json",
            "--fail-on",
            "never",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"analyze exited {result.returncode}\n"
            f"stdout: {result.stdout.strip()}\nstderr: {result.stderr.strip()}"
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"analyze did not emit JSON: {error}\n{result.stdout[:500]}") from error


# --------------------------------------------------------------------------- #
# Snapshot checks
# --------------------------------------------------------------------------- #


def snapshot_tables(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {table["name"]: table for table in snapshot["tables"]}


def check_snapshot(checks: Checks, connection: Any, snapshot: dict[str, Any]) -> None:
    """Compare the collected snapshot against the same facts read live."""
    tables = snapshot_tables(snapshot)

    print("\n[snapshot] every geometry table is present")
    for name in FIXTURE_TABLES:
        qualified = f"{SCHEMA}.{name}"
        checks.check(
            qualified in tables,
            f"table {qualified} collected",
            "present" if qualified in tables else "missing from the snapshot",
        )
    if not all(f"{SCHEMA}.{name}" in tables for name in FIXTURE_TABLES):
        return

    check_extents(checks, tables)
    check_include_index(checks, connection, tables)
    check_table_sizes(checks, connection, tables)
    check_sampling(checks, tables)
    check_index_fidelity(checks, connection, tables)
    check_activity(checks, tables)


def check_extents(checks: Checks, tables: dict[str, dict[str, Any]]) -> None:
    """Extents must be populated.

    ST_EstimatedExtent returns a box2d. Casting it to text gives BOX(x1 y1,x2 y2),
    which the collector's parser reads; rendering it through ST_AsText instead
    gives a five-vertex POLYGON that parsed to nothing, so every extent came back
    null and every layer-relative selectivity estimate silently lost its
    denominator.
    """
    print("\n[snapshot] extents are populated, not null")
    for name in FIXTURE_TABLES:
        table = tables[f"{SCHEMA}.{name}"]
        for column in table["geometry_columns"]:
            width = column.get("extent_width")
            height = column.get("extent_height")
            label = f"extent on {name}.{column['name']}"
            if width is None or height is None:
                checks.fail(label, "extent_width/extent_height are null")
            elif width <= 0 or height <= 0:
                checks.fail(label, f"degenerate extent {width} x {height}")
            else:
                checks.ok(label, f"{width:.1f} x {height:.1f}")


def check_include_index(
    checks: Checks, connection: Any, tables: dict[str, dict[str, Any]]
) -> None:
    """Key-column count must exclude INCLUDE payload columns.

    pg_index.indnatts counts INCLUDE columns from PostgreSQL 11 on; indnkeyatts
    does not. Reading indnatts made "(a) INCLUDE (b)" look like a two-column
    index, which broke both duplicate detection and leading-prefix detection.
    """
    print("\n[snapshot] INCLUDE payload columns are not counted as index keys")
    row = query_one(
        connection,
        """
        SELECT idx.indnatts, idx.indnkeyatts
        FROM pg_index idx
        JOIN pg_class i ON i.oid = idx.indexrelid
        WHERE i.relname = 'districts_code_covering_idx'
        """,
    )
    if row is None:
        checks.fail("covering index present on the server", "districts_code_covering_idx not found")
        return
    indnatts, indnkeyatts = int(row[0]), int(row[1])
    checks.check(
        indnatts == 2 and indnkeyatts == 1,
        "fixture really exercises the INCLUDE case",
        f"pg_index reports indnatts={indnatts}, indnkeyatts={indnkeyatts}",
    )

    indexes = {index["name"]: index for index in tables[f"{SCHEMA}.districts"]["indexes"]}
    covering = indexes.get("districts_code_covering_idx")
    if covering is None:
        checks.fail("covering index collected", "not present in the snapshot")
        return
    columns = covering["columns"]
    checks.check(
        len(columns) == 1 and columns[0].lower() == "district_code",
        "covering index reports key columns only",
        f"collected columns={columns} (expected ['district_code'], not the INCLUDE payload)",
    )


def check_table_sizes(
    checks: Checks, connection: Any, tables: dict[str, dict[str, Any]]
) -> None:
    """table_bytes must be the main fork, not the main fork plus TOAST.

    The cost model turns table_bytes into a heap page count for sequential-scan
    costing, and a seq scan reads only the main fork. pg_table_size adds TOAST,
    which for a table of large polygons is most of the bytes and none of the
    scanned pages, inflating every sequential-scan cost by an order of magnitude.
    """
    print("\n[snapshot] table sizes come from the main fork")
    for name in FIXTURE_TABLES:
        row = query_one(
            connection,
            "SELECT pg_relation_size(%s::regclass), pg_table_size(%s::regclass)",
            [f"{SCHEMA}.{name}", f"{SCHEMA}.{name}"],
        )
        relation_size, table_size = int(row[0]), int(row[1])
        collected = int(tables[f"{SCHEMA}.{name}"]["table_bytes"])
        checks.check(
            collected == relation_size,
            f"table_bytes on {name} is pg_relation_size",
            f"collected={collected} pg_relation_size={relation_size} pg_table_size={table_size}",
        )

    # The land_cover fixture only proves anything if TOAST really does dominate.
    row = query_one(
        connection,
        "SELECT pg_relation_size(%s::regclass), pg_table_size(%s::regclass)",
        [f"{SCHEMA}.land_cover", f"{SCHEMA}.land_cover"],
    )
    relation_size, table_size = int(row[0]), int(row[1])
    checks.check(
        table_size > relation_size * 4,
        "land_cover fixture actually TOASTs",
        f"pg_table_size={table_size} is {table_size / max(1, relation_size):.1f}x "
        f"pg_relation_size={relation_size}; the two must differ for this check to bite",
    )


def check_sampling(checks: Checks, tables: dict[str, dict[str, Any]]) -> None:
    """The TABLESAMPLE step must return data rather than nothing.

    TABLESAMPLE SYSTEM accepts each page independently, so a flat 1% selects
    nothing at all on a small table often enough to matter -- a 250-page table
    comes back empty roughly 8% of the time, silently dropping the feature-size
    statistic and with it every average_area_fraction the cost model derives from
    it. The collector raises the rate on small tables to clear a floor of
    MIN_SAMPLE_PAGES expected pages.
    """
    print("\n[snapshot] geometry sampling returns data")
    for name in FIXTURE_TABLES:
        table = tables[f"{SCHEMA}.{name}"]
        for column in table["geometry_columns"]:
            width = column.get("avg_bbox_width")
            height = column.get("avg_bbox_height")
            label = f"avg feature bbox on {name}.{column['name']}"
            if width is None or height is None:
                pages = max(1, int(table["table_bytes"]) // 8192)
                checks.fail(
                    label,
                    f"sampling returned no rows ({pages} heap pages, "
                    f"rate {sample_percent(int(table['table_bytes'])):.2f}%)",
                )
            else:
                checks.ok(label, f"{width:.2f} x {height:.2f}")

    # Assert the small-table floor is genuinely engaged, so the check above is
    # testing the mitigation rather than passing by luck on a big table.
    districts_bytes = int(tables[f"{SCHEMA}.districts"]["table_bytes"])
    pages = max(1, districts_bytes // 8192)
    rate = sample_percent(districts_bytes)
    checks.check(
        pages < MIN_SAMPLE_PAGES and rate > SAMPLE_PERCENT,
        "small-table sampling floor is engaged",
        f"districts has {pages} heap pages, so the rate is raised from "
        f"{SAMPLE_PERCENT}% to {rate:.1f}%",
    )


def check_index_fidelity(
    checks: Checks, connection: Any, tables: dict[str, dict[str, Any]]
) -> None:
    """Every index on every fixture table is collected, with its shape intact."""
    print("\n[snapshot] indexes are collected faithfully")
    for name in FIXTURE_TABLES:
        live = {
            row[0]
            for row in query(
                connection,
                """
                SELECT i.relname
                FROM pg_index idx
                JOIN pg_class i ON i.oid = idx.indexrelid
                JOIN pg_class t ON t.oid = idx.indrelid
                JOIN pg_namespace n ON n.oid = t.relnamespace
                WHERE n.nspname = %s AND t.relname = %s
                """,
                [SCHEMA, name],
            )
        }
        collected = {index["name"] for index in tables[f"{SCHEMA}.{name}"]["indexes"]}
        checks.check(
            live == collected,
            f"index set on {name}",
            f"server has {sorted(live)}, snapshot has {sorted(collected)}",
        )

    parcels = {index["name"]: index for index in tables[f"{SCHEMA}.parcels"]["indexes"]}

    gist = parcels.get("parcels_geom_gist")
    checks.check(
        gist is not None and gist["method"].lower() == "gist",
        "access method on parcels_geom_gist",
        f"collected method={gist['method'] if gist else None!r} (expected 'gist')",
    )

    expression = parcels.get("parcels_owner_lower_idx")
    if expression is None:
        checks.fail("expression index collected", "parcels_owner_lower_idx missing")
    else:
        columns = expression["columns"]
        checks.check(
            len(columns) == 1 and "lower" in columns[0].lower(),
            "expression index key is rendered, not blank",
            f"collected columns={columns}",
        )

    duplicates = [
        index
        for index in tables[f"{SCHEMA}.parcels"]["indexes"]
        if index["name"] in {"parcels_code_idx", "parcels_code_dup_idx"}
    ]
    checks.check(
        len(duplicates) == 2
        and duplicates[0]["columns"] == duplicates[1]["columns"]
        and duplicates[0]["method"] == duplicates[1]["method"],
        "duplicate index pair collected as identical",
        f"collected {[(d['name'], d['method'], d['columns']) for d in duplicates]}",
    )

    # Nothing collected may carry a null or empty key list: an index whose columns
    # vanish is invisible to every redundancy rule.
    for name in FIXTURE_TABLES:
        empty = [
            index["name"]
            for index in tables[f"{SCHEMA}.{name}"]["indexes"]
            if not index["columns"] or any(not str(c).strip() for c in index["columns"])
        ]
        checks.check(
            not empty,
            f"no index on {name} lost its key columns",
            f"indexes with empty key lists: {empty}" if empty else "all key lists populated",
        )


def check_activity(checks: Checks, tables: dict[str, dict[str, Any]]) -> None:
    """Write counters and physical correlation must survive collection."""
    print("\n[snapshot] write activity and correlation")
    sensors = tables[f"{SCHEMA}.sensor_readings"]
    checks.check(
        int(sensors["inserts"]) >= SENSOR_ROWS,
        "insert counter on sensor_readings",
        f"collected inserts={sensors['inserts']} (expected at least {SENSOR_ROWS})",
    )
    checks.check(
        bool(sensors["append_only"]),
        "sensor_readings classified append-only",
        f"inserts={sensors['inserts']} updates={sensors['updates']} "
        f"deletes={sensors['deletes']}",
    )
    correlation = sensors["column_correlation"].get("recorded_at")
    checks.check(
        correlation is not None and correlation > 0.9,
        "recorded_at correlation collected",
        f"collected correlation={correlation} (rows were loaded in timestamp order)",
    )
    checks.check(
        int(sensors["row_count"]) > SENSOR_ROWS * 0.9,
        "row_count on sensor_readings",
        f"collected row_count={sensors['row_count']} (expected around {SENSOR_ROWS})",
    )


# --------------------------------------------------------------------------- #
# DDL checks
# --------------------------------------------------------------------------- #


def split_statements(ddl: str) -> list[str]:
    """Split a recommendation's DDL into individual statements.

    A few recommendations carry more than one statement (a CREATE EXTENSION
    before a composite index, an ANALYZE after a CLUSTER). None of the emitted
    statements contains a semicolon inside a literal, so a plain split is enough.
    """
    return [f"{part.strip()};" for part in ddl.split(";") if part.strip()]


def execute_emitted_ddl(
    checks: Checks, connection: Any, report: dict[str, Any]
) -> None:
    """Run every emitted statement against the server to prove it is valid SQL."""
    print("\n[ddl] every emitted statement runs on this server")
    recommendations = report.get("recommendations", [])
    kinds = sorted({r["kind"] for r in recommendations})
    print(f"  analyze produced {len(recommendations)} recommendation(s): {', '.join(kinds)}")

    # The fixture and workload are built so that these rules must fire. If one
    # stops firing, this job is no longer exercising the DDL it was written for,
    # even though every statement it does emit still runs.
    required = {"missing_gist", "knn_gist", "partial_index", "composite_index", "rewrite",
                "redundant_index"}
    missing = sorted(required - set(kinds))
    checks.check(
        not missing,
        "workload provokes every rule this job covers",
        f"rules that did not fire: {missing}" if missing else f"fired: {', '.join(kinds)}",
    )

    statements: list[tuple[str, str]] = []
    seen: set[str] = set()
    for recommendation in recommendations:
        ddl = recommendation.get("ddl")
        if not ddl:
            continue
        for statement in split_statements(ddl):
            if statement in seen:
                continue
            seen.add(statement)
            statements.append((recommendation["kind"], statement))

    if not checks.check(
        bool(statements),
        "analyze emitted DDL to execute",
        f"{len(statements)} distinct statement(s)",
    ):
        return

    creates = [s for _, s in statements if s.upper().startswith("CREATE INDEX")]
    drops = [s for _, s in statements if s.upper().startswith("DROP INDEX")]
    checks.check(
        bool(creates) and bool(drops),
        "both CREATE INDEX and DROP INDEX are covered",
        f"{len(creates)} CREATE INDEX, {len(drops)} DROP INDEX",
    )

    for kind, statement in statements:
        collapsed = " ".join(statement.split())
        try:
            with connection.cursor() as cursor:
                cursor.execute(statement)
        except Exception as error:  # noqa: BLE001 - any driver error is a failure
            checks.fail(f"execute [{kind}]", f"{collapsed}\n          {error}")
        else:
            checks.ok(f"execute [{kind}]", collapsed)


def check_rewrite_equivalence(checks: Checks, connection: Any) -> None:
    """The ST_Distance -> ST_DWithin advisory must not change the answer.

    The advisory is only worth acting on if the rewritten predicate matches the
    same rows, so compare the two row sets rather than just the counts. The
    boundary differs -- ST_Distance(...) < r is strict where ST_DWithin(...) is
    inclusive -- so rows at exactly the radius are excluded from the comparison.
    """
    print("\n[rewrite] ST_Distance(...) < r is equivalent to ST_DWithin(..., r)")
    point, radius = PROBE_POINT, PROBE_RADIUS

    original = {
        row[0]
        for row in query(
            connection,
            f"SELECT id FROM poi WHERE ST_Distance(geom, {point}) < {radius}",
        )
    }
    rewritten = {
        row[0]
        for row in query(
            connection,
            f"SELECT id FROM poi WHERE ST_DWithin(geom, {point}, {radius})",
        )
    }
    on_boundary = {
        row[0]
        for row in query(
            connection,
            f"SELECT id FROM poi WHERE ST_Distance(geom, {point}) = {radius}",
        )
    }

    checks.check(
        bool(original),
        "the rewrite comparison covers a non-empty result",
        f"{len(original)} row(s) match the original predicate",
    )
    difference = original.symmetric_difference(rewritten) - on_boundary
    checks.check(
        not difference,
        "row sets match",
        f"original={len(original)} rewritten={len(rewritten)} "
        f"on-boundary={len(on_boundary)} differing={len(difference)}",
    )

    # The point of the rewrite is that the rewritten form can use the index the
    # original cannot. Confirm the planner agrees on this server.
    with connection.cursor() as cursor:
        cursor.execute("CREATE INDEX IF NOT EXISTS poi_geom_gist_check ON poi USING GIST (geom)")
        cursor.execute("ANALYZE poi")
        cursor.execute(
            f"EXPLAIN (COSTS OFF) SELECT id FROM poi WHERE ST_DWithin(geom, {point}, {radius})"
        )
        plan = "\n".join(str(row[0]) for row in cursor.fetchall())
    checks.check(
        "poi_geom_gist_check" in plan or "Index Scan" in plan or "Bitmap" in plan,
        "the rewritten predicate is index-accelerated",
        f"plan: {' / '.join(line.strip() for line in plan.splitlines())}",
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def server_banner(connection: Any) -> str:
    row = query_one(connection, "SELECT version(), postgis_lib_version()")
    version = str(row[0]).split(" on ")[0]
    return f"{version} / PostGIS {row[1]}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a fixture schema, verify the collect snapshot against it and run "
            "every emitted DDL statement on a live PostgreSQL/PostGIS server."
        )
    )
    parser.add_argument(
        "--dsn",
        required=True,
        help="libpq connection string of a database this script may create tables in",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="leave the fixture schema in place afterwards for manual inspection",
    )
    args = parser.parse_args(argv)

    try:
        import psycopg
    except ImportError:
        print("error: psycopg is required; install it with 'pip install -r requirements.txt'")
        return EXIT_SETUP

    try:
        # Autocommit throughout: CREATE INDEX CONCURRENTLY and DROP INDEX
        # CONCURRENTLY cannot run inside a transaction block, and those are the
        # statements this script exists to execute.
        connection = psycopg.connect(args.dsn, autocommit=True)
    except Exception as error:  # noqa: BLE001
        print(f"error: cannot connect: {error}")
        return EXIT_SETUP

    checks = Checks()
    try:
        print(f"Server: {server_banner(connection)}")

        print("\n[setup] building the fixture schema")
        build_fixture(connection)
        if not wait_for_statistics(connection):
            print("error: pg_stat_user_tables never reported the fixture inserts")
            return EXIT_SETUP
        print("  fixture built and statistics visible")

        with tempfile.TemporaryDirectory() as directory:
            workdir = Path(directory)
            catalog = workdir / "snapshot.json"
            workload = workdir / "workload.sql"
            workload.write_text(WORKLOAD_SQL, encoding="utf-8")

            run_collect(args.dsn, catalog)
            snapshot = json.loads(catalog.read_text(encoding="utf-8"))
            check_snapshot(checks, connection, snapshot)

            report = run_analyze(workload, catalog)
            execute_emitted_ddl(checks, connection, report)

        check_rewrite_equivalence(checks, connection)
    except RuntimeError as error:
        print(f"\nerror: {error}")
        return EXIT_SETUP
    finally:
        if not args.keep:
            with connection.cursor() as cursor:
                cursor.execute(
                    "DROP TABLE IF EXISTS "
                    + ", ".join(FIXTURE_TABLES)
                    + " CASCADE"
                )
        connection.close()

    print(f"\n{checks.passed} check(s) passed, {len(checks.failures)} failed.")
    if checks.failures:
        print("\nFAILED:")
        for failure in checks.failures:
            print(f"  - {failure}")
        return EXIT_FAILED
    print("All live checks passed.")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
