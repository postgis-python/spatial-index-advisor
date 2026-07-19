# spatial-index-advisor

Reads a PostgreSQL query workload and a PostGIS catalog snapshot, and tells you which spatial
indexes to build, which to drop, and which queries no index will ever save.

## What it does

Point it at a workload — a `pg_stat_statements` export, a PostgreSQL CSV log, or just a file of
SQL — plus a JSON snapshot of your tables, and it produces a ranked list of recommendations.
Each one carries the DDL, the reason, an estimated index size, a modelled benefit, a confidence
level, and the statement fingerprints that motivated it.

It covers:

- **Missing GiST indexes** on geometry columns that sargable spatial predicates filter on.
- **BRIN** on very large, append-mostly tables whose geometry correlates with physical order —
  with an explicit list of the conditions that make BRIN a bad idea.
- **GiST for KNN** on columns whose traffic is dominated by `ORDER BY geom <-> ...`. GiST is the
  only PostGIS method that can answer these from the index at all — see the note on SP-GiST under
  [the cost model](#the-cost-model).
- **Partial indexes** when a high-frequency statement always carries the same constant filter.
- **Composite (`btree_gist`) indexes** when a parameterised equality always co-occurs with the
  spatial predicate, including the operator-class caveat.
- **`CLUSTER` / physical reordering** for range-heavy scans against an uncorrelated heap.
- **Redundant indexes** — exact duplicates and leading-column prefixes — with the drop DDL.
- **Rewrite advisories** where indexing cannot help: `ST_Distance(...) < r` instead of
  `ST_DWithin`, `ST_Transform` applied to the indexed column, negated predicates.

Output is a terminal report, JSON, or bare SQL. The exit code reflects whether anything at or
above a chosen severity was found, so it drops into CI.

## Why it exists

Spatial index problems are hard to spot from a query plan alone, because the plan tells you what
happened, not what the workload as a whole needs. A `Seq Scan on vehicle_positions` in one
`EXPLAIN` looks survivable; the same scan executed 1.3 million times a day is the single largest
thing your database does. Conversely a query that looks pathological in isolation may run twice
a week and deserve no attention at all.

The other half of the problem is that PostGIS index selection is genuinely subtle, and the rules
are not obvious from the documentation:

- `ST_Intersects`, `ST_Contains` and friends silently emit a `&&` bounding-box check, so a GiST
  index helps. `ST_Distance(a, b) < r` does not, and no index will ever be used for it — the
  query has to become `ST_DWithin`.
- Wrapping the indexed column in `ST_Transform` discards the index entirely, and nothing in the
  plan says "you did this to yourself".
- BRIN can be four orders of magnitude smaller than GiST and just as fast — until an out-of-order
  backfill destroys the correlation it depends on, at which point every query reads the whole
  table and nothing warns you.
- A composite GiST index needs the `btree_gist` extension, and a partial index only helps if the
  filter appears as a literal in the query rather than a bind parameter.

This tool encodes those rules, applies them across the whole workload at once, and shows its
working. It does not replace `EXPLAIN (ANALYZE, BUFFERS)` — it tells you where to point it.

## Install

Python 3.10 or newer. There is nothing to install from a package index; clone and run.

```bash
git clone https://github.com/postgis-python/spatial-index-advisor.git
cd spatial-index-advisor
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Then run it as a module:

```bash
.venv/bin/python -m spatial_index_advisor analyze --help
```

`psycopg` is only used by the `collect` subcommand. The analysis engine never imports it, so if
you generate the catalog snapshot elsewhere you can drop that line from `requirements.txt`.

## Usage

### 1. Analyse a `pg_stat_statements` export

The shipped example is a fleet-tracking / geofencing schema: a 412-million-row position table, a
trips table, geofence polygons, driver pings and zone visits.

```bash
.venv/bin/python -m spatial_index_advisor analyze \
  -w examples/pg_stat_statements.csv \
  -c examples/catalog.json \
  --top 2
```

```
Workload: 10 statement fingerprints, 414,198,900 calls, 2,011,910,441 ms total execution time
Sources: examples/pg_stat_statements.csv(pgss)
                                 Spatial index recommendations
┏━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃   # ┃ Severity ┃ Recommendation                   ┃ Type           ┃ Saving/speedup ┃ Conf.  ┃
┡━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│   1 │ CRITICAL │ Add a GiST index on              │ GiST           │  6797.2T / 99x │ medium │
│     │          │ public.vehicle_positions.geom    │                │                │        │
│   2 │ CRITICAL │ Rewrite the predicate on         │ -              │   165.5T / 99x │ high   │
│     │          │ public.vehicle_positions.geom:   │                │                │        │
│     │          │ ST_Distance                      │                │                │        │
└─────┴──────────┴──────────────────────────────────┴────────────────┴────────────────┴────────┘
            Saving is modelled cost units across the workload; speedup is per call.
╭─────────────── 1. CRITICAL  Add a GiST index on public.vehicle_positions.geom ───────────────╮
│ 1,330,300 executions filter public.vehicle_positions.geom with ST_Contains, ST_DWithin, but  │
│ the column has no spatial index. Every one of those calls scans all 412,000,000 rows and     │
│ evaluates the predicate per row.                                                             │
│                                                                                              │
│ Why this index type: GiST is the general-purpose PostGIS index: it handles any geometry      │
│ type, supports the bounding-box operators these predicates expand to, and is orderable for   │
│ KNN. SP-GiST is only competitive for point data and BRIN only for physically clustered       │
│ tables.                                                                                      │
│                                                                                              │
│ CREATE INDEX CONCURRENTLY idx_vehicle_positions_geom_gist ON public.vehicle_positions USING  │
│ GIST (geom);                                                                                 │
│                                                                                              │
│ estimated size 17.1 GB; basis: sequential scan vs GiST scan at 0.5000% selectivity (default  │
│ assumption, no geometry statistics available); confidence: medium                            │
│                                                                                              │
│ Caveats:                                                                                     │
│   - CONCURRENTLY avoids an ACCESS EXCLUSIVE lock but cannot run inside a transaction block,  │
│ and leaves an INVALID index behind if it fails.                                              │
│   - Building a GiST index on a large table is I/O heavy; consider raising                    │
│ maintenance_work_mem for the session.                                                        │
│                                                                                              │
│ Statements: 50c70d6b1fda, 82e94541628a                                                       │
│                                                                                              │
│ Further reading: https://www.postgis-python.com/advanced-gist-indexing-optimization/         │
╰──────────────────────────────────────────────────────────────────────────────────────────────╯
╭────── 2. CRITICAL  Rewrite the predicate on public.vehicle_positions.geom: ST_Distance ──────╮
│ 32,400 executions use ST_Distance against geom in a form that cannot use any index:          │
│ ST_Distance does not emit a bounding-box check, so no index applies. No index will help      │
│ until the statement changes.                                                                 │
│                                                                                              │
│ basis: sequential scan today vs an indexed scan after the rewrite; confidence: high          │
│                                                                                              │
│ Caveats:                                                                                     │
│   - Replace the distance comparison with ST_DWithin(geom, search_geom, radius); only         │
│ ST_DWithin can be answered from a GiST index.                                                │
│                                                                                              │
│ Statements: 9f6821b296e6                                                                     │
│                                                                                              │
│ Further reading: https://www.postgis-python.com/mastering-core-spatial-query-patterns/       │
╰──────────────────────────────────────────────────────────────────────────────────────────────╯
1 referenced table(s) are absent from the catalog snapshot and were ignored: vehicles.
Benefit and size figures are heuristic estimates from a static cost model, not measurements.
Verify each change with EXPLAIN (ANALYZE, BUFFERS).
```

Note the `vehicles` warning: the workload references a table the snapshot does not describe, so
the advisor says so rather than guessing. The `INSERT` statement in the same export — 412 million
calls of it — is correctly ignored, because it carries no spatial predicate.

### 2. Get just the DDL

`--format sql` emits the recommendations in rank order as a reviewable script.

```bash
.venv/bin/python -m spatial_index_advisor analyze \
  -w examples/postgresql-2026-07-14.csv \
  -c examples/catalog.json \
  --format sql --top 5
```

```sql
-- Generated by spatial-index-advisor.
-- Benefit and size figures are heuristic estimates from a static cost model, not measurements. Verify each change with EXPLAIN (ANALYZE, BUFFERS).
-- Review every statement before running it. CREATE/DROP INDEX CONCURRENTLY
-- cannot run inside a transaction block, so apply these one at a time.

-- [1] CRITICAL (high confidence): Add a GiST index on public.vehicle_positions.geom
--     estimated benefit: 1,259,884,874 cost units over 11 calls (~3660x per call, estimated)
--     estimated size: 17.1 GB
--     caveat: CONCURRENTLY avoids an ACCESS EXCLUSIVE lock but cannot run inside a transaction block, and leaves an INVALID index behind if it fails.
--     caveat: Building a GiST index on a large table is I/O heavy; consider raising maintenance_work_mem for the session.
CREATE INDEX CONCURRENTLY idx_vehicle_positions_geom_gist ON public.vehicle_positions USING GIST (geom);

-- [3] CRITICAL (high confidence): Add a GiST index on public.trips.geom
--     estimated benefit: 41,103,140 cost units over 15 calls (~17x per call, estimated)
--     estimated size: 399.1 MB
--     caveat: CONCURRENTLY avoids an ACCESS EXCLUSIVE lock but cannot run inside a transaction block, and leaves an INVALID index behind if it fails.
--     caveat: Building a GiST index on a large table is I/O heavy; consider raising maintenance_work_mem for the session.
CREATE INDEX CONCURRENTLY idx_trips_geom_gist ON public.trips USING GIST (geom);

-- [4] HIGH (medium confidence): Add a partial GiST index on public.trips.geom WHERE status = 'active'
--     estimated benefit: 43,114,840 cost units over 15 calls (~87x per call, estimated)
--     estimated size: 79.8 MB
--     caveat: The 20% coverage figure is an assumption; check the real selectivity of `status = 'active'` before relying on the size estimate.
--     caveat: Statements that pass the filter value as a bind parameter will not match the index predicate and will fall back to the full index or a seq scan.
CREATE INDEX CONCURRENTLY idx_trips_geom_gist_partial ON public.trips USING GIST (geom) WHERE status = 'active';

-- [5] MEDIUM (medium confidence): Consider BRIN on public.vehicle_positions.geom as a low-cost alternative
--     estimated benefit: 1,184,493,358 cost units over 11 calls (~17x per call, estimated)
--     estimated size: 4.0 MB
--     caveat: BRIN is a bad idea as soon as the physical ordering breaks down: heavy UPDATEs, out-of-order backfills, or a VACUUM FULL with a different ordering will silently turn every scan into a full heap read.
--     caveat: BRIN cannot answer highly selective single-row lookups efficiently, and it is not orderable, so it cannot serve KNN queries.
--     caveat: New page ranges are only summarised by autovacuum or brin_summarize_new_values(); until then they are always scanned.
--     caveat: Keep a GiST index as well if any statement needs precise, selective access.
CREATE INDEX CONCURRENTLY idx_vehicle_positions_geom_brin ON public.vehicle_positions USING BRIN (geom);
```

Rank 2 is missing from the script because it is a rewrite advisory with no DDL — the numbering
keeps referring to the full report. Note also that this run reaches **high** confidence on the
`vehicle_positions` index, where the `pg_stat_statements` run above only managed **medium**: a
server-side statement export normalises `ST_DWithin(geom, $1, $2)`, whereas the log preserves the
literal 250-metre radius, so the selectivity estimate stops being a guess.

### 3. Different workload sources

The parser is chosen per file by sniffing, and several `-w` files can be combined in one run.
`examples/queries.sql` is a plain file of statements with optional `-- calls: N` hints — the
lowest-friction way to start, and because it keeps its literals it exercises every rule:

```bash
.venv/bin/python -m spatial_index_advisor analyze \
  -w examples/queries.sql \
  -c examples/catalog.json \
  --top 6
```

```
Workload: 7 statement fingerprints, 2,075,700 calls, 0 ms total execution time
Sources: examples/queries.sql(sql)
                                 Spatial index recommendations
┏━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃   # ┃ Severity ┃ Recommendation                   ┃ Type           ┃ Saving/speedup ┃ Conf.  ┃
┡━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│   1 │ CRITICAL │ Add a GiST index on              │ GiST           │ 152.4T / 3660x │ high   │
│     │          │ public.vehicle_positions.geom    │                │                │        │
│   2 │ CRITICAL │ Rewrite the predicate on         │ -              │   3.7T / 3660x │ high   │
│     │          │ public.vehicle_positions.geom:   │                │                │        │
│     │          │ ST_Distance                      │                │                │        │
│   3 │ CRITICAL │ Add a GiST index on              │ GiST           │   662.6G / 17x │ high   │
│     │          │ public.trips.geom                │                │                │        │
│   4 │ CRITICAL │ Rewrite the predicate on         │ expression     │    20.0G / 17x │ high   │
│     │          │ public.trips.geom: ST_Intersects │ GiST           │                │        │
│   5 │ HIGH     │ Add a partial GiST index on      │ GiST (partial) │   695.0G / 87x │ medium │
│     │          │ public.trips.geom WHERE fleet_id │                │                │        │
│     │          │ = 12                             │                │                │        │
│   6 │ HIGH     │ CLUSTER public.zone_visits on    │ -              │    167.1G / 8x │ medium │
│     │          │ idx_zone_visits_geom_gist        │                │                │        │
└─────┴──────────┴──────────────────────────────────┴────────────────┴────────────────┴────────┘
            Saving is modelled cost units across the workload; speedup is per call.
```

There is no timing data in a SQL file, so the report warns that ranking is structural only. Note
finding 4: because the file carries a literal SRID, the advisor can emit a runnable expression
index — `CREATE INDEX CONCURRENTLY ... USING GIST (ST_Transform(geom, 4326))` — instead of a
placeholder.

### 4. JSON for pipelines

```bash
.venv/bin/python -m spatial_index_advisor analyze \
  -w examples/pg_stat_statements.csv -c examples/catalog.json \
  --format json --top 1
```

```json
{
  "disclaimer": "Benefit and size figures are heuristic estimates from a static cost model, not measurements. Verify each change with EXPLAIN (ANALYZE, BUFFERS).",
  "workload": {
    "sources": [
      "examples/pg_stat_statements.csv(pgss)"
    ],
    "statement_count": 10,
    "total_calls": 414198900,
    "total_exec_time_ms": 2011910441.12
  },
  "warnings": [
    "1 referenced table(s) are absent from the catalog snapshot and were ignored: vehicles."
  ],
  "unparsed_fingerprints": [],
  "unknown_tables": [
    "vehicles"
  ],
  "recommendation_count": 11,
  "has_critical": true,
  "recommendations": [
    {
      "kind": "missing_gist",
      "title": "Add a GiST index on public.vehicle_positions.geom",
      "table": "public.vehicle_positions",
      "severity": "critical",
      "confidence": "medium",
      "rationale": "1,330,300 executions filter public.vehicle_positions.geom with ST_Contains, ST_DWithin, but the column has no spatial index. Every one of those calls scans all 412,000,000 rows and evaluates the predicate per row.",
      "index_type": "GiST",
      "type_rationale": "GiST is the general-purpose PostGIS index: it handles any geometry type, supports the bounding-box operators these predicates expand to, and is orderable for KNN. SP-GiST is only competitive for point data and BRIN only for physically clustered tables.",
      "ddl": "CREATE INDEX CONCURRENTLY idx_vehicle_positions_geom_gist ON public.vehicle_positions USING GIST (geom);",
      "estimated_size_bytes": 18342961152,
      "caveats": [
        "CONCURRENTLY avoids an ACCESS EXCLUSIVE lock but cannot run inside a transaction block, and leaves an INVALID index behind if it fails.",
        "Building a GiST index on a large table is I/O heavy; consider raising maintenance_work_mem for the session."
      ],
      "fingerprints": [
        "50c70d6b1fda",
        "82e94541628a"
      ],
      "docs_url": "https://www.postgis-python.com/advanced-gist-indexing-optimization/",
      "estimate_is_heuristic": true,
      "benefit": {
        "current_cost_per_call": 5161566289.0,
        "projected_cost_per_call": 52060317.56,
        "calls": 1330300,
        "total_cost_saved": 6797175793808893.0,
        "speedup": 99.1,
        "basis": "sequential scan vs GiST scan at 0.5000% selectivity (default assumption, no geometry statistics available)"
      }
    }
  ]
}
```

### 5. Collect a snapshot from a live database

```bash
.venv/bin/python -m spatial_index_advisor collect \
  --dsn 'postgresql://reader@db.internal/fleet' \
  --schema public \
  --output catalog.json
```

This reads `geometry_columns`, `pg_class`, `pg_index`, `pg_stat_user_tables` and `pg_stats`, plus
`ST_EstimatedExtent` and a 1% `TABLESAMPLE` per geometry column to measure average feature size.
Pass `--no-sample` to skip the sampling queries on a busy system. A read-only role is enough.

### 6. Use it in CI

```bash
.venv/bin/python -m spatial_index_advisor analyze \
  -w workload.csv -c catalog.json --format json --fail-on high
```

Exit codes: `0` nothing at or above the threshold, `2` something was found, `1` the tool could not
run. `--fail-on` accepts `critical` (the default), `high`, `medium`, `low` or `never`.

### As a library

```python
from pathlib import Path

from spatial_index_advisor import analyse, load_catalog, load_workload

report = analyse(
    load_workload([Path("examples/pg_stat_statements.csv")]),
    load_catalog(Path("examples/catalog.json")),
)
for recommendation in report.top(5):
    print(recommendation.severity.value, recommendation.title)
    print(recommendation.ddl)
```

## How it works

### Normalization and fingerprinting

Every statement is parsed with [sqlglot](https://github.com/tobymao/sqlglot) and rewritten with
all literals, bind parameters and `IN` lists replaced by `?`. The SHA-1 of the result, truncated
to 12 hex characters, is the fingerprint; variants that differ only in their constants are folded
into one entry whose counters are summed. Statements sqlglot cannot model fall back to a regex
normalization so they still group with their own variants instead of fragmenting the workload.

When the same fingerprint arrives from two sources, the representative sample kept for analysis is
the one with the fewest bind parameters — a log line with a literal radius is strictly more
informative than the server-normalised form of the same query.

### Predicate analysis

The advisor extends sqlglot's PostgreSQL dialect with the two PostGIS operators it does not model
(`<#>` and `&&&`) rather than hand-rolling a parser. From the resulting AST it extracts, per
statement: the tables and aliases, the geometry columns in each spatial predicate and the
functions wrapping them, whether each predicate is sargable, the co-occurring non-spatial filters
and whether their right-hand side is a constant, any KNN `ORDER BY`, and the `LIMIT`.

Sargability follows PostGIS semantics: `ST_Intersects`, `ST_Contains`, `ST_Within`, `ST_Covers`,
`ST_DWithin` and their relatives emit an internal `&&` and are indexable; `ST_Distance`,
`ST_Disjoint`, `ST_Relate` and friends are not. A predicate under `NOT`, or with the column
wrapped in `ST_Transform`, `ST_Buffer`, `ST_Centroid` and similar, is marked non-sargable with the
reason recorded.

Only top-level `AND` operands are eligible as partial-index predicates, because anything under an
`OR` is not guaranteed to hold for every matching row.

### The cost model

`spatial_index_advisor/costmodel.py` holds every formula and constant, each documented in place.
It uses PostgreSQL's default planner parameters (`seq_page_cost = 1.0`, `random_page_cost = 4.0`,
`cpu_tuple_cost = 0.01`, `cpu_operator_cost = 0.0025`) so the figures are broadly comparable to
the `cost=` values in an `EXPLAIN` plan, plus one model constant of its own: spatial predicates
are charged 5000× a scalar comparison, which is the `procost` PostGIS declares for the
geometry/geometry forms of `ST_Intersects`, `ST_Contains` and `ST_DWithin` (`SELECT proname,
procost FROM pg_proc`). Checked against PostgreSQL 16 / PostGIS 3.4, a modelled sequential scan
reproduces the planner's own total cost exactly on three of four test tables; the fourth differs
by 2× because its predicate wraps the column in `ST_Transform`, so the planner charges two costly
functions per row where the model charges one.

- A **sequential scan** costs `pages × seq_page_cost + rows × (cpu_tuple_cost + spatial predicate
  cost)`.
- A **GiST scan** costs tree descent, plus per-matched-row index and recheck CPU, plus heap
  fetches weighted by physical correlation — correlated matches are read sequentially,
  uncorrelated ones cost a random page each.
- A **BRIN scan** reads `selectivity + (1 − |correlation|) × (1 − selectivity)` of the heap. At
  correlation 1.0 that is just the matching fraction; at correlation 0 it is the whole table, and
  BRIN is strictly worse than a sequential scan.
- **Selectivity** comes from geometry statistics where they exist: for `ST_DWithin(geom, p, r)`
  the `(2r)²` search window is compared against the layer extent. Failing that, the mean feature
  bounding box area against the extent. Failing that, a flat 0.5% assumption — and the
  recommendation is downgraded to medium or low confidence and says so in its `basis` field.
- **Index sizes** come from bytes-per-entry estimates: 40 for a 2D GiST box, 24 for an SP-GiST
  quadtree leaf, one 64-byte summary per 128-page range for BRIN, at a 0.9 fill factor. These hold
  up well when built for real: against PostGIS 3.4 the model predicted 233.5 MB for a GiST index
  that came out at 217 MB, 17.0 MB for one that came out at 16 MB, and 48 kB for a BRIN index that
  came out at 40 kB. The exception is the partial index, which inherits the 20% coverage guess
  described under [limitations](#limitations) and was 4× low.

**On SP-GiST and KNN.** Earlier versions of this tool recommended SP-GiST for point columns whose
traffic was dominated by `ORDER BY geom <-> point`. That advice was wrong. Only
`gist_geometry_ops_2d` registers `<->` and `<#>` as *ordering* operators — in
`pg_amop`, `amoppurpose = 'o'` — and `spgist_geometry_ops_2d` registers none. An SP-GiST index
therefore cannot return rows in distance order under any circumstances: measured on 150,000 point
rows, adding one left the plan an unchanged sequential scan with a top-N sort even with
`enable_seqscan = off`, while the equivalent GiST index turned the same query into an ordered
index scan and took it from 25.5 ms to 0.12 ms. SP-GiST remains a fine choice for *containment*
searches on points — it measured 0.23 ms against GiST's 0.25 ms on the same data — but this tool
no longer proposes it, because the case it used to propose it for is the one case it cannot serve.

Severity is capped twice. A rule cap encodes triage: "there is no index at all" can be CRITICAL,
"a better index exists" cannot exceed HIGH, and "you have a duplicate index" is always LOW. A
table-size cap on top of that means a finding about a 24,000-row table can never be critical
however often it is hit. CRITICAL additionally requires a large per-call speedup, not just a large
aggregate — a big number spread over millions of already-fast calls is not an emergency.

### Limitations

Be aware of these before trusting a number:

- **The estimates are model output, not measurements.** There is no planner, no `EXPLAIN`, no
  runtime feedback. They exist to rank recommendations against each other and give an order of
  magnitude. Measure before and after.
- **The ranking is trustworthy; the magnitudes are not.** Every recommendation the tool produced
  for a 6-table, 6.2-million-row PostGIS 3.4 schema was applied and measured with
  `EXPLAIN (ANALYZE, BUFFERS)`. The ordering held up well — Spearman ρ = 0.87 against the real
  ranking by total time saved, with the tool's top three being exactly the real top three (they
  were within 8% of one another, so their internal order is a coin toss) and the bottom three in
  exactly the right places. Individual speedups were much rougher: the ratio of measured to
  estimated speedup ranged from 0.06× to 12×, median 0.56×. Treat a modelled speedup as "this is
  the biggest win available", never as "this will be 99× faster".
- **The model inherits the planner's optimism about `procost`.** Because the spatial predicate
  cost now matches what PostGIS declares, the modelled sequential scan reproduces the planner's
  `cost=` exactly — including where the planner is wrong. Both charge the full 5000× function cost
  for every row, but a selective `&&` bounding-box test rejects most rows long before the exact
  predicate runs. That is why the two worst overestimates in the calibration run were the largest
  table, where the model predicted 100–200× and the measured win was 11×.
- **Partial-index detection needs literals.** `pg_stat_statements` normalises constants to `$n`
  server-side, so a partial index can only be proposed from a log or SQL-file workload. This is
  why the tool supports several sources.
- **Selectivity of the non-spatial column is assumed**, at 10% for a composite index and 20%
  coverage for a partial index. Check `pg_stats.n_distinct` before sizing anything on those. These
  two are pure guesses and they can be badly wrong in the direction that matters: in the
  calibration run the dominant filter was `status = 'open'`, which covered 89% of the table, not
  20%. The partial index was still a large win — but only because it replaced a sequential scan,
  not because it was selective, and its size estimate was off by more than 4×. The assumption is
  left in place because nothing in a workload file can reveal the true fraction; only the catalog
  can, and the snapshot does not carry per-value statistics.
- **CTEs and subqueries are flattened** into a single alias namespace. An unqualified column in a
  multi-table statement is left unresolved rather than guessed, so it is silently skipped.
- **`<->` and `<#>` share an AST node**; a statement mixing both is reported under `<->`. The full
  operator set is preserved on the analysis object.
- **No `pg_stat_user_indexes` feedback.** A "redundant" index may still be serving a constraint or
  a quarterly report; the recommendation says to check `idx_scan` first, and it means it.
- **Write cost is not modelled.** Every index slows down `INSERT`/`UPDATE` and adds vacuum work.
  The advisor accounts for this only qualitatively, by refusing to index small tables.
- **One geometry column at a time.** Multi-column spatial indexes beyond `(scalar, geometry)` are
  out of scope.

## Configuration

There is no config file; behaviour is controlled by flags and by the catalog snapshot.

| Flag | Effect |
| --- | --- |
| `-w/--workload PATH` | Workload file; repeat for several. |
| `-c/--catalog PATH` | JSON catalog snapshot. |
| `--workload-format` | `auto` (default), `pgss`, `csvlog`, `sql`. |
| `-f/--format` | `rich` (default), `json`, `sql`. |
| `-n/--top N` | Show only the N highest-ranked recommendations. |
| `--fail-on` | `critical` (default), `high`, `medium`, `low`, `never`. |
| `--no-color` | Plain text output. |

The thresholds that drive the rules are module-level constants in
`spatial_index_advisor/rules.py`, each with a comment explaining the judgement behind it —
`BRIN_MIN_ROWS`, `FILTER_DOMINANCE`, `CLUSTER_MAX_CORRELATION`, `MIN_ROWS_FOR_INDEX` and the
severity bands. The cost model constants live in `spatial_index_advisor/costmodel.py`.

### Catalog snapshot format

Only `name`, `row_count` and `table_bytes` are required per table. Everything else improves the
estimate; where a statistic is missing, the affected recommendation drops in confidence rather
than disappearing.

```json
{
  "database": "fleet",
  "collected_at": "2026-07-14T09:12:03+00:00",
  "postgis_version": "3.4.2",
  "tables": [
    {
      "name": "public.vehicle_positions",
      "row_count": 412000000,
      "table_bytes": 61000000000,
      "inserts": 412000000, "updates": 0, "deletes": 0,
      "append_only": true,
      "column_correlation": { "recorded_at": 0.999 },
      "geometry_columns": [
        {
          "name": "geom", "geometry_type": "POINT", "srid": 3857,
          "avg_bbox_width": 0.0, "avg_bbox_height": 0.0,
          "extent_width": 62000.0, "extent_height": 41000.0,
          "correlation": 0.94
        }
      ],
      "indexes": [
        { "name": "vehicle_positions_pkey", "method": "btree",
          "columns": ["id"], "unique": true, "size_bytes": 9250000000 }
      ]
    }
  ]
}
```

`tables` may also be an object keyed by table name. `append_only` is derived from the write
counters when it is not given explicitly.

## Testing

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

225 tests, no database, no network, no Docker. The collector is exercised through a fake DB-API
connection that replays recorded catalog rows, and the rest of the suite runs from the fixtures in
`examples/` and in-memory builders.

Coverage:

```bash
.venv/bin/python -m pytest --cov=spatial_index_advisor --cov-report=term-missing
```

## Further reading

The guides behind the rules this tool applies:

- [Mastering core spatial query patterns](https://www.postgis-python.com/mastering-core-spatial-query-patterns/)
  — bounding-box `&&`, `ST_DWithin`, KNN `<->` and spatial joins, and why `ST_Distance(...) < r`
  is the wrong shape.
- [Advanced GiST indexing and optimization](https://www.postgis-python.com/advanced-gist-indexing-optimization/)
  — partial and composite indexes, index-only scans, and GiST vs SP-GiST vs BRIN.
- [Spatial schema migrations and evolution](https://www.postgis-python.com/spatial-schema-migrations-and-evolution/)
  — concurrent index builds, in-place SRID reprojection and zero-downtime backfills.
- [Spatial performance monitoring and observability](https://www.postgis-python.com/spatial-performance-monitoring-and-observability/)
  — getting a good `pg_stat_statements` export, GiST bloat detection and autovacuum tuning.

## License

MIT. See [LICENSE](LICENSE).
