"""spatial-index-advisor: spatial index recommendations for PostgreSQL/PostGIS.

Library use::

    from pathlib import Path
    from spatial_index_advisor import analyse, load_catalog, load_workload

    report = analyse(
        load_workload([Path("examples/pg_stat_statements.csv")]),
        load_catalog(Path("examples/catalog.json")),
    )

Command line use::

    python -m spatial_index_advisor analyze -w workload.csv -c catalog.json
"""

from __future__ import annotations

__version__ = "1.0.0"

from .analysis import analyze_statement
from .catalog import load_catalog, parse_catalog
from .engine import analyse, analyse_statements
from .errors import AdvisorError, CatalogError, CollectorError, WorkloadParseError
from .models import (
    AdvisorReport,
    BenefitEstimate,
    CatalogSnapshot,
    Confidence,
    Recommendation,
    Severity,
    StatementAnalysis,
    TableStats,
    Workload,
    WorkloadStatement,
)
from .workload import load_workload

__all__ = [
    "AdvisorError",
    "AdvisorReport",
    "BenefitEstimate",
    "CatalogError",
    "CatalogSnapshot",
    "CollectorError",
    "Confidence",
    "Recommendation",
    "Severity",
    "StatementAnalysis",
    "TableStats",
    "Workload",
    "WorkloadParseError",
    "WorkloadStatement",
    "__version__",
    "analyse",
    "analyse_statements",
    "analyze_statement",
    "load_catalog",
    "load_workload",
    "parse_catalog",
]
