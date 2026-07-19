"""Documentation links attached to findings.

Each recommendation kind points at the guide that explains the underlying
technique, so a user who hits a finding can read about it rather than reverse
engineer the advisor's reasoning.
"""

from __future__ import annotations

from typing import Final

CORE_QUERY_PATTERNS: Final[str] = (
    "https://www.postgis-python.com/mastering-core-spatial-query-patterns/"
)
GIST_OPTIMIZATION: Final[str] = (
    "https://www.postgis-python.com/advanced-gist-indexing-optimization/"
)
SCHEMA_MIGRATIONS: Final[str] = (
    "https://www.postgis-python.com/spatial-schema-migrations-and-evolution/"
)
PERFORMANCE_MONITORING: Final[str] = (
    "https://www.postgis-python.com/spatial-performance-monitoring-and-observability/"
)
