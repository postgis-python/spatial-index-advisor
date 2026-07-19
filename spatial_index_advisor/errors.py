"""Exception hierarchy for spatial-index-advisor.

Every error raised deliberately by this package derives from :class:`AdvisorError`
so that the CLI can turn it into a clean one-line message instead of a traceback.
"""

from __future__ import annotations


class AdvisorError(Exception):
    """Base class for all errors raised by this package."""


class WorkloadParseError(AdvisorError):
    """A workload source could not be parsed.

    Carries the source path and, when known, the offending line number so the
    user can go and look at the input rather than guess.
    """

    def __init__(self, source: str, message: str, line: int | None = None) -> None:
        location = f"{source}:{line}" if line is not None else source
        super().__init__(f"{location}: {message}")
        self.source = source
        self.message = message
        self.line = line


class CatalogError(AdvisorError):
    """A catalog snapshot was missing, malformed, or internally inconsistent."""


class CollectorError(AdvisorError):
    """Collecting a snapshot from a live database failed."""
