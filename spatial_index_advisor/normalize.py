"""Statement normalization and fingerprinting.

Two statements that differ only in their literals describe the same workload and
must be counted together. Normalization rewrites every literal, bind parameter
and ``IN`` list to a single ``?`` placeholder; the fingerprint is a short hash of
the result. ``pg_stat_statements`` already does this server side, but log files
and ad-hoc SQL files do not, so the advisor does it for every source uniformly.
"""

from __future__ import annotations

import hashlib
import re
from typing import Final

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from .dialect import PostGIS

_COMMENT_RE: Final[re.Pattern[str]] = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)
_STRING_RE: Final[re.Pattern[str]] = re.compile(r"'(?:[^']|'')*'")
_NUMBER_RE: Final[re.Pattern[str]] = re.compile(r"\b\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\b")
_PARAM_RE: Final[re.Pattern[str]] = re.compile(r"\$\d+")
_WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")
_IN_LIST_RE: Final[re.Pattern[str]] = re.compile(r"IN\s*\((?:\s*\?\s*,)+\s*\?\s*\)", re.IGNORECASE)

#: Length of the hex fingerprint. 12 hex chars is 48 bits: collision-free for the
#: few thousand statement classes a real workload contains, short enough to read.
FINGERPRINT_LENGTH: Final[int] = 12


def _placeholder() -> exp.Var:
    """A bare ``?`` placeholder node.

    ``exp.Var`` is used rather than ``exp.Placeholder`` because the PostgreSQL
    generator renders placeholders as ``%s``, which is noisier to read.
    """
    return exp.Var(this="?")


def _mask_literals(node: exp.Expression) -> exp.Expression:
    """sqlglot transform callback replacing constants with placeholders."""
    if isinstance(node, (exp.Literal, exp.Parameter, exp.Placeholder)):
        return _placeholder()
    if isinstance(node, exp.In) and len(node.args.get("expressions") or []) > 1:
        return exp.In(this=node.this, expressions=[_placeholder()])
    return node


def normalize_sql(sql: str) -> str:
    """Return ``sql`` with all literals replaced by ``?`` and whitespace collapsed.

    Falls back to a textual normalization when the statement cannot be parsed,
    so that an exotic statement still groups with its own variants instead of
    fragmenting the workload.
    """
    stripped = sql.strip().rstrip(";").strip()
    if not stripped:
        return ""
    try:
        tree = sqlglot.parse_one(stripped, dialect=PostGIS)
    except SqlglotError:
        return _normalize_textually(stripped)
    if tree is None or isinstance(tree, exp.Command):
        # sqlglot falls back to an opaque Command node for syntax it does not
        # model; its payload is the raw text, so masking must be textual.
        return _normalize_textually(stripped)
    try:
        return tree.transform(_mask_literals).sql(dialect=PostGIS, normalize_functions=False)
    except SqlglotError:
        return _normalize_textually(stripped)


def _normalize_textually(sql: str) -> str:
    """Regex-only normalization used when sqlglot cannot parse the statement."""
    text = _COMMENT_RE.sub(" ", sql)
    text = _STRING_RE.sub("?", text)
    text = _PARAM_RE.sub("?", text)
    text = _NUMBER_RE.sub("?", text)
    text = _WHITESPACE_RE.sub(" ", text).strip().rstrip(";").strip()
    return _IN_LIST_RE.sub("IN (?)", text)


def fingerprint(normalized: str) -> str:
    """Stable short hash of a normalized statement."""
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    return digest[:FINGERPRINT_LENGTH]


def normalize_and_fingerprint(sql: str) -> tuple[str, str]:
    """Convenience wrapper returning ``(normalized_sql, fingerprint)``."""
    normalized = normalize_sql(sql)
    return normalized, fingerprint(normalized)
