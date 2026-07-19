"""A sqlglot dialect that understands the PostGIS index operators.

sqlglot's ``postgres`` dialect already parses ``&&`` and ``<->``, but not the
box-distance operator ``<#>`` or the n-dimensional overlap operator ``&&&``.
Rather than hand-rolling a parser we extend the dialect with the two missing
tokens so the full statement still parses into a normal sqlglot AST.

``<#>`` is deliberately mapped onto ``exp.Distance`` (the same node ``<->``
produces): both are KNN-orderable GiST operators and the recommendation logic
treats them identically. The *reported* operator name never comes from the AST —
it comes from :func:`operators_in_text`, which reads the statement text — so the
output stays faithful to what the user actually wrote.
"""

from __future__ import annotations

import re
from typing import Final

from sqlglot import exp
from sqlglot.dialects.postgres import Postgres
from sqlglot.tokens import TokenType


class PostGIS(Postgres):
    """PostgreSQL dialect extended with the PostGIS index operator tokens."""

    class Tokenizer(Postgres.Tokenizer):
        """Adds ``<#>`` and ``&&&`` to the PostgreSQL operator table."""

        KEYWORDS = {
            **Postgres.Tokenizer.KEYWORDS,
            "<#>": TokenType.HASH_DASH,
            "&&&": TokenType.DAMP,
        }

    class Parser(Postgres.Parser):
        """Binds the added tokens to existing binary expression nodes."""

        FACTOR = {**Postgres.Parser.FACTOR, TokenType.HASH_DASH: exp.Distance}
        BITWISE = {**Postgres.Parser.BITWISE, TokenType.DAMP: exp.ArrayOverlaps}


#: Spatial operators recognised in raw statement text, longest first so that
#: ``<<->>`` is not mistaken for ``<->`` and ``&&&`` not for ``&&``.
SPATIAL_OPERATORS: Final[tuple[str, ...]] = ("<<->>", "&&&", "<#>", "<->", "&&")

#: KNN-orderable distance operators. Only these justify an ORDER BY index scan.
KNN_OPERATORS: Final[frozenset[str]] = frozenset({"<->", "<#>", "<<->>"})

#: Bounding-box overlap operators, the classic GiST-accelerated predicate.
OVERLAP_OPERATORS: Final[frozenset[str]] = frozenset({"&&", "&&&"})

_OPERATOR_RE: Final[re.Pattern[str]] = re.compile(
    "|".join(re.escape(op) for op in SPATIAL_OPERATORS)
)
_STRING_OR_COMMENT_RE: Final[re.Pattern[str]] = re.compile(
    r"'(?:[^']|'')*'|--[^\n]*|/\*.*?\*/", re.DOTALL
)


def operators_in_text(sql: str) -> frozenset[str]:
    """Return the spatial operators literally present in ``sql``.

    String literals and comments are blanked out first so that an operator
    appearing inside a quoted string is not counted.
    """
    masked = _STRING_OR_COMMENT_RE.sub(lambda m: " " * len(m.group(0)), sql)
    return frozenset(_OPERATOR_RE.findall(masked))
