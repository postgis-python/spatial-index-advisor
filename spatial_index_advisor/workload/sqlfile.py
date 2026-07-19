"""Parser for a plain file of semicolon-separated SQL statements.

This is the lowest-friction source: paste the queries your application issues
into a file and run the advisor against it. There are no execution counts, so
every statement is credited with a single call and no timing; the cost model
falls back to structural ranking for such a workload, which the report states.

An optional trailing comment of the form ``-- calls: 5000`` immediately before a
statement supplies a call count, which makes frequency-based rules (partial
indexes in particular) usable without a real workload capture.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from ..errors import WorkloadParseError
from .base import RawStatement, WorkloadSourceParser

_CALLS_HINT_RE: Final[re.Pattern[str]] = re.compile(
    r"--\s*calls\s*[:=]\s*(\d+)", re.IGNORECASE
)
_LINE_COMMENT_RE: Final[re.Pattern[str]] = re.compile(r"--[^\n]*")
_BLOCK_COMMENT_RE: Final[re.Pattern[str]] = re.compile(r"/\*.*?\*/", re.DOTALL)


def split_statements(text: str) -> list[str]:
    """Split SQL text on semicolons that are not inside a string or comment.

    Dollar-quoted bodies are not supported; a file containing a function
    definition should not be used as a workload source.
    """
    chunks: list[str] = []
    buffer: list[str] = []
    in_string = False
    in_line_comment = False
    in_block_comment = False
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        nxt = text[index + 1] if index + 1 < length else ""
        if in_line_comment:
            if char == "\n":
                in_line_comment = False
            buffer.append(char)
        elif in_block_comment:
            buffer.append(char)
            if char == "*" and nxt == "/":
                buffer.append(nxt)
                index += 1
                in_block_comment = False
        elif in_string:
            buffer.append(char)
            if char == "'":
                if nxt == "'":
                    buffer.append(nxt)
                    index += 1
                else:
                    in_string = False
        elif char == "'":
            in_string = True
            buffer.append(char)
        elif char == "-" and nxt == "-":
            in_line_comment = True
            buffer.append(char)
        elif char == "/" and nxt == "*":
            in_block_comment = True
            buffer.append(char)
        elif char == ";":
            chunks.append("".join(buffer))
            buffer = []
        else:
            buffer.append(char)
        index += 1
    chunks.append("".join(buffer))
    return [chunk for chunk in chunks if _strip_comments(chunk).strip()]


def _strip_comments(text: str) -> str:
    """Remove SQL comments, used to decide whether a chunk holds any statement."""
    return _LINE_COMMENT_RE.sub(" ", _BLOCK_COMMENT_RE.sub(" ", text))


class SqlFileParser(WorkloadSourceParser):
    """Reads a plain ``.sql`` file of statements."""

    name = "sql"

    def sniff(self, text: str, path: Path) -> bool:
        """Recognise any file whose first meaningful token starts a statement."""
        stripped = _strip_comments(text).strip()
        return bool(
            re.match(r"^(SELECT|WITH|INSERT|UPDATE|DELETE)\b", stripped, re.IGNORECASE)
        )

    def parse(self, text: str, path: Path) -> list[RawStatement]:
        """Split the file into statements, honouring ``-- calls:`` hints."""
        chunks = split_statements(text)
        if not chunks:
            raise WorkloadParseError(str(path), "file contains no SQL statements")
        statements: list[RawStatement] = []
        for chunk in chunks:
            hint = _CALLS_HINT_RE.search(chunk)
            calls = int(hint.group(1)) if hint else 1
            sql = _strip_comments(chunk).strip()
            if not sql:
                continue
            statements.append(RawStatement(sql=sql, calls=max(1, calls)))
        if not statements:
            raise WorkloadParseError(str(path), "file contains no SQL statements")
        return statements
