"""Tests for statement normalization and fingerprinting."""

from __future__ import annotations

import pytest

from spatial_index_advisor.dialect import operators_in_text
from spatial_index_advisor.normalize import (
    fingerprint,
    normalize_and_fingerprint,
    normalize_sql,
)


def test_literal_variants_collapse_to_one_fingerprint() -> None:
    first = normalize_and_fingerprint(
        "SELECT id FROM t WHERE ST_DWithin(geom, ST_MakePoint(1, 2), 500) AND kind = 'bus'"
    )
    second = normalize_and_fingerprint(
        "SELECT id FROM t WHERE ST_DWithin(geom, ST_MakePoint(9, 9), 25) AND kind = 'van'"
    )
    assert first == second


def test_bind_parameters_and_literals_normalize_alike() -> None:
    parameterised, _ = normalize_and_fingerprint("SELECT id FROM t WHERE fleet_id = $1")
    literal, _ = normalize_and_fingerprint("SELECT id FROM t WHERE fleet_id = 42")
    assert parameterised == literal == "SELECT id FROM t WHERE fleet_id = ?"


def test_in_lists_of_different_lengths_collapse() -> None:
    short, _ = normalize_and_fingerprint("SELECT 1 FROM t WHERE id IN (1)")
    long, _ = normalize_and_fingerprint("SELECT 1 FROM t WHERE id IN (1, 2, 3, 4, 5)")
    assert short == long


def test_differing_structure_keeps_distinct_fingerprints() -> None:
    _, one = normalize_and_fingerprint("SELECT id FROM t WHERE a = 1")
    _, two = normalize_and_fingerprint("SELECT id FROM t WHERE b = 1")
    assert one != two


def test_trailing_semicolon_and_whitespace_are_irrelevant() -> None:
    assert normalize_sql("  SELECT 1 FROM t ;  ") == normalize_sql("SELECT 1 FROM t")


def test_empty_input_normalizes_to_empty_string() -> None:
    assert normalize_sql("   ") == ""


def test_unparseable_input_falls_back_to_textual_normalization() -> None:
    normalized = normalize_sql("GRANT WEIRD ?? 'x' 42 TO nobody")
    assert "42" not in normalized
    assert "'x'" not in normalized


def test_fingerprint_is_stable_and_short() -> None:
    digest = fingerprint("SELECT 1")
    assert digest == fingerprint("SELECT 1")
    assert len(digest) == 12


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT a && b FROM t", {"&&"}),
        ("SELECT a <-> b FROM t", {"<->"}),
        ("SELECT a <#> b FROM t", {"<#>"}),
        ("SELECT a <<->> b FROM t", {"<<->>"}),
        ("SELECT a &&& b FROM t", {"&&&"}),
        ("SELECT '<->' FROM t", set()),
        ("SELECT 1 -- a && b\nFROM t", set()),
    ],
)
def test_operator_text_scan(sql: str, expected: set[str]) -> None:
    assert operators_in_text(sql) == expected
