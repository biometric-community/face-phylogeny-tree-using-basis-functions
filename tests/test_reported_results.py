"""REPORT.md, the committed output of the full run, against the published results."""

from __future__ import annotations

import re

import pytest


def percent(report: str, label: str) -> float:
    match = re.search(rf"{re.escape(label)}: \*\*([\d.]+)%\*\*", report)
    assert match, f"{label} missing from REPORT.md"
    return float(match.group(1))


@pytest.mark.parametrize("label,key", [("Rank-1 root", "rank1"), ("Rank-2 root", "rank2"), ("Rank-3 root", "rank3")])
def test_root_identification_reaches_tables_4_and_5(report, targets, label, key):
    assert percent(report, label) >= targets["chebyshev_floor"][key]


def test_ipt_reconstruction_reaches_tables_4_and_5(report, targets):
    assert percent(report, "IPT edge accuracy") >= targets["chebyshev_floor"]["ipt"]


def test_rank_1_is_well_above_chance_for_four_nodes(report):
    assert percent(report, "Rank-1 root") >= 40.0
