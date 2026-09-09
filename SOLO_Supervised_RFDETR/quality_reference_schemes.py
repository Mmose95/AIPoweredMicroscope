"""Count-based reference definitions shared by downstream evaluation and reporting.

The workbook Tag column is not a classification reference. It may retain explicit
exclusion markers for compatibility with the study workbook.
"""
from __future__ import annotations
import re
from typing import Any

COUNT_BINS = ("0-9", "10-25", "26+")


def normalized_count_bin(value: Any) -> str:
    """Normalize the expert count categories without inventing exact counts."""
    text = str(value if value is not None else "").strip().casefold().replace("–", "-").replace("—", "-")
    compact = re.sub(r"\s+", "", text.replace("til", "-"))
    if compact in {"0-9", "0to9"}:
        return "0-9"
    if compact in {"10-25", "10to25"}:
        return "10-25"
    if compact in {"26+", ">25", "26"}:
        return "26+"
    try:
        number = int(float(compact))
    except (TypeError, ValueError):
        raise ValueError(f"Unsupported expert count category: {value!r}")
    if number <= 9:
        return "0-9"
    if number <= 25:
        return "10-25"
    return "26+"


def count_bin_from_integer(value: int) -> str:
    number = float(value)
    if not number.is_integer() or number < 0:
        raise ValueError(f"Expected a nonnegative integer cell count, got {value!r}")
    value = int(number)
    if value <= 9:
        return "0-9"
    if value <= 25:
        return "10-25"
    return "26+"


def geckler_class(epithelial_bin: str, leucocyte_bin: str) -> str:
    """Return Geckler group 1-6 from the study's three expert count bins."""
    if epithelial_bin == "26+":
        return {"0-9": "G1", "10-25": "G2", "26+": "G3"}[leucocyte_bin]
    if epithelial_bin == "10-25" and leucocyte_bin == "26+":
        return "G4"
    if epithelial_bin == "0-9" and leucocyte_bin == "26+":
        return "G5"
    return "G6"


def collapsed_geckler_label(group: str) -> str:
    """Collapse Geckler to acceptable, unacceptable, and unknown."""
    if group in {"G4", "G5"}:
        return "Acceptable"
    if group in {"G1", "G2", "G3"}:
        return "Unacceptable"
    return "Unknown"


def murray_washington_label(epithelial_bin: str, leucocyte_bin: str) -> str:
    """Binary Murray-Washington culture-quality interpretation."""
    return (
        "Acceptable"
        if epithelial_bin == "0-9" and leucocyte_bin == "26+"
        else "Unacceptable"
    )


