"""
Chemistry parsing utilities for perovskite solar-cell datasets.

This module converts heterogeneous absorber-material strings into
structured chemistry features for downstream statistical analysis
and machine-learning models.
"""

import re
import pandas as pd


def element_symbols(text):
    """Extract element-like symbols from a material string."""

    if not isinstance(text, str):
        return set()

    return set(
        re.findall(
            r"[A-Z][a-z]?",
            text
        )
    )


def has_element_or_word(text, symbol, words):
    """
    Detect an element either from its chemical symbol
    or from explicit textual aliases.
    """

    text = str(text)

    symbols = element_symbols(text)

    low = text.lower()

    if symbol in symbols:
        return 1

    for word in words:
        if word in low:
            return 1

    return 0


def detect_fa(text):
    """Detect formamidinium (FA) chemistry."""

    text = str(text)
    low = text.lower()

    if "formamidinium" in low:
        return 1

    patterns = [
        r"FAPb",
        r"FASn",
        r"FA\d",
        r"\(FA",
        r"FA\(",
        r"FA[0-9.]",
    ]

    return int(
        any(
            re.search(pattern, text)
            for pattern in patterns
        )
    )


def detect_ma(text):
    """Detect methylammonium (MA) chemistry."""

    text = str(text)
    low = text.lower()

    if "methylammonium" in low:
        return 1

    patterns = [
        r"MAPb",
        r"MASn",
        r"MA\d",
        r"\(MA",
        r"MA\(",
        r"MA[0-9.]",
    ]

    return int(
        any(
            re.search(pattern, text)
            for pattern in patterns
        )
    )


def assign_cation_family(row):
    """
    Assign absorber to an FA/MA/Cs cation family.
    """

    fa = row["has_FA"]
    ma = row["has_MA"]
    cs = row["has_Cs"]

    n_present = fa + ma + cs

    if n_present == 0:
        return "OTHER_OR_UNCLEAR"

    if n_present >= 2:
        return "MIXED_FA_MA_CS"

    if fa:
        return "FA"

    if ma:
        return "MA"

    if cs:
        return "Cs"

    return "OTHER_OR_UNCLEAR"


def assign_halide_family(row):
    """Assign I/Br/Cl halide family."""

    present = []

    if row["has_I"]:
        present.append("I")

    if row["has_Br"]:
        present.append("Br")

    if row["has_Cl"]:
        present.append("Cl")

    if len(present) == 0:
        return "OTHER_OR_UNCLEAR"

    if len(present) == 1:
        return present[0]

    return "+".join(present)


def add_chemistry_features(df, absorber_col="absorber_final"):
    """
    Convert absorber strings into structured chemistry features.

    Parameters
    ----------
    df : pandas.DataFrame
        Input perovskite dataset.

    absorber_col : str
        Column containing absorber-material descriptions.

    Returns
    -------
    pandas.DataFrame
        Copy of the input DataFrame with chemistry features added.
    """

    df = df.copy()

    absorber = (
        df[absorber_col]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["has_Pb"] = absorber.apply(
        lambda x: has_element_or_word(
            x,
            "Pb",
            ["lead", "plumb"]
        )
    )

    df["has_Sn"] = absorber.apply(
        lambda x: has_element_or_word(
            x,
            "Sn",
            ["tin"]
        )
    )

    df["has_Cs"] = absorber.apply(
        lambda x: has_element_or_word(
            x,
            "Cs",
            ["cesium", "caesium"]
        )
    )

    df["has_I"] = absorber.apply(
        lambda x: has_element_or_word(
            x,
            "I",
            ["iodide", "iodine"]
        )
    )

    df["has_Br"] = absorber.apply(
        lambda x: has_element_or_word(
            x,
            "Br",
            ["bromide", "bromine"]
        )
    )

    df["has_Cl"] = absorber.apply(
        lambda x: has_element_or_word(
            x,
            "Cl",
            ["chloride", "chlorine"]
        )
    )

    df["has_FA"] = absorber.apply(
        detect_fa
    )

    df["has_MA"] = absorber.apply(
        detect_ma
    )

    df["n_FA_MA_Cs"] = (
        df[
            [
                "has_FA",
                "has_MA",
                "has_Cs"
            ]
        ]
        .sum(axis=1)
    )

    df["mixed_FA_MA_Cs"] = (
        df["n_FA_MA_Cs"] >= 2
    ).astype(int)

    df["n_halides"] = (
        df[
            [
                "has_I",
                "has_Br",
                "has_Cl"
            ]
        ]
        .sum(axis=1)
    )

    df["mixed_halide"] = (
        df["n_halides"] >= 2
    ).astype(int)

    df["Pb_Sn_mixed_detected"] = (
        (df["has_Pb"] == 1)
        &
        (df["has_Sn"] == 1)
    ).astype(int)

    df["cation_family"] = df.apply(
        assign_cation_family,
        axis=1
    )

    df["halide_family"] = df.apply(
        assign_halide_family,
        axis=1
    )

    return df
