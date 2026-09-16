"""
Feature-engineering utilities for perovskite solar-cell datasets.

This module standardizes ETL/HTL labels, groups rare categories,
and prepares leakage-safe features for PCE prediction.
"""

import re
import numpy as np
import pandas as pd


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_material_text(x):
    """Normalize whitespace in material labels."""

    if pd.isna(x):
        return np.nan

    return re.sub(
        r"\s+",
        " ",
        str(x).strip()
    )


# ============================================================
# ETL CANONICALIZATION
# ============================================================

def canonicalize_etl(x):
    """
    Canonicalize common ETL naming variants.
    Conservative by design: only well-supported aliases are merged.
    """

    if pd.isna(x):
        return np.nan

    s = str(x).strip()
    low = s.lower().replace(" ", "")

    if low == "sno2":
        return "SnO2"

    if low in [
        "tio2",
        "c-tio2",
        "compacttio2"
    ]:
        return "TiO2"

    compact_present = (
        "compacttio2" in low
        or "c-tio2" in low
        or "ctio2" in low
    )

    meso_present = (
        "mesoporous" in low
        or "m-tio2" in low
        or "mp-tio2" in low
        or "mtio2" in low
        or "mptio2" in low
    )

    if (
        "tio2" in low
        and compact_present
        and meso_present
    ):
        return "compact/mesoporous TiO2"

    if low == "c60/bcp":
        return "C60/BCP"

    if low == "pcbm/bcp":
        return "PCBM/BCP"

    if low == "pc61bm/bcp":
        return "PC61BM/BCP"

    if low == "pcbm":
        return "PCBM"

    if low == "pc61bm":
        return "PC61BM"

    return s


# ============================================================
# HTL CANONICALIZATION
# ============================================================

def canonicalize_htl(x):
    """
    Canonicalize common HTL naming variants.
    """

    if pd.isna(x):
        return np.nan

    s = str(x).strip()

    low = (
        s.lower()
        .replace(" ", "")
        .replace("-", "")
    )

    if low in [
        "spiroometad",
        "spiromeotad"
    ]:
        return "Spiro-OMeTAD"

    if low == "pedot:pss":
        return "PEDOT:PSS"

    if low == "ptaa":
        return "PTAA"

    if low in [
        "nio",
        "niox"
    ]:
        return "NiOx"

    if low == "meo2pacz":
        return "MeO-2PACz"

    if low == "2pacz":
        return "2PACz"

    if low == "p3ht":
        return "P3HT"

    if low == "cuscn":
        return "CuSCN"

    return s


# ============================================================
# RARE CATEGORY GROUPING
# ============================================================

def group_rare_categories(
    series,
    min_count=20,
    missing_label="MISSING",
    other_label="OTHER"
):
    """
    Group low-frequency categorical values into OTHER.

    Parameters
    ----------
    series : pandas.Series
        Categorical feature.

    min_count : int
        Minimum number of occurrences required to keep a category.

    missing_label : str
        Label used for missing values.

    other_label : str
        Label used for rare categories.
    """

    counts = series.value_counts(
        dropna=False
    )

    def convert(x):

        if pd.isna(x):
            return missing_label

        if counts.get(x, 0) < min_count:
            return other_label

        return x

    return series.apply(
        convert
    )


# ============================================================
# TRANSPORT-LAYER STANDARDIZATION
# ============================================================

def standardize_transport_layers(
    df,
    etl_col="ETL",
    htl_col="HTL"
):
    """
    Add cleaned and canonical ETL/HTL columns.
    """

    df = df.copy()

    df["ETL_clean"] = (
        df[etl_col]
        .apply(clean_material_text)
    )

    df["HTL_clean"] = (
        df[htl_col]
        .apply(clean_material_text)
    )

    df["ETL_canonical"] = (
        df["ETL_clean"]
        .apply(canonicalize_etl)
    )

    df["HTL_canonical"] = (
        df["HTL_clean"]
        .apply(canonicalize_htl)
    )

    return df


# ============================================================
# MODEL CATEGORY PREPARATION
# ============================================================

def add_grouped_transport_features(
    df,
    min_count=20
):
    """
    Add grouped ETL/HTL features for machine learning.
    """

    df = df.copy()

    df["ETL_grouped"] = (
        group_rare_categories(
            df["ETL_canonical"],
            min_count=min_count
        )
    )

    df["HTL_grouped"] = (
        group_rare_categories(
            df["HTL_canonical"],
            min_count=min_count
        )
    )

    return df


# ============================================================
# LEAKAGE-SAFE FEATURE DEFINITIONS
# ============================================================

CATEGORICAL_FEATURES = [
    "architecture_final",
    "lead_status_final",
    "ETL_grouped",
    "HTL_grouped",
    "cation_family",
    "halide_family",
]


NUMERIC_FEATURES = [
    "has_Pb",
    "has_Sn",
    "has_Cs",
    "has_FA",
    "has_MA",
    "has_I",
    "has_Br",
    "has_Cl",
    "mixed_FA_MA_Cs",
    "mixed_halide",
    "Pb_Sn_mixed_detected",
    "year",
]


LEAKAGE_COLUMNS = [
    "Voc",
    "Jsc",
    "Jsc_model",
    "FF",
    "PCE_calc",
    "relative_error",
]


def get_model_features():
    """
    Return the leakage-safe feature list used for PCE prediction.
    """

    return (
        CATEGORICAL_FEATURES
        + NUMERIC_FEATURES
    )


def build_ml_dataset(
    df,
    target_col="PCE",
    doi_col="doi_canonical"
):
    """
    Build the modeling table used for PCE prediction.

    Direct performance-derived variables such as Voc, Jsc and FF
    are deliberately excluded to avoid target leakage.
    """

    features = get_model_features()

    for forbidden in LEAKAGE_COLUMNS:
        if forbidden in features:
            raise ValueError(
                f"Leakage feature detected: {forbidden}"
            )

    columns = [
        doi_col,
        target_col
    ] + features

    return df[
        columns
    ].copy()
