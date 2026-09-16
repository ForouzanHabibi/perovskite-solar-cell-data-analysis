"""
Rule-based semantic curation for perovskite solar-cell datasets.

This module implements the deterministic part of the curation
pipeline before external-evidence retrieval and LLM-assisted review.

Main responsibilities
---------------------
- photovoltaic performance QC
- architecture normalization
- conservative scope refinement
- lead-chemistry classification
- semantic-review flagging

Ambiguous cases are deliberately retained for later evidence-grounded
review rather than being aggressively auto-classified.
"""

import re
import numpy as np
import pandas as pd


# ============================================================
# TRUSTED PERFORMANCE LIMITS
# ============================================================

PCE_MIN = 0.1
PCE_MAX = 35.0

VOC_MIN = 0.3
VOC_MAX = 1.6

JSC_MIN_ABS = 1.0
JSC_MAX_ABS = 35.0

FF_MIN = 0.2
FF_MAX = 0.95

MAX_RELATIVE_ERROR = 0.20


# ============================================================
# PERFORMANCE QC
# ============================================================

def add_performance_qc(
    df,
    pce_col="PCE",
    voc_col="Voc",
    jsc_col="Jsc",
    ff_col="FF",
):
    """
    Add photovoltaic consistency metrics.

    PCE_calc is numerically:

        Voc [V] × |Jsc| [mA/cm²] × FF

    which gives PCE in percent under standard 100 mW/cm² illumination.
    """

    df = df.copy()

    df["Jsc_abs"] = (
        pd.to_numeric(
            df[jsc_col],
            errors="coerce"
        )
        .abs()
    )

    df["PCE_calc"] = (
        pd.to_numeric(
            df[voc_col],
            errors="coerce"
        )
        *
        df["Jsc_abs"]
        *
        pd.to_numeric(
            df[ff_col],
            errors="coerce"
        )
    )

    pce = pd.to_numeric(
        df[pce_col],
        errors="coerce"
    )

    denominator = pce.abs().replace(
        0,
        np.nan
    )

    df["relative_error"] = (
        (
            pce
            - df["PCE_calc"]
        )
        .abs()
        / denominator
    )

    return df


def trusted_performance_mask(
    df,
    pce_col="PCE",
    voc_col="Voc",
    jsc_abs_col="Jsc_abs",
    ff_col="FF",
    relative_error_col="relative_error",
):
    """
    Return the trusted photovoltaic-performance QC mask.
    """

    pce = pd.to_numeric(
        df[pce_col],
        errors="coerce"
    )

    voc = pd.to_numeric(
        df[voc_col],
        errors="coerce"
    )

    jsc = pd.to_numeric(
        df[jsc_abs_col],
        errors="coerce"
    )

    ff = pd.to_numeric(
        df[ff_col],
        errors="coerce"
    )

    err = pd.to_numeric(
        df[relative_error_col],
        errors="coerce"
    )

    return (
        pce.between(
            PCE_MIN,
            PCE_MAX,
            inclusive="both"
        )
        &
        (
            (voc >= VOC_MIN)
            &
            (voc < VOC_MAX)
        )
        &
        jsc.between(
            JSC_MIN_ABS,
            JSC_MAX_ABS,
            inclusive="both"
        )
        &
        ff.between(
            FF_MIN,
            FF_MAX,
            inclusive="both"
        )
        &
        (
            err <= MAX_RELATIVE_ERROR
        )
    )


def apply_trusted_performance_filter(df):
    """
    Apply the deterministic photovoltaic QC layer.
    """

    qc_df = add_performance_qc(
        df
    )

    mask = trusted_performance_mask(
        qc_df
    )

    return (
        qc_df
        .loc[mask]
        .copy()
        .reset_index(drop=True)
    )


# ============================================================
# ARCHITECTURE NORMALIZATION
# ============================================================

def normalize_architecture(value):
    """
    Normalize architecture labels to:

        NIP
        PIN
        UNKNOWN
    """

    if pd.isna(value):
        return "UNKNOWN"

    text = (
        str(value)
        .strip()
        .lower()
        .replace(" ", "")
    )

    nip_aliases = {
        "nip",
        "n-i-p",
        "normal",
        "conventional",
    }

    pin_aliases = {
        "pin",
        "p-i-n",
        "inverted",
        "inverse",
    }

    if text in nip_aliases:
        return "NIP"

    if text in pin_aliases:
        return "PIN"

    if "n-i-p" in text:
        return "NIP"

    if "p-i-n" in text:
        return "PIN"

    return "UNKNOWN"


def add_normalized_architecture(
    df,
    architecture_col="architecture",
):
    """Add architecture_norm."""

    df = df.copy()

    df["architecture_norm"] = (
        df[architecture_col]
        .apply(
            normalize_architecture
        )
    )

    return df


# ============================================================
# TEXT HELPERS
# ============================================================

def _clean_text(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def _contains_any(
    text,
    patterns,
):
    text = text.lower()

    return any(
        pattern.lower() in text
        for pattern in patterns
    )


# ============================================================
# QUANTUM-DOT DETECTION
# ============================================================

QD_PATTERNS = [
    "quantum dot",
    "quantum-dot",
    "quantum dots",
    "pqd",
    "pqds",
    "qd solar cell",
    "qd film",
]


def detect_quantum_dot_case(
    absorber="",
    cell_stack="",
    title="",
):
    """
    Conservative quantum-dot detection.
    """

    text = " ".join(
        [
            _clean_text(absorber),
            _clean_text(cell_stack),
            _clean_text(title),
        ]
    )

    return _contains_any(
        text,
        QD_PATTERNS
    )


# ============================================================
# SCOPE REFINEMENT
# ============================================================

def refine_scope_rule(
    absorber,
    cell_stack="",
    title="",
    explicit_non_psc=False,
):
    """
    Conservative PSC scope refinement.

    Ambiguous records are sent to review rather than excluded.

    Returns
    -------
    KEEP_PSC_CANDIDATE
    REVIEW_MISSING_ABSORBER
    REVIEW_QD
    EXCLUDE_NON_PSC
    """

    absorber_text = _clean_text(
        absorber
    )

    if explicit_non_psc:
        return "EXCLUDE_NON_PSC"

    if detect_quantum_dot_case(
        absorber=absorber,
        cell_stack=cell_stack,
        title=title,
    ):
        return "REVIEW_QD"

    if absorber_text == "":
        return "REVIEW_MISSING_ABSORBER"

    return "KEEP_PSC_CANDIDATE"


# ============================================================
# LEAD-CHEMISTRY CLASSIFICATION
# ============================================================

LEAD_FREE_ELEMENT_PATTERNS = [
    r"\bSn\b",
    r"\bGe\b",
    r"\bBi\b",
    r"\bSb\b",
    r"\bAg\b",
    r"\bCu\b",
    r"\bFe\b",
]


def _has_symbol(
    text,
    symbol,
):
    """
    Detect a chemical element symbol conservatively.
    """

    text = _clean_text(
        text
    )

    return bool(
        re.search(
            rf"(?<![A-Za-z]){re.escape(symbol)}(?![a-z])",
            text
        )
    )


def assign_lead_status(
    absorber,
    scope=None,
):
    """
    Assign deterministic initial lead-chemistry status.

    Possible labels
    ---------------
    LEAD_BASED
    MIXED_PB_SN
    LEAD_FREE_CANDIDATE
    NO_PB_AMBIGUOUS
    UNCLEAR
    NOT_APPLICABLE
    """

    if scope == "EXCLUDE_NON_PSC":
        return "NOT_APPLICABLE"

    text = _clean_text(
        absorber
    )

    if text == "":
        return "UNCLEAR"

    low = text.lower()

    has_pb = (
        _has_symbol(
            text,
            "Pb"
        )
        or "lead" in low
    )

    has_sn = (
        _has_symbol(
            text,
            "Sn"
        )
        or "tin" in low
    )

    if has_pb and has_sn:
        return "MIXED_PB_SN"

    if has_pb:
        return "LEAD_BASED"

    if (
        "lead-free" in low
        or "lead free" in low
    ):
        return "LEAD_FREE_CANDIDATE"

    for pattern in LEAD_FREE_ELEMENT_PATTERNS:

        if re.search(
            pattern,
            text
        ):
            return "LEAD_FREE_CANDIDATE"

    # Formula/text exists, but absence of Pb alone is not
    # considered sufficient proof of lead-free chemistry.
    return "NO_PB_AMBIGUOUS"


# ============================================================
# CORE SEMANTIC REVIEW
# ============================================================

def determine_core_review_reason(
    scope_refined,
    lead_status,
    architecture_norm,
):
    """
    Determine why a record still requires semantic review.
    """

    reasons = []

    if scope_refined == "REVIEW_QD":
        reasons.append(
            "QUANTUM_DOT_SCOPE"
        )

    if (
        scope_refined
        == "REVIEW_MISSING_ABSORBER"
    ):
        reasons.append(
            "MISSING_ABSORBER"
        )

    if lead_status in [
        "UNCLEAR",
        "NO_PB_AMBIGUOUS",
    ]:
        reasons.append(
            "UNCLEAR_CHEMISTRY"
        )

    if architecture_norm == "UNKNOWN":
        reasons.append(
            "UNKNOWN_ARCHITECTURE"
        )

    if not reasons:
        return ""

    return " + ".join(
        reasons
    )


def needs_core_review(
    scope_refined,
    lead_status,
    architecture_norm,
):
    """
    Boolean semantic-review flag.
    """

    return (
        determine_core_review_reason(
            scope_refined,
            lead_status,
            architecture_norm,
        )
        != ""
    )


# ============================================================
# FULL RULE-BASED SEMANTIC PASS
# ============================================================

def add_semantic_rule_columns(
    df,
    absorber_col="absorber",
    architecture_col="architecture",
    stack_col="cell_stack",
    title_col="paper_title",
):
    """
    Apply the deterministic semantic curation layer.

    This function intentionally stops before external evidence
    retrieval and LLM-assisted resolution.
    """

    df = df.copy()

    # Architecture
    df["architecture_norm"] = (
        df[architecture_col]
        .apply(
            normalize_architecture
        )
    )


    # Scope
    df["scope_refined"] = (
        df.apply(
            lambda row:
            refine_scope_rule(
                absorber=row.get(
                    absorber_col,
                    ""
                ),
                cell_stack=row.get(
                    stack_col,
                    ""
                ),
                title=row.get(
                    title_col,
                    ""
                ),
            ),
            axis=1,
        )
    )


    # Lead chemistry
    df["lead_status_rule"] = (
        df.apply(
            lambda row:
            assign_lead_status(
                absorber=row.get(
                    absorber_col,
                    ""
                ),
                scope=row[
                    "scope_refined"
                ],
            ),
            axis=1,
        )
    )


    # Review reason
    df["core_review_reason"] = (
        df.apply(
            lambda row:
            determine_core_review_reason(
                scope_refined=row[
                    "scope_refined"
                ],
                lead_status=row[
                    "lead_status_rule"
                ],
                architecture_norm=row[
                    "architecture_norm"
                ],
            ),
            axis=1,
        )
    )


    df["needs_core_semantic_review"] = (
        df[
            "core_review_reason"
        ]
        .ne("")
    )


    return df
