"""
Evidence-grounded semantic curation utilities.

This module implements the deterministic policy layer around
external scholarly evidence and LLM-assisted semantic review.

Core principles
---------------
1. DOI values are normalized before evidence matching.
2. Repaired DOI values are tracked explicitly.
3. Evidence is categorized by strength.
4. TITLE_STACK_ONLY evidence is never auto-accepted or auto-excluded.
5. Only ABSTRACT_GROUNDED evidence may support automatic decisions.
6. LLM outputs are treated as structured evidence-assisted proposals,
   not as unrestricted replacements for curated data.
"""

import re
import pandas as pd


# ============================================================
# CONFIDENCE THRESHOLDS
# ============================================================

AUTO_THRESHOLD = 0.90
SCOPE_THRESHOLD = 0.90


# ============================================================
# DOI NORMALIZATION
# ============================================================

def normalize_doi(value):
    """
    Normalize DOI strings for deterministic matching.

    Removes common DOI URL prefixes and surrounding whitespace.
    """

    if pd.isna(value):
        return None

    doi = str(value).strip().lower()

    if doi in [
        "",
        "nan",
        "none",
    ]:
        return None

    prefixes = [
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi:",
    ]

    for prefix in prefixes:
        if doi.startswith(prefix):
            doi = doi[
                len(prefix):
            ]

    return doi.strip()


# ============================================================
# DOI FORMAT CHECK
# ============================================================

DOI_PATTERN = re.compile(
    r"^10\.\d{4,9}/\S+$",
    flags=re.IGNORECASE,
)


def looks_like_doi(value):
    """
    Lightweight DOI syntax check.

    This validates structure only; it does not verify that the DOI
    exists in Crossref, OpenAlex, or another registry.
    """

    doi = normalize_doi(
        value
    )

    if doi is None:
        return False

    return bool(
        DOI_PATTERN.match(
            doi
        )
    )


# ============================================================
# DOI MAPPING
# ============================================================

def build_doi_mapping(
    doi_qc_df,
    original_col="doi_original",
    resolved_col="doi_resolved",
):
    """
    Build original -> resolved DOI mapping from a DOI QC table.
    """

    mapping = {}

    for _, row in doi_qc_df.iterrows():

        original = normalize_doi(
            row.get(
                original_col
            )
        )

        resolved = normalize_doi(
            row.get(
                resolved_col
            )
        )

        if (
            original is not None
            and resolved is not None
        ):
            mapping[
                original
            ] = resolved

    return mapping


def add_canonical_doi(
    df,
    doi_col="doi",
    mapping=None,
):
    """
    Add normalized and canonical DOI columns.

    If a DOI-repair mapping is supplied, repaired DOI values are used
    as canonical identifiers.
    """

    df = df.copy()

    df["doi_norm"] = (
        df[doi_col]
        .apply(
            normalize_doi
        )
    )

    if mapping is None:
        df["doi_canonical"] = (
            df["doi_norm"]
        )

        return df

    df["doi_canonical"] = (
        df["doi_norm"]
        .map(
            mapping
        )
        .fillna(
            df["doi_norm"]
        )
    )

    return df


# ============================================================
# EVIDENCE LEVEL
# ============================================================

EVIDENCE_ABSTRACT = (
    "ABSTRACT_GROUNDED"
)

EVIDENCE_TITLE_STACK = (
    "TITLE_STACK_ONLY"
)


def classify_evidence_level(
    abstract_text,
):
    """
    Classify available scholarly evidence.

    ABSTRACT_GROUNDED:
        usable abstract available.

    TITLE_STACK_ONLY:
        no usable abstract; interpretation must remain manual.
    """

    if pd.isna(
        abstract_text
    ):
        return EVIDENCE_TITLE_STACK

    text = str(
        abstract_text
    ).strip()

    if len(text) == 0:
        return EVIDENCE_TITLE_STACK

    return EVIDENCE_ABSTRACT


def auto_accept_eligible(
    evidence_level,
):
    """
    Only abstract-grounded records may be auto-resolved.
    """

    return (
        evidence_level
        == EVIDENCE_ABSTRACT
    )


# ============================================================
# TASK ROUTING
# ============================================================

TASK_ABSORBER = (
    "RESOLVE_ABSORBER"
)

TASK_CHEMISTRY = (
    "RESOLVE_CHEMISTRY"
)

TASK_ARCHITECTURE = (
    "RESOLVE_ARCHITECTURE"
)

TASK_QD_SCOPE = (
    "RESOLVE_QD_SCOPE"
)


def build_task_list(
    scope_value=None,
    lead_status=None,
    architecture=None,
):
    """
    Determine which semantic fields require evidence-based review.
    """

    tasks = []

    if scope_value == "REVIEW_MISSING_ABSORBER":
        tasks.append(
            TASK_ABSORBER
        )

        tasks.append(
            TASK_CHEMISTRY
        )

    if lead_status in [
        "UNCLEAR",
        "NO_PB_AMBIGUOUS",
    ]:

        if (
            TASK_CHEMISTRY
            not in tasks
        ):
            tasks.append(
                TASK_CHEMISTRY
            )

    if architecture == "UNKNOWN":
        tasks.append(
            TASK_ARCHITECTURE
        )

    if scope_value == "REVIEW_QD":
        tasks.append(
            TASK_QD_SCOPE
        )

    return tasks


# ============================================================
# STRUCTURED LLM OUTPUT HELPERS
# ============================================================

VALID_SCOPE_LABELS = {
    "PSC_COMPATIBLE",
    "NON_PSC",
    "UNCLEAR",
}

VALID_CHEMISTRY_LABELS = {
    "LEAD_BASED",
    "LEAD_FREE",
    "MIXED_PB_SN",
    "UNCLEAR",
    "NOT_REQUESTED",
}

VALID_ARCHITECTURE_LABELS = {
    "NIP",
    "PIN",
    "UNKNOWN",
    "NOT_REQUESTED",
}

VALID_QD_LABELS = {
    "KEEP_PSC_QD",
    "EXCLUDE_NON_PSC_QD",
    "UNCLEAR",
    "NOT_REQUESTED",
}


def _safe_float(
    value,
    default=0.0,
):
    try:
        return float(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return default


# ============================================================
# DETERMINISTIC ABSTENTION
# ============================================================

def derive_abstention(
    tasks,
    absorber=None,
    chemistry=None,
    architecture=None,
    qd_scope=None,
):
    """
    Derive abstention deterministically from unresolved requested tasks.
    """

    abstention = {}


    if TASK_ABSORBER in tasks:

        abstention[
            TASK_ABSORBER
        ] = (
            absorber is None
            or str(
                absorber
            ).strip() == ""
        )


    if TASK_CHEMISTRY in tasks:

        abstention[
            TASK_CHEMISTRY
        ] = (
            chemistry
            in [
                None,
                "UNCLEAR",
                "NOT_REQUESTED",
            ]
        )


    if TASK_ARCHITECTURE in tasks:

        abstention[
            TASK_ARCHITECTURE
        ] = (
            architecture
            in [
                None,
                "UNKNOWN",
                "NOT_REQUESTED",
            ]
        )


    if TASK_QD_SCOPE in tasks:

        abstention[
            TASK_QD_SCOPE
        ] = (
            qd_scope
            in [
                None,
                "UNCLEAR",
                "NOT_REQUESTED",
            ]
        )


    return abstention


# ============================================================
# FIELD-LEVEL RESOLUTION
# ============================================================

def resolved_requested_tasks(
    tasks,
    absorber=None,
    absorber_confidence=0.0,
    chemistry=None,
    chemistry_confidence=0.0,
    architecture=None,
    architecture_confidence=0.0,
    qd_scope=None,
    qd_scope_confidence=0.0,
    threshold=AUTO_THRESHOLD,
):
    """
    Determine which requested semantic tasks were resolved
    above the configured confidence threshold.
    """

    resolved = []


    if TASK_ABSORBER in tasks:

        if (
            absorber is not None
            and str(
                absorber
            ).strip() != ""
            and _safe_float(
                absorber_confidence
            ) >= threshold
        ):
            resolved.append(
                TASK_ABSORBER
            )


    if TASK_CHEMISTRY in tasks:

        if (
            chemistry
            in [
                "LEAD_BASED",
                "LEAD_FREE",
                "MIXED_PB_SN",
            ]
            and _safe_float(
                chemistry_confidence
            ) >= threshold
        ):
            resolved.append(
                TASK_CHEMISTRY
            )


    if TASK_ARCHITECTURE in tasks:

        if (
            architecture
            in [
                "NIP",
                "PIN",
            ]
            and _safe_float(
                architecture_confidence
            ) >= threshold
        ):
            resolved.append(
                TASK_ARCHITECTURE
            )


    if TASK_QD_SCOPE in tasks:

        if (
            qd_scope
            in [
                "KEEP_PSC_QD",
                "EXCLUDE_NON_PSC_QD",
            ]
            and _safe_float(
                qd_scope_confidence
            ) >= threshold
        ):
            resolved.append(
                TASK_QD_SCOPE
            )


    return resolved


# ============================================================
# FINAL CURATION DECISION
# ============================================================

def determine_curation_decision(
    evidence_level,
    tasks,
    scope_guard,
    scope_confidence,
    absorber=None,
    absorber_confidence=0.0,
    chemistry=None,
    chemistry_confidence=0.0,
    architecture=None,
    architecture_confidence=0.0,
    qd_scope=None,
    qd_scope_confidence=0.0,
    auto_threshold=AUTO_THRESHOLD,
    scope_threshold=SCOPE_THRESHOLD,
):
    """
    Deterministic decision policy used after structured LLM review.

    Possible outputs
    ----------------
    AUTO_EXCLUDE_NON_PSC
    AUTO_ACCEPT_COMPLETE
    PARTIAL_MANUAL_REVIEW
    MANUAL_REVIEW
    """

    # --------------------------------------------------------
    # Title/stack-only evidence is never auto accepted.
    # --------------------------------------------------------

    if (
        evidence_level
        != EVIDENCE_ABSTRACT
    ):
        return (
            "MANUAL_REVIEW"
        )


    scope_confidence = (
        _safe_float(
            scope_confidence
        )
    )


    # --------------------------------------------------------
    # Strong evidence that the paper/device is outside PSC scope
    # --------------------------------------------------------

    if (
        scope_guard
        == "NON_PSC"
        and scope_confidence
        >= scope_threshold
    ):
        return (
            "AUTO_EXCLUDE_NON_PSC"
        )


    # --------------------------------------------------------
    # If scope itself remains uncertain, retain manual review.
    # --------------------------------------------------------

    if (
        scope_guard
        == "UNCLEAR"
        or scope_confidence
        < scope_threshold
    ):
        return (
            "MANUAL_REVIEW"
        )


    resolved = (
        resolved_requested_tasks(
            tasks=tasks,
            absorber=absorber,
            absorber_confidence=(
                absorber_confidence
            ),
            chemistry=chemistry,
            chemistry_confidence=(
                chemistry_confidence
            ),
            architecture=architecture,
            architecture_confidence=(
                architecture_confidence
            ),
            qd_scope=qd_scope,
            qd_scope_confidence=(
                qd_scope_confidence
            ),
            threshold=auto_threshold,
        )
    )


    if len(tasks) == 0:
        return (
            "AUTO_ACCEPT_COMPLETE"
        )


    if len(resolved) == len(tasks):
        return (
            "AUTO_ACCEPT_COMPLETE"
        )


    if len(resolved) > 0:
        return (
            "PARTIAL_MANUAL_REVIEW"
        )


    return (
        "MANUAL_REVIEW"
    )


# ============================================================
# SAFE FIELD UPDATE POLICY
# ============================================================

def should_update_field(
    evidence_level,
    confidence,
    threshold=AUTO_THRESHOLD,
):
    """
    Decide whether an LLM-derived field may overwrite the existing
    semantic value.

    TITLE_STACK_ONLY values are never used for automatic overwrite.
    """

    if (
        evidence_level
        != EVIDENCE_ABSTRACT
    ):
        return False

    return (
        _safe_float(
            confidence
        )
        >= threshold
    )


# ============================================================
# RECORD-LEVEL REVIEW POLICY
# ============================================================

def review_policy_label(
    evidence_level,
):
    """
    Human-readable review policy label.
    """

    if (
        evidence_level
        == EVIDENCE_ABSTRACT
    ):
        return (
            "LLM_RESOLVE_WITH_CONFIDENCE_POLICY"
        )

    return (
        "LLM_ASSIST_ONLY_MANUAL_REVIEW"
    )
