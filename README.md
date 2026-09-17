# Evidence-Grounded Scientific Data Curation with LLMs  
## A Perovskite Solar Cell Case Study

An end-to-end scientific data curation and machine-learning pipeline for experimental perovskite solar-cell datasets.

The project combines deterministic quality-control rules, semantic normalization, DOI-linked scholarly evidence, structured LLM-assisted review, chemistry-aware feature engineering, and temporally held-out machine-learning validation.

The central goal is not simply to train a predictive model, but to build a traceable workflow that converts heterogeneous literature-derived records into a scientifically curated and model-ready dataset.

---

## Project Overview

Scientific datasets extracted from the literature often contain:

- inconsistent material naming,
- ambiguous device architectures,
- missing or malformed absorber descriptions,
- mixed conventions for photovoltaic parameters,
- duplicate or malformed DOI identifiers,
- inconsistent ETL/HTL terminology,
- uncertain scope,
- and records that cannot safely be resolved using simple rules.

This project addresses these issues through a staged curation pipeline:

```text
Raw PerovskiteNet data
        │
        ▼
Photovoltaic QC
        │
        ▼
Rule-based semantic curation
        │
        ▼
Semantic review queue
        │
        ▼
DOI normalization and evidence retrieval
        │
        ▼
Evidence-grounded structured LLM review
        │
        ▼
Trusted semantic dataset
        │
        ▼
Chemistry-aware feature engineering
        │
        ▼
Leakage-safe PCE prediction
        │
        ▼
Random + temporal validation
