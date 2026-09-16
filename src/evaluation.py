"""
Evaluation and robustness utilities for perovskite PCE models.

Includes:
- permutation importance
- error analysis by publication year
- temporal ablation studies
- categorical support analysis
- binary chemistry support analysis
"""

import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# REGRESSION METRICS
# ============================================================

def regression_metrics(y_true, y_pred):
    """Return MAE, RMSE, R² and mean prediction error."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return {
        "MAE": mean_absolute_error(
            y_true,
            y_pred
        ),

        "RMSE": np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        ),

        "R2": r2_score(
            y_true,
            y_pred
        ),

        "mean_error": np.mean(
            y_pred - y_true
        ),
    }


# ============================================================
# PERMUTATION IMPORTANCE
# ============================================================

def temporal_permutation_importance(
    fitted_pipeline,
    X_test,
    y_test,
    feature_columns,
    n_repeats=20,
    random_state=42,
):
    """
    Calculate permutation importance on a held-out temporal test set.

    Importance is measured as the decrease in R² after shuffling
    one raw feature at a time.
    """

    result = permutation_importance(
        fitted_pipeline,
        X_test,
        y_test,
        scoring="r2",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance_mean":
            result.importances_mean,
        "importance_std":
            result.importances_std,
    })

    return (
        importance_df
        .sort_values(
            "importance_mean",
            ascending=False
        )
        .reset_index(drop=True)
    )


# ============================================================
# ERROR ANALYSIS BY PUBLICATION YEAR
# ============================================================

def error_by_year(
    test_df,
    predictions,
    target_col="PCE",
    year_col="year",
    doi_col="doi_canonical",
):
    """
    Summarize prediction errors by publication year.
    """

    pred_df = test_df[
        [
            doi_col,
            year_col,
            target_col
        ]
    ].copy()

    pred_df["PCE_pred"] = predictions

    pred_df["error"] = (
        pred_df["PCE_pred"]
        - pred_df[target_col]
    )

    pred_df["abs_error"] = (
        pred_df["error"].abs()
    )


    summary = (
        pred_df
        .groupby(year_col)
        .agg(
            n=(target_col, "size"),

            mean_true_PCE=(
                target_col,
                "mean"
            ),

            mean_pred_PCE=(
                "PCE_pred",
                "mean"
            ),

            MAE=(
                "abs_error",
                "mean"
            ),

            mean_error=(
                "error",
                "mean"
            ),
        )
        .reset_index()
    )


    rmse_by_year = (
        pred_df
        .groupby(year_col)["error"]
        .apply(
            lambda x:
            np.sqrt(
                np.mean(
                    x ** 2
                )
            )
        )
        .reset_index(
            name="RMSE"
        )
    )


    summary = summary.merge(
        rmse_by_year,
        on=year_col,
        how="left"
    )

    return pred_df, summary


# ============================================================
# PREPROCESSOR FOR ABLATION STUDIES
# ============================================================

def _build_preprocessor(
    categorical_features,
    numeric_features,
):
    """Internal preprocessing helper."""

    transformers = []


    if categorical_features:

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )


    if numeric_features:

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            )
        )


    return ColumnTransformer(
        transformers=transformers
    )


# ============================================================
# TEMPORAL ABLATION
# ============================================================

def temporal_ablation_study(
    train_df,
    test_df,
    feature_sets,
    categorical_candidates,
    numeric_candidates,
    target_col="PCE",
    alpha=1.0,
):
    """
    Evaluate multiple feature sets using Ridge regression.

    Parameters
    ----------
    feature_sets : dict
        Example:
        {
            "FULL": [...],
            "NO_YEAR": [...],
            "CHEMISTRY_ONLY": [...]
        }
    """

    results = []


    for set_name, features in feature_sets.items():

        categorical_features = [
            f for f in features
            if f in categorical_candidates
        ]

        numeric_features = [
            f for f in features
            if f in numeric_candidates
        ]


        preprocessor = _build_preprocessor(
            categorical_features,
            numeric_features,
        )


        model = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "ridge",
                    Ridge(
                        alpha=alpha
                    ),
                ),
            ]
        )


        model.fit(
            train_df[features],
            train_df[target_col],
        )


        predictions = model.predict(
            test_df[features]
        )


        metrics = regression_metrics(
            test_df[target_col],
            predictions,
        )


        results.append({
            "feature_set":
                set_name,

            "n_features_raw":
                len(features),

            **metrics,
        })


    return (
        pd.DataFrame(results)
        .sort_values(
            "R2",
            ascending=False
        )
        .reset_index(drop=True)
    )


# ============================================================
# CATEGORICAL SUPPORT ANALYSIS
# ============================================================

def categorical_support(
    train_df,
    test_df,
    categorical_columns,
    target_col="PCE",
    low_support_threshold=50,
):
    """
    Calculate category counts and mean PCE in train/test periods.
    """

    tables = []


    for col in categorical_columns:

        train_stats = (
            train_df
            .groupby(col)
            .agg(
                n_train=(
                    target_col,
                    "size"
                ),

                mean_PCE_train=(
                    target_col,
                    "mean"
                ),
            )
            .reset_index()
        )


        test_stats = (
            test_df
            .groupby(col)
            .agg(
                n_test=(
                    target_col,
                    "size"
                ),

                mean_PCE_test=(
                    target_col,
                    "mean"
                ),
            )
            .reset_index()
        )


        merged = train_stats.merge(
            test_stats,
            on=col,
            how="outer"
        )


        merged = merged.rename(
            columns={
                col: "category"
            }
        )


        merged["feature_group"] = col


        merged["n_total"] = (
            merged["n_train"].fillna(0)
            + merged["n_test"].fillna(0)
        )


        merged["low_support_flag"] = (
            merged["n_total"]
            < low_support_threshold
        )


        tables.append(
            merged
        )


    return pd.concat(
        tables,
        ignore_index=True
    )


# ============================================================
# BINARY FEATURE SUPPORT
# ============================================================

def binary_feature_support(
    train_df,
    test_df,
    binary_features,
    target_col="PCE",
):
    """
    Compare mean PCE for binary chemistry features.
    """

    rows = []


    for feature in binary_features:

        for value in [0, 1]:

            train_subset = train_df[
                train_df[feature] == value
            ]

            test_subset = test_df[
                test_df[feature] == value
            ]


            rows.append({
                "feature":
                    feature,

                "value":
                    value,

                "n_train":
                    len(train_subset),

                "n_test":
                    len(test_subset),

                "mean_PCE_train":
                    train_subset[
                        target_col
                    ].mean(),

                "mean_PCE_test":
                    test_subset[
                        target_col
                    ].mean(),
            })


    return pd.DataFrame(
        rows
    )


# ============================================================
# RIDGE COEFFICIENT EXTRACTION
# ============================================================

def extract_ridge_coefficients(
    fitted_pipeline,
):
    """
    Extract transformed feature names and Ridge coefficients
    from a fitted sklearn Pipeline.
    """

    preprocessor = (
        fitted_pipeline
        .named_steps[
            "preprocessor"
        ]
    )

    model = (
        fitted_pipeline
        .named_steps[
            "model"
        ]
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    coefficients = (
        model.coef_
    )


    coef_df = pd.DataFrame({
        "feature_encoded":
            feature_names,

        "coefficient":
            coefficients,
    })


    coef_df["feature"] = (
        coef_df[
            "feature_encoded"
        ]
        .str.replace(
            "categorical__",
            "",
            regex=False
        )
        .str.replace(
            "numeric__",
            "",
            regex=False
        )
    )


    coef_df[
        "abs_coefficient"
    ] = (
        coef_df[
            "coefficient"
        ].abs()
    )


    return (
        coef_df
        .sort_values(
            "abs_coefficient",
            ascending=False
        )
        .reset_index(drop=True)
    )
