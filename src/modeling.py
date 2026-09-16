"""
Modeling utilities for perovskite solar-cell PCE prediction.

This module provides:
- leakage-safe preprocessing
- random train/test splitting
- temporal validation
- baseline, Ridge and Random Forest models
- regression metrics

The primary target is power conversion efficiency (PCE).
"""

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# DEFAULT FEATURE DEFINITIONS
# ============================================================

DEFAULT_CATEGORICAL_FEATURES = [
    "architecture_final",
    "lead_status_final",
    "ETL_grouped",
    "HTL_grouped",
    "cation_family",
    "halide_family",
]


DEFAULT_NUMERIC_FEATURES = [
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


# ============================================================
# PREPROCESSING
# ============================================================

def build_preprocessor(
    categorical_features=None,
    numeric_features=None,
):
    """
    Build the preprocessing pipeline.

    Categorical features:
        missing-value imputation + one-hot encoding

    Numeric features:
        median imputation + standard scaling
    """

    if categorical_features is None:
        categorical_features = DEFAULT_CATEGORICAL_FEATURES

    if numeric_features is None:
        numeric_features = DEFAULT_NUMERIC_FEATURES


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


    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
        ]
    )


# ============================================================
# DEFAULT MODELS
# ============================================================

def get_default_models():
    """
    Return the baseline models used in the project.
    """

    return {
        "Dummy_Mean": DummyRegressor(
            strategy="mean"
        ),

        "Ridge": Ridge(
            alpha=1.0
        ),

        "Random_Forest": RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }


# ============================================================
# PIPELINE
# ============================================================

def build_model_pipeline(
    estimator,
    categorical_features=None,
    numeric_features=None,
):
    """
    Combine preprocessing and regression model.
    """

    preprocessor = build_preprocessor(
        categorical_features=categorical_features,
        numeric_features=numeric_features,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                estimator,
            ),
        ]
    )


# ============================================================
# METRICS
# ============================================================

def regression_metrics(
    y_true,
    y_pred,
):
    """
    Calculate regression performance metrics.
    """

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    mean_error = np.mean(
        np.asarray(y_pred)
        - np.asarray(y_true)
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "mean_error": mean_error,
    }


# ============================================================
# MODEL EVALUATION
# ============================================================

def fit_and_evaluate(
    estimator,
    X_train,
    y_train,
    X_test,
    y_test,
    categorical_features=None,
    numeric_features=None,
):
    """
    Train a regression pipeline and evaluate it on a held-out set.
    """

    pipeline = build_model_pipeline(
        estimator=clone(estimator),
        categorical_features=categorical_features,
        numeric_features=numeric_features,
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    predictions = pipeline.predict(
        X_test
    )

    metrics = regression_metrics(
        y_test,
        predictions,
    )

    return (
        pipeline,
        predictions,
        metrics,
    )


# ============================================================
# RANDOM SPLIT
# ============================================================

def random_split(
    df,
    feature_columns,
    target_col="PCE",
    test_size=0.20,
    random_state=42,
):
    """
    Create a conventional random held-out split.
    """

    X = df[
        feature_columns
    ].copy()

    y = df[
        target_col
    ].copy()

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )


# ============================================================
# TEMPORAL SPLIT
# ============================================================

def temporal_split(
    df,
    train_end_year=2022,
    test_start_year=2023,
    year_col="year",
):
    """
    Split data chronologically.

    Training:
        year <= train_end_year

    Testing:
        year >= test_start_year

    Rows without publication year are excluded.
    """

    temporal_df = df[
        df[year_col].notna()
    ].copy()

    train_df = temporal_df[
        temporal_df[year_col]
        <= train_end_year
    ].copy()

    test_df = temporal_df[
        temporal_df[year_col]
        >= test_start_year
    ].copy()

    return (
        train_df,
        test_df,
    )


# ============================================================
# BENCHMARK MULTIPLE MODELS
# ============================================================

def benchmark_models(
    train_df,
    test_df,
    feature_columns,
    categorical_features=None,
    numeric_features=None,
    target_col="PCE",
    split_name="TEMPORAL",
    models=None,
):
    """
    Benchmark multiple regression models on a fixed split.
    """

    if models is None:
        models = get_default_models()

    X_train = train_df[
        feature_columns
    ]

    y_train = train_df[
        target_col
    ]

    X_test = test_df[
        feature_columns
    ]

    y_test = test_df[
        target_col
    ]


    results = []
    fitted_models = {}
    prediction_tables = {}


    for model_name, estimator in models.items():

        (
            pipeline,
            predictions,
            metrics,
        ) = fit_and_evaluate(
            estimator=estimator,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            categorical_features=categorical_features,
            numeric_features=numeric_features,
        )


        result = {
            "split": split_name,
            "model": model_name,
            "n_train": len(train_df),
            "n_test": len(test_df),
            **metrics,
        }

        results.append(
            result
        )

        fitted_models[
            model_name
        ] = pipeline


        prediction_tables[
            model_name
        ] = pd.DataFrame(
            {
                "y_true":
                    y_test.values,
                "y_pred":
                    predictions,
            }
        )


    results_df = pd.DataFrame(
        results
    )

    return (
        results_df,
        fitted_models,
        prediction_tables,
    )


# ============================================================
# TEMPORAL RIDGE
# ============================================================

def fit_temporal_ridge(
    df,
    feature_columns,
    categorical_features=None,
    numeric_features=None,
    target_col="PCE",
    train_end_year=2022,
    test_start_year=2023,
):
    """
    Train the primary Ridge model using temporal validation.
    """

    train_df, test_df = temporal_split(
        df,
        train_end_year=train_end_year,
        test_start_year=test_start_year,
    )

    estimator = Ridge(
        alpha=1.0
    )

    pipeline, predictions, metrics = (
        fit_and_evaluate(
            estimator=estimator,
            X_train=train_df[feature_columns],
            y_train=train_df[target_col],
            X_test=test_df[feature_columns],
            y_test=test_df[target_col],
            categorical_features=categorical_features,
            numeric_features=numeric_features,
        )
    )

    return {
        "pipeline": pipeline,
        "train_df": train_df,
        "test_df": test_df,
        "predictions": predictions,
        "metrics": metrics,
    }
