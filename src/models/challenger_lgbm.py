# src/models/challenger_lgbm.py
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from lightgbm import LGBMClassifier


def build_challenger_pipeline(numeric_features, categorical_features) -> Pipeline:
    """
    Challenger model = LightGBM (gradient boosted trees).
    Why:
    - Strong tabular performance (captures non-linearities/interactions)
    - Comparable deployment pattern (predict_proba)
    - Interpretable via SHAP (global drivers)
    """
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        # No scaling needed for tree-based models
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Conservative hyperparameters (defensible, avoids obvious overfit)
    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=200,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocess", pre), ("model", model)])


def fit_challenger(pipe: Pipeline, train_df: pd.DataFrame, y_col: str, numeric_features, categorical_features) -> Pipeline:
    X = train_df[numeric_features + categorical_features]
    y = train_df[y_col].astype(int)
    return pipe.fit(X, y)


def predict_pd(pipe: Pipeline, df: pd.DataFrame, numeric_features, categorical_features) -> np.ndarray:
    X = df[numeric_features + categorical_features]
    return pipe.predict_proba(X)[:, 1]