# src/models/champion_logit.py
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import NUM_FEATURES, CAT_FEATURES, ENGINEERED_FEATURES, TARGET_COL


def build_champion_pipeline() -> Pipeline:
    """
    Champion = logistic regression (scorecard-style).
    Why logistic?
    - Widely accepted in credit risk as a baseline / champion due to interpretability & stability.
    - Easier to validate (coefficients, monotonic intuition) vs black-box models.

    Why impute instead of dropping rows?
    - Dropping can bias the sample (missingness is not random in credit data).
    - In regulated environments, you want a deterministic treatment of missing values.
    """

    numeric_features = NUM_FEATURES + ENGINEERED_FEATURES
    categorical_features = CAT_FEATURES

    # Numeric preprocessing:
    # - Median imputation (robust to outliers) fitted on TRAIN only.
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    # Categorical preprocessing:
    # - Missing -> explicit "Missing" category (keeps info; avoids row drops)
    # - OneHotEncode with handle_unknown to survive unseen categories in OOT.
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop",  # Why drop others? Avoid accidental leakage and keep feature policy tight.
        verbose_feature_names_out=False,
    )

    # Class imbalance exists (defaults ~15-20%), class_weight helps avoid trivial "all non-default" solutions.
    model = LogisticRegression(
        max_iter=200,
        class_weight="balanced",
        n_jobs=None,  # safer cross-platform
        solver="lbfgs"
    )

    pipe = Pipeline(steps=[("preprocess", pre), ("model", model)])
    return pipe


def fit_champion(pipe: Pipeline, train_df: pd.DataFrame) -> Pipeline:
    X = train_df[NUM_FEATURES + CAT_FEATURES + ENGINEERED_FEATURES]
    y = train_df[TARGET_COL].astype(int)
    return pipe.fit(X, y)


def predict_pd(pipe: Pipeline, df: pd.DataFrame) -> np.ndarray:
    X = df[NUM_FEATURES + CAT_FEATURES + ENGINEERED_FEATURES]
    # PD = P(default=1)
    return pipe.predict_proba(X)[:, 1]