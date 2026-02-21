# src/models/champion_logit.py
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import NUM_FEATURES, CAT_FEATURES, ENGINEERED_FEATURES, TARGET_COL


MISSINGNESS_DROP_THRESHOLD = 0.30  # 30% missing in TRAIN -> exclude feature from model inputs


def select_features_with_missingness_policy(train_df: pd.DataFrame):
    """
    Data quality policy (Model Risk):
    - If a feature has too much missingness in TRAIN, it is high model-risk (imputation dominates).
    - We exclude it from model inputs but keep it in the dataset for reporting.

    Threshold choice:
    - 30% is a common pragmatic cutoff: beyond this, the model is often learning 'missingness' more than signal.
    """
    num_candidates = NUM_FEATURES + ENGINEERED_FEATURES
    cat_candidates = CAT_FEATURES

    missing_num = train_df[num_candidates].isna().mean()
    missing_cat = train_df[cat_candidates].isna().mean()

    kept_num = [c for c in num_candidates if missing_num.get(c, 0.0) <= MISSINGNESS_DROP_THRESHOLD]
    kept_cat = [c for c in cat_candidates if missing_cat.get(c, 0.0) <= MISSINGNESS_DROP_THRESHOLD]

    dropped = {
        "dropped_num": sorted([c for c in num_candidates if c not in kept_num]),
        "dropped_cat": sorted([c for c in cat_candidates if c not in kept_cat]),
        "missing_num": missing_num.sort_values(ascending=False),
        "missing_cat": missing_cat.sort_values(ascending=False),
    }
    return kept_num, kept_cat, dropped


def build_champion_pipeline(numeric_features, categorical_features) -> Pipeline:
    """
    Champion = logistic regression baseline.

    Key points for Model Risk:
    - Fit preprocessing on TRAIN only (pipeline ensures this).
    - Scale numerics to improve convergence/stability of logistic.
    - Do NOT use class_weight="balanced" if we want PDs as probabilities;
      instead we evaluate imbalance-aware metrics (PR-AUC, Top-K capture).
    """

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),  # deterministic, robust
            ("scaler", StandardScaler()),                   # improves convergence
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),  # deterministic
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop",  # drop everything else to avoid accidental leakage
        verbose_feature_names_out=False,
    )

    model = LogisticRegression(
        max_iter=500,        # higher cap to reduce convergence warnings
        solver="lbfgs",
    )

    return Pipeline(steps=[("preprocess", pre), ("model", model)])


def fit_champion(train_df: pd.DataFrame):
    kept_num, kept_cat, dropped_info = select_features_with_missingness_policy(train_df)

    pipe = build_champion_pipeline(kept_num, kept_cat)

    X = train_df[kept_num + kept_cat]
    y = train_df[TARGET_COL].astype(int)

    pipe.fit(X, y)
    return pipe, kept_num, kept_cat, dropped_info


def predict_pd(pipe: Pipeline, df: pd.DataFrame, kept_num, kept_cat) -> np.ndarray:
    X = df[kept_num + kept_cat]
    return pipe.predict_proba(X)[:, 1]