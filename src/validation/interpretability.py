# src/validation/interpretability.py
import numpy as np
import shap


def shap_global_summary(pipe, X_df, max_samples=20000, random_state=42):
    """
    Computes SHAP values for a tree model inside a sklearn Pipeline.

    Returns:
    - shap_values (for class 1 if binary)
    - feature_names
    - X_trans (transformed feature matrix used for SHAP)

    Note:
    - We run SHAP on the post-preprocessing matrix (OHE expanded).
    - We subsample for speed on large datasets.
    """
    rng = np.random.default_rng(random_state)
    if len(X_df) > max_samples:
        idx = rng.choice(len(X_df), size=max_samples, replace=False)
        X_df = X_df.iloc[idx].copy()

    pre = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]

    X_trans = pre.transform(X_df)
    feat_names = pre.get_feature_names_out()

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_trans)

    # For binary models, SHAP may return [class0, class1]
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    return shap_vals, feat_names, X_trans