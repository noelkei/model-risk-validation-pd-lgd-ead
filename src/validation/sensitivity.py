# src/validation/sensitivity.py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def sensitivity_shocks(
    df_base: pd.DataFrame,
    predict_fn,
    feature: str,
    shocks=(-0.10, -0.05, 0.05, 0.10),
    cap_100_features=("revol_util",),
):
    """
    df_base must already contain baseline predictions in column 'pd_pred'.
    predict_fn(df) must return pd_pred array for the given df.
    """
    rows = []
    pd_base = df_base["pd_pred"].values

    for s in shocks:
        df_s = df_base.copy()
        df_s[feature] = df_s[feature] * (1.0 + s)
        if feature in cap_100_features:
            df_s[feature] = df_s[feature].clip(upper=100)

        pd_new = predict_fn(df_s)

        rows.append({
            "feature": feature,
            "shock": s,
            "mean_pd_base": float(np.mean(pd_base)),
            "mean_pd_new": float(np.mean(pd_new)),
            "delta_mean_pd": float(np.mean(pd_new) - np.mean(pd_base)),
            "p95_base": float(np.quantile(pd_base, 0.95)),
            "p95_new": float(np.quantile(pd_new, 0.95)),
            "delta_p95": float(np.quantile(pd_new, 0.95) - np.quantile(pd_base, 0.95)),
            "spearman_rank_corr": float(spearmanr(pd_base, pd_new).correlation),
        })

    return pd.DataFrame(rows)