# src/validation/stress.py
import numpy as np
import pandas as pd


def compute_ead_proxy(df: pd.DataFrame) -> pd.Series:
    if "total_rec_prncp" in df.columns:
        ead = df["funded_amnt"] - df["total_rec_prncp"]
        return ead.clip(lower=0).astype(float)
    return df["funded_amnt"].astype(float)


def compute_lgd_proxy(df: pd.DataFrame, default_col: str = "default_flag") -> pd.Series:
    ead = compute_ead_proxy(df)
    lgd = pd.Series(np.nan, index=df.index, dtype=float)

    mask = df[default_col] == 1
    lgd.loc[mask] = (1.0 - (df.loc[mask, "recoveries"] / (ead.loc[mask] + 1e-6))).clip(0, 1)
    return lgd


def apply_scenario(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    df_s = df.copy()
    if scenario == "mild":
        df_s["annual_inc"] = df_s["annual_inc"] * 0.95
        df_s["dti"] = df_s["dti"] * 1.10
    elif scenario == "severe":
        df_s["annual_inc"] = df_s["annual_inc"] * 0.90
        df_s["dti"] = df_s["dti"] * 1.20
        if "revol_util" in df_s.columns:
            df_s["revol_util"] = (df_s["revol_util"] * 1.10).clip(upper=100)
    else:
        raise ValueError("unknown scenario")
    return df_s


def el_proxy_mean(df: pd.DataFrame, lgd_avg: float, ead_col: str = "EAD_proxy", pd_col: str = "pd_pred") -> float:
    return float(np.mean(df[pd_col] * lgd_avg * df[ead_col]))


def stress_test(
    df_base: pd.DataFrame,
    predict_fn,
    default_col: str = "default_flag",
    scenarios=("mild", "severe")
) -> pd.DataFrame:
    """
    df_base must contain 'pd_pred'. We compute EAD/LGD proxies and evaluate scenario deltas.
    predict_fn(df) must return pd_pred array for scenario df.
    """
    df0 = df_base.copy()
    df0["EAD_proxy"] = compute_ead_proxy(df0)
    df0["LGD_proxy"] = compute_lgd_proxy(df0, default_col=default_col)

    mask = df0[default_col] == 1
    lgd_avg = float(df0.loc[mask, "LGD_proxy"].mean())

    base_mean = float(np.mean(df0["pd_pred"]))
    base_p95 = float(np.quantile(df0["pd_pred"], 0.95))
    base_el = el_proxy_mean(df0, lgd_avg, ead_col="EAD_proxy", pd_col="pd_pred")

    rows = []
    for sc in scenarios:
        df_sc = apply_scenario(df0, sc)
        df_sc = df_sc.copy()
        df_sc["pd_pred"] = predict_fn(df_sc)

        rows.append({
            "scenario": sc,
            "mean_pd": float(np.mean(df_sc["pd_pred"])),
            "delta_mean_pd": float(np.mean(df_sc["pd_pred"]) - base_mean),
            "p95_pd": float(np.quantile(df_sc["pd_pred"], 0.95)),
            "delta_p95": float(np.quantile(df_sc["pd_pred"], 0.95) - base_p95),
            "EL_proxy_mean": el_proxy_mean(df_sc, lgd_avg, ead_col="EAD_proxy", pd_col="pd_pred"),
            "delta_EL_proxy": el_proxy_mean(df_sc, lgd_avg, ead_col="EAD_proxy", pd_col="pd_pred") - base_el,
            "LGD_avg_defaults": lgd_avg,
        })

    return pd.DataFrame(rows), df0