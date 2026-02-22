# src/reporting/tables.py
from __future__ import annotations

import pandas as pd


def round_table(df: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    """
    Standard rounding for report tables.
    """
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].round(decimals)
    return out


def traffic_light_psi(psi_table: pd.DataFrame) -> pd.DataFrame:
    """
    Keep columns in a report-ready order and sort by PSI descending.
    Expects: feature, type, psi, flag
    """
    cols = [c for c in ["feature", "type", "psi", "flag"] if c in psi_table.columns]
    out = psi_table[cols].copy().sort_values("psi", ascending=False)
    return out


def compare_oot_table(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Extract OOT rows and keep key columns for the executive comparison.
    Expects the output of model_summary_table concatenation (model, set, metrics...).
    """
    oot = summary[summary["set"] == "OOT"].copy().set_index("model")
    keep = [c for c in [
        "AUC", "PR_AUC", "KS", "Brier",
        "Top10pct_Default_Capture",
        "Mean_PD", "Obs_Default_Rate",
        "Calib_Intercept", "Calib_Slope"
    ] if c in oot.columns]
    return oot[keep].reset_index()


def champion_oot_table(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Same as compare_oot_table but for champion only.
    """
    oot = summary[(summary["set"] == "OOT") & (summary["model"].str.contains("Champion"))].copy()
    return oot