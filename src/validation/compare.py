# src/validation/compare.py
import pandas as pd

from src.validation.metrics import performance_report
from src.validation.calibration import calibration_slope_intercept
from src.validation.stability import compute_psi_table, score_ks_2sample


def model_summary_table(name: str, y_train, p_train, y_oot, p_oot) -> pd.DataFrame:
    """
    Single table: performance + calibration for Train and OOT.
    Used in Notebook 03 and in the final report.
    """
    rep_tr = performance_report(y_train, p_train)
    rep_oot = performance_report(y_oot, p_oot)
    cal_tr = calibration_slope_intercept(y_train, p_train)
    cal_oot = calibration_slope_intercept(y_oot, p_oot)

    return pd.DataFrame([
        {"model": name, "set": "Train", **rep_tr, **cal_tr},
        {"model": name, "set": "OOT", **rep_oot, **cal_oot},
    ])


def stability_summary(name: str, train_df: pd.DataFrame, oot_df: pd.DataFrame, kept_num, kept_cat, score_col="pd_pred") -> dict:
    """
    Score stability summary:
    - PSI of the score
    - two-sample KS statistic on the score distribution
    """
    psi_table = compute_psi_table(train_df, oot_df, kept_num, kept_cat, score_col=score_col)
    score_psi = float(psi_table.loc[psi_table["feature"] == score_col, "psi"].iloc[0])
    ks = score_ks_2sample(train_df[score_col].values, oot_df[score_col].values)
    return {"model": name, "score_psi": score_psi, "score_ks_stat": ks["ks_stat"], "score_ks_p": ks["p_value"]}