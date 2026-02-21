# src/data/features.py
import pandas as pd

def _robust_parse_mon_yr(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, format="%b-%Y", errors="coerce")
    if s.isna().any():
        s2 = pd.to_datetime(series, errors="coerce")
        s = s.fillna(s2)
    return s

def add_credit_history_length(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Underwriting-time field -> safe feature engineering
    df["earliest_cr_line"] = _robust_parse_mon_yr(df["earliest_cr_line"])

    # If earliest_cr_line is missing, credit_history_length becomes NaN and will be imputed later.
    df["credit_history_length_years"] = (df["issue_d"] - df["earliest_cr_line"]).dt.days / 365.25
    return df