# src/data/features.py
import pandas as pd

def add_credit_history_length(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # earliest_cr_line is underwriting-time; safe to use.
    df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], errors="coerce")

    # Engineered feature: how long the borrower has had credit history at origination.
    df["credit_history_length_years"] = (df["issue_d"] - df["earliest_cr_line"]).dt.days / 365.25
    return df