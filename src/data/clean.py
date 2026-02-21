# src/data/clean.py
import pandas as pd
from src.config import STATUS_COL, TARGET_COL, DEFAULT_STATUSES, NONDEFAULT_STATUSES

def filter_closed_loans_and_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Only closed outcomes: otherwise PD label is not observed.
    keep = DEFAULT_STATUSES | NONDEFAULT_STATUSES
    df = df[df[STATUS_COL].isin(keep)].copy()

    # Binary PD target: 1=default, 0=non-default.
    df[TARGET_COL] = (df[STATUS_COL].isin(DEFAULT_STATUSES)).astype(int)
    return df