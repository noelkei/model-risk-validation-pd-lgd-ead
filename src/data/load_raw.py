# src/data/load_raw.py
import pandas as pd
from src.config import (
    RAW_LOAN_CSV, DATE_COL, STATUS_COL,
    POST_ORIGINATION_COLS, NUM_FEATURES, CAT_FEATURES
)

# Load only required columns:
# - Underwriting-time features for PD modeling
# - Small set of post-origination fields for LGD/EAD proxy (never model inputs)
USECOLS = sorted(set(
    [DATE_COL, STATUS_COL, "earliest_cr_line"] +
    NUM_FEATURES + CAT_FEATURES + POST_ORIGINATION_COLS
))

def _robust_parse_mon_yr(series: pd.Series) -> pd.Series:
    """
    issue_d / earliest_cr_line are typically like 'Dec-2018'.
    We parse with explicit format first (fast + consistent),
    then fallback for any leftovers (robustness).
    """
    s = pd.to_datetime(series, format="%b-%Y", errors="coerce")
    if s.isna().any():
        s2 = pd.to_datetime(series, errors="coerce")
        s = s.fillna(s2)
    return s

def load_raw_loans(path=RAW_LOAN_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=USECOLS, low_memory=False)

    df[DATE_COL] = _robust_parse_mon_yr(df[DATE_COL])

    # Data quality evidence: report parse failures (should be near 0)
    nat_rate = float(df[DATE_COL].isna().mean())
    if nat_rate > 0.001:  # >0.1% NaT is suspicious
        print(f"[WARN] {DATE_COL} NaT rate: {nat_rate:.4%}")

    return df