# src/data/load_raw.py
import pandas as pd
from src.config import (
    RAW_LOAN_CSV, DATE_COL, STATUS_COL,
    POST_ORIGINATION_COLS, NUM_FEATURES, CAT_FEATURES
)

# Load only what we need:
# - Pre-origination features for PD modeling (policy-compliant)
# - A small set of post-origination fields for LGD/EAD proxies (NOT model inputs)
# This keeps memory sane and avoids accidental leakage.
USECOLS = sorted(set(
    [DATE_COL, STATUS_COL, "earliest_cr_line"] +
    NUM_FEATURES + CAT_FEATURES + POST_ORIGINATION_COLS
))

def load_raw_loans(path=RAW_LOAN_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=USECOLS, low_memory=False)
    # issue_d comes as month-based; parse to datetime for true OOT split.
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    return df