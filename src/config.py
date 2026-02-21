# src/config.py
from pathlib import Path

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RAW_LOAN_CSV = DATA_RAW_DIR / "loan.csv"
RAW_DICTIONARY_XLSX = DATA_RAW_DIR / "LCDataDictionary.xlsx"

DATE_COL = "issue_d"
STATUS_COL = "loan_status"
TARGET_COL = "default_flag"

# We only have data up to 2018-12, so OOT must be 2018 (true time-based generalization).
TRAIN_START = "2013-01-01"
TRAIN_END   = "2017-12-31"
OOT_START   = "2018-01-01"
OOT_END     = "2018-12-31"

# Baseline PD: use closed outcomes only. We exclude "Current"/"Late" to avoid label noise and horizon ambiguity.
DEFAULT_STATUSES = {"Charged Off", "Default"}
NONDEFAULT_STATUSES = {"Fully Paid"}

# Pre-origination features only: available at underwriting time (no future info).
NUM_FEATURES = [
    "annual_inc", "dti", "int_rate", "delinq_2yrs", "inq_last_6mths",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "installment", "funded_amnt", "loan_amnt"
]

CAT_FEATURES = [
    "term", "grade", "sub_grade", "emp_length", "home_ownership",
    "verification_status", "purpose", "application_type"
]

# Engineered from pre-origination fields; adds signal while staying policy-compliant.
ENGINEERED_FEATURES = ["credit_history_length_years"]

# Post-origination fields: NOT allowed as PD model inputs (leakage), but useful for LGD/EAD proxies & diagnostics.
POST_ORIGINATION_COLS = [
    "recoveries", "total_rec_prncp", "out_prncp", "total_pymnt", "last_pymnt_amnt"
]

# Simple keyword-based leakage flagging: we want to prove we looked for it, even if we already restrict usecols.
LEAKAGE_PATTERNS = [
    "recover", "collection", "last_pymnt", "pymnt", "total_rec",
    "out_prncp", "next_pymnt", "settlement", "hardship", "paid",
]