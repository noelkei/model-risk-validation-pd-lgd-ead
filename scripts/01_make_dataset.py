# scripts/01_make_dataset.py
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH when running scripts directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_raw import load_raw_loans
from src.data.clean import filter_closed_loans_and_target
from src.data.features import add_credit_history_length
from src.data.split import time_split
from src.utils.checks import leakage_candidates

def main():
    df = load_raw_loans()
    print("Loaded (usecols) shape:", df.shape)

    # Report potential leakage fields (diagnostic, not selection).
    leaks = leakage_candidates(df.columns.tolist())
    print("Leakage candidates found:", len(leaks))
    print("Examples:", leaks[:25])

    df = filter_closed_loans_and_target(df)
    print("After closed-loans filter shape:", df.shape)
    print("Overall default rate:", df["default_flag"].mean())

    df = add_credit_history_length(df)

    train, oot = time_split(df)
    print("Train shape:", train.shape, " | OOT shape:", oot.shape)
    print("Train default rate:", train["default_flag"].mean() if len(train) else None)
    print("OOT default rate:", oot["default_flag"].mean() if len(oot) else None)

if __name__ == "__main__":
    main()