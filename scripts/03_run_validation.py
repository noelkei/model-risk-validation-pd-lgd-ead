# scripts/03_run_validation.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.load_raw import load_raw_loans
from src.data.clean import filter_closed_loans_and_target
from src.data.features import add_credit_history_length
from src.data.split import time_split

from src.models.champion_logit import fit_champion, predict_pd
from src.validation.stability import compute_psi_table, score_ks_2sample
from src.validation.sensitivity import sensitivity_shocks
from src.validation.stress import stress_test


def main():
    df = load_raw_loans()
    df = filter_closed_loans_and_target(df)
    df = add_credit_history_length(df)
    train, oot = time_split(df)

    pipe, kept_num, kept_cat, dropped = fit_champion(train)

    train = train.copy()
    oot = oot.copy()
    train["pd_pred"] = predict_pd(pipe, train, kept_num, kept_cat)
    oot["pd_pred"] = predict_pd(pipe, oot, kept_num, kept_cat)

    predict_fn = lambda d: predict_pd(pipe, d, kept_num, kept_cat)

    psi = compute_psi_table(train, oot, kept_num, kept_cat, score_col="pd_pred")
    ks = score_ks_2sample(train["pd_pred"].values, oot["pd_pred"].values)

    sens = []
    for f in [c for c in ["annual_inc", "dti", "revol_util"] if c in kept_num]:
        sens.append(sensitivity_shocks(oot, predict_fn, feature=f))
    sens = pd.concat(sens, ignore_index=True) if sens else pd.DataFrame()

    stress_table, oot_with_proxies = stress_test(oot, predict_fn)

    print("\nTop PSI:")
    print(psi.head(10))
    print("\nScore KS shift:", ks)
    print("\nSensitivity (head):")
    print(sens.head(10))
    print("\nStress:")
    print(stress_table)

if __name__ == "__main__":
    main()