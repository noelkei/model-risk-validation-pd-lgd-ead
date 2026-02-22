# scripts/04_run_challenger.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import TARGET_COL
from src.data.load_raw import load_raw_loans
from src.data.clean import filter_closed_loans_and_target
from src.data.features import add_credit_history_length
from src.data.split import time_split

from src.models.champion_logit import fit_champion, predict_pd as predict_champion_pd
from src.models.challenger_lgbm import build_challenger_pipeline, fit_challenger, predict_pd as predict_lgbm_pd

from src.validation.compare import model_summary_table, stability_summary


def main():
    df = load_raw_loans()
    df = filter_closed_loans_and_target(df)
    df = add_credit_history_length(df)
    train, oot = time_split(df)

    # Champion
    champ_pipe, kept_num, kept_cat, dropped = fit_champion(train)

    train_ch = train.copy()
    oot_ch = oot.copy()
    train_ch["pd_pred"] = predict_champion_pd(champ_pipe, train_ch, kept_num, kept_cat)
    oot_ch["pd_pred"] = predict_champion_pd(champ_pipe, oot_ch, kept_num, kept_cat)

    # Challenger (same kept features for fairness)
    lgbm_pipe = build_challenger_pipeline(kept_num, kept_cat)
    lgbm_pipe = fit_challenger(lgbm_pipe, train, TARGET_COL, kept_num, kept_cat)

    train_lb = train.copy()
    oot_lb = oot.copy()
    train_lb["pd_pred"] = predict_lgbm_pd(lgbm_pipe, train_lb, kept_num, kept_cat)
    oot_lb["pd_pred"] = predict_lgbm_pd(lgbm_pipe, oot_lb, kept_num, kept_cat)

    # Performance + calibration comparison
    summ_ch = model_summary_table("Champion_Logit", train[TARGET_COL].values, train_ch["pd_pred"].values,
                                  oot[TARGET_COL].values, oot_ch["pd_pred"].values)
    summ_lb = model_summary_table("Challenger_LGBM", train[TARGET_COL].values, train_lb["pd_pred"].values,
                                  oot[TARGET_COL].values, oot_lb["pd_pred"].values)

    summary = pd.concat([summ_ch, summ_lb], ignore_index=True)
    print("\n=== PERFORMANCE + CALIBRATION SUMMARY ===")
    print(summary)

    # Stability comparison (score)
    stab_ch = stability_summary("Champion_Logit", train_ch, oot_ch, kept_num, kept_cat, score_col="pd_pred")
    stab_lb = stability_summary("Challenger_LGBM", train_lb, oot_lb, kept_num, kept_cat, score_col="pd_pred")

    print("\n=== STABILITY SUMMARY (SCORE) ===")
    print(pd.DataFrame([stab_ch, stab_lb]))


if __name__ == "__main__":
    main()