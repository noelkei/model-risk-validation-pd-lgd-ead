# scripts/05_generate_report_artifacts.py
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
from src.validation.stability import compute_psi_table
from src.validation.sensitivity import sensitivity_shocks
from src.validation.stress import stress_test
from src.validation.interpretability import shap_global_summary

from src.reporting.tables import round_table, traffic_light_psi, compare_oot_table
from src.reporting.figures import (
    ensure_dir, savefig,
    plot_reliability_curve, plot_score_hist, plot_psi_topbar, plot_shap_bar
)


def main():
    reports_dir = PROJECT_ROOT / "reports"
    fig_dir = ensure_dir(reports_dir / "figures")
    tab_dir = ensure_dir(reports_dir / "tables")

    # --- Data + split ---
    df = load_raw_loans()
    df = filter_closed_loans_and_target(df)
    df = add_credit_history_length(df)
    train, oot = time_split(df)

    # --- Champion ---
    champ_pipe, kept_num, kept_cat, dropped = fit_champion(train)
    train_ch = train.copy()
    oot_ch = oot.copy()
    train_ch["pd_pred"] = predict_champion_pd(champ_pipe, train_ch, kept_num, kept_cat)
    oot_ch["pd_pred"] = predict_champion_pd(champ_pipe, oot_ch, kept_num, kept_cat)

    # --- Challenger ---
    lgbm_pipe = build_challenger_pipeline(kept_num, kept_cat)
    lgbm_pipe = fit_challenger(lgbm_pipe, train, TARGET_COL, kept_num, kept_cat)
    train_lb = train.copy()
    oot_lb = oot.copy()
    train_lb["pd_pred"] = predict_lgbm_pd(lgbm_pipe, train_lb, kept_num, kept_cat)
    oot_lb["pd_pred"] = predict_lgbm_pd(lgbm_pipe, oot_lb, kept_num, kept_cat)

    # --- Summary tables (performance + calibration) ---
    summ_ch = model_summary_table("Champion_Logit",
                                  train[TARGET_COL].values, train_ch["pd_pred"].values,
                                  oot[TARGET_COL].values, oot_ch["pd_pred"].values)
    summ_lb = model_summary_table("Challenger_LGBM",
                                  train[TARGET_COL].values, train_lb["pd_pred"].values,
                                  oot[TARGET_COL].values, oot_lb["pd_pred"].values)
    summary = pd.concat([summ_ch, summ_lb], ignore_index=True)

    round_table(summary, 4).to_csv(tab_dir / "model_summary_train_oot.csv", index=False)
    round_table(compare_oot_table(summary), 4).to_csv(tab_dir / "oot_compare_champion_vs_challenger.csv", index=False)

    # --- Stability (score) table ---
    stab_ch = stability_summary("Champion_Logit", train_ch, oot_ch, kept_num, kept_cat, score_col="pd_pred")
    stab_lb = stability_summary("Challenger_LGBM", train_lb, oot_lb, kept_num, kept_cat, score_col="pd_pred")
    stab = pd.DataFrame([stab_ch, stab_lb])
    round_table(stab, 6).to_csv(tab_dir / "score_stability_compare.csv", index=False)

    # --- PSI table (Champion inputs + score) ---
    psi = compute_psi_table(train_ch, oot_ch, kept_num, kept_cat, score_col="pd_pred")
    traffic_light_psi(round_table(psi, 4)).to_csv(tab_dir / "psi_table.csv", index=False)

    # --- Sensitivity table (Champion on OOT) ---
    predict_fn_ch = lambda d: predict_champion_pd(champ_pipe, d, kept_num, kept_cat)
    drivers = [c for c in ["annual_inc", "dti", "revol_util"] if c in kept_num]
    sens_tables = [sensitivity_shocks(oot_ch, predict_fn_ch, feature=f) for f in drivers]
    sens = pd.concat(sens_tables, ignore_index=True) if sens_tables else pd.DataFrame()
    round_table(sens, 6).to_csv(tab_dir / "sensitivity_table.csv", index=False)

    # --- Stress table + LGD by grade (Champion on OOT) ---
    stress_table, oot_with_proxies = stress_test(oot_ch, predict_fn_ch, default_col=TARGET_COL)
    round_table(stress_table, 6).to_csv(tab_dir / "stress_table.csv", index=False)

    default_mask = oot_with_proxies[TARGET_COL] == 1
    lgd_by_grade = (
        oot_with_proxies.loc[default_mask]
        .groupby("grade")["LGD_proxy"]
        .agg(["count", "mean", "median", lambda x: x.quantile(0.9)])
        .rename(columns={"<lambda_0>": "p90"})
        .sort_values("count", ascending=False)
    )
    round_table(lgd_by_grade.reset_index(), 6).to_csv(tab_dir / "lgd_by_grade.csv", index=False)

    # --- Figures ---
    # Reliability (Champion OOT)
    plot_reliability_curve(train[TARGET_COL].values, train_ch["pd_pred"].values, title="Champion reliability (Train)")
    savefig(fig_dir / "champion_reliability_train.png")

    plot_reliability_curve(oot[TARGET_COL].values, oot_ch["pd_pred"].values, title="Champion reliability (OOT)")
    savefig(fig_dir / "champion_reliability_oot.png")

    # Score histogram (Champion)
    plot_score_hist(train_ch["pd_pred"].values, oot_ch["pd_pred"].values, title="Champion score distribution (Train vs OOT)")
    savefig(fig_dir / "champion_score_hist_train_vs_oot.png")

    # PSI top bar
    plot_psi_topbar(psi, top_n=12, title="Top PSI drivers (Champion inputs + score)")
    savefig(fig_dir / "psi_top_drivers.png")

    # SHAP bar (Challenger, OOT sample)
    X_oot = oot_lb[kept_num + kept_cat]
    shap_vals, feat_names, X_trans = shap_global_summary(lgbm_pipe, X_oot, max_samples=20000)
    plot_shap_bar(shap_vals, feat_names, max_display=15, title="Challenger SHAP mean(|value|)")
    savefig(fig_dir / "challenger_shap_bar.png")

    print("Saved tables to:", tab_dir)
    print("Saved figures to:", fig_dir)


if __name__ == "__main__":
    main()
