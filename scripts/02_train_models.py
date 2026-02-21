# scripts/02_train_models.py
import sys
from pathlib import Path

# Ensure project root is importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_raw import load_raw_loans
from src.data.clean import filter_closed_loans_and_target
from src.data.features import add_credit_history_length
from src.data.split import time_split

from src.models.champion_logit import build_champion_pipeline, fit_champion, predict_pd
from src.validation.metrics import performance_report
from src.validation.calibration import calibration_slope_intercept


def main():
    # 1) Load minimal columns (memory-safe)
    df = load_raw_loans()

    # 2) Define PD target on closed loans only (avoid label noise)
    df = filter_closed_loans_and_target(df)

    # 3) Feature engineering (policy-safe)
    df = add_credit_history_length(df)

    # 4) OOT time split (true generalization)
    train, oot = time_split(df)

    # 5) Train champion
    pipe = build_champion_pipeline()
    pipe = fit_champion(pipe, train)

    # 6) Predict
    pd_train = predict_pd(pipe, train)
    pd_oot = predict_pd(pipe, oot)

    # 7) Reports
    rep_train = performance_report(train["default_flag"].values, pd_train)
    rep_oot = performance_report(oot["default_flag"].values, pd_oot)

    cal_train = calibration_slope_intercept(train["default_flag"].values, pd_train)
    cal_oot = calibration_slope_intercept(oot["default_flag"].values, pd_oot)

    print("\n=== CHAMPION PERFORMANCE (Train) ===")
    for k, v in rep_train.items():
        print(f"{k}: {v:.6f}")
    for k, v in cal_train.items():
        print(f"{k}: {v:.6f}")

    print("\n=== CHAMPION PERFORMANCE (OOT) ===")
    for k, v in rep_oot.items():
        print(f"{k}: {v:.6f}")
    for k, v in cal_oot.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()