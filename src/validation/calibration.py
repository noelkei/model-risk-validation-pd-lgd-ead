# src/validation/calibration.py
import numpy as np
import statsmodels.api as sm


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    # Avoid inf values; calibration should be stable numerically.
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def calibration_slope_intercept(y_true: np.ndarray, pd_pred: np.ndarray) -> dict:
    """
    Calibration-in-the-large (intercept) and calibration slope.
    Why?
    - In credit risk, a model can discriminate well but be miscalibrated (PDs too high/low).
    - Intercept ~ 0 and slope ~ 1 indicates good calibration.
    """
    y = np.asarray(y_true).astype(int)
    x = logit(np.asarray(pd_pred))
    X = sm.add_constant(x)

    model = sm.Logit(y, X).fit(disp=False)
    intercept, slope = model.params[0], model.params[1]
    return {
        "Calib_Intercept": float(intercept),
        "Calib_Slope": float(slope),
    }