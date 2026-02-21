# src/validation/metrics.py
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss


def gini_from_auc(auc: float) -> float:
    # Standard definition in credit risk: Gini = 2*AUC - 1
    return 2.0 * auc - 1.0


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    KS = max difference between CDFs of scores for goods vs bads.
    Why KS?
    - Common banking metric for score separation.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)

    # Sort by score
    order = np.argsort(y_score)
    y_true_sorted = y_true[order]

    # CDF for positives (defaults) and negatives (non-defaults)
    n_pos = (y_true_sorted == 1).sum()
    n_neg = (y_true_sorted == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan

    cdf_pos = np.cumsum(y_true_sorted == 1) / n_pos
    cdf_neg = np.cumsum(y_true_sorted == 0) / n_neg
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def performance_report(y_true: np.ndarray, pd_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    pd_pred = np.asarray(pd_pred)

    auc = roc_auc_score(y_true, pd_pred)
    return {
        "AUC": float(auc),
        "Gini": float(gini_from_auc(auc)),
        "KS": float(ks_statistic(y_true, pd_pred)),
        "Brier": float(brier_score_loss(y_true, pd_pred)),
        "Mean_PD": float(np.mean(pd_pred)),
        "Obs_Default_Rate": float(np.mean(y_true)),
    }