# src/validation/metrics.py
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score


def gini_from_auc(auc: float) -> float:
    return 2.0 * auc - 1.0


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)

    order = np.argsort(y_score)
    y_true_sorted = y_true[order]

    n_pos = (y_true_sorted == 1).sum()
    n_neg = (y_true_sorted == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan

    cdf_pos = np.cumsum(y_true_sorted == 1) / n_pos
    cdf_neg = np.cumsum(y_true_sorted == 0) / n_neg
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def topk_default_capture(y_true: np.ndarray, y_score: np.ndarray, k: float = 0.10) -> float:
    """
    Business-style metric:
    - Among the top k% highest-risk scores, what fraction of all defaults do we capture?
    Why:
    - Useful when actions are taken on top-risk segments (collections, manual review, tighter underwriting).
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)

    n = len(y_true)
    if n == 0:
        return np.nan

    cutoff = int(np.ceil((1 - k) * n))
    thresh = np.partition(y_score, cutoff)[cutoff]
    selected = y_score >= thresh

    total_defaults = (y_true == 1).sum()
    if total_defaults == 0:
        return np.nan

    captured = (y_true[selected] == 1).sum()
    return float(captured / total_defaults)


def performance_report(y_true: np.ndarray, pd_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    pd_pred = np.asarray(pd_pred)

    auc = roc_auc_score(y_true, pd_pred)
    pr_auc = average_precision_score(y_true, pd_pred)  # PR-AUC is more informative under class imbalance

    return {
        "AUC": float(auc),
        "PR_AUC": float(pr_auc),
        "Gini": float(gini_from_auc(auc)),
        "KS": float(ks_statistic(y_true, pd_pred)),
        "Brier": float(brier_score_loss(y_true, pd_pred)),
        "Top10pct_Default_Capture": float(topk_default_capture(y_true, pd_pred, k=0.10)),
        "Mean_PD": float(np.mean(pd_pred)),
        "Obs_Default_Rate": float(np.mean(y_true)),
    }