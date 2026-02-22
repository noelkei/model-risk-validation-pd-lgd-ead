# src/reporting/figures.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def savefig(path: Path, dpi: int = 200):
    """
    Centralized save to ensure consistent export settings across notebooks/scripts.
    """
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_reliability_curve(y_true, p_pred, n_bins: int = 10, title: str = "Reliability curve"):
    """
    Reliability curve (calibration plot) using equal-frequency bins by predicted probability.
    """
    y_true = np.asarray(y_true).astype(int)
    p_pred = np.asarray(p_pred)

    bins = np.quantile(p_pred, np.linspace(0, 1, n_bins + 1))
    bins[0] = -1.0
    bins[-1] = 2.0

    idx = np.digitize(p_pred, bins) - 1
    pred_mean, obs_rate, counts = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        pred_mean.append(float(p_pred[m].mean()))
        obs_rate.append(float(y_true[m].mean()))
        counts.append(int(m.sum()))

    plt.figure()
    plt.plot(pred_mean, obs_rate, marker="o", label="Observed vs predicted")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.title(title)
    plt.xlabel("Mean predicted PD in bin")
    plt.ylabel("Observed default rate in bin")
    plt.legend()

    return pd.DataFrame({"pred_mean": pred_mean, "obs_rate": obs_rate, "count": counts})


def plot_score_hist(train_scores, oot_scores, title="Score distribution: Train vs OOT"):
    """
    Simple score distribution comparison (histogram).
    """
    train_scores = np.asarray(train_scores)
    oot_scores = np.asarray(oot_scores)

    plt.figure()
    plt.hist(train_scores, bins=50, alpha=0.6, label="Train")
    plt.hist(oot_scores, bins=50, alpha=0.6, label="OOT")
    plt.title(title)
    plt.xlabel("Predicted PD")
    plt.ylabel("Count")
    plt.legend()


def plot_psi_topbar(psi_table: pd.DataFrame, top_n: int = 10, title: str = "Top PSI drivers"):
    """
    Bar plot of top PSI features (including score if present).
    Expects columns: feature, psi, flag
    """
    top = psi_table.sort_values("psi", ascending=False).head(top_n).copy()

    plt.figure(figsize=(8, 4.5))
    plt.barh(top["feature"][::-1], top["psi"][::-1])
    plt.title(title)
    plt.xlabel("PSI")
    plt.ylabel("Feature")


def plot_shap_bar(shap_vals, feature_names, max_display: int = 15, title: str = "SHAP mean(|value|)"):
    """
    Report-friendly SHAP bar plot:
    mean absolute SHAP value per feature.
    """
    shap_vals = np.asarray(shap_vals)
    mean_abs = np.mean(np.abs(shap_vals), axis=0)

    order = np.argsort(mean_abs)[-max_display:]
    feats = np.array(feature_names)[order]
    vals = mean_abs[order]

    plt.figure(figsize=(8, 5))
    plt.barh(feats, vals)
    plt.title(title)
    plt.xlabel("mean(|SHAP value|)")
    plt.ylabel("Feature")