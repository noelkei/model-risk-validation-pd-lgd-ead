# src/validation/stability.py
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def psi_from_bins(expected_counts, actual_counts, eps=1e-6):
    e = np.asarray(expected_counts, dtype=float)
    a = np.asarray(actual_counts, dtype=float)
    e = e / e.sum()
    a = a / a.sum()
    e = np.clip(e, eps, 1.0)
    a = np.clip(a, eps, 1.0)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_numeric(train_series: pd.Series, oot_series: pd.Series, n_bins=10):
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(train_series.dropna(), qs))
    if len(edges) < 3:
        return np.nan
    edges[0] = -np.inf
    edges[-1] = np.inf

    train_bins = pd.cut(train_series, bins=edges, include_lowest=True)
    oot_bins = pd.cut(oot_series, bins=edges, include_lowest=True)

    e_counts = train_bins.value_counts(sort=False).values
    a_counts = oot_bins.value_counts(sort=False).values
    return psi_from_bins(e_counts, a_counts)


def psi_categorical(train_series: pd.Series, oot_series: pd.Series):
    train_counts = train_series.fillna("MISSING").value_counts()
    oot_counts = oot_series.fillna("MISSING").value_counts()

    cats = set(train_counts.index.tolist())
    oot_other = oot_counts[~oot_counts.index.isin(list(cats))].sum()
    oot_counts = oot_counts[oot_counts.index.isin(list(cats))]
    if oot_other > 0:
        oot_counts.loc["OTHER"] = oot_other
        train_counts.loc["OTHER"] = 0

    idx = sorted(set(train_counts.index).union(set(oot_counts.index)))
    e = train_counts.reindex(idx, fill_value=0).values
    a = oot_counts.reindex(idx, fill_value=0).values
    return psi_from_bins(e, a)


def psi_flag(psi):
    if pd.isna(psi):
        return "NA"
    if psi < 0.10:
        return "GREEN"
    if psi < 0.25:
        return "AMBER"
    return "RED"


def compute_psi_table(train_df: pd.DataFrame, oot_df: pd.DataFrame, numeric_features, categorical_features, score_col=None):
    rows = []
    for c in numeric_features:
        psi = psi_numeric(train_df[c], oot_df[c], n_bins=10)
        rows.append({"feature": c, "type": "numeric", "psi": psi, "flag": psi_flag(psi)})

    for c in categorical_features:
        psi = psi_categorical(train_df[c], oot_df[c])
        rows.append({"feature": c, "type": "categorical", "psi": psi, "flag": psi_flag(psi)})

    if score_col is not None:
        psi = psi_numeric(train_df[score_col], oot_df[score_col], n_bins=10)
        rows.append({"feature": score_col, "type": "score", "psi": psi, "flag": psi_flag(psi)})

    return pd.DataFrame(rows).sort_values("psi", ascending=False)


def score_ks_2sample(train_scores: np.ndarray, oot_scores: np.ndarray):
    res = ks_2samp(train_scores, oot_scores)
    return {"ks_stat": float(res.statistic), "p_value": float(res.pvalue)}