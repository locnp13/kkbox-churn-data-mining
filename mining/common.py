"""Shared helpers for analysis/ and mining/: paths, feature groups, loading, result export.

Every script reads the cohort tables built by data_prep/01_build_features.py so that numbers
are consistent across techniques. Missing values are left as NaN here; each method handles
them itself (imputation inside sklearn pipelines, so no test-set statistics leak into training).
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("DATA_DIR", ROOT / "data"))
REPORT = Path(os.environ.get("REPORT_DIR", ROOT / "report"))
FIG = REPORT / "figures"
METRICS = REPORT / "metrics"
for p in (FIG, METRICS):
    p.mkdir(parents=True, exist_ok=True)
SEED = 42

PROFILE = ["age", "tenure_days", "has_member"]
CATEGORICAL = ["city", "registered_via", "gender", "last_payment_method"]
SUBSCRIPTION = ["n_tx", "n_tx_180d", "n_cancel", "auto_renew_rate", "discount_rate", "n_free_tx",
                "n_plan_changes", "n_payment_methods", "total_paid", "days_since_first_tx",
                "last_plan_days", "last_list_price", "last_paid", "last_auto_renew",
                "last_is_cancel", "days_since_last_tx", "days_to_expire"]
LISTENING = ["active_days", "active_days_last14", "total_songs", "complete_ratio", "skip_ratio",
             "avg_unique_songs", "total_hours", "mins_per_active_day", "listen_trend",
             "days_since_last_listen", "weekly_cv", "has_logs"]
NUMERIC = PROFILE + SUBSCRIPTION + LISTENING


def load(cohort="mar", sample=None, seed=SEED):
    """Cohort feature table. ``sample``: stratified subsample size (keeps the natural prior)."""
    df = pd.read_parquet(DATA / f"features_{cohort}.parquet")
    for c in ("has_logs", "has_tx", "has_member"):
        df[c] = df[c].astype(int)
    # counts are genuinely zero for users without logs / transactions
    zero = ["active_days", "active_days_last14", "total_songs", "total_hours", "n_tx", "n_tx_180d",
            "n_cancel", "n_free_tx", "n_plan_changes", "total_paid"]
    df[zero] = df[zero].fillna(0)
    if sample and sample < len(df):
        idx = [g.sample(int(round(sample * len(g) / len(df))), random_state=seed).index
               for _, g in df.groupby("is_churn")]
        df = df.loc[np.concatenate(idx)]
    return df.reset_index(drop=True)


def wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion k/n."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n
    den = 1 + z ** 2 / n
    mid = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return mid - half, mid + half


def churn_table(df, col, order=None):
    """Churn rate per level of ``col`` with Wilson CI and lift vs the overall rate."""
    g = df.groupby(col, observed=True)["is_churn"].agg(n="size", churners="sum")
    if order is not None:
        g = g.reindex([o for o in order if o in g.index])
    g["churn_rate"] = g.churners / g.n
    g["ci_low"], g["ci_high"] = wilson(g.churners, g.n)
    g["lift"] = g.churn_rate / df["is_churn"].mean()
    return g


def save_metrics(name, obj):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    (METRICS / f"{name}.json").write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=conv),
                                          encoding="utf-8")


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=130)


def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(p, float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out
