"""Bonus: temporal features from daily logs, subset comparison, dimensionality reduction (PCA/UMAP).

Temporal pattern: listening trend within the log window (last 10 days vs first 10 days) and
recency of last activity. Compared on listening/profile-only models to show what they add.
"""
import sys
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEATURES, FIG, NUM, load

SEED = 42
RAW = FEATURES.parent
TX = ["n_tx", "n_cancel", "auto_renew_rate", "avg_plan_days", "avg_paid", "avg_discount"]

df = load()
t = duckdb.sql(f"""
    SELECT msno,
      sum(CASE WHEN date % 100 >= 22 THEN total_secs ELSE 0 END) AS secs_late,
      sum(CASE WHEN date % 100 <= 10 THEN total_secs ELSE 0 END) AS secs_early,
      31 - max(date % 100) AS days_since_last
    FROM read_csv_auto('{RAW}/user_logs_v2.csv')
    WHERE msno IN (SELECT msno FROM '{FEATURES}')
    GROUP BY msno
""").df()
df = df.merge(t, on="msno", how="left")
df["listen_trend"] = np.log1p(df["secs_late"].fillna(0)) - np.log1p(df["secs_early"].fillna(0))
df["days_since_last"] = df["days_since_last"].fillna(31)  # no logs: treat as idle all window
df["secs_late"] = df["secs_late"].fillna(0)

base = [c for c in NUM if c not in TX] + ["tenure_days", "has_logs"]
eng = ["listen_trend", "days_since_last"]


def auc(cols, data):
    Xtr, Xte, ytr, yte = train_test_split(data[cols], data["is_churn"], test_size=0.25,
                                          stratify=data["is_churn"], random_state=SEED)
    m = RandomForestClassifier(200, min_samples_leaf=20, n_jobs=-1, random_state=SEED).fit(Xtr, ytr)
    return roc_auc_score(yte, m.predict_proba(Xte)[:, 1])


print("AUC listening/profile only      :", round(auc(base, df), 3))
print("AUC + temporal features         :", round(auc(base + eng, df), 3))
print("\nchurn by listening trend (users with logs):")
lg = df[df["has_logs"] == 1].copy()
lg["trend_bin"] = pd.cut(lg["listen_trend"], [-99, -2, -0.5, 0.5, 2, 99],
                         labels=["falling fast", "falling", "flat", "rising", "rising fast"])
trend = lg.groupby("trend_bin", observed=True)["is_churn"].agg(["mean", "size"])
print(trend.round(3).to_string())
trend.to_csv(FIG.parent / "bonus_trend.csv")
lg["idle_bin"] = pd.cut(lg["days_since_last"], [-1, 1, 7, 14, 31], labels=["0-1", "2-7", "8-14", "15+"])
idle = lg.groupby("idle_bin", observed=True)["is_churn"].agg(["mean", "size"])
print("\nchurn by days since last listen:\n", idle.round(3).to_string())
idle.to_csv(FIG.parent / "bonus_idle.csv")

print("\nsubset comparison (listening+temporal model AUC, churn rate):")
rows = []
for name, mask in {"registered_via 7": df["registered_via"] == 7,
                   "registered_via 9": df["registered_via"] == 9,
                   "registered_via 3/4": df["registered_via"].isin([3, 4]),
                   "tenure < 1y": df["tenure_days"] < 365,
                   "tenure >= 4y": df["tenure_days"] >= 1460}.items():
    sub = df[mask]
    rows.append((name, len(sub), sub["is_churn"].mean(), auc(base + eng, sub)))
subset = pd.DataFrame(rows, columns=["subset", "n", "churn_rate", "auc"])
print(subset.round(3).to_string(index=False))
subset.to_csv(FIG.parent / "bonus_subsets.csv", index=False)

Xdf = df[base + eng].clip(lower=0)
X = StandardScaler().fit_transform(np.log1p(Xdf.fillna(Xdf.median())))
pca = PCA(5, random_state=SEED).fit(X)
print("\nPCA explained variance (5 comps):", pca.explained_variance_ratio_.round(3))
idx = np.random.default_rng(SEED).choice(len(X), 15000, replace=False)
emb = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=SEED).fit_transform(X[idx])
plt.figure(figsize=(7, 6))
plt.scatter(emb[:, 0], emb[:, 1], c=df["is_churn"].values[idx], s=3, cmap="coolwarm", alpha=0.6)
plt.title("UMAP of listening/profile features (red = churn)")
plt.savefig(FIG / "umap_churn.png", dpi=120)
