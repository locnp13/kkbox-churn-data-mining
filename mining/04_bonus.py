"""Bonus / advanced techniques on the mar cohort.

1. Temporal behaviour: weekly listening trajectory of churners vs stayers over the 60 days
   before the cutoff, plus churn by listening trend and by idle days (95 % CI).
2. Anomaly detection: Isolation Forest on behaviour features; churn among anomalies.
3. Subset comparison: how well the out-of-time LightGBM ranks churners inside each segment.
4. Dimensionality reduction: PCA (variance, loadings) and UMAP (by churn and by cluster).

Outputs: report/bonus_*.csv, figures/bonus_*.png, umap_churn.png, metrics/bonus.json
"""
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_prep"))
from common import DATA, REPORT, SEED, churn_table, load, save_metrics, savefig
from config import COHORTS, LOG_WINDOW_DAYS

warnings.filterwarnings("ignore")
df = load("mar")
base = df.is_churn.mean()
out = {}

# ---- 1. temporal trajectory -----------------------------------------------------------------
cut = date.fromisoformat(COHORTS["mar"]["cutoff"])
start = cut - timedelta(days=LOG_WINDOW_DAYS - 1)
logs = DATA / "logs_window.parquet"
if logs.exists():
    lbl = DATA / "features_mar.parquet"
    traj = duckdb.sql(f"""
        WITH u AS (SELECT msno, is_churn FROM '{lbl}'),
        w AS (SELECT l.msno, {LOG_WINDOW_DAYS // 7 - 1}
                       - (DATE '{cut}' - strptime(CAST(date AS VARCHAR), '%Y%m%d')::DATE) // 7 AS week,
                     count(DISTINCT date) AS days, sum(total_secs) / 3600 AS hours
              FROM '{logs}' l WHERE date BETWEEN {start:%Y%m%d} AND {cut:%Y%m%d}
                AND msno IN (SELECT msno FROM u) GROUP BY 1, 2)
        SELECT u.is_churn, w.week, sum(w.days) / any_value(n.n) AS active_days_per_user,
               sum(w.hours) / any_value(n.n) AS hours_per_user
        FROM w JOIN u USING (msno)
        JOIN (SELECT is_churn, count(*) n FROM u GROUP BY 1) n USING (is_churn)
        WHERE week >= 0
        GROUP BY 1, 2 ORDER BY 1, 2""").df()
    traj.to_csv(REPORT / "bonus_trajectory.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for y, lab in [(0, "ở lại"), (1, "churn")]:
        t = traj[traj.is_churn == y]
        axes[0].plot(t.week + 1, t.active_days_per_user, marker="o", label=lab)
        axes[1].plot(t.week + 1, t.hours_per_user, marker="o", label=lab)
    axes[0].set_title("Số ngày nghe trung bình mỗi tuần"); axes[1].set_title("Số giờ nghe trung bình mỗi tuần")
    for a in axes:
        a.set_xlabel(f"tuần trước cutoff (tuần cuối = 7 ngày tới {cut})"); a.legend()
    savefig(fig, "bonus_trajectory.png")
    first, last = traj.week.min(), traj.week.max()
    out["trajectory_change"] = {
        lab: float(traj[(traj.is_churn == y) & (traj.week == last)].active_days_per_user.iloc[0] /
                   traj[(traj.is_churn == y) & (traj.week == first)].active_days_per_user.iloc[0] - 1)
        for y, lab in [(0, "stay"), (1, "churn")]}

lg = df[df.has_logs == 1].copy()
lg["trend"] = pd.cut(lg.listen_trend, [-99, -1, -0.25, 0.25, 1, 99],
                     labels=["falling fast", "falling", "flat", "rising", "rising fast"])
lg["idle"] = pd.cut(lg.days_since_last_listen, [-1, 0, 3, 7, 14, 60], labels=["0", "1-3", "4-7", "8-14", "15+"])
trend, idle = churn_table(lg, "trend"), churn_table(lg, "idle")
trend.to_csv(REPORT / "bonus_trend.csv"); idle.to_csv(REPORT / "bonus_idle.csv")
print("churn by trend:\n", trend.round(3).to_string(), "\nchurn by idle days:\n", idle.round(3).to_string())
out["trend"] = trend.churn_rate.to_dict(); out["idle"] = idle.churn_rate.to_dict()

# ---- 2. anomaly detection -------------------------------------------------------------------
FE = ["n_tx", "n_cancel", "auto_renew_rate", "discount_rate", "last_paid", "last_plan_days",
      "total_paid", "active_days", "total_hours", "avg_unique_songs", "skip_ratio", "complete_ratio"]
Xa = df[FE].fillna(0).clip(lower=0)
Xa = StandardScaler().fit_transform(np.log1p(Xa))
iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=SEED, n_jobs=-1).fit(Xa)
df["anomaly"] = np.where(iso.predict(Xa) == -1, "anomaly", "normal")
an = churn_table(df, "anomaly")
prof_an = df.groupby("anomaly")[FE].median().T
prof_an.to_csv(REPORT / "bonus_anomaly_profile.csv")
an.to_csv(REPORT / "bonus_anomaly.csv")
print("\nIsolation Forest:\n", an.round(3).to_string(), "\n", prof_an.round(2).to_string())
out["anomaly"] = an[["n", "churn_rate", "lift"]].to_dict("index")

# ---- 3. subset comparison ----------------------------------------------------------------------
pred = DATA / "pred_mar_lgbm.parquet"
cl = DATA / "clusters_mar.parquet"
if pred.exists():
    d = df.merge(pd.read_parquet(pred), on="msno")
    if cl.exists():
        d = d.merge(pd.read_parquet(cl), on="msno", how="left")
    subsets = {
        "all users": d.index == d.index,
        "registered_via 7": d.registered_via == 7, "registered_via 9": d.registered_via == 9,
        "registered_via 3/4": d.registered_via.isin([3, 4]),
        "tenure < 1y": d.tenure_days < 365, "tenure >= 4y": d.tenure_days >= 1460,
        "auto-renew on": d.last_auto_renew == 1, "auto-renew off": d.last_auto_renew == 0,
        "no listening logs": d.has_logs == 0, "with listening logs": d.has_logs == 1,
    }
    if "cluster" in d:
        subsets.update({f"segment {c}": d.cluster == c for c in sorted(d.cluster.dropna().unique())})
    rows = []
    for name, m in subsets.items():
        s = d[m]
        if s.is_churn.nunique() == 2 and len(s) >= 500:
            rows.append({"subset": name, "n": len(s), "share": len(s) / len(d), "churn_rate": s.is_churn.mean(),
                         "roc_auc": roc_auc_score(s.is_churn, s.p_churn),
                         "pr_auc": average_precision_score(s.is_churn, s.p_churn),
                         "mean_pred": s.p_churn.mean()})
    subset = pd.DataFrame(rows)
    subset.to_csv(REPORT / "bonus_subsets.csv", index=False)
    print("\n", subset.round(3).to_string(index=False))
    out["subsets"] = subset.to_dict("records")

# ---- 4. PCA / UMAP --------------------------------------------------------------------------
pca = PCA(min(8, Xa.shape[1]), random_state=SEED).fit(Xa)
load_df = pd.DataFrame(pca.components_[:3].T, index=FE, columns=["PC1", "PC2", "PC3"])
load_df.to_csv(REPORT / "bonus_pca_loadings.csv")
out["pca_explained"] = pca.explained_variance_ratio_.round(4)
print("\nPCA explained variance:", pca.explained_variance_ratio_.round(3))
rng = np.random.default_rng(SEED)
idx = rng.choice(len(Xa), min(15_000, len(Xa)), replace=False)
try:
    import umap
    emb = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=SEED).fit_transform(Xa[idx])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(emb[:, 0], emb[:, 1], c=df.is_churn.values[idx], s=3, cmap="coolwarm", alpha=0.6)
    axes[0].set_title("UMAP của đặc trưng hành vi (đỏ = churn)")
    if cl.exists():
        c = df[["msno"]].merge(pd.read_parquet(cl), on="msno", how="left").cluster.values[idx]
        sc = axes[1].scatter(emb[:, 0], emb[:, 1], c=c, s=3, cmap="tab10", alpha=0.6)
        axes[1].legend(*sc.legend_elements(), title="phân khúc", fontsize=7)
        axes[1].set_title("cùng phép chiếu, tô màu theo phân khúc K-Means")
    savefig(fig, "umap_churn.png")
except ImportError:
    print("umap-learn not installed; skipping UMAP")
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
ax.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), "r-o")
ax.set_xlabel("thành phần"); ax.set_ylabel("tỉ lệ phương sai giải thích"); ax.set_title("PCA: phương sai giải thích")
savefig(fig, "bonus_pca_scree.png")

save_metrics("bonus", out)
