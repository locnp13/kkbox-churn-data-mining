"""Temporal stability: do the patterns found on the mar cohort hold one month earlier (feb)?

1. Segments: the mar K-Means model (scaler + centroids) assigns feb users; compare segment
   shares and churn rates. Then K-Means is refitted on feb from scratch (own scaler) and
   compared with the mar model on the same feb users (ARI) and by centroid matching
   (Hungarian assignment on centroid distance).
2. Rules: every selected mar rule (churn and retention) is re-measured on feb with the item
   thresholds learnt on the mar train split, so the rule definition is identical.
Churn base rates differ (feb 6.4 %, mar 9.0 %), so lift (relative to each month's base) is the
comparable quantity.

Outputs: report/cluster_temporal.csv, rule_temporal.csv, figures/stability.png, metrics/stability.json
"""
import os
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, REPORT, SEED, churn_table, load, save_metrics, savefig
from transforms import itemise, prepare

N = int(os.environ.get("CLUSTER_SAMPLE", 200_000))
model = joblib.load(DATA / "cluster_model.joblib")
scaler, km, MED, names = model["scaler"], model["km"], model["median"], model["names"]
NOLOG = model.get("nolog_id", km.n_clusters)
CLF = model.get("features") or list(MED.index)
K = km.n_clusters + 1          # K-Means clusters + the rule-based "no listening log" segment

mar = load("mar").merge(pd.read_parquet(DATA / "clusters_mar.parquet")[["msno", "cluster"]], on="msno")
feb = load("feb")
feb["cluster"] = np.where(feb.has_logs == 1, km.predict(scaler.transform(prepare(feb)[CLF].fillna(MED))), NOLOG)
out = {"base_mar": mar.is_churn.mean(), "base_feb": feb.is_churn.mean(), "k": K}

# ---- 1a. same segments, one month earlier ----------------------------------------------------
cm, cf = churn_table(mar, "cluster"), churn_table(feb, "cluster")
seg = pd.DataFrame({
    "name": pd.Series(names), "share_mar": cm.n / cm.n.sum(), "share_feb": cf.n / cf.n.sum(),
    "churn_mar": cm.churn_rate, "churn_feb": cf.churn_rate, "ci_low_feb": cf.ci_low, "ci_high_feb": cf.ci_high,
    "lift_mar": cm.lift, "lift_feb": cf.lift})
seg.index.name = "cluster"
seg.to_csv(REPORT / "cluster_temporal.csv")
print(seg.round(3).to_string())
out["segment_lift_spearman"] = spearmanr(seg.lift_mar, seg.lift_feb).statistic
out["segment_share_max_abs_diff"] = float((seg.share_mar - seg.share_feb).abs().max())

# ---- 1b. refit on feb: does the same structure re-emerge? ---------------------------------------
fs = feb[feb.has_logs == 1].sample(min(N, int((feb.has_logs == 1).sum())), random_state=SEED)
Rf = prepare(fs)[CLF]
med_f = Rf.median()
sc_f = StandardScaler().fit(Rf.fillna(med_f))
km_f = KMeans(km.n_clusters, n_init=10, random_state=SEED).fit(sc_f.transform(Rf.fillna(med_f)))
out["ari_feb_refit_vs_mar_model"] = adjusted_rand_score(fs.cluster, km_f.labels_)
# centroids of the feb refit expressed in the mar feature space, matched to mar centroids
cent_f = scaler.transform(pd.DataFrame(sc_f.inverse_transform(km_f.cluster_centers_), columns=Rf.columns))
D = np.linalg.norm(km.cluster_centers_[:, None, :] - cent_f[None, :, :], axis=2)
r, c = linear_sum_assignment(D)
out["centroid_match_mean_dist"] = float(D[r, c].mean())
out["centroid_spread_mean_dist"] = float(np.linalg.norm(km.cluster_centers_ - km.cluster_centers_.mean(0), axis=1).mean())
print(f"ARI feb-refit vs mar model: {out['ari_feb_refit_vs_mar_model']:.3f}; matched centroid distance "
      f"{out['centroid_match_mean_dist']:.2f} vs centroid spread {out['centroid_spread_mean_dist']:.2f}")

# ---- 2. rules ---------------------------------------------------------------------------
tr, _ = train_test_split(mar, test_size=0.3, stratify=mar.is_churn, random_state=SEED)
I_feb = itemise(feb, tr)
y = feb.is_churn.values
rows = []
for kind, f in [("churn", "churn_rules.csv"), ("retain", "retention_rules.csv")]:
    for _, r_ in pd.read_csv(REPORT / f).iterrows():
        items = r_.rule.split(" & ")
        if not all(i in I_feb.columns for i in items):
            continue
        m = I_feb[items].all(axis=1).values
        conf = y[m].mean() if m.sum() else np.nan
        rows.append({"rule": r_.rule, "direction": kind, "support_mar": r_.support,
                     "conf_mar_holdout": r_.confidence_holdout, "lift_mar_holdout": r_.lift_holdout,
                     "support_feb": m.mean(), "conf_feb": conf, "lift_feb": conf / y.mean()})
rt = pd.DataFrame(rows)
rt["same_direction"] = np.where(rt.direction == "churn", rt.lift_feb > 1, rt.lift_feb < 1)
rt.to_csv(REPORT / "rule_temporal.csv", index=False)
ok = rt.dropna(subset=["lift_feb"])
ok = ok[(ok.lift_feb > 0) & (ok.lift_mar_holdout > 0)]   # log-scale comparison
out["rules_checked"] = len(rt)
out["rule_log_lift_pearson"] = pearsonr(np.log(ok.lift_mar_holdout), np.log(ok.lift_feb)).statistic
out["rule_share_same_direction"] = float(rt.same_direction.mean())
out["rule_share_lift_ge_1_5_feb"] = float((rt[rt.direction == "churn"].lift_feb >= 1.5).mean())
print(rt[["rule", "lift_mar_holdout", "lift_feb", "same_direction"]].round(2).head(15).to_string(index=False))
print({k: round(v, 3) if isinstance(v, float) else v for k, v in out.items()})

# ---- figure -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
x = np.arange(K)
axes[0].bar(x - 0.2, seg.churn_mar, 0.4, label="tháng 3 (mô hình gốc)")
axes[0].bar(x + 0.2, seg.churn_feb, 0.4, label="tháng 2 (áp mô hình tháng 3)")
axes[0].set_xticks(x, [f"{i}" for i in seg.index]); axes[0].set_xlabel("phân khúc")
axes[0].set_ylabel("tỉ lệ churn"); axes[0].legend()
axes[0].set_title(f"Churn theo phân khúc ở hai tháng (Spearman lift = {out['segment_lift_spearman']:.2f})")
for kind, col in [("churn", "#C44E52"), ("retain", "#55A868")]:
    s = ok[ok.direction == kind]
    axes[1].scatter(s.lift_mar_holdout, s.lift_feb, c=col, label="luật churn" if kind == "churn" else "luật giữ chân")
lim = [min(ok.lift_mar_holdout.min(), ok.lift_feb.min()) * 0.8, max(ok.lift_mar_holdout.max(), ok.lift_feb.max()) * 1.2]
axes[1].plot(lim, lim, "k--", lw=0.8); axes[1].set_xscale("log"); axes[1].set_yscale("log")
axes[1].set_xlabel("lift tháng 3 (holdout)"); axes[1].set_ylabel("lift tháng 2")
axes[1].set_title(f"Luật kết hợp: lift hai tháng (r log-lift = {out['rule_log_lift_pearson']:.2f})")
axes[1].legend()
savefig(fig, "stability.png")
save_metrics("stability", out)
