"""Sensitivity of the segmentation to feature design (users with listening logs, mar cohort).

Variants, all K-Means with the same k-selection rule as 01_clustering.py:
  A  baseline     16 features, equal weight (as in 01_clustering.py)
  B  group-weight each feature group (subscription / listening / tenure) gets the same total
                  variance: features scaled by 1/sqrt(group size), so the 8 listening features do
                  not outvote the 7 subscription ones
  C  reduced      drop one of every pair with |Spearman| > 0.8 (redundant features)
  D  mixed        B + one-hot registration channel and payment method as a fourth, equally
                  weighted group (a k-prototypes-style use of categorical attributes)
Silhouette is measured in each variant's own space, so it is comparable across k but only
roughly across variants; stability (bootstrap ARI), churn separation (post hoc) and agreement
with A are comparable.

Outputs: report/cluster_variants.csv, report/cluster_variants_k.csv, figures/cluster_variants.png,
metrics/cluster_variants.json
"""
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPORT, SEED, load, save_metrics, savefig
from transforms import FEATS, prepare

N = int(os.environ.get("VARIANT_SAMPLE", 100_000))
B = int(os.environ.get("CLUSTER_BOOT", 3))
SIL_TOL, ARI_MIN = float(os.environ.get("SIL_TOL", 0.75)), float(os.environ.get("ARI_MIN", 0.80))
K_RANGE = range(2, 9)
GROUPS = {
    "subscription": ["n_tx", "n_cancel", "auto_renew_rate", "discount_rate", "last_plan_days", "last_paid", "days_to_expire"],
    "tenure": ["tenure_days"],
    "listening": ["active_days", "total_hours", "complete_ratio", "skip_ratio", "avg_unique_songs", "listen_trend",
                  "days_since_last_listen", "weekly_cv"],
}
t0 = time.time()
rng = np.random.default_rng(SEED)
mar = load("mar")
df = mar[mar.has_logs == 1].sample(min(N, int((mar.has_logs == 1).sum())), random_state=SEED).reset_index(drop=True)
y = df.is_churn.values
R = prepare(df)
R = R.fillna(R.median())
Z = pd.DataFrame(StandardScaler().fit_transform(R), columns=FEATS)


def weighted(Zs, groups):
    out = []
    for cols in groups.values():
        out.append(Zs[cols].values / np.sqrt(len(cols)))
    return np.hstack(out)


# C: redundancy pruning, greedy in FEATS order
corr = R.corr(method="spearman").abs()
keep = []
for f in FEATS:
    if all(corr.loc[f, g] <= 0.8 for g in keep):
        keep.append(f)
dropped = [f for f in FEATS if f not in keep]

# D: categorical groups (top levels, rest -> "other")
cat = pd.DataFrame(index=df.index)
for c, top in [("registered_via", 6), ("last_payment_method", 8)]:
    v = df[c].astype("object").where(df[c].notna(), "NA").astype(str)
    lv = v.value_counts().index[:top]
    cat[c] = v.where(v.isin(lv), "other")
oh = pd.get_dummies(cat).astype(float)
oh = (oh - oh.mean()) / oh.std().replace(0, 1)

SPACES = {
    "A. baseline (16 biến, trọng số bằng nhau)": Z.values,
    "B. trọng số theo nhóm biến": weighted(Z, GROUPS),
    f"C. bỏ biến trùng ({len(dropped)} biến)": Z[keep].values,
    "D. B + kênh đăng ký và phương thức thanh toán": np.hstack([weighted(Z, GROUPS), oh.values / np.sqrt(oh.shape[1])]),
}
sil_idx = rng.choice(len(df), min(8_000, len(df)), replace=False)
ref_idx = rng.choice(len(df), min(15_000, len(df)), replace=False)
boots = [rng.choice(len(df), len(df), replace=True) for _ in range(B)]

rows_k, rows, labels = [], [], {}
for name, X in SPACES.items():
    best = None
    per_k = []
    for k in K_RANGE:
        km = KMeans(k, n_init=4, random_state=SEED).fit(X)
        ref = km.predict(X[ref_idx])
        ari = np.mean([adjusted_rand_score(ref, KMeans(k, n_init=2, random_state=SEED + b + 1).fit(X[bi]).predict(X[ref_idx]))
                       for b, bi in enumerate(boots)])
        r = {"variant": name, "k": k, "silhouette": silhouette_score(X[sil_idx], km.labels_[sil_idx]),
             "davies_bouldin": davies_bouldin_score(X[sil_idx], km.labels_[sil_idx]), "stability_ari": ari,
             "min_cluster_share": np.bincount(km.labels_).min() / len(km.labels_), "_km": km}
        per_k.append(r)
    pk = pd.DataFrame(per_k)
    ok = pk[(pk.k >= 3) & (pk.min_cluster_share >= 0.02)]
    ok = ok if len(ok) else pk
    macro = ok.loc[ok.silhouette.idxmax()]
    fine = ok[(ok.silhouette >= SIL_TOL * macro.silhouette) & (ok.stability_ari >= ARI_MIN)]
    chosen = fine.loc[fine.k.idxmax()] if len(fine) else macro
    lab = chosen["_km"].labels_
    labels[name] = lab
    churn_by = pd.Series(y).groupby(lab).mean()
    plan_by = df.groupby(lab).last_plan_days.mean()
    rows.append({"variant": name, "k": int(chosen.k), "silhouette": chosen.silhouette,
                 "davies_bouldin": chosen.davies_bouldin, "stability_ari": chosen.stability_ari,
                 "met_rule": bool(len(fine)), "churn_spread": churn_by.max() - churn_by.min(),
                 "nmi_with_churn": normalized_mutual_info_score(y, lab),
                 "long_plan_segment_found": bool((plan_by > 60).any()),
                 "churn_in_riskiest": churn_by.max(), "share_riskiest": float((lab == churn_by.idxmax()).mean())})
    rows_k += [{k_: v for k_, v in r.items() if k_ != "_km"} for r in per_k]
    print(f"{name}: k={int(chosen.k)} sil={chosen.silhouette:.3f} ARI={chosen.stability_ari:.3f} "
          f"spread={rows[-1]['churn_spread']:.3f}")
res = pd.DataFrame(rows)
base_lab = labels[next(iter(SPACES))]
res["ari_vs_A"] = [adjusted_rand_score(base_lab, labels[v]) for v in res.variant]
res.to_csv(REPORT / "cluster_variants.csv", index=False)
pd.DataFrame(rows_k).to_csv(REPORT / "cluster_variants_k.csv", index=False)
print(res.round(3).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
kd = pd.DataFrame(rows_k)
for name, g in kd.groupby("variant", sort=False):
    axes[0].plot(g.k, g.silhouette, marker="o", label=name.split(" (")[0])
    axes[1].plot(g.k, g.stability_ari, marker="o", label=name.split(" (")[0])
axes[0].set_title("Silhouette theo số cụm"); axes[1].set_title("Độ ổn định (ARI bootstrap) theo số cụm")
axes[1].axhline(ARI_MIN, color="grey", ls=":")
for a in axes:
    a.set_xlabel("số cụm"); a.legend(fontsize=7)
savefig(fig, "cluster_variants.png")

save_metrics("cluster_variants", {"n_users": len(df), "dropped_redundant": dropped, "kept": keep,
                                  "n_onehot": int(oh.shape[1]), "results": res.to_dict("records"),
                                  "runtime_s": time.time() - t0})
