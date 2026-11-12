"""Behavioural segmentation (unsupervised) on the mar cohort; churn is used only to profile.

Two-stage design:
  * Users with no listening log in the 60-day window (about 19 %) form a rule-based segment.
    Giving them constant "inactive" values and letting K-Means find them produced an artificial
    cluster (it only rediscovered the has_logs flag), so they are separated explicitly.
  * K-Means is fitted on users WITH logs only: subscription + listening features (redundant ones
    with |Spearman| > 0.8 dropped, 13 of 16 kept on the real data), log1p for
    heavy tails, median imputation, standardisation. The churn label is never used to fit or to
    choose the model.

Choice of k uses internal criteria only:
  * macro k  = best silhouette (clusters >= 2 % of users);
  * chosen k = the largest k (<= K_MAX = 7, so at most 8 segments in total) whose silhouette is >= SIL_TOL x best AND whose bootstrap stability
    (mean ARI over B refits of the whole pipeline) is >= ARI_MIN, clusters >= 2 % of users.
Stability: ARI per k, ARI and per-cluster Jaccard for the chosen k. Alternatives (Ward, GMM,
HDBSCAN) on a common subsample. The heat map shows centroids in the standardised space K-Means uses.

Outputs: report/cluster_quality.csv, cluster_algorithms.csv, cluster_profile*.csv, cluster_jaccard.csv,
cluster_zscores.csv, cluster_nesting.csv, figures/cluster_*.png, metrics/clustering.json,
data/clusters_mar.parquet, data/cluster_model.joblib
"""
import json
import os
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, REPORT, SEED, churn_table, load, save_metrics, savefig
from transforms import FEATS, prepare

N = int(os.environ.get("CLUSTER_SAMPLE", 200_000))
N_ALT = int(os.environ.get("CLUSTER_ALT_SAMPLE", 15_000))  # O(n^2) algorithms
B = int(os.environ.get("CLUSTER_BOOT", 5))                 # bootstrap refits per k
SIL_TOL = float(os.environ.get("SIL_TOL", 0.75))
ARI_MIN = float(os.environ.get("ARI_MIN", 0.80))
K_RANGE = range(2, 11)
t0 = time.time()
rng = np.random.default_rng(SEED)

full = load("mar")
loggers = full[full.has_logs == 1]
df = loggers.sample(min(N, len(loggers)), random_state=SEED).reset_index(drop=True)
R = prepare(df)
# drop redundant features: greedy in FEATS order, drop a feature whose |Spearman| with an already
# kept one exceeds 0.8 (decided on the features only; see 11_clustering_variants.py, variant C)
corr = R.fillna(R.median()).corr(method="spearman").abs()
CLF = []
for f in FEATS:
    if all(corr.loc[f, g] <= 0.8 for g in CLF):
        CLF.append(f)
DROPPED = [f for f in FEATS if f not in CLF]
print("features used:", len(CLF), "dropped as redundant:", DROPPED)
R = R[CLF]
MED = R.median()
scaler = StandardScaler().fit(R.fillna(MED))
X = scaler.transform(R.fillna(MED))
sil_idx = rng.choice(len(X), min(10_000, len(X)), replace=False)
ref_idx = rng.choice(len(X), min(20_000, len(X)), replace=False)
boots = [rng.choice(len(R), len(R), replace=True) for _ in range(B)]


def refit(k, bi, seed):
    """Refit imputer medians, scaler and K-Means on a bootstrap sample; label the reference set."""
    Rb = R.iloc[bi]
    med_b = Rb.median()
    sc_b = StandardScaler().fit(Rb.fillna(med_b))
    kb = KMeans(k, n_init=3, random_state=seed).fit(sc_b.transform(Rb.fillna(med_b)))
    return kb.predict(sc_b.transform(R.iloc[ref_idx].fillna(med_b)))


# ---- 1. internal quality and stability for every k -------------------------------------------
rows, models = [], {}
for k in K_RANGE:
    km = KMeans(k, n_init=5, random_state=SEED).fit(X)
    models[k] = km
    lab = km.labels_
    ref = km.predict(X[ref_idx])
    aris = [adjusted_rand_score(ref, refit(k, bi, SEED + b + 1)) for b, bi in enumerate(boots)]
    rows.append({"k": k, "inertia": km.inertia_,
                 "silhouette": silhouette_score(X[sil_idx], lab[sil_idx]),
                 "davies_bouldin": davies_bouldin_score(X[sil_idx], lab[sil_idx]),
                 "calinski_harabasz": calinski_harabasz_score(X[sil_idx], lab[sil_idx]),
                 "min_cluster_share": np.bincount(lab).min() / len(lab),
                 "stability_ari": np.mean(aris), "stability_ari_sd": np.std(aris)})
    print(f"k={k} sil={rows[-1]['silhouette']:.3f} DB={rows[-1]['davies_bouldin']:.3f} "
          f"ARI={rows[-1]['stability_ari']:.3f}")
quality = pd.DataFrame(rows)
K_MAX = int(os.environ.get("K_MAX", 7))   # at most 7 clusters + the rule-based segment (usability, aspect 7)
ok = quality[(quality.k >= 3) & (quality.k <= K_MAX) & (quality.min_cluster_share >= 0.02)]
ok = ok if len(ok) else quality
macro = ok.loc[ok.silhouette.idxmax()]
K_MACRO = int(macro.k)
fine = ok[(ok.silhouette >= SIL_TOL * macro.silhouette) & (ok.stability_ari >= ARI_MIN)]
best = fine.loc[fine.k.idxmax()] if len(fine) else macro
if os.environ.get("CLUSTER_K"):
    best = quality.loc[quality.k == int(os.environ["CLUSTER_K"])].iloc[0]
K = int(best.k)
NOLOG = K                      # id of the rule-based segment
km = models[K]
quality["chosen"] = np.where(quality.k == K, "chosen", np.where(quality.k == K_MACRO, "best silhouette", ""))
quality.to_csv(REPORT / "cluster_quality.csv", index=False)
print(f"best-silhouette k={K_MACRO} ({macro.silhouette:.3f}); chosen k={K} "
      f"(silhouette {best.silhouette:.3f}, stability ARI {best.stability_ari:.3f})")

# ---- 2. per-cluster stability (Jaccard) for the chosen k ----------------------------------------
ref = km.predict(X[ref_idx])
jac = np.zeros((B, K))
for b, bi in enumerate(boots):
    lb = refit(K, bi, SEED + 100 + b)
    for c in range(K):
        A = ref == c
        jac[b, c] = max(((A & (lb == j)).sum() / (A | (lb == j)).sum()) for j in range(K))
jaccard = pd.DataFrame({"cluster": range(K), "jaccard_mean": jac.mean(0), "jaccard_min": jac.min(0)})
jaccard.to_csv(REPORT / "cluster_jaccard.csv", index=False)
print(jaccard.round(3).to_string(index=False))

fig, axes = plt.subplots(1, 5, figsize=(20, 3.6))
for ax, m in zip(axes, ["inertia", "silhouette", "davies_bouldin", "calinski_harabasz", "stability_ari"]):
    ax.plot(quality.k, quality[m], marker="o"); ax.axvline(K, color="crimson", ls="--")
    ax.axvline(K_MACRO, color="grey", ls=":")
    ax.set_title(m + (" (càng thấp càng tốt)" if m in ("inertia", "davies_bouldin") else " (càng cao càng tốt)"),
                 fontsize=10)
    ax.set_xlabel("k")
savefig(fig, "cluster_selection.png")

# ---- 3. alternative algorithms on a shared subsample -----------------------------------------
alt_idx = rng.choice(len(X), min(N_ALT, len(X)), replace=False)
Xa = X[alt_idx]
algos = {
    "K-Means": lambda: km.predict(Xa),
    "Agglomerative (Ward)": lambda: AgglomerativeClustering(K, linkage="ward").fit_predict(Xa),
    "Gaussian mixture": lambda: GaussianMixture(K, covariance_type="full", random_state=SEED).fit(Xa).predict(Xa),
    "HDBSCAN": lambda: HDBSCAN(min_cluster_size=max(50, len(Xa) // 100)).fit_predict(Xa),
}
alt_rows, alt_labels = [], {}
y_alt = df.is_churn.values[alt_idx]
for name, fn in algos.items():
    s = time.time(); lab = fn(); el = time.time() - s
    alt_labels[name] = lab
    m = lab >= 0                       # HDBSCAN marks noise as -1
    n_cl = len(set(lab[m]))
    good = n_cl >= 2
    alt_rows.append({"algorithm": name, "n_clusters": n_cl, "noise_share": 1 - m.mean(),
                     "silhouette": silhouette_score(Xa[m], lab[m]) if good else np.nan,
                     "davies_bouldin": davies_bouldin_score(Xa[m], lab[m]) if good else np.nan,
                     "ari_vs_kmeans": adjusted_rand_score(alt_labels["K-Means"][m], lab[m]),
                     "churn_rate_spread": pd.Series(y_alt[m]).groupby(lab[m]).mean().agg(lambda v: v.max() - v.min()),
                     "runtime_s": el})
algos_df = pd.DataFrame(alt_rows)
algos_df.to_csv(REPORT / "cluster_algorithms.csv", index=False)
print(algos_df.round(3).to_string(index=False))

# ---- 4. centroids, names ---------------------------------------------------------------------
z = pd.DataFrame(km.cluster_centers_, columns=CLF)        # standardised space used by K-Means
LABELS = {
    ("active_days", 1): "nghe nhiều", ("active_days", -1): "ít ngày nghe",
    ("auto_renew_rate", 1): "tự gia hạn", ("auto_renew_rate", -1): "trả thủ công",
    ("n_cancel", 1): "từng hủy", ("last_plan_days", 1): "gói dài hạn",
    ("discount_rate", 1): "hay được giảm giá", ("tenure_days", 1): "lâu năm", ("tenure_days", -1): "mới",
    ("listen_trend", -1): "nghe giảm dần", ("listen_trend", 1): "nghe tăng",
    ("days_since_last_listen", 1): "lâu chưa nghe", ("skip_ratio", 1): "hay bỏ qua bài",
    ("n_tx", 1): "giao dịch nhiều", ("n_tx", -1): "ít giao dịch", ("days_to_expire", 1): "còn hạn dài",
    ("weekly_cv", 1): "nghe thất thường", ("total_hours", 1): "nghe nhiều giờ",
    ("complete_ratio", 1): "nghe hết bài", ("complete_ratio", -1): "ít nghe hết bài",
    ("last_paid", 1): "trả giá cao",
}
names = {}
for c in z.index:
    tags = []
    for f in z.loc[c].abs().sort_values(ascending=False).index:
        key = (f, int(np.sign(z.loc[c, f])))
        if key in LABELS and abs(z.loc[c, f]) > 0.35 and LABELS[key] not in tags:
            tags.append(LABELS[key])
        if len(tags) == 3:
            break
    names[c] = ", ".join(tags) if tags else "trung bình"
names[NOLOG] = "Không có nhật ký nghe (tách theo quy tắc)"
nf = REPORT / "segment_names.json"   # analyst-assigned business names, written after reading z-scores
if nf.exists():
    manual = {int(k): v for k, v in json.loads(nf.read_text(encoding="utf-8")).items()}
    if set(manual) == set(names):
        names = manual
z.index.name = "cluster"
z.to_csv(REPORT / "cluster_zscores.csv")

# ---- 5. assign every user of the cohort ------------------------------------------------------
Xfull = scaler.transform(prepare(full)[CLF].fillna(MED))
lab_full = np.where(full.has_logs == 1, km.predict(Xfull), NOLOG)
km_macro = models[K_MACRO]
macro_full = np.where(full.has_logs == 1, km_macro.predict(Xfull), K_MACRO)
pd.DataFrame({"msno": full.msno, "cluster": lab_full, "cluster_macro": macro_full}).to_parquet(
    DATA / "clusters_mar.parquet")
joblib.dump({"scaler": scaler, "km": km, "median": MED, "names": names, "nolog_id": NOLOG, "features": CLF},
            DATA / "cluster_model.joblib")
fa = full.assign(cluster=lab_full)
base = full.is_churn.mean()
prof = churn_table(fa, "cluster")
prof.insert(0, "name", pd.Series(names))
prof["share"] = prof.n / prof.n.sum()
prof = prof.join(fa.groupby("cluster")[FEATS].mean())
prof.to_csv(REPORT / "cluster_profile_full.csv")
pd.crosstab(pd.Series(lab_full, name="chosen"), pd.Series(macro_full, name="macro"), normalize="index") \
  .round(3).to_csv(REPORT / "cluster_nesting.csv")
churn_table(full.assign(cluster_macro=macro_full), "cluster_macro").to_csv(REPORT / "cluster_profile_macro.csv")
print("full cohort:", prof[["name", "n", "share", "churn_rate", "lift"]].round(3).to_string(), sep="\n")

dd = fa.assign(auto_renew=fa.last_auto_renew.map({1: "on", 0: "off"}).fillna("no tx"))
drill = dd.groupby(["cluster", "auto_renew"])["is_churn"].agg(n="size", churn_rate="mean").reset_index()
drill[drill.n >= 100].to_csv(REPORT / "cluster_drilldown.csv", index=False)

# ---- 6. figures -------------------------------------------------------------------------
pts = PCA(2, random_state=SEED).fit_transform(Xa)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
lab_a = km.predict(Xa)
for c in range(K):
    m = lab_a == c
    axes[0].scatter(pts[m, 0], pts[m, 1], s=3, alpha=0.5, label=f"{c}: {names[c]}")
axes[0].legend(markerscale=4, fontsize=7); axes[0].set_title(f"K-Means k={K} trên không gian PCA (khách có nhật ký)")
axes[1].scatter(pts[:, 0], pts[:, 1], c=y_alt, s=3, cmap="coolwarm", alpha=0.5)
axes[1].set_title("cùng các điểm, tô màu theo churn (đỏ)")
savefig(fig, "clusters_pca.png")

fig, ax = plt.subplots(figsize=(9.5, 0.55 * (K + 1) + 1.5))
order = prof.sort_values("churn_rate").index
ax.barh([f"{c}: {names[c]}" for c in order], prof.loc[order, "churn_rate"],
        xerr=[prof.loc[order, "churn_rate"] - prof.loc[order, "ci_low"],
              prof.loc[order, "ci_high"] - prof.loc[order, "churn_rate"]],
        color=["#B0B7C3" if c == NOLOG else "#4C72B0" for c in order])
ax.axvline(base, color="crimson", ls="--", label=f"mức chung {base:.1%}")
for i, c in enumerate(order):
    ax.text(prof.loc[c, "churn_rate"], i, f"  {prof.loc[c, 'churn_rate']:.1%} (n={prof.loc[c, 'n']:,})", va="center", fontsize=8)
ax.set_xlabel("tỉ lệ rời bỏ (CI 95%)"); ax.legend()
ax.set_title("Tỉ lệ rời bỏ theo nhóm (xám: nhóm tách theo quy tắc)")
savefig(fig, "cluster_churn.png")

fig, ax = plt.subplots(figsize=(11, 0.6 * K + 2))
im = ax.imshow(z.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
ax.set_xticks(range(len(CLF)), CLF, rotation=60, ha="right", fontsize=8)
ax.set_yticks(range(K), [f"{c}: {names[c]}" for c in z.index], fontsize=8)
fig.colorbar(im)
ax.set_title("Tâm cụm trong không gian đã chuẩn hoá (đơn vị: độ lệch chuẩn)")
savefig(fig, "cluster_heatmap.png")

save_metrics("clustering", {
    "features": CLF, "dropped_redundant": DROPPED,
    "n_users": len(df), "n_loggers": len(loggers), "n_nolog": int((full.has_logs == 0).sum()),
    "k": K, "k_macro": K_MACRO, "nolog_id": NOLOG, "silhouette_macro": macro.silhouette,
    "silhouette": best.silhouette, "davies_bouldin": best.davies_bouldin,
    "calinski_harabasz": best.calinski_harabasz, "stability_ari": best.stability_ari,
    "stability_ari_sd": best.stability_ari_sd, "jaccard_min_cluster": float(jaccard.jaccard_mean.min()),
    "sil_tol": SIL_TOL, "ari_min": ARI_MIN, "k_max": K_MAX, "base_churn": base, "names": names,
    "algorithms": algos_df.to_dict("records"), "runtime_s": time.time() - t0,
})
