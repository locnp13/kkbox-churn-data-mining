"""K-Means on behaviour features; pick k by silhouette, then profile clusters vs churn."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIG, NUM, load

SEED = 42
df = load()
X = np.log1p(df[[c for c in NUM if c not in ("age",)]].clip(lower=0))
X = StandardScaler().fit_transform(X)

scores = {}
for k in range(2, 9):
    labels = KMeans(k, n_init=10, random_state=SEED).fit_predict(X)
    scores[k] = silhouette_score(X, labels, sample_size=min(20000, len(X)), random_state=SEED)
    print(f"k={k} silhouette={scores[k]:.3f}")
best_k = max(scores, key=scores.get)

km = KMeans(best_k, n_init=10, random_state=SEED).fit(X)
df["cluster"] = km.labels_
print(f"\nbest k={best_k}")
# is_churn here is from a class-balanced sample: compare clusters relatively, not as absolute rates
profile = df.groupby("cluster").agg(size=("msno", "size"), churn_rate=("is_churn", "mean"),
                                    **{c: (c, "mean") for c in NUM})
print(profile.round(3).to_string())
profile.to_csv(FIG.parent / "cluster_profile.csv")

pts = PCA(2, random_state=SEED).fit_transform(X)
idx = np.random.default_rng(SEED).choice(len(X), min(10000, len(X)), replace=False)
plt.figure(figsize=(7, 6))
plt.scatter(pts[idx, 0], pts[idx, 1], c=km.labels_[idx], s=4, cmap="tab10")
plt.title(f"K-Means (k={best_k}) in PCA space")
plt.savefig(FIG / "clusters_pca.png", dpi=120)
