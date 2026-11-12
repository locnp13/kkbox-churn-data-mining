"""Association rules X -> CHURN (and protective X -> low churn) with FP-Growth, validated on holdout.

Design choices (and why):
* Items come from business thresholds or quantiles learnt on the TRAIN split only (no rank-based
  binning, which split ties arbitrarily and created spurious items in v1).
* FP-Growth enumerates frequent antecedents X (label excluded, max 3 items). For each X we
  compute confidence P(churn|X), lift, leverage and a one-sided binomial z-test against the base
  rate; p-values are corrected with Benjamini-Hochberg (thousands of hypotheses are tested).
* Productive-rule filter (Webb 2007): a rule is kept only if it beats every one of its sub-rules
  by at least MIN_IMPROVE in confidence (churn rules) or is lower by that much (protective).
* Every reported rule is re-measured on a 30 % holdout split never used for mining.
* Sensitivity: rule counts and top-10 stability across min_support values.

Outputs: report/churn_rules.csv, retention_rules.csv, rule_sensitivity.csv, metrics/rules.json,
figures/rules_scatter.png
"""
import os
import sys
import time
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from scipy.stats import norm
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, REPORT, SEED, bh_fdr, load, save_metrics, savefig
from transforms import itemise

MIN_SUPPORT = float(os.environ.get("MIN_SUPPORT", 0.01))   # of the antecedent X
MIN_IMPROVE = 0.02
MAX_LEN = 3

df = load("mar")
cl = DATA / "clusters_mar.parquet"
if cl.exists():  # segment label from clustering as an extra item (NaN for users outside the sample)
    df = df.merge(pd.read_parquet(cl), on="msno", how="left")
tr, ho = train_test_split(df, test_size=0.3, stratify=df.is_churn, random_state=SEED)


Itr, Iho = itemise(tr, tr), itemise(ho, tr)
Iho = Iho.reindex(columns=Itr.columns, fill_value=False)
ytr, yho = tr.is_churn.values, ho.is_churn.values
base = ytr.mean()


def evaluate(itemsets, I, y):
    out = []
    A = I.values
    col = {c: i for i, c in enumerate(I.columns)}
    for s in itemsets:
        m = A[:, [col[c] for c in s]].all(1)
        n = m.sum()
        k = y[m].sum()
        out.append((n, k))
    n, k = np.array(out).T if out else (np.array([]), np.array([]))
    return n, k


def mine(min_support):
    t = time.time()
    freq = fpgrowth(Itr, min_support=min_support, use_colnames=True, max_len=MAX_LEN)
    sets = [frozenset(s) for s in freq.itemsets]
    n, k = evaluate(sets, Itr, ytr)
    r = pd.DataFrame({"items": sets, "n": n, "churners": k})
    r["len"] = r["items"].map(len)
    r["support"] = r.n / len(Itr)
    r["rule_support"] = r.churners / len(Itr)
    r["confidence"] = r.churners / r.n
    r["lift"] = r.confidence / base
    r["leverage"] = r.rule_support - r.support * base
    r["conviction"] = (1 - base) / (1 - r.confidence).clip(lower=1e-9)
    # two-proportion z-test, X vs not-X (equivalent to the 2x2 chi-square test)
    N, K = len(Itr), ytr.sum()
    p1, p0 = r.confidence, (K - r.churners) / (N - r.n).clip(lower=1)
    pool = K / N
    zstat = (p1 - p0) / np.sqrt(pool * (1 - pool) * (1 / r.n + 1 / (N - r.n).clip(lower=1)))
    r["p_value"] = 2 * norm.sf(np.abs(zstat))
    r["q_value"] = bh_fdr(r.p_value)
    return r, time.time() - t


def productive(r, direction, conf):
    """Keep rules that improve on every sub-rule; ``conf`` covers ALL frequent itemsets."""
    keep = []
    for s, c in zip(r["items"], r.confidence):
        subs = [frozenset(x) for L in range(1, len(s)) for x in combinations(s, L)]
        subs = [conf[x] for x in subs if x in conf]
        if direction == "churn":
            keep.append(all(c >= sc + MIN_IMPROVE for sc in subs))
        else:
            keep.append(all(c <= sc - MIN_IMPROVE for sc in subs))
    return r[keep]


def select(r, direction, top=25):
    conf = dict(zip(r["items"], r.confidence))
    sig = r[r.q_value < 0.01]
    sig = sig[sig.lift >= 1.5] if direction == "churn" else sig[sig.lift <= 0.5]
    sig = productive(sig, direction, conf)
    return sig.sort_values("lift", ascending=direction != "churn").head(top)


rules, runtime = mine(MIN_SUPPORT)
print(f"{len(rules):,} frequent antecedents in {runtime:.1f}s, base churn {base:.3%}")
out = {}
for direction in ("churn", "retain"):
    s = select(rules, direction).copy()
    n, k = evaluate(list(s["items"]), Iho, yho)
    s["support_holdout"] = n / len(Iho)
    s["confidence_holdout"] = k / np.maximum(n, 1)
    s["lift_holdout"] = s.confidence_holdout / yho.mean()
    s["rule"] = s["items"].map(lambda x: " & ".join(sorted(x)))
    cols = ["rule", "len", "support", "confidence", "lift", "leverage", "conviction", "q_value",
            "support_holdout", "confidence_holdout", "lift_holdout"]
    out[direction] = s[cols]
    print(f"\n{direction} rules:\n", s[cols].head(12).round(3).to_string(index=False))
out["churn"].to_csv(REPORT / "churn_rules.csv", index=False)
out["retain"].to_csv(REPORT / "retention_rules.csv", index=False)

# ---- sensitivity to min_support -----------------------------------------------------------
top_ref = set(out["churn"].rule.head(10))
sens = []
for ms in [0.005, 0.01, 0.02, 0.05]:
    r, rt = mine(ms)
    s = select(r, "churn", top=10)
    names = {" & ".join(sorted(x)) for x in s["items"]}
    sens.append({"min_support": ms, "n_itemsets": len(r), "n_significant_churn_rules":
                 int(((r.q_value < 0.01) & (r.lift >= 1.5)).sum()), "n_productive_top": len(s),
                 "max_lift": s.lift.max() if len(s) else np.nan,
                 "jaccard_top10_vs_default": len(names & top_ref) / max(1, len(names | top_ref)),
                 "runtime_s": rt})
sens = pd.DataFrame(sens)
sens.to_csv(REPORT / "rule_sensitivity.csv", index=False)
print("\n", sens.round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(rules.support, rules.confidence, c=rules.lift, cmap="viridis", s=8, alpha=0.6)
ax.scatter(out["churn"].support, out["churn"].confidence, facecolors="none", edgecolors="red", s=40,
           label="luật churn được chọn")
ax.axhline(base, color="crimson", ls="--", lw=1, label=f"churn nền {base:.1%}")
ax.set_xscale("log"); ax.set_xlabel("support của vế trái X (thang log)"); ax.set_ylabel("P(churn | X)")
fig.colorbar(sc, label="lift"); ax.legend(); ax.set_title("Tập phổ biến: support và confidence churn")
savefig(fig, "rules_scatter.png")

c = out["churn"]
save_metrics("rules", {
    "n_train": len(Itr), "n_holdout": len(Iho), "n_items": Itr.shape[1], "base_churn": base,
    "min_support": MIN_SUPPORT, "n_itemsets": len(rules), "runtime_s": runtime,
    "n_churn_rules": len(c), "n_retention_rules": len(out["retain"]),
    "holdout_lift_corr": float(np.corrcoef(c.lift, c.lift_holdout)[0, 1]) if len(c) > 2 else None,
    "mean_abs_conf_drift": float((c.confidence - c.confidence_holdout).abs().mean()) if len(c) else None,
    "top_churn": c.head(5)[["rule", "confidence_holdout", "lift_holdout", "support"]].to_dict("records"),
    "top_retain": out["retain"].head(5)[["rule", "confidence_holdout", "lift_holdout", "support"]].to_dict("records"),
})
