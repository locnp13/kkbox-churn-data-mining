"""Exploratory and descriptive analysis on the mar cohort (natural churn prior).

Outputs: report/figures/eda_*.png, report/eda_*.csv, report/metrics/eda.json
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mining"))
from common import FIG, NUMERIC, REPORT, churn_table, load, save_metrics, savefig

df = load("mar")
feb = load("feb")
base = df.is_churn.mean()
print(f"mar cohort: {len(df):,} users, churn {base:.3%}; feb cohort: {len(feb):,}, churn {feb.is_churn.mean():.3%}")

# ---- data quality: missing values as they come from data_prep (before any imputation) ----
missing = df[NUMERIC + ["gender", "city", "registered_via"]].isna().mean().sort_values(ascending=False)
missing = pd.concat([missing.rename("missing_rate"),
                     df[NUMERIC].describe().T[["mean", "50%", "min", "max"]]], axis=1)
missing.round(4).to_csv(REPORT / "eda_quality.csv")
print(missing.head(12).round(3).to_string())

# ---- univariate: churn vs stay distributions + Mann-Whitney effect size ------------------
rows = []
for c in NUMERIC:
    a, b = df.loc[df.is_churn == 1, c].dropna(), df.loc[df.is_churn == 0, c].dropna()
    if len(a) > 10 and len(b) > 10 and a.nunique() > 1:
        u, p = mannwhitneyu(a, b)
        rows.append((c, a.median(), b.median(), u / (len(a) * len(b)), p))
effects = pd.DataFrame(rows, columns=["feature", "median_churn", "median_stay", "auc_effect", "p_value"])
effects["abs_effect"] = (effects.auc_effect - 0.5).abs()
effects = effects.sort_values("abs_effect", ascending=False)
effects.to_csv(REPORT / "eda_effects.csv", index=False)
print("\nunivariate effect (AUC of single feature, 0.5 = none):\n", effects.head(12).round(3).to_string(index=False))

# ---- churn by group (with 95% CI) --------------------------------------------------------
d = df.copy()
d["auto_renew"] = d.last_auto_renew.map({1: "on", 0: "off"}).fillna("no tx")
d["cancelled_last"] = d.last_is_cancel.map({1: "yes", 0: "no"}).fillna("no tx")
d["tenure_y"] = (d.tenure_days // 365).clip(0, 10)
d["plan"] = pd.cut(d.last_plan_days, [-1, 0, 7, 30, 10_000], labels=["0", "1-7", "8-30", ">30"])
d["idle"] = pd.cut(d.days_since_last_listen, [-1, 0, 3, 7, 14, 60], labels=["0", "1-3", "4-7", "8-14", "15+"])
d.loc[d.has_logs == 0, "idle"] = np.nan
d["idle"] = d["idle"].cat.add_categories("no logs").fillna("no logs")
d["trend"] = pd.cut(d.listen_trend, [-99, -1, -0.25, 0.25, 1, 99],
                    labels=["falling fast", "falling", "flat", "rising", "rising fast"])
d["expire"] = pd.cut(d.days_to_expire, [-10_000, 0, 15, 31, 10_000], labels=["<=0", "1-15", "16-31", ">31"])
groups = {"auto_renew": None, "cancelled_last": None, "registered_via": None, "gender": None,
          "tenure_y": None, "plan": None, "idle": None, "trend": None, "expire": None}
tables, chi = [], {}
for col in groups:
    t = churn_table(d, col)
    t.insert(0, "feature", col)
    tables.append(t.reset_index().rename(columns={col: "level"}))
    ct = pd.crosstab(d[col], d.is_churn)
    if min(ct.shape) < 2:
        continue
    chi2, p, dof, _ = chi2_contingency(ct)
    chi[col] = {"chi2": chi2, "p": p, "cramers_v": float(np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1))))}
groups_df = pd.concat(tables)
groups_df["level"] = groups_df["level"].astype(str)
groups_df.to_csv(REPORT / "eda_churn_by_group.csv", index=False)
chi_df = pd.DataFrame(chi).T.sort_values("cramers_v", ascending=False)
chi_df.to_csv(REPORT / "eda_association.csv")
print("\nCramer's V with churn:\n", chi_df.round(3).to_string())

fig, axes = plt.subplots(3, 3, figsize=(14, 11))
for ax, col in zip(axes.flat, groups):
    t = groups_df[groups_df.feature == col]
    t = t[t.n >= 200]
    ax.bar(t.level, t.churn_rate, yerr=[t.churn_rate - t.ci_low, t.ci_high - t.churn_rate],
           color="#4C72B0", capsize=3)
    ax.axhline(base, color="crimson", ls="--", lw=1)
    ax.set_title(f"churn theo {col} (V={chi.get(col, {}).get('cramers_v', float('nan')):.2f})")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
fig.suptitle(f"Tỉ lệ churn theo nhóm, cohort tháng 3 (nét đứt = mức chung {base:.1%}, thanh lỗi = CI 95%)")
savefig(fig, "eda_churn_by_group.png")

# ---- distributions of key behaviours ------------------------------------------------------
keys = ["active_days", "total_hours", "listen_trend", "days_since_last_listen", "n_tx", "tenure_days"]
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, c in zip(axes.flat, keys):
    for y, col in [(0, "#4C72B0"), (1, "#DD8452")]:
        v = df.loc[df.is_churn == y, c].dropna()
        ax.hist(v.clip(upper=v.quantile(0.99)), bins=40, density=True, alpha=0.55, color=col,
                label="churn" if y else "ở lại")
    ax.set_title(c)
axes.flat[0].legend()
savefig(fig, "eda_distributions.png")

# ---- correlation -------------------------------------------------------------------------
corr = df[NUMERIC + ["is_churn"]].corr(method="spearman")
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
ax.set_yticks(range(len(corr)), corr.columns, fontsize=7)
fig.colorbar(im, shrink=0.7)
ax.set_title("Tương quan Spearman (cohort tháng 3)")
savefig(fig, "eda_correlation.png")
high = (corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack().abs().sort_values(ascending=False))

# ---- trend over tenure -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
t = churn_table(d, "tenure_y")
ax.errorbar(t.index, t.churn_rate, yerr=[t.churn_rate - t.ci_low, t.ci_high - t.churn_rate], marker="o")
ax.set_xlabel("thâm niên (năm, chặn ở 10)"); ax.set_ylabel("tỉ lệ churn")
ax.set_title("Tỉ lệ churn theo thâm niên")
savefig(fig, "eda_churn_by_tenure.png")

save_metrics("eda", {
    "n_mar": len(df), "churn_mar": base, "n_feb": len(feb), "churn_feb": feb.is_churn.mean(),
    "share_no_logs": 1 - df.has_logs.mean(), "share_no_tx": 1 - df.has_tx.mean(),
    "age_missing": df.age.isna().mean(), "gender_unknown": (df.gender == "unknown").mean(),
    "top_effects": effects.head(8)[["feature", "auc_effect"]].to_dict("records"),
    "cramers_v": chi_df.cramers_v.to_dict(),
    "top_correlated_pairs": [(a, b, v) for (a, b), v in high.head(6).items()],
})
