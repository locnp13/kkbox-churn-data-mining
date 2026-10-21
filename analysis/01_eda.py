"""EDA: churn rate by group and distributions. Figures go to report/figures/."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mining"))
from common import FIG, load

df = load()
print(df.describe().T.to_string())
print("missing rate:\n", df.isna().mean().round(3).to_string())

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, col in zip(axes.flat, ["auto_renew_rate", "registered_via", "gender", "has_logs"]):
    df.groupby(col)["is_churn"].mean().plot.bar(ax=ax, title=f"churn rate by {col}")
plt.tight_layout()
plt.savefig(FIG / "eda_churn_by_group.png", dpi=120)

df["tenure_bin"] = (df["tenure_days"] // 365).clip(0, 8)
df.groupby("tenure_bin")["is_churn"].mean().plot.bar(title="churn rate by tenure (years)")
plt.savefig(FIG / "eda_churn_by_tenure.png", dpi=120)
