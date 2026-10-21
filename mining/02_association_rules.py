"""FP-Growth on discretised user attributes; keep rules whose consequent is churn."""
import sys
from pathlib import Path

import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIG, load

MIN_SUPPORT, MIN_LIFT = 0.02, 1.2  # tune: lower support if too few rules

df = load()


def q(col, labels):
    return pd.qcut(df[col].rank(method="first"), len(labels), labels=[f"{col}={l}" for l in labels])


items = pd.DataFrame({
    "auto_renew": (df["auto_renew_rate"] > 0.5).map({True: "auto_renew=on", False: "auto_renew=off"}),
    "cancelled": (df["n_cancel"] > 0).map({True: "cancelled_before", False: "never_cancelled"}),
    "logs": df["has_logs"].map({1: "has_logs", 0: "no_logs"}),
    "via": "via=" + df["registered_via"].astype(str),
    "gender": "gender=" + df["gender"].astype(str),
    "listening": q("total_secs" if "total_secs" in df else "avg_secs", ["low", "mid", "high"]),
    "skip": q("skip_ratio", ["low", "mid", "high"]),
    "plan": pd.cut(df["avg_plan_days"], [-1, 0, 30, 10_000], labels=["plan=none", "plan=<=30d", "plan=>30d"]),
    "churn": df["is_churn"].map({1: "CHURN", 0: "stay"}),
})
onehot = pd.get_dummies(items.astype(str), prefix="", prefix_sep="").astype(bool)

freq = fpgrowth(onehot, min_support=MIN_SUPPORT, use_colnames=True)
rules = association_rules(freq, metric="lift", min_threshold=MIN_LIFT)
rules = rules[rules["consequents"].apply(lambda s: s == frozenset({"CHURN"}))]
rules = rules.sort_values("lift", ascending=False)
rules["antecedents"] = rules["antecedents"].apply(lambda s: " & ".join(sorted(s)))
out = rules[["antecedents", "support", "confidence", "lift"]]
print(out.head(20).round(3).to_string(index=False))
out.to_csv(FIG.parent / "churn_rules.csv", index=False)
