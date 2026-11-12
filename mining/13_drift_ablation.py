"""Does calendar-driven drift explain the drop from in-month to out-of-time performance?

Hypothesis (from feature_drift.csv): some features shift with the calendar between the feb and
mar cohorts (PSI > 0.1), most of all days_since_first_tx, which is capped by the start of the
transaction file (01/2015). Test: train LightGBM on feb without the drifting features and compare
  * in-month AUC  : 80/20 split inside feb (same split for every variant)
  * out-of-time AUC: train on all feb, score mar
  * the gap between the two, with a paired bootstrap CI of the out-of-time difference.
If removing the drifting features narrows the gap without hurting out-of-time AUC, the hypothesis
is supported; if the gap stays, the drop has other causes (e.g. changed relation between
features and churn).

Outputs: report/drift_ablation.csv, metrics/drift_ablation.json
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CATEGORICAL, NUMERIC, REPORT, SEED, load, save_metrics

warnings.filterwarnings("ignore")
src = Path(__file__).with_name("03_classification.py").read_text(encoding="utf-8")
ns = {**globals(), "lgb": __import__("lightgbm")}
exec(src[src.index("def to_str"):src.index("ALL_NUM, CAT")], ns)
LGBM, as_cat = ns["LGBM"], ns["as_cat"]

drift = pd.read_csv(REPORT / "feature_drift.csv")
drifting = drift[(drift.psi > 0.1) & drift.feature.isin(NUMERIC)].feature.tolist()
top = drift.feature.iloc[0]
tr, te = as_cat(load("feb"), CATEGORICAL), as_cat(load("mar"), CATEGORICAL)
a, b = train_test_split(tr, test_size=0.2, stratify=tr.is_churn, random_state=SEED)
sets = {"Tất cả đặc trưng": NUMERIC,
        f"Bỏ {top}": [f for f in NUMERIC if f != top],
        f"Bỏ mọi đặc trưng PSI > 0,1 ({len(drifting)})": [f for f in NUMERIC if f not in drifting]}
rows, P = [], {}
for name, num in sets.items():
    p_in = LGBM(num, CATEGORICAL).fit(a, a.is_churn.values).predict_proba(b)[:, 1]
    p_oot = LGBM(num, CATEGORICAL).fit(tr, tr.is_churn.values).predict_proba(te)[:, 1]
    P[name] = p_oot
    rows.append({"feature_set": name, "n_features": len(num) + len(CATEGORICAL),
                 "auc_in_month": roc_auc_score(b.is_churn, p_in), "auc_out_of_time": roc_auc_score(te.is_churn, p_oot)})
    rows[-1]["gap"] = rows[-1]["auc_in_month"] - rows[-1]["auc_out_of_time"]
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()})
res = pd.DataFrame(rows)
y = te.is_churn.values
rng = np.random.default_rng(SEED)
base = P[next(iter(P))]
d = {k: [] for k in P}
for _ in range(100):
    i = rng.integers(0, len(y), len(y))
    a0 = roc_auc_score(y[i], base[i])
    for k, p in P.items():
        d[k].append(roc_auc_score(y[i], p[i]) - a0)
res["oot_delta_vs_all"] = res.auc_out_of_time - res.auc_out_of_time.iloc[0]
res["oot_delta_ci_low"] = [np.percentile(d[k], 2.5) for k in res.feature_set]
res["oot_delta_ci_high"] = [np.percentile(d[k], 97.5) for k in res.feature_set]
res.to_csv(REPORT / "drift_ablation.csv", index=False)
print(res.round(4).to_string(index=False))
save_metrics("drift_ablation", {"drifting_features": drifting, "results": res.to_dict("records")})
