"""Cross-method validation.

1. Leakage audit (mar cohort, one fixed 70/30 split, same LightGBM settings): adds each
   post-cutoff source in turn to the clean feature set, so the AUC difference is attributable
   to that source alone (ablation, as opposed to comparing two different pipelines).
   Also shows how a single post-cutoff item ("no transaction in March") produces a
   near-deterministic association rule, the main artefact of the v1 results.
2. Algorithm comparison table across techniques from their metrics/*.json.

Outputs: report/leakage_audit.csv, algorithm_comparison.csv, figures/leakage_audit.png,
metrics/validation.json
"""
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CATEGORICAL, DATA, METRICS, NUMERIC, REPORT, SEED, load, save_metrics, savefig

warnings.filterwarnings("ignore")
# reuse the LightGBM wrapper without re-running the classification script
src = Path(__file__).with_name("03_classification.py").read_text(encoding="utf-8")
ns = {**globals(), "lgb": __import__("lightgbm")}
exec(src[src.index("def to_str"):src.index("ALL_NUM, CAT")], ns)
LGBM = ns["LGBM"]

out = {}
leak_file = DATA / "leakage_mar.parquet"
if leak_file.exists():
    df = load("mar").merge(pd.read_parquet(leak_file), on="msno", how="left")
    for c in CATEGORICAL:
        df[c] = df[c].astype("object").where(df[c].notna(), "NA").astype(str)
    tr, te = train_test_split(df, test_size=0.3, stratify=df.is_churn, random_state=SEED)
    POST_LOG = ["post_log_active_days", "post_log_hours", "post_log_days_idle"]
    POST_TX = ["post_tx_n", "post_tx_any_cancel", "post_tx_auto_renew"]
    V1 = ["v1_n_tx", "v1_n_cancel", "v1_auto_renew_rate", "v1_avg_plan_days", "v1_avg_paid"]
    sets = {
        "clean (cutoff T)": (NUMERIC, CATEGORICAL),
        "clean + March listening logs": (NUMERIC + POST_LOG, CATEGORICAL),
        "clean + March transactions": (NUMERIC + POST_TX, CATEGORICAL),
        "v1 recipe (unfiltered *_v2 files)": (V1 + POST_LOG, ["registered_via", "gender"]),
        "listening only, clean": ([c for c in NUMERIC if c in
                                   ["active_days", "total_hours", "listen_trend", "days_since_last_listen",
                                    "active_days_last14", "skip_ratio", "complete_ratio", "weekly_cv"]], []),
        "listening only, March logs (v1)": (POST_LOG, []),
    }
    rows, P = [], {}
    for name, (num, cat) in sets.items():
        p = LGBM(num, cat).fit(tr, tr.is_churn.values).predict_proba(te)[:, 1]
        P[name] = p
        rows.append({"feature_set": name, "roc_auc": roc_auc_score(te.is_churn, p),
                     "pr_auc": average_precision_score(te.is_churn, p)})
        print(f"{name:38s} AUC={rows[-1]['roc_auc']:.4f} PR-AUC={rows[-1]['pr_auc']:.4f}")
    audit = pd.DataFrame(rows)
    audit["auc_inflation_vs_clean"] = audit.roc_auc - audit.roc_auc.iloc[0]
    # paired bootstrap CI of the AUC difference vs the clean model (same resampled users)
    y = te.is_churn.values
    rng = np.random.default_rng(SEED)
    clean = P[audit.feature_set.iloc[0]]
    diffs = {k: [] for k in P}
    for _ in range(100):
        i = rng.integers(0, len(y), len(y))
        a0 = roc_auc_score(y[i], clean[i])
        for k, p in P.items():
            diffs[k].append(roc_auc_score(y[i], p[i]) - a0)
    audit["delta_ci_low"] = [np.percentile(diffs[k], 2.5) for k in audit.feature_set]
    audit["delta_ci_high"] = [np.percentile(diffs[k], 97.5) for k in audit.feature_set]
    audit.to_csv(REPORT / "leakage_audit.csv", index=False)

    # the deterministic rule behind v1's top associations
    no_post = df.post_tx_n == 0
    rule = {"support": float(no_post.mean()), "confidence": float(df.is_churn[no_post].mean()),
            "lift": float(df.is_churn[no_post].mean() / df.is_churn.mean()),
            "share_of_churners_covered": float(no_post[df.is_churn == 1].mean())}
    print("rule 'no March transaction -> churn':", {k: round(v, 3) for k, v in rule.items()})
    out["leakage_audit"] = audit.to_dict("records"); out["leaky_rule_no_march_tx"] = rule

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(audit.feature_set[::-1], audit.roc_auc[::-1],
            color=["#55A868" if "clean" in s and "+" not in s else "#C44E52" for s in audit.feature_set[::-1]])
    for i, v in enumerate(audit.roc_auc[::-1]):
        ax.text(v, i, f" {v:.3f}", va="center")
    ax.set_xlim(0.5, 1.02); ax.set_xlabel("ROC-AUC (cùng người dùng, cùng phép chia)")
    ax.set_title("Audit rò rỉ: xanh = thông tin có tại cutoff, đỏ = dùng dữ liệu tháng nhãn")
    savefig(fig, "leakage_audit.png")
else:
    print("leakage_mar.parquet missing: run data_prep/03_build_leakage_audit.py")

# ---- algorithm comparison ------------------------------------------------------------------


def m(name):
    p = METRICS / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


cl, ru, cf, sq = m("clustering"), m("rules"), m("classification"), m("sequential")
best = next((r for r in cf.get("out_of_time", []) if r["model"] == cf.get("best_model")), {})
alt_name = "LightGBM" if cf.get("best_model") == "Logistic regression" else "Logistic regression"
lr = next((r for r in cf.get("out_of_time", []) if r["model"] == alt_name), {})
rows = [
    {"technique": f"K-Means (k={cl.get('k')})", "type": "unsupervised / clustering",
     "quality": f"silhouette {cl.get('silhouette', np.nan):.3f}, DB {cl.get('davies_bouldin', np.nan):.2f}, "
                f"bootstrap ARI {cl.get('bootstrap_ari_mean', np.nan):.2f}",
     "runtime_s": cl.get("runtime_s"), "interpretability": "high (centroid profiles)",
     "business_use": "segment-level strategy, campaign targeting"},
    {"technique": "FP-Growth association rules", "type": "pattern-based",
     "quality": f"{ru.get('n_churn_rules')} productive churn rules (q<0.01, lift>=1.5), "
                f"holdout lift corr {ru.get('holdout_lift_corr') or np.nan:.2f}",
     "runtime_s": ru.get("runtime_s"), "interpretability": "very high (if-then rules)",
     "business_use": "trigger conditions for retention actions"},
    {"technique": "PrefixSpan sequential patterns", "type": "pattern-based / sequential",
     "quality": f"{sq.get('n_patterns', sq.get('selected_patterns'))} significant patterns",
     "runtime_s": sq.get("runtime_s"), "interpretability": "high (ordered event chains)",
     "business_use": "early-warning journeys before churn"},
    {"technique": f"{cf.get('best_model')} (supporting)", "type": "supervised / classification",
     "quality": f"out-of-time ROC-AUC {best.get('roc_auc', np.nan):.3f}, PR-AUC {best.get('pr_auc', np.nan):.3f}, "
                f"lift@10% {best.get('lift@10%', np.nan):.2f}",
     "runtime_s": best.get("fit_s"),
     "interpretability": "high (coefficients)" if cf.get("best_model") == "Logistic regression" else "medium (SHAP needed)",
     "business_use": "ranking users for a limited retention budget"},
    {"technique": f"{alt_name} (supporting)", "type": "supervised / classification",
     "quality": f"out-of-time ROC-AUC {lr.get('roc_auc', np.nan):.3f}, PR-AUC {lr.get('pr_auc', np.nan):.3f}",
     "runtime_s": lr.get("fit_s"),
     "interpretability": "high (coefficients)" if alt_name == "Logistic regression" else "medium (SHAP needed)",
     "business_use": "transparent baseline" if alt_name == "Logistic regression" else "non-linear benchmark"},
]
comp = pd.DataFrame(rows)
comp.to_csv(REPORT / "algorithm_comparison.csv", index=False)
print(comp.to_string(index=False))
out["comparison"] = rows
save_metrics("validation", out)
