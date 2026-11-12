"""Can the supporting classifier be improved? A controlled 2x2 experiment.

Factors: (a) payment-history features (new, from transactions <= cutoff only);
         (b) LightGBM hyper-parameters tuned with Optuna (TPE) vs the defaults of 03_classification.
Protocol: all tuning happens inside the feb cohort (80/20 split of feb, AUC on the 20 %); the mar
cohort is touched once per configuration for the final out-of-time score, so the comparison is
not tuned on the test month. Paired bootstrap CI of the AUC gain vs the baseline.

Outputs: report/model_improvement.csv, model_tuning_trials.csv, metrics/model_improvement.json,
figures/model_improvement.png
"""
import os
import sys
import time
import warnings
from pathlib import Path

import duckdb
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CATEGORICAL, DATA, NUMERIC, REPORT, SEED, load, save_metrics, savefig

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
N_TRIALS = int(os.environ.get("N_TRIALS", 30))
TUNE_SAMPLE = int(os.environ.get("TUNE_SAMPLE", 400_000))
t0 = time.time()

ORDER = ("transaction_date, membership_expire_date, is_cancel, payment_method_id, payment_plan_days, "
         "plan_list_price, actual_amount_paid, is_auto_renew")


def payment_history(cohort):
    """Payment-history features from transactions_<cohort>.parquet (already cut at the cohort's T)."""
    f = DATA / f"transactions_{cohort}.parquet"
    return duckdb.execute(f"""
    WITH t0 AS (   -- same data-quality rules as data_prep/01_build_features.py
      SELECT * REPLACE (CASE WHEN payment_plan_days > 0 THEN payment_plan_days END AS payment_plan_days,
                        CASE WHEN edate >= DATE '2015-01-01' THEN edate END AS edate)
      FROM '{f}'),
    t AS (
      SELECT *, row_number() OVER (PARTITION BY msno ORDER BY {ORDER.replace(', ', ' DESC, ')} DESC) AS rk,
             lag(is_auto_renew) OVER w AS prev_auto, lag(edate IGNORE NULLS) OVER w AS prev_exp,
             lag(tdate) OVER w AS prev_t, lag(payment_method_id) OVER w AS prev_method
      FROM t0 WINDOW w AS (PARTITION BY msno ORDER BY {ORDER}))
    SELECT msno,
      sum(CASE WHEN prev_auto IS NOT NULL AND prev_auto <> is_auto_renew THEN 1 ELSE 0 END) AS n_autorenew_switches,
      sum(CASE WHEN prev_auto = 1 AND is_auto_renew = 0 THEN 1 ELSE 0 END) AS n_auto_to_manual,
      sum(CASE WHEN prev_exp IS NOT NULL AND tdate - prev_exp > 0 THEN 1 ELSE 0 END) AS n_lapses,
      max(CASE WHEN prev_exp IS NOT NULL THEN tdate - prev_exp END) AS max_lapse_days,
      avg(CASE WHEN prev_t IS NOT NULL THEN tdate - prev_t END) AS mean_days_between_tx,
      stddev_pop(CASE WHEN prev_t IS NOT NULL THEN tdate - prev_t END) AS sd_days_between_tx,
      sum(CASE WHEN prev_method IS NOT NULL AND prev_method <> payment_method_id THEN 1 ELSE 0 END) AS n_method_switches,
      avg(actual_amount_paid / nullif(payment_plan_days, 0)) AS mean_price_per_day,
      max(CASE WHEN rk = 1 THEN actual_amount_paid / nullif(payment_plan_days, 0) END) AS last_price_per_day,
      avg(CASE WHEN rk <= 3 THEN is_auto_renew END) AS auto_renew_last3,
      sum(CASE WHEN rk <= 3 THEN is_cancel ELSE 0 END) AS cancels_last3,
      count(DISTINCT payment_plan_days) AS n_distinct_plans,
      max(edate) - min(tdate) AS covered_span_days
    FROM t GROUP BY msno""").df()


NEW = ["n_autorenew_switches", "n_auto_to_manual", "n_lapses", "max_lapse_days", "mean_days_between_tx",
       "sd_days_between_tx", "n_method_switches", "mean_price_per_day", "last_price_per_day",
       "auto_renew_last3", "cancels_last3", "n_distinct_plans", "covered_span_days"]


def prep(cohort):
    d = load(cohort).merge(payment_history(cohort), on="msno", how="left")
    for c in CATEGORICAL:
        d[c] = pd.Categorical(d[c].astype("object").where(d[c].notna(), "NA").astype(str))
    return d


tr, te = prep("feb"), prep("mar")
for c in CATEGORICAL:   # same category codes in both months
    cats = sorted(set(tr[c].cat.categories) | set(te[c].cat.categories))
    tr[c], te[c] = tr[c].cat.set_categories(cats), te[c].cat.set_categories(cats)
ytr, yte = tr.is_churn.values, te.is_churn.values
print(f"features ready in {time.time() - t0:.0f}s; new feature coverage feb {tr[NEW[0]].notna().mean():.1%}")

DEFAULT = dict(learning_rate=0.05, num_leaves=63, min_child_samples=200, subsample=0.8, subsample_freq=1,
               colsample_bytree=0.8, reg_lambda=1.0)


def fit(cols, params, X, y, n_max=3000):
    """Early stopping on a fixed 10 % slice of the given training data (feb only)."""
    val = np.random.default_rng(SEED).random(len(X)) < 0.1
    m = lgb.LGBMClassifier(n_estimators=n_max, random_state=SEED, verbose=-1, n_jobs=-1, **params)
    m.fit(X.loc[~val, cols], y[~val], eval_set=[(X.loc[val, cols], y[val])],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


# ---- Optuna tuning inside feb ---------------------------------------------------------------
def tune(cols, tag):
    sub = tr.sample(min(TUNE_SAMPLE, len(tr)), random_state=SEED)
    a, b = train_test_split(sub, test_size=0.2, stratify=sub.is_churn, random_state=SEED)

    def objective(trial):
        params = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 2000, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            min_split_gain=trial.suggest_float("min_split_gain", 0.0, 1.0))
        m = fit(cols, params, a, a.is_churn.values, n_max=2000)
        trial.set_user_attr("n_trees", m.best_iteration_)
        return roc_auc_score(b.is_churn, m.predict_proba(b[cols])[:, 1])

    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    st.enqueue_trial({k: v for k, v in DEFAULT.items() if k != "subsample_freq"} | {"reg_alpha": 1e-3, "min_split_gain": 0.0})
    st.optimize(objective, n_trials=N_TRIALS)
    trials = st.trials_dataframe()
    trials["feature_set"] = tag
    print(f"[{tag}] best feb-validation AUC {st.best_value:.4f} (default {st.trials[0].value:.4f})")
    return st.best_params | {"subsample_freq": 1}, trials


BASE = NUMERIC + CATEGORICAL
configs, all_trials = {}, []
p_base_tuned, t1 = tune(BASE, "baseline features")
p_new_tuned, t2 = tune(BASE + NEW, "+ payment-history features")
all_trials = pd.concat([t1, t2])
all_trials.to_csv(REPORT / "model_tuning_trials.csv", index=False)
configs = {
    "A. baseline (03_classification)": (BASE, DEFAULT),
    "B. + payment-history features": (BASE + NEW, DEFAULT),
    "C. tuned hyper-parameters": (BASE, p_base_tuned),
    "D. features + tuning": (BASE + NEW, p_new_tuned),
}

# ---- final out-of-time scores ----------------------------------------------------------------
preds, rows = {}, []
for name, (cols, params) in configs.items():
    s = time.time()
    m = fit(cols, params, tr, ytr)
    p = m.predict_proba(te[cols])[:, 1]
    preds[name] = p
    k = int(len(p) * 0.1)
    top = np.argsort(-p)[:k]
    rows.append({"configuration": name, "n_features": len(cols), "n_trees": m.best_iteration_,
                 "roc_auc": roc_auc_score(yte, p), "pr_auc": average_precision_score(yte, p),
                 "lift@10%": yte[top].mean() / yte.mean(), "recall@10%": yte[top].sum() / yte.sum(),
                 "fit_s": time.time() - s})
    print(f"{name:34s} AUC={rows[-1]['roc_auc']:.4f} PR-AUC={rows[-1]['pr_auc']:.4f}")
res = pd.DataFrame(rows)
rng = np.random.default_rng(SEED)
base = preds["A. baseline (03_classification)"]
deltas = {n: [] for n in preds}
for _ in range(100):
    i = rng.integers(0, len(yte), len(yte))
    a0 = roc_auc_score(yte[i], base[i])
    for n, p in preds.items():
        deltas[n].append(roc_auc_score(yte[i], p[i]) - a0)
res["delta_auc"] = res.roc_auc - res.roc_auc.iloc[0]
res["delta_ci_low"] = [np.percentile(deltas[n], 2.5) for n in res.configuration]
res["delta_ci_high"] = [np.percentile(deltas[n], 97.5) for n in res.configuration]
res.to_csv(REPORT / "model_improvement.csv", index=False)
print(res.round(4).to_string(index=False))

# importance of the new features in configuration D
mD = fit(*configs["D. features + tuning"], tr, ytr)
gain = pd.Series(mD.booster_.feature_importance("gain"), index=configs["D. features + tuning"][0])
gain = (gain / gain.sum()).sort_values(ascending=False)
new_share = float(gain[NEW].sum())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].barh(res.configuration[::-1], res.delta_auc[::-1] * 1000,
             xerr=[(res.delta_auc - res.delta_ci_low)[::-1] * 1000, (res.delta_ci_high - res.delta_auc)[::-1] * 1000],
             color="#4C72B0")
axes[0].axvline(0, color="k", lw=0.8)
axes[0].set_xlabel("thay đổi ROC-AUC so với baseline (×0,001, CI 95%)")
axes[0].set_title("Thí nghiệm cải tiến mô hình (test ngoài thời gian, tháng 3)")
g = gain.head(15)[::-1]
axes[1].barh(g.index, g.values, color=["#DD8452" if c in NEW else "#4C72B0" for c in g.index])
axes[1].set_title(f"Tỉ trọng gain, cấu hình D (cam = đặc trưng mới, tổng {new_share:.0%})")
savefig(fig, "model_improvement.png")

save_metrics("model_improvement", {
    "results": res.to_dict("records"), "tuned_params_baseline": p_base_tuned, "tuned_params_new": p_new_tuned,
    "n_trials": N_TRIALS, "tune_sample": TUNE_SAMPLE, "new_features": NEW, "new_feature_gain_share": new_share,
    "top_gain": gain.head(10).to_dict(), "runtime_s": time.time() - t0,
})
