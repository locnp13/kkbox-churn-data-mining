"""Supporting method: churn classification with an out-of-time evaluation.

* Train on the feb cohort (features at 2017-01-31), test on the mar cohort (features at
  2017-02-28): the model never sees the test month, as in deployment.
* 5-fold stratified CV inside the train cohort for model comparison (mean +- sd).
* Models: prior baseline, logistic regression, decision tree, random forest, LightGBM.
  All preprocessing (imputation, scaling, one-hot) is fitted inside a Pipeline on train only.
* Metrics for an imbalanced problem: ROC-AUC, PR-AUC, log-loss, Brier, lift and recall in the
  top decile (the campaign budget view); bootstrap 95 % CI for the best model.
* Interpretation: shallow decision-tree rules, SHAP values, feature-group ablation.

Outputs: report/classification_*.csv, tree_rules.txt, figures/clf_*.png, metrics/classification.json
"""
import os
import sys
import time
from pathlib import Path

import warnings

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CATEGORICAL, DATA, LISTENING, NUMERIC, PROFILE, REPORT, SEED, SUBSCRIPTION, load,
                    save_metrics, savefig)

warnings.filterwarnings("ignore")
RF_SAMPLE = int(os.environ.get("RF_SAMPLE", 300_000))
FAST = os.environ.get("FAST") == "1"
t0 = time.time()
tr, te = load("feb"), load("mar")
ytr, yte = tr.is_churn.values, te.is_churn.values
print(f"train feb {len(tr):,} (churn {ytr.mean():.3%}) -> test mar {len(te):,} (churn {yte.mean():.3%})")


def top_decile(y, p, q=0.1):
    k = max(1, int(len(p) * q))
    top = np.argsort(-p)[:k]
    prec = y[top].mean()
    return {"precision@10%": prec, "recall@10%": y[top].sum() / y.sum(), "lift@10%": prec / y.mean()}


def scores(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "log_loss": log_loss(y, p), "brier": brier_score_loss(y, p), **top_decile(y, p)}


def sk_pipe(model, num, cat, scale=False):
    num_steps = [SimpleImputer(strategy="median", add_indicator=True)] + ([StandardScaler()] if scale else [])
    pre = ColumnTransformer([
        ("num", make_pipeline(*num_steps), num),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=200)), cat)])
    return make_pipeline(pre, model)


def to_str(x):
    """Category labels as strings; missing -> 'NA' (pandas 3 keeps NaN through astype(str))."""
    return x.astype("object").where(x.notna(), "NA").astype(str)


def as_cat(d, cat):
    d = d.copy()
    for c in cat:
        d[c] = to_str(d[c])
    return d


class LGBM:
    """LightGBM with categorical columns; early stopping on an internal 10 % validation split."""

    def __init__(self, num, cat, **kw):
        self.num, self.cat, self.kw = num, cat, kw

    def _X(self, d):
        X = d[self.num + self.cat].copy()
        for c in self.cat:
            X[c] = pd.Categorical(to_str(X[c]), categories=self.levels[c])
        return X

    def fit(self, d, y):
        self.levels = {c: sorted(to_str(d[c]).unique()) for c in self.cat}
        X = self._X(d)
        rng = np.random.default_rng(SEED)
        val = rng.random(len(X)) < 0.1
        params = dict(n_estimators=2000, learning_rate=0.05, num_leaves=63, min_child_samples=200,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                      random_state=SEED, verbose=-1, n_jobs=-1)
        params.update(self.kw)
        self.m = lgb.LGBMClassifier(**params)
        self.m.fit(X[~val], y[~val], eval_set=[(X[val], y[val])],
                   callbacks=[lgb.early_stopping(100, verbose=False)])
        return self

    def predict_proba(self, d):
        return self.m.predict_proba(self._X(d))


ALL_NUM, CAT = NUMERIC, CATEGORICAL
trc, tec = as_cat(tr, CAT), as_cat(te, CAT)


def models(num=ALL_NUM, cat=CAT):
    return {
        "Baseline (prior)": lambda: DummyClassifier(strategy="prior"),
        "Logistic regression": lambda: sk_pipe(LogisticRegression(C=0.5, max_iter=2000), num, cat, scale=True),
        "Decision tree (depth 6)": lambda: sk_pipe(DecisionTreeClassifier(max_depth=6, min_samples_leaf=500,
                                                                          random_state=SEED), num, cat),
        "Random forest": lambda: sk_pipe(RandomForestClassifier(300 if not FAST else 50, min_samples_leaf=50,
                                         max_features="sqrt", n_jobs=-1, random_state=SEED), num, cat),
        "LightGBM": lambda: LGBM(num, cat),
    }


def fit_predict(name, make, dtr, y, dte):
    m = make()
    if name == "Random forest" and len(dtr) > RF_SAMPLE:   # RF on 1M rows is slow; subsample
        idx = np.random.default_rng(SEED).choice(len(dtr), RF_SAMPLE, replace=False)
        dtr, y = dtr.iloc[idx], y[idx]
    X = dtr if name == "LightGBM" else dtr[ALL_NUM + CAT]
    m.fit(X, y)
    return m, m.predict_proba(dte if name == "LightGBM" else dte[ALL_NUM + CAT])[:, 1]


# ---- 1. out-of-time comparison ------------------------------------------------------------
rows, preds, fitted = [], {}, {}
for name, make in models().items():
    s = time.time()
    m, p = fit_predict(name, make, trc, ytr, tec)
    preds[name], fitted[name] = p, m
    rows.append({"model": name, **scores(yte, p), "fit_s": time.time() - s})
    print(f"{name:26s} AUC={rows[-1]['roc_auc']:.4f} PR-AUC={rows[-1]['pr_auc']:.4f} ({time.time() - s:.0f}s)")
oot = pd.DataFrame(rows)
oot.to_csv(REPORT / "classification_out_of_time.csv", index=False)

# ---- 2. 5-fold CV on the train cohort (model selection view) --------------------------------
cv_rows = []
skf = StratifiedKFold(3 if FAST else 5, shuffle=True, random_state=SEED)
for name in ["Logistic regression", "Decision tree (depth 6)", "LightGBM"]:
    for f, (a, b) in enumerate(skf.split(trc, ytr)):
        _, p = fit_predict(name, models()[name], trc.iloc[a], ytr[a], trc.iloc[b])
        cv_rows.append({"model": name, "fold": f, "roc_auc": roc_auc_score(ytr[b], p),
                        "pr_auc": average_precision_score(ytr[b], p)})
cv = pd.DataFrame(cv_rows).groupby("model")[["roc_auc", "pr_auc"]].agg(["mean", "std"])
cv.columns = ["_".join(c) for c in cv.columns]
cv.to_csv(REPORT / "classification_cv.csv")
print(cv.round(4).to_string())

# ---- 3. bootstrap CI for the model selected by CV -----------------------------------------------------
best = cv.roc_auc_mean.idxmax()   # chosen on feb CV only; the mar test set is not used to choose
rng = np.random.default_rng(SEED)
pb = preds[best]
boot = []
for _ in range(100 if not FAST else 20):
    i = rng.integers(0, len(yte), len(yte))
    boot.append((roc_auc_score(yte[i], pb[i]), average_precision_score(yte[i], pb[i])))
boot = np.array(boot)
ci = {"roc_auc_ci": np.percentile(boot[:, 0], [2.5, 97.5]), "pr_auc_ci": np.percentile(boot[:, 1], [2.5, 97.5])}
print(f"best={best} AUC 95% CI {ci['roc_auc_ci'].round(4)}")

# ---- 4. feature-group ablation (LightGBM, out-of-time) ---------------------------------------
groups = {"profile only": (PROFILE, ["city", "registered_via", "gender"]),
          "listening only": (LISTENING, []),
          "subscription only": (SUBSCRIPTION, ["last_payment_method"]),
          "subscription + profile": (SUBSCRIPTION + PROFILE, CAT),
          "all features": (ALL_NUM, CAT)}
abl = []
for g, (num, cat) in groups.items():
    p = LGBM(num, cat).fit(trc, ytr).predict_proba(tec)[:, 1]
    abl.append({"feature_set": g, "n_features": len(num) + len(cat), **scores(yte, p)})
abl = pd.DataFrame(abl)
abl.to_csv(REPORT / "classification_ablation.csv", index=False)
print(abl[["feature_set", "roc_auc", "pr_auc", "lift@10%"]].round(4).to_string(index=False))

# ---- 5. interpretation ----------------------------------------------------------------------
tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=2000, random_state=SEED)
Xt = tr[ALL_NUM].fillna(-1)
tree.fit(Xt, ytr)
(REPORT / "tree_rules.txt").write_text(export_text(tree, feature_names=ALL_NUM, show_weights=True),
                                        encoding="utf-8")

lg_model = fitted["LightGBM"]
imp_gain = pd.Series(lg_model.m.booster_.feature_importance("gain"), index=lg_model.num + lg_model.cat)
try:
    import shap
    S = tec.sample(min(5000, len(tec)), random_state=SEED)
    Xs = lg_model._X(S)
    sv = shap.TreeExplainer(lg_model.m).shap_values(Xs)
    sv = sv[1] if isinstance(sv, list) else sv
    shap_imp = pd.Series(np.abs(sv).mean(0), index=Xs.columns)
    plt.figure()
    shap.summary_plot(sv, Xs, max_display=15, show=False)
    plt.title("Giá trị SHAP, LightGBM (mẫu cohort tháng 3)")
    savefig(plt.gcf(), "clf_shap_summary.png")
    plt.close("all")
except Exception as e:  # shap is optional
    print("SHAP skipped:", e)
    shap_imp = imp_gain / imp_gain.sum()
fi = pd.DataFrame({"mean_abs_shap": shap_imp, "gain_share": imp_gain / imp_gain.sum()}).sort_values(
    "mean_abs_shap", ascending=False)
fi.to_csv(REPORT / "feature_importance.csv")
print(fi.head(12).round(4).to_string())

# ---- 6. figures ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for name, p in preds.items():
    if name.startswith("Baseline"):
        continue
    fpr, tpr, _ = roc_curve(yte, p); axes[0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(yte, p):.3f})")
    pr, rc, _ = precision_recall_curve(yte, p)
    axes[1].plot(rc, pr, label=f"{name} ({average_precision_score(yte, p):.3f})")
    fp, mp = calibration_curve(yte, p, n_bins=15, strategy="quantile")
    axes[2].plot(mp, fp, marker=".", label=name)
axes[0].plot([0, 1], [0, 1], "k--", lw=0.8); axes[0].set_title("ROC (ngoài thời gian, cohort tháng 3)")
axes[1].axhline(yte.mean(), color="k", ls="--", lw=0.8); axes[1].set_title("Precision-Recall")
axes[2].plot([0, 1], [0, 1], "k--", lw=0.8); axes[2].set_title("Hiệu chuẩn")
axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR"); axes[1].set_xlabel("recall (độ phủ)"); axes[1].set_ylabel("precision (độ chính xác)")
axes[2].set_xlabel("xác suất dự báo"); axes[2].set_ylabel("tỉ lệ thực tế")
for a in axes:
    a.legend(fontsize=7)
savefig(fig, "clf_curves.png")

# gains chart: share of churners captured when contacting the top x % riskiest users
fig, ax = plt.subplots(figsize=(6, 5))
for name in ["Logistic regression", "LightGBM"]:
    o = np.argsort(-preds[name]); cum = np.cumsum(yte[o]) / yte.sum()
    ax.plot(np.arange(1, len(o) + 1) / len(o), cum, label=name)
ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="ngẫu nhiên")
ax.set_xlabel("tỉ lệ người dùng được liên hệ"); ax.set_ylabel("tỉ lệ người churn bắt được")
ax.set_title("Cumulative gains (cohort tháng 3)"); ax.legend()
savefig(fig, "clf_gains.png")

pd.DataFrame({"msno": te.msno, "p_churn": preds["LightGBM"]}).to_parquet(DATA / "pred_mar_lgbm.parquet")
save_metrics("classification", {
    "n_train": len(tr), "n_test": len(te), "churn_train": ytr.mean(), "churn_test": yte.mean(),
    "out_of_time": oot.to_dict("records"), "cv": cv.reset_index().to_dict("records"),
    "best_model": best, **{k: v for k, v in ci.items()}, "ablation": abl.to_dict("records"),
    "top_features": fi.head(10).mean_abs_shap.to_dict(), "runtime_s": time.time() - t0,
})
