"""Probability recalibration under prior shift, and the business value of a retention campaign.

1. Prior-shift correction (Saerens, Latinne & Decaestecker 2002): the model is trained on feb
   (churn 6.4 %) and applied to mar. The EM algorithm re-estimates the new month's churn prior
   from the model's own predictions on mar (no mar labels used) and rescales the posteriors.
   Calibration before/after: mean prediction, Brier, log-loss, expected calibration error. An
   "oracle" row rescales with the true mar prior (not available at deployment) to show the best
   calibration a pure prior correction could give; if EM misses the true prior, the shift is not
   pure label shift (P(x|y) changed too).
2. Profit curve: contact the top x % of users ranked by predicted risk. Illustrative,
   configurable assumptions (env vars), with sensitivity over the campaign success rate:
     monthly revenue per user  ARPU = last_paid / plan_days * 30 (NTD)
     value of a saved churner  = ARPU * VALUE_MONTHS
     cost per contacted user   = CONTACT_COST + DISCOUNT * ARPU for everyone who stays
                                 (non-churners also redeem the offer)
     a contacted churner is saved with probability SUCCESS
   Realised mar labels are used to evaluate the curve, as a back-test.

Outputs: report/calibration.csv, profit_curve.csv, figures/calibration.png, figures/profit_curve.png,
metrics/business_value.json
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, METRICS, REPORT, load, save_metrics, savefig

CONTACT_COST = float(os.environ.get("CONTACT_COST", 5))      # NTD per contact (message + handling)
DISCOUNT = float(os.environ.get("DISCOUNT", 0.2))            # share of one month's fee offered
VALUE_MONTHS = float(os.environ.get("VALUE_MONTHS", 6))      # months of revenue kept per saved churner
SUCCESS = [float(s) for s in os.environ.get("SUCCESS", "0.1,0.2,0.3").split(",")]

d = load("mar")[["msno", "is_churn", "last_paid", "last_plan_days"]].merge(
    pd.read_parquet(DATA / "pred_mar_lgbm.parquet"), on="msno")
y, p = d.is_churn.values, np.clip(d.p_churn.values, 1e-6, 1 - 1e-6)
prior_train = json.loads((METRICS / "classification.json").read_text(encoding="utf-8"))["churn_train"]


# ---- 1. EM prior-shift correction -----------------------------------------------------------
def adjust(p, old, new):
    a = new / old * p
    b = (1 - new) / (1 - old) * (1 - p)
    return a / (a + b)


prior = prior_train
for it in range(100):
    q = adjust(p, prior_train, prior)
    new = q.mean()
    if abs(new - prior) < 1e-7:
        break
    prior = new
p_adj = adjust(p, prior_train, prior)
print(f"train prior {prior_train:.4f}; EM-estimated mar prior {prior:.4f} ({it} iterations); actual {y.mean():.4f}")


def ece(y, p, bins=15):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    return sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())


cal = pd.DataFrame([
    {"version": name, "mean_pred": q.mean(), "actual_rate": y.mean(), "brier": brier_score_loss(y, q),
     "log_loss": log_loss(y, q), "ece": ece(y, q)}
    for name, q in [("original (feb prior)", p), ("EM prior-shift corrected", p_adj),
                    ("oracle: true mar prior (not deployable)", adjust(p, prior_train, y.mean()))]])
cal.to_csv(REPORT / "calibration.csv", index=False)
print(cal.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(6, 5.5))
for name, q in [("gốc (tỉ lệ nền tháng 2)", p), ("đã hiệu chỉnh EM", p_adj),
                ("oracle: tỉ lệ nền thật tháng 3", adjust(p, prior_train, y.mean()))]:
    edges = np.quantile(q, np.linspace(0, 1, 21))
    idx = np.clip(np.searchsorted(edges, q, side="right") - 1, 0, 19)
    ax.plot([q[idx == b].mean() for b in range(20)], [y[idx == b].mean() for b in range(20)], marker="o", label=name)
ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="hiệu chuẩn hoàn hảo")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("xác suất dự báo (trung bình nhóm)"); ax.set_ylabel("tỉ lệ churn thực tế")
ax.set_title("Đường hiệu chuẩn trên tháng 3 (20 nhóm phân vị)"); ax.legend()
savefig(fig, "calibration.png")

# ---- 2. profit curve --------------------------------------------------------------------------
arpu = (d.last_paid / d.last_plan_days.replace(0, np.nan) * 30).replace([np.inf, -np.inf], np.nan)
arpu = arpu.fillna(arpu.median()).clip(0, 1000).values
order = np.argsort(-p)
shares = np.round(np.arange(0.01, 1.0001, 0.01), 2)
rows = []
for s in SUCCESS:
    yo, ao = y[order], arpu[order]
    gain = np.cumsum(yo * s * ao * VALUE_MONTHS)
    cost = np.cumsum(CONTACT_COST + DISCOUNT * ao * (yo * s + (1 - yo)))
    net = gain - cost
    rnd_net = (y * s * arpu * VALUE_MONTHS - CONTACT_COST - DISCOUNT * arpu * (y * s + (1 - y))).mean()
    for x in shares:
        k = int(len(order) * x) - 1
        rows.append({"success_rate": s, "share_contacted": x, "n_contacted": k + 1,
                     "net_value_ntd": net[k], "random_targeting_ntd": rnd_net * (k + 1),
                     "churners_reached": yo[:k + 1].sum() / y.sum()})
pc = pd.DataFrame(rows)
pc.to_csv(REPORT / "profit_curve.csv", index=False)
best = pc.loc[pc.groupby("success_rate").net_value_ntd.idxmax()]
print(best.round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
for s in SUCCESS:
    t = pc[pc.success_rate == s]
    ax.plot(t.share_contacted * 100, t.net_value_ntd / 1e6, label=f"mô hình, tỉ lệ cứu được {s:.0%}")
    ax.plot(t.share_contacted * 100, t.random_targeting_ntd / 1e6, ls=":", color=ax.lines[-1].get_color(),
            label=f"chọn ngẫu nhiên, {s:.0%}")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("% người dùng được liên hệ (xếp theo rủi ro dự báo)")
ax.set_ylabel("giá trị ròng (triệu NTD)")
ax.set_title(f"Đường lợi nhuận chiến dịch giữ chân, tháng 3 (giả định: {CONTACT_COST:.0f} NTD/liên hệ, "
             f"giảm {DISCOUNT:.0%} một tháng, giữ {VALUE_MONTHS:.0f} tháng)", fontsize=9)
ax.legend(fontsize=8)
savefig(fig, "profit_curve.png")

save_metrics("business_value", {
    "prior_train": prior_train, "prior_em_estimate": prior, "prior_actual": y.mean(), "em_iterations": it,
    "calibration": cal.to_dict("records"),
    "assumptions": {"contact_cost_ntd": CONTACT_COST, "discount_share_of_month": DISCOUNT,
                    "value_months": VALUE_MONTHS, "success_rates": SUCCESS},
    "median_arpu_ntd": float(np.median(arpu)),
    "optimum": best[["success_rate", "share_contacted", "net_value_ntd", "churners_reached"]].to_dict("records"),
})
