"""Evaluation framework: does the project answer its business questions, and how reliably?

Metrics are grouped into seven aspects, each tied to a question about the problem (not to a
technique). Besides collecting the metrics produced by earlier steps, this script computes the ones
that only make sense across techniques:
  * churner coverage of the top risk rules / segments (how much of the problem an action reaches);
  * population stability index (PSI) of each feature between the feb and mar cohorts (drift);
  * average rule / pattern length (interpretability).

Outputs: report/metrics_summary.csv, report/factor_summary.csv, report/feature_drift.csv,
report/rule_coverage.csv, metrics/metrics_summary.json
Thresholds are common rules of thumb and are written next to each metric so they can be challenged.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CATEGORICAL, DATA, METRICS, NUMERIC, REPORT, SEED, load, save_metrics
from transforms import itemise


def j(name):
    p = METRICS / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


cl, ru, sq, cf, va, st, bv, mi, bo = (j(n) for n in ["clustering", "rules", "sequential", "classification",
                                                      "validation", "stability", "business_value",
                                                      "model_improvement", "bonus"])
manifest = json.loads((DATA / "manifest.json").read_text()) if (DATA / "manifest.json").exists() else {}
oot = {r["model"]: r for r in cf.get("out_of_time", [])}
lgb = oot.get("LightGBM", {})
cv = {r["model"]: r for r in cf.get("cv", [])}
audit = {r["feature_set"]: r for r in va.get("leakage_audit", [])}
cal = {r["version"]: r for r in bv.get("calibration", [])}
mar, feb = load("mar"), load("feb")
base = mar.is_churn.mean()

# ---- cross-technique metrics -----------------------------------------------------------------
# churner coverage of the top-k risk rules, re-measured on the 30 % holdout split
tr, ho = train_test_split(mar.merge(pd.read_parquet(DATA / "clusters_mar.parquet")[["msno", "cluster"]], on="msno"),
                          test_size=0.3, stratify=mar.is_churn, random_state=SEED)
I = itemise(ho, tr)
y = ho.is_churn.values
rules = pd.read_csv(REPORT / "churn_rules.csv")
cov = []
covered = np.zeros(len(ho), bool)
for i, r in enumerate(rules.head(10).itertuples(), 1):
    items = r.rule.split(" & ")
    if all(it in I.columns for it in items):
        covered |= I[items].all(axis=1).values
    cov.append({"top_k_rules": i, "share_users_flagged": covered.mean(),
                "share_churners_covered": y[covered].sum() / y.sum(),
                "churn_rate_in_flagged": y[covered].mean() if covered.any() else np.nan})
cov = pd.DataFrame(cov)
cov.to_csv(REPORT / "rule_coverage.csv", index=False)
cov5 = cov.loc[cov.top_k_rules == min(5, len(cov))].iloc[0] if len(cov) else {}

# churner coverage of segments riskier than average
seg = pd.read_csv(REPORT / "cluster_profile_full.csv").set_index("cluster")
risky = seg[seg.churn_rate > base]
seg_cov = risky.churners.sum() / seg.churners.sum()
seg_share = risky.n.sum() / seg.n.sum()


def psi(a, b, bins=10):
    """Population stability index of b (mar) against a (feb), quantile bins of a."""
    a, b = a.dropna(), b.dropna()
    if a.nunique() < 2:
        return np.nan
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.unique(np.r_[a.min() - 1, a.unique(), a.max() + 1])
    pa = np.histogram(a, edges)[0] / len(a) + 1e-6
    pb = np.histogram(b.clip(edges[0], edges[-1]), edges)[0] / len(b) + 1e-6
    return float(((pb - pa) * np.log(pb / pa)).sum())


def psi_cat(a, b):
    """PSI for a categorical feature; missing values are their own category."""
    pa = a.astype("object").where(a.notna(), "NA").astype(str).value_counts(normalize=True)
    pb = b.astype("object").where(b.notna(), "NA").astype(str).value_counts(normalize=True)
    idx = pa.index.union(pb.index)
    pa, pb = pa.reindex(idx, fill_value=0) + 1e-6, pb.reindex(idx, fill_value=0) + 1e-6
    return float(((pb - pa) * np.log(pb / pa)).sum())


drift = pd.DataFrame([{"feature": f, "type": "số", "psi": psi(feb[f], mar[f]),
                       "missing_feb": feb[f].isna().mean(), "missing_mar": mar[f].isna().mean()} for f in NUMERIC] +
                     [{"feature": f, "type": "phân loại", "psi": psi_cat(feb[f], mar[f]),
                       "missing_feb": feb[f].isna().mean(), "missing_mar": mar[f].isna().mean()} for f in CATEGORICAL]
                     ).sort_values("psi", ascending=False)
drift.to_csv(REPORT / "feature_drift.csv", index=False)
sp = pd.read_csv(REPORT / "sequential_patterns.csv") if (REPORT / "sequential_patterns.csv").exists() else pd.DataFrame()

# bootstrap CI of precision / recall at a 10 % contact budget
pred = mar[["msno", "is_churn"]].merge(pd.read_parquet(DATA / "pred_mar_lgbm.parquet"), on="msno")
yy, pp = pred.is_churn.values, pred.p_churn.values
rng = np.random.default_rng(SEED)
pr_b = []
for _ in range(100):
    i = rng.integers(0, len(yy), len(yy))
    top = np.argsort(-pp[i])[: len(i) // 10]
    pr_b.append((yy[i][top].mean(), yy[i][top].sum() / yy[i].sum()))
pr_b = np.array(pr_b)
prec_ci, rec_ci = np.percentile(pr_b[:, 0], [2.5, 97.5]), np.percentile(pr_b[:, 1], [2.5, 97.5])

# ---- the framework -------------------------------------------------------------------------
rows = []


def add(aspect, question, metric, value, ref, ok, meaning):
    verdict = "tham khảo" if ok is None else ("đạt" if ok else "chưa đạt")
    rows.append({"aspect": aspect, "question": question, "metric": metric, "value": value,
                 "reference": ref, "verdict": verdict, "meaning": meaning})


g = lambda d, k, default=None: d.get(k, default) if d else default
A1, Q1 = "1. Tính hợp lệ", "Kết quả có dùng nhầm thông tin tương lai hoặc dữ liệu lỗi không?"
dq = manifest.get("data_quality", {})
add(A1, Q1, "Giao dịch muộn nhất so với mốc quan sát", manifest.get("cohorts", {}).get("mar", {}).get("max_transaction_date"),
    "≤ 20170228", (manifest.get("cohorts", {}).get("mar", {}).get("max_transaction_date") or 99999999) <= 20170228,
    "Không có giao dịch nào sau mốc được dùng")
clean = audit.get("clean (cutoff T)", {}).get("roc_auc")
if clean:
    add(A1, Q1, "AUC tăng ảo nếu dùng giao dịch tháng nhãn", audit["clean + March transactions"]["roc_auc"] - clean,
        "phải đo được rồi loại bỏ", None, "Độ lớn của lỗi mà thiết kế mốc quan sát tránh được")
    add(A1, Q1, "AUC tăng ảo của cách làm phiên bản 1", audit["v1 recipe (unfiltered *_v2 files)"]["roc_auc"] - clean,
        "phải đo được rồi loại bỏ", None, "Mức kết quả cũ bị thổi phồng")
add(A1, Q1, "Bản ghi gói 0 ngày trong dữ liệu gốc", dq.get("tx_plan_days_zero"),
    "sau xử lý không còn gói 0 ngày trong đặc trưng", bool((mar.last_plan_days.dropna() > 0).all()),
    "Kiểm tra: đặc trưng “độ dài gói cuối” không còn giá trị 0")
add(A1, Q1, "Ngày hết hạn hỏng (trước 2015) trong dữ liệu gốc", dq.get("tx_expiry_before_2015"),
    "sau xử lý không còn hạn trước 2015", bool((mar.days_to_expire.dropna() > -800).all()),
    "Kiểm tra: số ngày còn hạn không còn giá trị trước 01/2015")
add(A1, Q1, "Thâm niên âm sau xử lý", int((mar.tenure_days < 0).sum()), "0", int((mar.tenure_days < 0).sum()) == 0,
    "Không còn ngày đăng ký sau mốc quan sát")

A2, Q2 = "2. Chất lượng kỹ thuật", "Bản thân từng kỹ thuật có cho kết quả tốt không?"
add(A2, Q2, "Silhouette (phân cụm)", g(cl, "silhouette"), "> 0,5 rõ; 0,25–0,5 vừa; < 0,25 yếu",
    (g(cl, "silhouette") or 0) >= 0.25, "Các nhóm khách tách nhau rõ đến đâu")
add(A2, Q2, "Davies–Bouldin (phân cụm)", g(cl, "davies_bouldin"), "< 1 tốt; càng thấp càng tốt",
    (g(cl, "davies_bouldin") or 9) < 1, "Mức chồng lấn giữa các nhóm")
sens = pd.read_csv(REPORT / "rule_sensitivity.csv")
n_sig = int(sens.loc[sens.min_support.round(3) == 0.01, "n_significant_churn_rules"].iloc[0])
add(A2, Q2, "Số luật lift ≥ 1,5 vượt kiểm định FDR (support ≥ 1%)", n_sig, "> 0", n_sig > 0,
    f"Báo cáo giữ {g(ru, 'n_churn_rules')} luật mạnh nhất sau khi bỏ luật thừa")
add(A2, Q2, "Số mẫu tuần tự có ý nghĩa / số kiểm định", f"{g(sq, 'n_selected')}/{g(sq, 'n_tested')}",
    "tham khảo", None, "Sau hiệu chỉnh nhiều kiểm định")
add(A2, Q2, "ROC-AUC tháng 3 (LightGBM)", lgb.get("roc_auc"), "0,5 ngẫu nhiên; > 0,8 tốt",
    (lgb.get("roc_auc") or 0) > 0.8, "Khả năng xếp hạng khách theo rủi ro")
add(A2, Q2, "PR-AUC (average precision) tháng 3", lgb.get("pr_auc"), f"ngẫu nhiên = {base:.3f}; ≥ 3 lần là tốt",
    (lgb.get("pr_auc") or 0) >= 3 * base, "Chất lượng khi tìm khách rời bỏ (lớp hiếm)")

A3, Q3 = "3. Độ tin cậy", "Kết quả có giữ được trên dữ liệu chưa thấy và ở tháng khác không?"
base_ll = -(base * np.log(base) + (1 - base) * np.log(1 - base))
add(A2, Q2, "Log-loss tháng 3 (LightGBM)", lgb.get("log_loss"), f"dự báo theo tỉ lệ chung = {base_ll:.3f}",
    (lgb.get("log_loss") or 9) < base_ll, "Chất lượng xác suất (phạt nặng dự báo tự tin mà sai)")
add(A3, Q3, "Ổn định phân cụm (ARI bootstrap)", g(cl, "stability_ari"), "≥ 0,8 ổn định; 0,6–0,8 vừa",
    (g(cl, "stability_ari") or 0) >= 0.8, "Chạy lại trên mẫu khác có ra cùng cách chia")
add(A3, Q3, "Cụm kém ổn định nhất (Jaccard)", g(cl, "jaccard_min_cluster"), "≥ 0,75 ổn định; < 0,5 không ổn định",
    (g(cl, "jaccard_min_cluster") or 0) >= 0.75, "Nhóm yếu nhất có được tìm lại")
add(A3, Q3, "Phân cụm lại ở tháng 2 (ARI)", g(st, "ari_feb_refit_vs_mar_model"), "≥ 0,8",
    (g(st, "ari_feb_refit_vs_mar_model") or 0) >= 0.8, "Cấu trúc nhóm lặp lại ở tháng khác")
add(A3, Q3, "Tương quan lift luật train – holdout", g(ru, "holdout_lift_corr"), "≥ 0,9",
    (g(ru, "holdout_lift_corr") or 0) >= 0.9, "Luật đúng trên dữ liệu chưa dùng để tìm")
add(A3, Q3, "Luật giữ hướng ở tháng 2", g(st, "rule_share_same_direction"), "≥ 0,9",
    (g(st, "rule_share_same_direction") or 0) >= 0.9, "Luật lặp lại ở tháng khác")
add(A3, Q3, "Mẫu tuần tự giữ hướng ở tháng 2", g(sq, "share_same_direction_feb"), "≥ 0,9",
    (g(sq, "share_same_direction_feb") or 0) >= 0.9, "Chuỗi lặp lại ở tháng khác")
add(A3, Q3, "Khoảng tin cậy 95% ROC-AUC", f"{cf['roc_auc_ci'][0]:.3f}–{cf['roc_auc_ci'][1]:.3f}" if cf.get("roc_auc_ci") else None,
    "càng hẹp càng chắc", None, "Độ bất định của kết quả mô hình")

A4, Q4 = "4. Phân biệt rủi ro", "Có tách được khách rủi ro cao khỏi khách rủi ro thấp không?"
add(A4, Q4, "Chênh lệch tỉ lệ rời bỏ giữa các nhóm", seg.churn_rate.max() - seg.churn_rate.min(), "càng lớn càng có ích",
    seg.churn_rate.max() - seg.churn_rate.min() > 0.1, "Phân nhóm có ý nghĩa về rủi ro")
add(A4, Q4, "Lift cao nhất của luật (holdout)", rules.lift_holdout.max() if len(rules) else None, "> 2 mạnh",
    (rules.lift_holdout.max() if len(rules) else 0) > 2, "Nhóm khách rủi ro nhất gấp bao nhiêu lần bình thường")
add(A4, Q4, "Lift của luật “an toàn” nhất (holdout)",
    pd.read_csv(REPORT / "retention_rules.csv").lift_holdout.min() if (REPORT / "retention_rules.csv").exists() else None,
    "< 0,5 rõ rệt", None, "Nhóm khách ít rủi ro nhất")
add(A4, Q4, "Lift ở top 10% (mô hình)", lgb.get("lift@10%"), "> 3 tốt", (lgb.get("lift@10%") or 0) > 3,
    "Danh sách 10% rủi ro nhất tốt gấp bao nhiêu lần chọn ngẫu nhiên")

A5, Q5 = "5. Độ phủ và tính hành động", "Các phát hiện bao trùm được bao nhiêu phần vấn đề?"
if len(cov):
    add(A5, Q5, "Khách rời bỏ được 5 luật đầu bao phủ", cov5["share_churners_covered"], "càng cao càng tốt", None,
        f"Bắt được phần này khi chỉ gắn cờ {cov5['share_users_flagged']:.1%} khách (tỉ lệ rời bỏ {cov5['churn_rate_in_flagged']:.1%})")
add(A5, Q5, "Khách rời bỏ nằm trong các nhóm rủi ro hơn trung bình", seg_cov, "càng cao càng tốt", None,
    f"Các nhóm này chiếm {seg_share:.1%} tổng số khách")
add(A5, Q5, "Recall ở top 10% (mô hình)", lgb.get("recall@10%"), "ngẫu nhiên = 10%", (lgb.get("recall@10%") or 0) > 0.3,
    f"Liên hệ 10% khách thì tiếp cận được bao nhiêu khách sẽ rời bỏ (CI 95% {rec_ci[0]:.3f}–{rec_ci[1]:.3f})")
add(A5, Q5, "Precision ở top 10% (mô hình)", lgb.get("precision@10%"), f"ngẫu nhiên = {base:.3f}", None,
    f"Trong số khách được liên hệ, bao nhiêu thật sự sắp rời đi (CI 95% {prec_ci[0]:.3f}–{prec_ci[1]:.3f})")

A6, Q6 = "6. Giá trị kinh doanh và vận hành", "Áp dụng thật thì có lợi và có bền không?"
opt = {round(o["success_rate"], 2): o for o in bv.get("optimum", [])}
pcv = pd.read_csv(REPORT / "profit_curve.csv")
m20 = pcv.success_rate.round(2) == 0.2
opt_rand = float(pcv.loc[pcv.loc[m20, "net_value_ntd"].idxmax(), "random_targeting_ntd"]) if m20.any() else 0.0
if 0.2 in opt:
    add(A6, Q6, "Lãi ròng tối ưu/tháng (giả định cứu được 20%)", opt[0.2]["net_value_ntd"], "> 0 và > chọn ngẫu nhiên",
        opt[0.2]["net_value_ntd"] > max(0, opt_rand),
        f"Khi liên hệ {opt[0.2]['share_contacted']:.0%} khách rủi ro nhất; chọn ngẫu nhiên cùng số khách: {opt_rand:,.0f} NTD")
add(A6, Q6, "Sai số hiệu chuẩn (ECE)", (cal.get("original (feb prior)") or {}).get("ece"), "< 0,02 tốt",
    ((cal.get("original (feb prior)") or {}).get("ece") or 1) < 0.02, "Xác suất dự báo có dùng trực tiếp được không")
add(A6, Q6, "Đánh giá quá cao nếu chỉ kiểm tra cùng tháng (ΔAUC)",
    (cv.get("LightGBM", {}).get("roc_auc_mean") or 0) - (lgb.get("roc_auc") or 0), "càng nhỏ càng tốt", None,
    "Hậu quả của dữ liệu thay đổi theo thời gian")
add(A6, Q6, "Số đặc trưng thay đổi đáng kể giữa 2 tháng (PSI > 0,1)", int((drift.psi > 0.1).sum()),
    "PSI < 0,1 ổn định; 0,1–0,25 thay đổi vừa; > 0,25 thay đổi lớn", None,
    f"Thay đổi lớn nhất: {drift.feature.iloc[0]} (PSI {drift.psi.iloc[0]:.2f})")
runtime = sum((j(n).get("runtime_s") or 0) for n in ["clustering", "rules", "classification", "sequential",
                                                      "model_improvement"]) / 60
add(A6, Q6, "Thời gian chạy các bước khai phá chính (phút)", runtime, "chạy được trên máy cá nhân 16 GB", None,
    "Chưa gồm tạo đặc trưng (~6 phút) và đọc dữ liệu lần đầu (~13 phút)")

A7, Q7 = "7. Khả năng giải thích", "Người làm kinh doanh có hiểu và dùng được không?"
add(A7, Q7, "Số nhóm khách", int(len(seg)), "3–8 nhóm dễ dùng", 3 <= len(seg) <= 8, "Đủ ít để thiết kế chương trình riêng")
add(A7, Q7, "Độ dài trung bình của luật (số điều kiện)", float(rules.len.mean()) if len(rules) else None, "≤ 3 dễ hiểu",
    (rules.len.mean() if len(rules) else 9) <= 3, "Luật dạng “nếu … thì …” ngắn")
add(A7, Q7, "Độ dài trung bình của chuỗi (số bước)", float(sp.len.mean()) if len(sp) else None, "≤ 4", None,
    "Chuỗi sự kiện ngắn, dễ theo dõi")
add(A7, Q7, "Số yếu tố chiếm 80% ảnh hưởng (SHAP)", None, "càng ít càng dễ giải thích", None, "")

fi = pd.read_csv(REPORT / "feature_importance.csv", index_col=0)
share = (fi.mean_abs_shap / fi.mean_abs_shap.sum()).sort_values(ascending=False).cumsum()
rows[-1]["value"] = int((share < 0.8).sum() + 1)
ms = pd.DataFrame(rows)
ms.to_csv(REPORT / "metrics_summary.csv", index=False)
print(ms[["aspect", "metric", "value", "verdict"]].to_string(index=False))

# ---- factor ranking ---------------------------------------------------------------------------
eff = pd.read_csv(REPORT / "eda_effects.csv").set_index("feature")
rows = []
for f in fi.index:
    r = {"feature": f, "shap": fi.loc[f, "mean_abs_shap"], "psi": drift.set_index("feature").psi.get(f)}
    if f in eff.index:
        a = eff.loc[f, "auc_effect"]
        r["separation"] = abs(a - 0.5) + 0.5
        r["direction"] = "cao hơn ở khách rời bỏ" if a > 0.5 else "thấp hơn ở khách rời bỏ"
    if f in NUMERIC:
        if mar[f].nunique() > 2:
            v = mar[f].dropna()
            rk = v.rank(method="first", pct=True)          # exactly 10 % of users at each end (ties split)
            r["churn_low"] = mar.loc[rk[rk <= 0.1].index, "is_churn"].mean()
            r["churn_high"] = mar.loc[rk[rk > 0.9].index, "is_churn"].mean()
        else:
            r["churn_low"] = mar.loc[mar[f] == 0, "is_churn"].mean()
            r["churn_high"] = mar.loc[mar[f] == 1, "is_churn"].mean()
    rows.append(r)
fs = pd.DataFrame(rows).sort_values("shap", ascending=False)
fs.insert(0, "shap_rank", np.arange(1, len(fs) + 1))
fs.to_csv(REPORT / "factor_summary.csv", index=False)
print(fs.head(12).round(3).to_string(index=False))
save_metrics("metrics_summary", {
    "n_metrics": len(ms), "n_pass": int((ms.verdict == "đạt").sum()), "n_fail": int((ms.verdict == "chưa đạt").sum()),
    "rule_coverage_top5": cov5.to_dict() if len(cov) else None, "segment_churner_coverage": seg_cov,
    "segment_user_share": seg_share, "n_features_psi_gt_0_1": int((drift.psi > 0.1).sum()),
    "top_drift": drift.head(5).to_dict("records"),
})
