"""Build report/report.docx from the pipeline outputs (metrics/*.json, *.csv, figures/).

Written for a reader who does not know the code: short sentences, every term explained, rule and
pattern codes translated (report/vi_labels.py). Main numbers are read from the result files; a few
interpretive sentences quote numbers too, so re-read the text after a re-run with new data.
Run `python report/make_diagrams.py` first for the two explanatory diagrams.
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vi_labels as vi
from docx_helpers import bullets, figure, new_document, para, table

R = Path(os.environ.get("REPORT_DIR", Path(__file__).resolve().parent))
F = R / "figures"
M = R / "metrics"
DATA = Path(os.environ.get("DATA_DIR", R.parent / "data"))
OUT = Path(os.environ.get("REPORT_OUT", R / "report.docx"))


def j(name):
    p = M / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def csv(name, **kw):
    p = R / name
    return pd.read_csv(p, **kw) if p.exists() else pd.DataFrame()


def pct(x, d=1):
    return "–" if x is None or pd.isna(x) else f"{x * 100:.{d}f}%".replace(".", ",")


def num(x, d=3):
    return "–" if x is None or pd.isna(x) else f"{x:.{d}f}".replace(".", ",")


def k(x):
    return f"{int(round(x)):,}".replace(",", ".")


eda, cl, ru, cf, sq, va, st, bv, mi_m, vol, msm = (j(n) for n in [
    "eda", "clustering", "rules", "classification", "sequential", "validation", "stability",
    "business_value", "model_improvement", "data_volume", "metrics_summary"])
manifest = json.loads((DATA / "manifest.json").read_text()) if (DATA / "manifest.json").exists() else {}
dq = manifest.get("data_quality", {})
oot = pd.DataFrame(cf.get("out_of_time", []))
lg = oot.set_index("model").loc["LightGBM"] if len(oot) else {}
grp = csv("eda_churn_by_group.csv")
seg = csv("cluster_profile_full.csv").set_index("cluster")
base = eda.get("churn_mar", 0.09)
NOLOG = cl.get("nolog_id", len(seg) - 1)
risk_seg = seg.churn_rate.idxmax()
cov = csv("rule_coverage.csv")
cov5 = cov[cov.top_k_rules == 5].iloc[0] if len(cov) >= 5 else None
cvt = csv("classification_cv.csv")
cv_lgb = cvt.set_index("model").loc["LightGBM", "roc_auc_mean"] if len(cvt) else None
audit = {r["feature_set"]: r for r in va.get("leakage_audit", [])}
fsum = csv("factor_summary.csv").set_index("feature")
sens_all = csv("rule_sensitivity.csv")
n_sig = int(sens_all.loc[sens_all.min_support.round(3) == 0.01, "n_significant_churn_rules"].iloc[0]) if len(sens_all) else None
plan17 = grp[(grp.feature == "plan") & (grp.level.astype(str) == "1-7")]
clean_auc = audit.get("clean (cutoff T)", {}).get("roc_auc")


def g(feature, level):
    r = grp[(grp.feature == feature) & (grp.level.astype(str) == level)]
    return float(r.churn_rate.iloc[0]) if len(r) else None


def sname(c):
    return f"Nhóm {c} – {seg.loc[c, 'name']}"


n_tab, n_fig = [0], [0]


def T(df, caption, **kw):
    n_tab[0] += 1
    return table(d, df, f"Bảng {n_tab[0]}. {caption}", **kw)


def FIG(path, caption, width=6.0):
    n_fig[0] += 1
    figure(d, path, f"Hình {n_fig[0]}. {caption}", width=width)


def P(text):
    return para(d, text)


def H(text, level):
    d.add_heading(text, level)


d = new_document()
d.add_heading("Khám phá hành vi người dùng và các mẫu rời bỏ dịch vụ nghe nhạc thuê bao KKBox", 0)
para(d, "Môn DAM501 – Data Mining · Đồ án cuối kỳ", bold=True)
para(d, "Nhóm thực hiện: Lộc Nguyễn Phúc, Khánh Phạm Duy Văn, Bình Chu Đức, Quân Đoàn Đàm, Hạnh Nguyễn Thị Hồng, "
        "Tùng Trần, Phùng Quý Quốc.")

# =============================================================================================
H("Tóm tắt", 1)
P(f"KKBox là dịch vụ nghe nhạc trả phí theo tháng ở Đài Loan. Nhóm dùng dữ liệu thật của KKBox (khoảng "
  f"{num(vol.get('unique_users_both_cohorts', 1_082_190) / 1e6, 2)} triệu người dùng, "
  f"{k(vol.get('user_logs_rows_scanned', 392e6) / 1e6)} triệu dòng nhật ký nghe nhạc) để trả lời câu hỏi: nhóm "
  "khách hàng nào hay bỏ dịch vụ, và những dấu hiệu nào báo trước việc đó.")
P("Kết quả chính:")
bullets(d, [x for x in [
    f"Việc khách có rời đi hay không gắn với cách họ trả tiền nhiều hơn là với việc họ nghe nhạc nhiều hay ít. "
    f"Khách đã tắt tự gia hạn rời đi {pct(g('auto_renew', 'off'), 0)}, khách bật tự gia hạn chỉ {pct(g('auto_renew', 'on'))}.",
    f"Trong các nhóm khách hàng tìm được, nhóm rủi ro nhất là khách mua gói dài hạn và trả tiền thủ công ({pct(seg.loc[risk_seg, 'share'])} số khách, "
    f"{pct(seg.loc[risk_seg, 'churn_rate'], 0)} không gia hạn). Nhóm này được phân cụm, luật kết hợp, mẫu tuần tự và "
    "phát hiện bất thường cùng tìm ra.",
    f"Chỉ 5 luật “nếu … thì …” đơn giản, gắn cờ {pct(cov5.share_users_flagged)} khách, đã bao phủ "
    f"{pct(cov5.share_churners_covered, 0)} số khách rời bỏ." if cov5 is not None else None,
    f"Mô hình dự báo xếp hạng khách theo rủi ro khá tốt (ROC-AUC {num(lg.get('roc_auc'))}): liên hệ 10% khách rủi "
    f"ro nhất sẽ tiếp cận {pct(lg.get('recall@10%'), 0)} số khách sắp rời đi.",
    "Phiên bản trước của nhóm công bố ROC-AUC 0,977, nhưng có dùng nhầm dữ liệu của tháng cần dự báo. Đo trên "
    f"cùng một phép chia, lỗi này làm ROC-AUC tăng ảo khoảng {num(audit.get('v1 recipe (unfiltered *_v2 files)', {}).get('roc_auc', 0) - (clean_auc or 0), 2)}. "
    "Bản này sửa lỗi đó và làm sạch thêm các bản ghi lỗi.",
] if x])

# =============================================================================================
H("Giải thích một số thuật ngữ", 1)
T(pd.DataFrame([
    ("Rời bỏ (churn)", "Khách không mua gói mới trong vòng 30 ngày sau khi gói cũ hết hạn (định nghĩa của KKBox)."),
    ("Mốc quan sát", "Ngày mà ta “đứng” để nhìn về quá khứ. Mọi thông tin dùng để phân tích phải xảy ra trước ngày này."),
    ("Đặc trưng", "Một con số mô tả khách hàng, ví dụ số lần hủy gói hay số ngày nghe nhạc trong 60 ngày."),
    ("Rò rỉ dữ liệu", "Vô tình dùng thông tin của tương lai (sau mốc quan sát) khiến kết quả trông tốt hơn thực tế."),
    ("Support", "Tỉ lệ khách hàng thỏa mãn một điều kiện. Support 2% nghĩa là 2 trong 100 khách."),
    ("Confidence", "Trong số khách thỏa điều kiện, bao nhiêu phần trăm rời bỏ."),
    ("Lift", f"Confidence chia cho tỉ lệ rời bỏ chung ({pct(base)}). Lift 3 nghĩa là rủi ro gấp 3 lần bình thường; "
             "lift dưới 1 nghĩa là nhóm đó ít rời bỏ hơn bình thường."),
    ("Tập holdout", "Một phần dữ liệu (30%) được giữ lại, không dùng khi tìm mẫu, chỉ dùng để kiểm tra lại."),
    ("Silhouette", "Điểm đo các cụm tách nhau rõ đến đâu, từ −1 đến 1. Trên 0,5 là rõ, 0,25–0,5 là vừa phải."),
    ("ARI, Jaccard", "Đo hai lần phân cụm giống nhau đến đâu; càng gần 1 càng ổn định."),
    ("ROC-AUC", "Xác suất mô hình xếp một khách sẽ rời đi cao hơn một khách ở lại. 0,5 tương đương dự báo ngẫu nhiên."),
    ("PR-AUC", "Độ chính xác trung bình khi tìm khách sẽ rời đi (tính bằng average precision); dự báo ngẫu nhiên chỉ đạt bằng tỉ lệ rời bỏ chung."),
    ("Recall ở top 10%", "Trong 10% khách bị đánh giá rủi ro nhất có bao nhiêu phần trăm tổng số khách rời bỏ."),
    ("PSI", "Đo một đặc trưng thay đổi phân bố giữa hai tháng. Dưới 0,1 là ổn định, trên 0,25 là thay đổi lớn."),
    ("SHAP", "Cách đo mỗi đặc trưng đẩy dự báo của mô hình lên hay xuống bao nhiêu."),
], columns=["Thuật ngữ", "Ý nghĩa"]), "Các thuật ngữ dùng trong báo cáo")

# =============================================================================================
H("1. Giới thiệu và bài toán kinh doanh", 1)
P("Phần lớn gói thuê bao của KKBox có thời hạn 30 ngày, nên việc duy trì khách hàng phụ thuộc vào quyết định gia "
  "hạn hằng tháng. Giữ một khách cũ rẻ hơn nhiều so với tìm một khách mới. Vì vậy công ty cần biết không chỉ ai "
  "sắp rời đi, mà còn vì sao, để chọn cách giữ chân phù hợp cho từng nhóm.")
P("Nhóm đặt ra ba câu hỏi:")
bullets(d, [
    "Khách hàng chia thành những nhóm hành vi nào, và nhóm nào hay rời bỏ? (phân cụm)",
    "Những tổ hợp đặc điểm nào, và những chuỗi giao dịch nào, đi kèm với việc rời bỏ? (luật kết hợp, mẫu tuần tự)",
    "Có thể dự báo trước ai sẽ rời bỏ không, và yếu tố nào quan trọng nhất? (phân lớp – phương pháp bổ trợ)",
])
P("Đề bài coi trọng việc khám phá mẫu hơn việc dự báo, nên phân lớp chỉ đóng vai trò hỗ trợ giải thích.")
P("Cách làm quan trọng nhất của đồ án là chọn một mốc quan sát cho mỗi nhóm khách và chỉ dùng dữ liệu trước "
  "mốc đó (Hình 2). Làm vậy thì kết quả phản ánh đúng tình huống thật: công ty phải đoán trước khi khách quyết định.")
P("Báo cáo sử dụng định nghĩa rời bỏ và danh sách khách hàng hằng tháng do KKBox cung cấp. Dữ liệu chỉ thuộc thị "
  "trường Đài Loan; thông tin khách nào rời bỏ, khách nào ở lại chỉ có cho tháng 2 và tháng 3 năm 2017.")
FIG(F / "diagram_pipeline.png", "Các bước thực hiện dự án, từ dữ liệu gốc đến khuyến nghị")
FIG(F / "diagram_timeline.png", "Cách cắt dữ liệu theo mốc quan sát cho hai nhóm khách hàng")

# =============================================================================================
H("2. Mô tả dữ liệu", 1)
P("Dữ liệu lấy từ cuộc thi “WSDM – KKBox's Churn Prediction Challenge” trên Kaggle (2017). Đây là dữ liệu thật, "
  "mã khách hàng đã được ẩn danh. Các bảng nối với nhau qua mã khách hàng.")
T(pd.DataFrame([
    ("train.csv", "Khách hết hạn tháng 2/2017 và nhãn rời bỏ", "Huấn luyện mô hình, kiểm tra độ ổn định"),
    ("train_v2.csv", "Khách hết hạn tháng 3/2017 và nhãn rời bỏ", "Khai phá mẫu, kiểm tra mô hình"),
    ("members_v3.csv", "Thành phố, tuổi, giới tính, kênh đăng ký, ngày đăng ký", "Thông tin hồ sơ"),
    ("transactions.csv và _v2", "Từng lần mua gói: độ dài gói, giá, tự gia hạn, hủy, ngày hết hạn", "Đặc trưng thuê bao, chuỗi giao dịch"),
    ("user_logs.csv (28 GB)", "Mỗi ngày nghe bao nhiêu bài, nghe hết hay bỏ qua, bao nhiêu giây", "Đặc trưng nghe nhạc"),
    ("user_logs_v2.csv", "Nhật ký nghe tháng 3/2017", "Chỉ dùng để đo mức rò rỉ dữ liệu"),
], columns=["Bảng", "Nội dung", "Dùng để"]), "Các bảng dữ liệu")
if manifest:
    c = manifest["cohorts"]
    T(pd.DataFrame([
        ("Tháng 2 (huấn luyện)", "31/01/2017", k(c["feb"]["n_users"]), pct(c["feb"]["churn_rate"]), pct(c["feb"]["share_has_logs"], 0)),
        ("Tháng 3 (khai phá, kiểm tra)", "28/02/2017", k(c["mar"]["n_users"]), pct(c["mar"]["churn_rate"]), pct(c["mar"]["share_has_logs"], 0)),
    ], columns=["Nhóm khách", "Mốc quan sát", "Số khách", "Tỉ lệ rời bỏ", "Có nhật ký nghe"]), "Hai nhóm khách hàng sau xử lý")
P("Dữ liệu có nhiều kiểu thuộc tính: dạng phân loại (thành phố, kênh đăng ký, giới tính), dạng số (giá, số giờ "
  "nghe), dạng đếm (số giao dịch, số ngày nghe), dạng ngày tháng và dạng chuỗi sự kiện (lịch sử giao dịch).")

# =============================================================================================
H("3. Tiền xử lý và làm sạch dữ liệu", 1)
H("3.1 Các lỗi dữ liệu phát hiện được và cách xử lý", 2)
T(pd.DataFrame([
    ("Giao dịch có độ dài gói 0 ngày và giá niêm yết 0 (nhưng vẫn trả tiền)", k(dq.get("tx_plan_days_zero", 0)),
     "Bản ghi cũ thiếu thông tin gói. Coi là “không rõ gói”: không tính là đổi gói, không tính giảm giá."),
    ("Ngày hết hạn trước năm 2015", k(dq.get("tx_expiry_before_2015", 0)), "Ngày hỏng. Chuyển thành thiếu."),
    ("Ngày hết hạn sớm hơn ngày giao dịch", k(dq.get("tx_expiry_before_tx_date", 0)),
     "Phần lớn là giao dịch hủy (rút ngắn hạn). Giữ lại vì có thể hợp lệ."),
    ("Khách có ngày đăng ký sau mốc quan sát", k((dq.get("users_registered_after_cutoff") or {}).get("mar", 0)) + " (tháng 3)",
     "Thông tin không thể có tại mốc quan sát. Chuyển thâm niên thành thiếu."),
    ("Tuổi ngoài khoảng 10–80 (âm, 0, trên 100)", pct(eda.get("age_missing"), 0) + " khách",
     "Coi là thiếu; không dùng tuổi trong phân cụm."),
    ("Giới tính bỏ trống", pct(eda.get("gender_unknown"), 0) + " khách", "Ghi là “không rõ”."),
    ("Thời gian nghe trong ngày âm hoặc trên 24 giờ", "10.442 dòng nhật ký", "Giới hạn trong khoảng 0–24 giờ."),
    ("Khách không có hồ sơ thành viên", "11% khách", "Giữ lại, thêm cờ “có hồ sơ”."),
    ("Khách không có nhật ký nghe trong 60 ngày", pct(eda.get("share_no_logs"), 0) + " khách",
     "Giữ lại, thêm cờ; trong phân cụm được tách thành nhóm riêng (mục 5.1)."),
    ("Giao dịch trùng giữa hai file giao dịch", "–", "Gộp hai file và bỏ dòng trùng."),
], columns=["Vấn đề", "Số lượng", "Cách xử lý"]), "Các lỗi dữ liệu và cách xử lý")
P("Lỗi đáng kể nhất là các bản ghi gói 0 ngày. Tuy chỉ chiếm khoảng 3,5% giao dịch, chúng làm khoảng 78% số lần "
  "“đổi gói” trước khi sửa là đổi giả (30 ngày → 0 → 30 ngày). Lỗi này được phát hiện khi nhóm rà soát lại phương "
  "pháp, và đã được sửa ở mọi bước dùng thông tin gói.")
H("3.2 Biến đổi dữ liệu", 2)
bullets(d, [
    f"Dữ liệu lớn: file nhật ký 28 GB không được giải nén ra ổ đĩa. Chương trình đọc từng phần trong lúc giải nén "
    f"và chỉ giữ những ngày cần dùng, từ {k(vol.get('user_logs_rows_scanned', 392e6) / 1e6)} triệu dòng còn "
    f"{num(vol.get('log_rows_kept_in_window', 40.9e6) / 1e6, 1)} triệu dòng.",
    "Gộp hai file giao dịch và bỏ mọi giao dịch sau mốc quan sát. Khi một khách có nhiều giao dịch cùng ngày, thứ "
    "tự được cố định theo một quy tắc chung để “giao dịch cuối” luôn xác định.",
    "Tạo 32 đặc trưng số cho mỗi khách. Nhóm thuê bao (17): số giao dịch, số lần hủy, tỉ lệ tự gia hạn, giảm giá, đổi "
    "gói và thông tin giao dịch cuối (độ dài gói, giá, có tự gia hạn không, còn bao nhiêu ngày tới hạn). Nhóm nghe "
    "nhạc (12, trong 60 ngày trước mốc): số ngày nghe, số giờ, tỉ lệ nghe hết bài, tỉ lệ bỏ qua, nghe tăng hay giảm, "
    "bao lâu chưa nghe, nghe đều hay thất thường. Nhóm hồ sơ (3): tuổi, thời gian sử dụng dịch vụ, có hồ sơ hay "
    "không. Cộng 4 biến phân loại (thành phố, kênh đăng ký, giới tính, phương thức thanh toán) là 36 biến đầu vào.",
    "Tạo chuỗi giao dịch cho mỗi khách (tối đa 12 giao dịch gần nhất), mỗi giao dịch được mô tả bằng các sự kiện "
    "như “trả tự động”, “trả thủ công”, “hủy”, “đổi gói”.",
    f"Giữ nguyên tỉ lệ rời bỏ thật (khoảng {pct(base, 0)}), không lấy mẫu cân bằng, để các tỉ lệ trong báo cáo đúng "
    "với thực tế.",
    "Các giá trị thiếu chỉ được điền sau khi chia dữ liệu, và chỉ dựa trên phần huấn luyện. Với phân cụm, các biến "
    "có đuôi dài (số giờ nghe, số giao dịch…) được lấy log rồi chuẩn hoá.",
])

# =============================================================================================
H("4. Phân tích khám phá dữ liệu", 1)
eff = csv("eda_effects.csv")
if len(eff):
    e = eff.head(8)[["feature", "median_churn", "median_stay", "auc_effect"]].copy()
    e["feature"] = e.feature.map(vi.feature)
    e["auc_effect"] = (e.auc_effect - 0.5).abs() + 0.5
    e.columns = ["Đặc trưng", "Trung vị (rời bỏ)", "Trung vị (ở lại)", "Khả năng phân biệt (0,5–1)"]
    T(e, "Tám đặc trưng phân biệt khách rời bỏ và khách ở lại rõ nhất")
assoc = csv("eda_association.csv", index_col=0)
if len(assoc):
    a = assoc.reset_index()
    a = pd.DataFrame({"Yếu tố": a.iloc[:, 0].map(lambda x: vi.GROUPS.get(x, x)), "Mức liên hệ (Cramér's V, 0–1)": a.cramers_v})
    T(a, "Mức liên hệ giữa từng yếu tố và việc rời bỏ (càng gần 1 càng mạnh)")
FIG(F / "eda_churn_by_group.png", "Tỉ lệ rời bỏ theo từng nhóm; vạch đứt là mức chung, thanh nhỏ là khoảng tin cậy 95%")
FIG(F / "eda_distributions.png", "So sánh phân bố một số hành vi giữa khách rời bỏ và khách ở lại")
FIG(F / "eda_churn_by_tenure.png", "Tỉ lệ rời bỏ theo số năm sử dụng", width=4.5)
FIG(F / "eda_correlation.png", "Mức tương quan giữa các đặc trưng", width=5.5)
P("Những gì dữ liệu cho thấy:")
bullets(d, [
    f"Cách trả tiền là yếu tố mạnh nhất. Khách tắt tự gia hạn ở lần mua cuối rời đi {pct(g('auto_renew', 'off'))}, "
    f"khách bật tự gia hạn chỉ {pct(g('auto_renew', 'on'))}. Khách có lần giao dịch cuối là hủy gói rời đi "
    f"{pct(g('cancelled_last', 'yes'), 0)}.",
    f"Độ dài gói cũng quan trọng. Gói thông thường (8–30 ngày) có tỉ lệ rời bỏ {pct(g('plan', '8-30'))}; gói rất ngắn "
    f"(1–7 ngày) là {pct(g('plan', '1-7'), 0)} (nhóm nhỏ, {k(plan17.n.iloc[0]) if len(plan17) else '–'} khách) và gói dài hơn "
    f"30 ngày là {pct(g('plan', '>30'), 0)}.",
    f"Kênh đăng ký tạo khác biệt lớn: kênh 7, chiếm gần một nửa số khách, có tỉ lệ rời bỏ {pct(g('registered_via', '7.0'))}, "
    f"còn kênh 4 là {pct(g('registered_via', '4.0'), 0)} và kênh 3 là {pct(g('registered_via', '3.0'), 0)}.",
    "Khi xét riêng từng yếu tố, số ngày không nghe gần đây, xu hướng nghe tăng hay giảm và thời gian sử dụng dịch "
    "vụ đều có mức liên hệ thấp với việc rời bỏ (Cramér's V dưới 0,05). Đây là điểm khác lớn nhất so với phiên bản "
    "trước, vì phiên bản trước đã dùng nhầm nhật ký nghe của tháng 3, lúc nhiều khách đã ngừng dịch vụ.",
])
P("Từ đó nhóm chọn kỹ thuật: luật kết hợp để tìm các tổ hợp giữa cách trả tiền, loại gói và kênh; mẫu tuần tự "
  "vì thứ tự các lần mua có thể chứa tín hiệu; phân cụm để kết hợp cả thông tin thuê bao lẫn thói quen nghe.")

# =============================================================================================
H("5. Các phương pháp khai phá và kết quả", 1)
H("5.1 Phân cụm khách hàng (K-Means)", 2)
P(f"Phân cụm được làm theo hai bước. Bước 1: {pct(eda.get('share_no_logs'), 0)} khách không có nhật ký nghe nào "
  "trong 60 ngày được tách thành một nhóm riêng theo quy tắc. Ở phiên bản trước, những khách này được gán giá trị "
  "cố định (“60 ngày không nghe”) rồi đưa vào K-Means, và thuật toán chỉ tìm lại đúng nhóm đó: 99% khách của một cụm "
  "là khách không có nhật ký. Cụm như vậy là do cách gán giá trị tạo ra, không phải cấu trúc thật, nên được tách ra "
  "bằng quy tắc.")
P(f"Bước 2: với {k(cl.get('n_loggers', 0))} khách có nhật ký, K-Means dùng các đặc trưng về thuê bao và nghe nhạc (đã "
  f"lấy log và chuẩn hoá), trên mẫu {k(cl.get('n_users', 200000))} khách. Trong 16 đặc trưng ban đầu, những đặc trưng gần "
  f"như trùng nhau (tương quan trên 0,8) chỉ giữ lại một, nên bỏ {len(cl.get('dropped_redundant', []))} đặc trưng (tổng số "
  "giờ nghe, tỉ lệ bỏ qua bài, độ thất thường), còn 13. Lý do: nếu giữ cả, chiều “mức độ nghe” bị tính nhiều lần và "
  "lấn át thông tin thanh toán; mục 6.7 so sánh cách làm này với các cách khác. Nhãn rời bỏ không được dùng ở bất kỳ "
  "bước nào của phân cụm, kể cả khi chọn số cụm.")
P(f"Số cụm được chọn chỉ dựa trên tiêu chí nội tại (không dùng tỉ lệ rời bỏ): lấy số cụm lớn nhất, không quá "
  f"{cl.get('k_max', 7)} (để tổng số nhóm không quá 8, còn dùng được trong thực tế), có silhouette ít nhất 75% điểm cao nhất "
  f"và độ ổn định (ARI qua 5 lần chạy lại toàn bộ trên mẫu khác) ít nhất 0,8. Silhouette cao nhất là "
  f"{num(cl.get('silhouette_macro'))} ở {cl.get('k_macro')} cụm; lời giải {cl.get('k')} cụm đạt cả hai điều kiện (silhouette "
  f"{num(cl.get('silhouette'))}, ARI {num(cl.get('stability_ari'), 3)}) nên được chọn (Bảng {n_tab[0] + 1}).")
q = csv("cluster_quality.csv")
if len(q):
    t = q[["k", "silhouette", "davies_bouldin", "stability_ari", "min_cluster_share"]].copy()
    t.columns = ["Số cụm", "Silhouette", "Davies–Bouldin", "Độ ổn định (ARI)", "Cụm nhỏ nhất"]
    T(t, "Chất lượng và độ ổn định theo số cụm (khách có nhật ký nghe)",
      fmt={"Silhouette": "{:.3f}", "Davies–Bouldin": "{:.2f}", "Độ ổn định (ARI)": "{:.2f}", "Cụm nhỏ nhất": "{:.1%}"})
FIG(F / "cluster_selection.png", "Các chỉ số chọn số cụm; vạch đỏ là số cụm được chọn")
s = seg.reset_index()[["cluster", "name", "n", "share", "churn_rate", "lift"]].sort_values("churn_rate")
jac = csv("cluster_jaccard.csv").set_index("cluster")
s["stab"] = s.cluster.map(lambda c: jac.jaccard_mean.get(c, float("nan")))
s.columns = ["Nhóm", "Đặc điểm", "Số khách", "Tỉ trọng", "Tỉ lệ rời bỏ", "Lift", "Độ ổn định (Jaccard)"]
T(s, "Các nhóm khách hàng tháng 3 (nhóm không có nhật ký nghe được tách theo quy tắc nên không có Jaccard)",
  fmt={"Số khách": "{:,.0f}", "Tỉ trọng": "{:.1%}", "Tỉ lệ rời bỏ": "{:.1%}", "Lift": "{:.2f}", "Độ ổn định (Jaccard)": "{:.2f}"})
FIG(F / "cluster_churn.png", "Tỉ lệ rời bỏ của từng nhóm")
FIG(F / "cluster_heatmap.png", "Tâm của từng cụm (đỏ: cao hơn trung bình, xanh: thấp hơn trung bình, đơn vị độ lệch chuẩn)")
P("Cách đọc các nhóm:")
bullets(d, [f"{sname(c)}: {pct(seg.loc[c, 'share'], 0)} khách, tỉ lệ rời bỏ {pct(seg.loc[c, 'churn_rate'])}." for c in seg.index] + [
    "Nhóm gói dài hạn khác hẳn phần còn lại: độ dài gói và số tiền trả ở lần mua cuối cao hơn trung bình 4–6 độ lệch "
    "chuẩn, tỉ lệ tự gia hạn thấp hơn 2,2 độ lệch chuẩn, rất ít giao dịch.",
    "Nhóm “nghe nhiều nhưng trả thủ công” nghe nhiều hơn trung bình nhưng tỉ lệ rời bỏ gấp khoảng 2,5 lần mức chung; "
    "nhóm “trung thành” cũng nghe nhiều nhưng tự gia hạn và chỉ rời bỏ khoảng 3%. Hai nhóm có thói quen nghe giống nhau, "
    "khác nhau ở cách trả tiền, và rủi ro chênh nhau khoảng 7 lần.",
    "Nhóm “mới hoặc quay lại” có thâm niên thấp, ít giao dịch và mức nghe tăng mạnh; tỉ lệ rời bỏ khoảng 17%. Nhóm "
    "“nghe thưa, lâu chưa nghe” lại ít rời bỏ hơn mức chung, vì phần lớn vẫn tự gia hạn.",
])
alg = csv("cluster_algorithms.csv")
if len(alg):
    a = alg[["algorithm", "n_clusters", "noise_share", "silhouette", "ari_vs_kmeans", "churn_rate_spread"]].copy()
    a["algorithm"] = a.algorithm.replace({"Agglomerative (Ward)": "Phân cụm phân cấp (Ward)", "Gaussian mixture": "Hỗn hợp Gauss (GMM)"})
    a.columns = ["Thuật toán", "Số cụm", "Điểm bị coi là nhiễu", "Silhouette", "Giống K-Means (ARI)", "Chênh lệch tỉ lệ rời bỏ"]
    T(a, "So sánh bốn thuật toán phân cụm trên cùng 15.000 khách có nhật ký",
      fmt={"Điểm bị coi là nhiễu": "{:.0%}", "Silhouette": "{:.3f}", "Giống K-Means (ARI)": "{:.2f}", "Chênh lệch tỉ lệ rời bỏ": "{:.1%}"})
alg_i = alg.set_index("algorithm") if len(alg) else None
jac_s = jac.jaccard_mean.sort_values()
weak = jac_s.index[0]
P(f"Đánh giá: các cụm tách nhau yếu (silhouette {num(cl.get('silhouette'), 2)}, Davies–Bouldin "
  f"{num(cl.get('davies_bouldin'), 2)}) nhưng rất ổn định khi chạy lại trên mẫu khác (ARI {num(cl.get('stability_ari'), 3)}). "
  f"Phân cụm phân cấp Ward cho cách chia khá giống K-Means (ARI {num(alg_i.loc['Agglomerative (Ward)', 'ari_vs_kmeans'], 2)}) "
  f"và cùng mức chênh lệch rủi ro; HDBSCAN chỉ tìm được {int(alg_i.loc['HDBSCAN', 'n_clusters'])} vùng dày đặc và coi "
  f"{pct(alg_i.loc['HDBSCAN', 'noise_share'], 0)} điểm là nhiễu, nghĩa là khách hàng không tụ thành các đám tách biệt rõ. "
  f"Xét từng cụm, cụm kém ổn định nhất là “{seg.loc[weak, 'name']}” (Jaccard {num(jac_s.iloc[0], 2)}), nên dùng thận trọng. "
  "Các nhóm gói dài hạn, nghe nhiều nhưng trả thủ công, mới hoặc quay lại đều rất ổn định (Jaccard trên 0,9).")

H("5.2 Luật kết hợp (FP-Growth)", 2)
P(f"Mỗi khách được mô tả bằng một tập đặc điểm dạng có/không, ví dụ “tắt tự gia hạn”, “kênh đăng ký 9”, “ít ngày "
  f"nghe” ({ru.get('n_items', 45)} đặc điểm). FP-Growth tìm mọi tổ hợp tối đa 3 đặc điểm xuất hiện ở ít nhất 1% khách "
  f"({k(ru.get('n_itemsets', 3000))} tổ hợp). Với mỗi tổ hợp, nhóm tính tỉ lệ rời bỏ và lift.")
P("Để tránh chọn nhầm những luật chỉ đúng do may mắn, nhóm làm thêm ba bước: (1) kiểm định thống kê có hiệu chỉnh "
  "cho việc thử hàng nghìn luật cùng lúc; (2) bỏ những luật dài mà không tốt hơn luật ngắn hơn chứa trong nó; (3) "
  "tìm luật trên 70% dữ liệu rồi đo lại trên 30% còn lại.")
cr = csv("churn_rules.csv")
t = cr.head(12)[["rule", "support", "confidence", "lift", "confidence_holdout"]].copy()
t["rule"] = t.rule.map(vi.rule)
t.columns = ["Nếu khách có đặc điểm", "Support", "Tỉ lệ rời bỏ", "Lift", "Tỉ lệ rời bỏ khi kiểm tra lại"]
T(t, "Các tổ hợp đặc điểm đi kèm tỉ lệ rời bỏ cao",
  fmt={"Support": "{:.1%}", "Tỉ lệ rời bỏ": "{:.1%}", "Lift": "{:.1f}", "Tỉ lệ rời bỏ khi kiểm tra lại": "{:.1%}"})
rr = csv("retention_rules.csv")
t = rr.head(6)[["rule", "support", "confidence", "lift", "confidence_holdout"]].copy()
t["rule"] = t.rule.map(vi.rule)
t.columns = ["Nếu khách có đặc điểm", "Support", "Tỉ lệ rời bỏ", "Lift", "Tỉ lệ rời bỏ khi kiểm tra lại"]
T(t, "Các tổ hợp đặc điểm đi kèm tỉ lệ rời bỏ thấp",
  fmt={"Support": "{:.1%}", "Tỉ lệ rời bỏ": "{:.1%}", "Lift": "{:.2f}", "Tỉ lệ rời bỏ khi kiểm tra lại": "{:.1%}"})
FIG(F / "rules_scatter.png", "Mỗi điểm là một tổ hợp đặc điểm; điểm viền đỏ là luật được chọn")
P(f"Kết quả: {k(n_sig) if n_sig else '–'} luật có lift ≥ 1,5 vượt kiểm định thống kê. Sau khi bỏ các luật thừa, báo cáo "
  f"giữ {ru.get('n_churn_rules')} luật rủi ro mạnh nhất và {ru.get('n_retention_rules')} luật có tỉ lệ rời bỏ thấp (một số luật "
  f"trong nhóm này trùng nhau về tập khách). Khi đo lại trên 30% dữ liệu giữ riêng, tỉ lệ rời bỏ của các luật lệch trung "
  f"bình chỉ {num(ru.get('mean_abs_conf_drift', 0.01) * 100, 1)} điểm phần trăm.")
bullets(d, [
    "Rủi ro nhất là khách mua gói dài hơn 30 ngày: khoảng 83% không gia hạn. Đây cũng chính là nhóm gói dài hạn ở phần phân cụm.",
    "Nhóm khách tắt tự gia hạn có tỉ lệ rời bỏ cao gấp khoảng 4,6 lần mức chung. Nếu khách vừa tắt tự gia hạn vừa ít "
    "nghe, tỉ lệ rời bỏ vượt 50%. Như vậy thói quen nghe có ích khi đi kèm thông tin thanh toán, dù riêng nó thì yếu.",
    "Khách đã hết hạn ngay tại mốc quan sát rời đi khoảng 50%.",
    "Tỉ lệ rời bỏ thấp nhất thuộc về khách mới (dưới 1 năm) đăng ký qua kênh 7: chỉ 2,2%.",
])
if len(cov):
    t = cov[["top_k_rules", "share_users_flagged", "share_churners_covered", "churn_rate_in_flagged"]].copy()
    t.columns = ["Dùng k luật đầu", "Khách bị gắn cờ", "Khách rời bỏ được bao phủ", "Tỉ lệ rời bỏ trong nhóm gắn cờ"]
    T(t, "Độ phủ của các luật trên 30% dữ liệu giữ riêng",
      fmt={"Khách bị gắn cờ": "{:.1%}", "Khách rời bỏ được bao phủ": "{:.1%}", "Tỉ lệ rời bỏ trong nhóm gắn cờ": "{:.1%}"})
if cov5 is not None:
    P(f"Bảng độ phủ cho thấy giá trị thực tế của luật: chỉ cần 5 luật đầu, gắn cờ {pct(cov5.share_users_flagged)} khách, "
      f"là đã bao phủ {pct(cov5.share_churners_covered, 0)} số khách rời bỏ, và {pct(cov5.churn_rate_in_flagged, 0)} khách bị "
      "gắn cờ thật sự rời đi. Thêm luật thì độ phủ tăng chậm và độ chính xác giảm dần.")
sens = csv("rule_sensitivity.csv")
if len(sens):
    P("Khi đổi ngưỡng support tối thiểu từ 0,5% lên 5%, số luật đạt yêu cầu giảm từ "
      f"{k(sens.n_significant_churn_rules.iloc[0])} xuống {k(sens.n_significant_churn_rules.iloc[-1])}. Các chủ đề chính "
      "(tắt tự gia hạn, gói dài hạn, kênh 9) vẫn xuất hiện, nhưng thứ tự từng luật thay đổi theo ngưỡng. Ngưỡng 1% được "
      "chọn để mỗi luật dựa trên ít nhất khoảng 6.800 khách.")

H("5.3 Mẫu tuần tự (PrefixSpan)", 2)
P("Luật kết hợp chỉ nhìn trạng thái hiện tại. Mẫu tuần tự xét thêm thứ tự: điều gì xảy ra trước, điều gì xảy ra sau "
  "trong lịch sử mua gói. Mỗi lần giao dịch được mô tả bằng các sự kiện như trả tự động, trả thủ công, hủy, đổi gói "
  "(chỉ tính giữa các gói có thông tin), gói dài, lỡ hạn. PrefixSpan tìm các chuỗi xuất hiện ở ít nhất 1% khách, rồi "
  f"kiểm định và đo lại trên phần dữ liệu giữ riêng ({k(sq.get('n_users', 300000))} khách, {k(sq.get('n_tested', 0))} "
  f"chuỗi được kiểm tra, {sq.get('n_selected')} chuỗi có ý nghĩa).")
sp = csv("sequential_patterns.csv")
t = sp.groupby("direction", group_keys=False).head(7)[["pattern", "direction", "support_train", "conf_train",
                                                        "lift_train", "conf_holdout"]].copy()
t["pattern"] = t.pattern.map(vi.pattern)
t["direction"] = t.direction.map({"churn": "rời bỏ cao", "retain": "rời bỏ thấp"})
t.columns = ["Chuỗi sự kiện (→ là “sau đó”)", "Loại", "Support", "Tỉ lệ rời bỏ", "Lift", "Khi kiểm tra lại"]
T(t, "Các chuỗi giao dịch nổi bật", fmt={"Support": "{:.1%}", "Tỉ lệ rời bỏ": "{:.1%}", "Lift": "{:.1f}", "Khi kiểm tra lại": "{:.1%}"})
FIG(F / "seq_top_patterns.png", "Lift của các chuỗi rủi ro nhất trên tháng 3 và tháng 2")
bullets(d, [
    "Chuỗi gắn với tỉ lệ rời bỏ cao nhất gồm các giao dịch theo thứ tự: trả tiền thủ công, sau đó mua gói dài, rồi "
    "tiếp tục mua gói dài. Khoảng 70% khách có chuỗi này rời bỏ.",
    "Đổi gói (giữa các gói thật) xuất hiện ở khoảng 7% khách và đi kèm tỉ lệ rời bỏ khoảng 33%; hủy rồi đổi gói, hoặc "
    "trả tự động rồi đổi gói, khoảng 40%. Sau khi sửa lỗi gói 0 ngày, tín hiệu này yếu hơn trước nhưng vẫn rõ.",
    "Các chuỗi có “lỡ hạn” (mua lại sau khi gói cũ hết hạn hơn 30 ngày) và trả thủ công đi kèm tỉ lệ rời bỏ khoảng 35%.",
    "Ngược lại, khách trả tự động liên tục chỉ rời bỏ khoảng 5%.",
])
P("Ưu điểm của mẫu tuần tự là các sự kiện này xảy ra ngay lúc khách giao dịch, nên có thể dùng làm tín hiệu cảnh báo "
  "sớm. Nhóm chưa đo được các sự kiện đó xảy ra trước ngày hết hạn bao lâu.")

H("5.4 Dự báo khách rời bỏ (phương pháp bổ trợ)", 2)
P(f"Mô hình được huấn luyện bằng dữ liệu của {k(cf.get('n_train', 0))} khách tháng 2, sau đó được đánh giá trên "
  f"{k(cf.get('n_test', 0))} khách tháng 3 để mô phỏng việc dự báo cho tháng tiếp theo. Nhóm so sánh năm cách: đoán "
  "theo tỉ lệ chung, hồi quy logistic, cây quyết định, rừng ngẫu nhiên và LightGBM.")
t = oot[["model", "roc_auc", "pr_auc", "precision@10%", "recall@10%"]].copy()
t["model"] = t.model.map(vi.MODELS)
t.columns = ["Mô hình", "ROC-AUC", "PR-AUC", "Độ chính xác ở top 10%", "Recall ở top 10%"]
T(t, "Kết quả dự báo trên tháng 3", fmt={"ROC-AUC": "{:.3f}", "PR-AUC": "{:.3f}", "Độ chính xác ở top 10%": "{:.1%}", "Recall ở top 10%": "{:.1%}"})
FIG(F / "clf_curves.png", "Đường ROC, đường độ chính xác – độ phủ và đường hiệu chuẩn trên tháng 3")
FIG(F / "clf_gains.png", "Liên hệ x% khách rủi ro nhất thì bắt được bao nhiêu phần trăm khách sẽ rời đi", width=4.5)
P(f"LightGBM đạt kết quả tốt nhất, với ROC-AUC {num(lg.get('roc_auc'))} và PR-AUC {num(lg.get('pr_auc'))}. Để so sánh, "
  f"dự báo ngẫu nhiên chỉ đạt PR-AUC khoảng {num(base, 3)}. Trong 10% khách bị đánh giá rủi ro nhất có "
  f"{pct(lg.get('precision@10%'), 0)} thật sự rời đi, và nhóm này chứa {pct(lg.get('recall@10%'), 0)} tổng số khách rời bỏ.")
if cv_lgb:
    P(f"Khi chỉ kiểm tra trong cùng tháng 2, ROC-AUC là {num(cv_lgb)}; sang tháng 3 giảm còn {num(lg.get('roc_auc'))}. Như "
      f"vậy đánh giá trong cùng tháng cho kết quả cao hơn đánh giá trên tháng kế tiếp khoảng "
      f"{num(cv_lgb - lg.get('roc_auc'))} ROC-AUC (nguyên nhân phân tích ở mục 6.5).")
abl = pd.DataFrame(cf.get("ablation", []))
t = abl[["feature_set", "n_features", "roc_auc", "pr_auc"]].copy()
t["feature_set"] = t.feature_set.map(vi.ABLATION)
t.columns = ["Nhóm đặc trưng được dùng", "Số biến", "ROC-AUC", "PR-AUC"]
T(t, "Mô hình dùng từng nhóm đặc trưng riêng lẻ", fmt={"ROC-AUC": "{:.3f}", "PR-AUC": "{:.3f}"})
P("Chỉ dùng thông tin thuê bao đã đạt gần như toàn bộ kết quả; chỉ dùng thói quen nghe thì chỉ nhỉnh hơn dự báo ngẫu "
  "nhiên một chút. Điều này khớp với phần khám phá dữ liệu.")
FIG(F / "clf_shap_summary.png", "Mức ảnh hưởng của từng đặc trưng lên dự báo của LightGBM (SHAP)", width=5.5)

mi = csv("model_improvement.csv")
if len(mi):
    H("5.5 Thử cải thiện mô hình", 2)
    P("Nhóm thử ba phương án: (1) thêm 13 đặc trưng về lịch sử thanh toán, như số lần bật/tắt tự gia hạn, số lần lỡ "
      "hạn, khoảng cách giữa các lần mua, trạng thái 3 lần mua gần nhất; (2) tinh chỉnh tham số LightGBM bằng công cụ "
      "Optuna (30 lần thử); (3) kết hợp cả hai. Việc tinh chỉnh chỉ dùng dữ liệu tháng 2; tháng 3 chỉ dùng để chấm điểm "
      "cuối cùng.")
    t = mi[["configuration", "n_features", "roc_auc", "pr_auc", "recall@10%", "delta_auc"]].copy()
    t["configuration"] = t.configuration.map(vi.CONFIGS)
    t.columns = ["Cách làm", "Số biến", "ROC-AUC", "PR-AUC", "Recall ở top 10%", "Tăng so với A"]
    T(t, "Kết quả thử cải thiện mô hình trên tháng 3",
      fmt={"ROC-AUC": "{:.4f}", "PR-AUC": "{:.4f}", "Recall ở top 10%": "{:.1%}", "Tăng so với A": "{:+.4f}"})
    FIG(F / "model_improvement.png", "Mức tăng ROC-AUC và mức đóng góp của các đặc trưng mới (màu cam)")
    b = mi.iloc[-1]
    P(f"Thêm đặc trưng thanh toán giúp tăng {num(mi.delta_auc.iloc[1], 4)}; tinh chỉnh tham số gần như không giúp "
      f"({num(mi.delta_auc.iloc[2], 4)}); kết hợp cả hai đạt ROC-AUC {num(b.roc_auc, 4)}. Mức tăng có thật về mặt thống kê "
      "nhưng rất nhỏ, và số khách rời bỏ bắt được ở top 10% không đổi. Đặc trưng mới hữu ích nhất là tỉ lệ tự gia hạn "
      "trong 3 lần mua gần nhất, khớp với chuỗi “trả tự động → trả thủ công” ở phần mẫu tuần tự. Hướng tiếp theo nên là "
      "bổ sung dữ liệu mới (lý do khách tắt tự gia hạn, lịch sử chăm sóc khách hàng) hơn là thay thuật toán. Các phần "
      "sau vẫn dùng mô hình LightGBM ban đầu để thống nhất số liệu.")

# =============================================================================================
H("6. Đánh giá và diễn giải", 1)
H("6.1 Kiểm tra rò rỉ dữ liệu", 2)
P("Để biết phiên bản trước bị thổi phồng bao nhiêu, nhóm giữ nguyên cùng một nhóm khách tháng 3 và cùng cách chia "
  "dữ liệu, rồi lần lượt thêm vào các thông tin xảy ra sau mốc quan sát.")
t = pd.DataFrame(va.get("leakage_audit", []))[["feature_set", "roc_auc", "auc_inflation_vs_clean"]]
t["feature_set"] = t.feature_set.map(vi.AUDIT)
t.columns = ["Dữ liệu dùng", "ROC-AUC", "Tăng ảo so với dữ liệu sạch"]
T(t, "Kết quả tăng lên bao nhiêu khi dùng nhầm dữ liệu sau mốc quan sát", fmt={"ROC-AUC": "{:.3f}", "Tăng ảo so với dữ liệu sạch": "{:+.3f}"})
FIG(F / "leakage_audit.png", "Xanh: chỉ dùng thông tin có trước mốc; đỏ: có dùng thông tin của tháng cần dự báo")
P(f"Thêm giao dịch tháng 3 làm ROC-AUC tăng từ {num(clean_auc)} lên "
  f"{num(audit.get('clean + March transactions', {}).get('roc_auc'))}. Lý do: khách không mua gói trong tháng 3 thì phần "
  "lớn chính là khách rời bỏ, tức mô hình đang “nhìn thấy” đáp án. Cách làm của phiên bản 1 cũng được lợi theo kiểu "
  f"này ({num(audit.get('v1 recipe (unfiltered *_v2 files)', {}).get('roc_auc'))}). Riêng với thói quen nghe, dùng nhật ký "
  f"tháng 3 làm điểm tăng từ {num(audit.get('listening only, clean', {}).get('roc_auc'))} lên "
  f"{num(audit.get('listening only, March logs (v1)', {}).get('roc_auc'))}: phần lớn tín hiệu nghe nhạc mà phiên bản 1 tìm "
  "được thực ra xuất hiện sau khi khách đã rời đi.")

H("6.2 Kết quả có lặp lại ở tháng khác không?", 2)
P(f"Nhóm lấy đúng các nhóm khách, luật và chuỗi tìm được trên tháng 3, áp lên khách tháng 2 rồi so sánh. Vì tỉ lệ rời "
  f"bỏ chung của hai tháng khác nhau ({pct(st.get('base_feb'))} và {pct(st.get('base_mar'))}), nhóm so sánh bằng lift.")
ct = csv("cluster_temporal.csv")
t = ct[["cluster", "name", "share_mar", "share_feb", "churn_mar", "churn_feb"]].copy()
t.columns = ["Nhóm", "Đặc điểm", "Tỉ trọng T3", "Tỉ trọng T2", "Rời bỏ T3", "Rời bỏ T2"]
T(t, "Các nhóm khách hàng ở tháng 3 và tháng 2", fmt={c: "{:.1%}" for c in ["Tỉ trọng T3", "Tỉ trọng T2", "Rời bỏ T3", "Rời bỏ T2"]})
FIG(F / "stability.png", "Trái: tỉ lệ rời bỏ của từng nhóm ở hai tháng. Phải: lift của các luật ở hai tháng (càng gần đường chéo càng ổn định)")
bullets(d, [
    f"Nhóm khách hàng: tỉ trọng mỗi nhóm gần như không đổi và thứ tự rủi ro gần như giữ nguyên (tương quan Spearman "
    f"của lift {num(st.get('segment_lift_spearman'), 2)}). Phân cụm lại từ đầu trên tháng 2 cho cách chia giống ở mức khá "
    f"(ARI {num(st.get('ari_feb_refit_vs_mar_model'), 2)}): chia càng chi tiết thì ranh giới giữa các nhóm gần nhau càng dễ "
    "dịch chuyển giữa các tháng (lời giải 3 nhóm lặp lại tốt hơn, ARI 0,98, nhưng kém hữu ích).",
    f"Luật kết hợp: {round(st.get('rule_share_same_direction', 0) * st.get('rules_checked', 0))} trên "
    f"{st.get('rules_checked')} luật vẫn gắn với mức rủi ro cao hoặc thấp như ở tháng 3 (tương quan log-lift "
    f"{num(st.get('rule_log_lift_pearson'), 2)}).",
    f"Chuỗi giao dịch: {pct(sq.get('share_same_direction_feb'), 0)} số chuỗi được báo cáo giữ nguyên ý nghĩa (tương quan "
    f"log-lift {num(sq.get('corr_lift_mar_feb'), 2)}).",
    f"Mức rủi ro tuyệt đối thì thay đổi: nhóm gói dài hạn rời bỏ {pct(ct.churn_mar.max(), 0)} ở tháng 3 nhưng "
    f"{pct(ct.loc[ct.churn_mar.idxmax(), 'churn_feb'], 0)} ở tháng 2, dù vẫn là nhóm rủi ro nhất. Vì vậy khi áp dụng nên "
    "dựa vào thứ hạng rủi ro, không nên dựa vào một ngưỡng phần trăm cố định.",
])

H("6.3 Xác suất dự báo và giá trị kinh doanh", 2)
calt = csv("calibration.csv")
P("Mô hình học ở tháng 2 (6,4% rời bỏ) nên dự báo trung bình 6,4%, trong khi tháng 3 thực tế là 9,0%. Nhóm thử điều "
  "chỉnh bằng thuật toán EM, một cách ước lượng lại tỉ lệ chung mà không cần biết đáp án, nhưng thuật toán vẫn ước "
  "lượng khoảng 6,5% nên gần như không cải thiện. Nếu biết trước tỉ lệ thật thì sai số giảm hơn một nửa, nhưng điều "
  "này không làm được trong thực tế. Kết luận: mô hình dùng tốt để xếp hạng khách theo rủi ro; con số phần trăm thì "
  "cần điều chỉnh lại khi đã biết tỉ lệ rời bỏ thật của tháng gần nhất.")
t = calt[["version", "mean_pred", "actual_rate", "ece"]].copy()
t["version"] = t.version.map(vi.CALIB)
t.columns = ["Cách tính", "Dự báo trung bình", "Thực tế", "Sai số hiệu chuẩn (ECE)"]
T(t, "Xác suất dự báo so với thực tế trên tháng 3", fmt={"Dự báo trung bình": "{:.1%}", "Thực tế": "{:.1%}", "Sai số hiệu chuẩn (ECE)": "{:.3f}"})
P(f"Giá trị kinh doanh được ước tính bằng một ví dụ với giả định rõ ràng: mỗi lần liên hệ tốn 5 NTD; khách được liên "
  f"hệ và ở lại nhận ưu đãi giảm 20% phí một tháng; giữ được một khách đem lại 6 tháng doanh thu (phí tháng phổ biến "
  f"là {k(bv.get('median_arpu_ntd', 149))} NTD); tỉ lệ cứu được khách là 10%, 20% hoặc 30%.")
pcv = csv("profit_curve.csv")
b = pcv.loc[pcv.groupby("success_rate").net_value_ntd.idxmax()]
t = b[["success_rate", "share_contacted", "churners_reached", "net_value_ntd", "random_targeting_ntd"]].copy()
t.columns = ["Tỉ lệ cứu được", "Nên liên hệ", "Khách rời bỏ được tiếp cận", "Lãi ròng (NTD)", "Nếu chọn ngẫu nhiên (NTD)"]
T(t, "Số khách nên liên hệ để có lãi cao nhất, theo từng giả định",
  fmt={"Tỉ lệ cứu được": "{:.0%}", "Nên liên hệ": "{:.0%}", "Khách rời bỏ được tiếp cận": "{:.0%}", "Lãi ròng (NTD)": "{:,.0f}", "Nếu chọn ngẫu nhiên (NTD)": "{:,.0f}"})
FIG(F / "profit_curve.png", "Lãi ròng theo phần trăm khách được liên hệ")
P(f"Theo các giả định này, chỉ nên liên hệ khoảng {pct(b.share_contacted.min(), 0)}–{pct(b.share_contacted.max(), 0)} khách "
  "rủi ro nhất. Liên hệ nhiều hơn thì lãi giảm, vì ưu đãi bị dùng bởi cả những khách vốn không định rời đi; liên hệ "
  "tất cả thì lỗ. Chọn khách ngẫu nhiên thì lỗ ở mọi mức. Các con số tiền chỉ để minh họa cách ra quyết định; tỉ lệ cứu "
  "được thật cần đo bằng thử nghiệm A/B.")

# ---- 6.4 evaluation framework ----------------------------------------------------------------
H("6.4 Đánh giá tổng hợp theo 7 khía cạnh", 2)
P("Các chỉ số ở trên được gom lại theo câu hỏi mà mỗi chỉ số giúp trả lời, thay vì theo từng kỹ thuật. Mỗi chỉ số có "
  "ngưỡng tham chiếu (theo thông lệ, có thể tranh luận) và kết luận đạt hay chưa đạt. Những chỉ số chưa đạt được giữ "
  "nguyên để thấy rõ giới hạn của kết quả.")
ms = csv("metrics_summary.csv")


def fmtv(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if abs(x) >= 1000:
        return k(x)
    return num(x, 3) if abs(x) < 10 else num(x, 1)


for aspect, grp_ in ms.groupby("aspect", sort=True):
    para(d, f"{aspect}: {grp_.question.iloc[0]}", bold=True)
    t = grp_[["metric", "value", "reference", "verdict", "meaning"]].copy()
    t["value"] = t.value.map(fmtv)
    t["meaning"] = t.meaning.fillna("")
    t.columns = ["Chỉ số", "Giá trị", "Ngưỡng tham chiếu", "Kết luận", "Ý nghĩa"]
    T(t, f"Khía cạnh “{aspect.split('. ', 1)[1]}”")
P("Nhận xét tổng hợp:")
bullets(d, [x for x in [
    "Tính hợp lệ: đạt. Không có thông tin sau mốc quan sát; các lỗi dữ liệu lớn đã được xử lý; mức thổi phồng của cách "
    "làm cũ đã được đo (0,05–0,07 ROC-AUC).",
    f"Chất lượng kỹ thuật: mô hình dự báo, luật kết hợp và mẫu tuần tự đạt. Phân cụm chưa đạt về độ tách (dưới ngưỡng 0,25) "
    f"(silhouette {num(cl.get('silhouette'), 2)}, Davies–Bouldin {num(cl.get('davies_bouldin'), 1)}): khách hàng không chia "
    "thành các nhóm tách biệt rõ.",
    f"Độ tin cậy: luật kết hợp và mẫu tuần tự rất đáng tin (đúng trên dữ liệu giữ riêng và lặp lại ở tháng 2). Phân cụm "
    f"rất ổn định khi lấy mẫu lại (ARI {num(cl.get('stability_ari'), 2)}), trừ một cụm nhỏ (Jaccard "
    f"{num(cl.get('jaccard_min_cluster'), 2)}); khi phân cụm lại trên tháng 2 thì độ giống chỉ ở mức khá "
    f"(ARI {num(st.get('ari_feb_refit_vs_mar_model'), 2)}).",
    f"Phân biệt rủi ro: đạt. Chênh lệch giữa luật rủi ro nhất và an toàn nhất rất lớn (lift khoảng 9 so với "
    f"{num(csv('retention_rules.csv').lift_holdout.min(), 2)} trên dữ liệu giữ riêng).",
    f"Độ phủ: 5 luật đơn giản bao phủ {pct(cov5.share_churners_covered, 0)} khách rời bỏ khi gắn cờ "
    f"{pct(cov5.share_users_flagged)} khách; mô hình bao phủ {pct(lg.get('recall@10%'), 0)} khi liên hệ 10% khách. Luật dễ "
    "giải thích, nhưng độ phủ thấp hơn mô hình." if cov5 is not None else None,
    "Giá trị kinh doanh và vận hành: có lãi và hơn hẳn chọn ngẫu nhiên theo giả định, nhưng xác suất bị lệch (ECE "
    "0,026) và một số đặc trưng thay đổi giữa hai tháng, nên cần theo dõi khi vận hành.",
    f"Khả năng giải thích: đạt. {len(seg)} nhóm có tên nghiệp vụ, luật trung bình 2 điều kiện, chuỗi khoảng 2–3 bước.",
] if x])

# ---- 6.5 factors -------------------------------------------------------------------------------
H("6.5 Các yếu tố ảnh hưởng tới việc rời bỏ và giải thích", 2)
P("Bảng dưới xếp hạng các yếu tố theo mức ảnh hưởng trong mô hình (SHAP), kèm khả năng phân biệt khi xét riêng, chiều "
  "tác động, tỉ lệ rời bỏ ở hai đầu giá trị, và mức thay đổi giữa hai tháng (PSI). Một yếu tố đáng tin khi nhiều nguồn "
  "bằng chứng cùng chỉ về một hướng.")
fs = csv("factor_summary.csv").head(14)
t = fs[["shap_rank", "feature", "shap", "separation", "direction", "churn_low", "churn_high", "psi"]].copy()
t["feature"] = t.feature.map(vi.feature)
t["direction"] = t.direction.fillna("(biến phân loại)")
t.columns = ["Hạng", "Yếu tố", "SHAP", "Phân biệt (0,5–1)", "Chiều", "Rời bỏ ở mức thấp", "Rời bỏ ở mức cao", "PSI"]
T(t, "Các yếu tố ảnh hưởng nhiều nhất (mức thấp/cao: 10% khách thấp nhất/cao nhất, hoặc 0/1 với biến có/không)",
  fmt={"SHAP": "{:.3f}", "Phân biệt (0,5–1)": "{:.2f}", "Rời bỏ ở mức thấp": "{:.1%}", "Rời bỏ ở mức cao": "{:.1%}", "PSI": "{:.2f}"})
P("Giải thích từng nhóm yếu tố. Đây là các giải thích hợp lý dựa trên dữ liệu và hiểu biết về dịch vụ thuê bao; dữ "
  "liệu cho thấy mối liên hệ, chưa chứng minh được nguyên nhân.")
fact = [
    ("1. Tự gia hạn (yếu tố mạnh nhất)",
     f"Bằng chứng: hạng 1 SHAP; rời bỏ {pct(g('auto_renew', 'off'))} khi tắt so với {pct(g('auto_renew', 'on'))} khi bật; "
     "tỉ lệ tự gia hạn ở 3 lần mua gần nhất là đặc trưng mới hữu ích nhất; chuỗi “trả tự động liên tục” có tỉ lệ rời bỏ "
     "thấp nhất. Giải thích: khi bật tự gia hạn, việc tiếp tục là mặc định, khách phải chủ động mới rời đi được. Khi trả "
     "thủ công, mỗi tháng khách phải tự quyết định lại, và chỉ cần quên hoặc lưỡng lự là gói hết hạn. Việc chủ động tắt "
     "tự gia hạn cũng thường là dấu hiệu khách đã có ý định ngừng."),
    ("2. Hủy ở lần giao dịch cuối",
     f"Bằng chứng: rời bỏ {pct(g('cancelled_last', 'yes'), 0)} khi lần giao dịch cuối là hủy. Giải thích: đây là hành động "
     "rời đi rõ ràng nhất; khách hủy rồi quay lại trong 30 ngày là thiểu số."),
    ("3. Độ dài gói và số tiền trả (gói dài hạn)",
     f"Bằng chứng: gói trên 30 ngày rời bỏ {pct(g('plan', '>30'), 0)}; 10% khách trả nhiều nhất ở lần mua cuối rời bỏ "
     f"{pct(fsum.churn_high.get('last_paid'), 0)}; nhóm gói dài hạn được cả 4 kỹ thuật tìm ra; nhóm bất thường của Isolation "
     "Forest có trung vị là 1 giao dịch, gói khoảng 200 ngày, giá 894 NTD. Giải thích (giả thuyết): đây có thể là các lần "
     "mua một lần, ví dụ theo khuyến mãi hoặc gói đi kèm thiết bị; dữ liệu không ghi nguồn gốc gói nên chưa kiểm chứng "
     "được. Khách loại này không hình thành thói quen thanh toán hằng tháng, và khi hết gói phải trả một khoản lớn hơn "
     "nhiều so với phí tháng, nên phần lớn không mua lại."),
    ("4. Phương thức thanh toán",
     "Bằng chứng: hạng 2 SHAP. Giải thích (giả thuyết): phương thức thanh toán có thể giới hạn khả năng tự gia hạn (ví "
     "dụ một số hình thức trả trước chỉ trả được một lần), nên nó đi kèm với yếu tố 1 và 3. Dữ liệu chỉ có mã phương "
     "thức, không có tên, nên chưa nói được phương thức nào cụ thể. SHAP cho biết mô hình dựa vào yếu tố này, không "
     "chứng minh nó là nguyên nhân."),
    ("5. Mức độ giao dịch gần đây và thời hạn còn lại",
     f"Bằng chứng: 10% khách ít giao dịch nhất trong 180 ngày rời bỏ {pct(fsum.churn_low.get('n_tx_180d'), 0)}; khách đã hết hạn tại mốc quan "
     "sát rời bỏ khoảng 50%. Giải thích: ít giao dịch nghĩa là khách mới, khách gói dài, hoặc khách mua ngắt quãng; "
     "khách đã để gói hết hạn mà chưa mua lại thì đã nửa bước rời đi."),
    ("6. Kênh đăng ký",
     f"Bằng chứng: kênh 7 rời bỏ {pct(g('registered_via', '7.0'))}, kênh 3 và 4 khoảng 17–23% (Cramér's V 0,20). Giải "
     "thích (giả thuyết): mỗi kênh có thể mang về một kiểu khách khác nhau (ví dụ qua đối tác, chương trình khuyến "
     "mãi). Người không khai giới tính ít rời bỏ hơn; điều này có thể liên quan đến kênh đăng ký hơn là đến giới tính, "
     "nhưng chưa được kiểm chứng."),
    ("7. Thói quen nghe nhạc (ảnh hưởng yếu)",
     "Bằng chứng: chỉ dùng hành vi nghe thì ROC-AUC chỉ khoảng 0,56; khách rời bỏ không nghe ít đi trong 8 tuần cuối. "
     "Giải thích (giả thuyết): ở dịch vụ trả phí, quyết định ở lại có thể phụ thuộc vào cách trả tiền nhiều hơn mức sử "
     "dụng. Khách vẫn nghe nhiều có thể rời đi vì giá hoặc chuyển sang dịch vụ khác (dữ liệu không có thông tin này); "
     "khách không nghe vẫn ở lại khi gói tự gia hạn. Thói quen nghe chỉ có ích khi kết hợp với trạng thái thanh toán "
     "(luật “tắt tự gia hạn + ít nghe”)."),
    ("8. Số ngày từ giao dịch đầu tiên (cần thận trọng)",
     "Bằng chứng: nằm trong nhóm 5 yếu tố mạnh nhất theo SHAP nhưng thay đổi rất mạnh giữa hai tháng (PSI 0,86). Giải "
     "thích: dữ liệu giao dịch bắt đầu từ 01/2015, nên với khách lâu năm giá trị này chỉ là “số ngày kể từ khi dữ liệu "
     "bắt đầu” và tăng đều theo lịch. Thí nghiệm ở mục 6.7 cho thấy bỏ đặc trưng này gần như không làm thay đổi kết quả, "
     "nên nó không phải nguyên nhân chính khiến mô hình kém hơn ở tháng mới; dù vậy khi triển khai vẫn nên thay bằng "
     "thời gian sử dụng dịch vụ từ hồ sơ để tránh lệch theo lịch."),
]
for title, text in fact:
    para(d, title, bold=True)
    P(text)
dab = csv("drift_ablation.csv")
P("Tại sao kết quả ở tháng 3 kém hơn khi kiểm tra trong cùng tháng 2 (ROC-AUC giảm khoảng 0,03)? Nhóm đã kiểm tra "
  "một giả thuyết: một số đặc trưng đổi phân bố theo lịch (rõ nhất là số ngày từ giao dịch đầu tiên, PSI 0,86). Bỏ "
  f"các đặc trưng có PSI trên 0,1 chỉ làm khoảng chênh giảm từ {num(dab.gap.iloc[0])} xuống {num(dab.gap.iloc[-1])} (mục "
  "6.7), tức thu hẹp được khoảng một phần mười. Như vậy các đặc trưng lệch theo lịch không phải nguyên nhân chính, "
  "nhưng thí nghiệm này chưa xác định được nguyên nhân còn lại. Một khả năng là quan hệ giữa đặc trưng và việc rời bỏ "
  "thay đổi giữa hai tháng (ví dụ nhóm gói dài hạn rời bỏ 48% ở tháng 2 nhưng 84% ở tháng 3); muốn khẳng định cần "
  "dữ liệu nhiều tháng hơn. Riêng việc tỉ lệ rời bỏ tăng từ 6,4% lên 9,0% tự nó không làm giảm ROC-AUC, nhưng làm xác "
  "suất dự báo bị lệch (mục 6.3).")

H("6.6 So sánh các kỹ thuật", 2)
comp = pd.DataFrame([
    ("Phân cụm (K-Means)", f"silhouette {num(cl.get('silhouette'), 2)}, ổn định khi lấy mẫu lại (ARI {num(cl.get('stability_ari'), 2)})",
     "Cao", f"Thứ tự rủi ro gần như giữ nguyên; phân cụm lại ở tháng 2 giống ở mức khá (ARI {num(st.get('ari_feb_refit_vs_mar_model'), 2)})",
     "Chia ngân sách, thiết kế chương trình theo nhóm"),
    ("Luật kết hợp (FP-Growth)", f"{ru.get('n_churn_rules')} luật, kiểm tra lại gần như khớp; 5 luật bao phủ "
     f"{pct(cov5.share_churners_covered, 0) if cov5 is not None else '–'} khách rời bỏ",
     "Rất cao (nếu… thì…)", f"{pct(st.get('rule_share_same_direction'), 0)} giữ ý nghĩa", "Đặt điều kiện kích hoạt hành động"),
    ("Mẫu tuần tự (PrefixSpan)", f"{sq.get('n_selected')} chuỗi có ý nghĩa, kiểm tra lại gần như khớp", "Cao",
     f"{pct(sq.get('share_same_direction_feb'), 0)} giữ ý nghĩa", "Cảnh báo sớm ngay khi khách giao dịch"),
    ("Phân lớp (LightGBM)", f"ROC-AUC {num(lg.get('roc_auc'))}, recall {pct(lg.get('recall@10%'), 0)} ở top 10%",
     "Trung bình (cần SHAP)", "Xếp hạng tốt, xác suất lệch", "Chọn khách để liên hệ khi ngân sách có hạn"),
], columns=["Kỹ thuật", "Chất lượng", "Dễ hiểu", "Ổn định theo thời gian", "Dùng để"])
T(comp, "So sánh bốn kỹ thuật")
P("Bốn kỹ thuật đi đến cùng một kết luận: nhóm mua gói dài hạn và trả thủ công là rủi ro nhất; tắt tự gia hạn là dấu "
  "hiệu rủi ro chính; thói quen nghe nhạc chỉ bổ sung. Vì các kỹ thuật dùng chung một bộ dữ liệu nên đây là sự nhất "
  "quán, chưa phải bằng chứng độc lập. Mỗi kỹ thuật có điểm yếu riêng: phân cụm chỉ cho các nhóm tách nhau yếu; luật kết hợp "
  "phụ thuộc cách chia ngưỡng; mẫu tuần tự phải thử rất nhiều chuỗi nên cần kiểm định chặt; mô hình dự báo khó giải "
  "thích và nhạy với thay đổi giữa các tháng. Toàn bộ quy trình chạy trên một máy tính cá nhân 16 GB RAM trong khoảng "
  "50 phút.")

H("6.7 Phân tích độ nhạy: kết quả có phụ thuộc vào các lựa chọn của nhóm không?", 2)
P("Ba lựa chọn của nhóm có thể ảnh hưởng tới kết quả: cách chọn đặc trưng cho phân cụm, quy tắc sắp xếp các giao dịch "
  "cùng ngày, và việc giữ các đặc trưng có phân bố thay đổi theo lịch. Mỗi lựa chọn được thử lại theo cách khác.")
para(d, "a) Các cách chọn đặc trưng cho phân cụm", bold=True)
cv_ = csv("cluster_variants.csv")
if len(cv_):
    t = cv_[["variant", "k", "silhouette", "stability_ari", "churn_spread", "ari_vs_A"]].copy()
    t.columns = ["Cách làm", "Số cụm chọn được", "Silhouette", "Độ ổn định (ARI)", "Chênh lệch tỉ lệ rời bỏ", "Giống cách A (ARI)"]
    T(t, "So sánh 4 cách chọn đặc trưng (mẫu 100.000 khách có nhật ký, cùng quy tắc chọn số cụm, chưa giới hạn tối đa 7)",
      fmt={"Silhouette": "{:.3f}", "Độ ổn định (ARI)": "{:.2f}", "Chênh lệch tỉ lệ rời bỏ": "{:.1%}", "Giống cách A (ARI)": "{:.2f}"})
    FIG(F / "cluster_variants.png", "Silhouette và độ ổn định theo số cụm của 4 cách chọn đặc trưng")
    P("Khi dùng đủ 16 đặc trưng với trọng số bằng nhau (cách A), nhóm thông tin nghe nhạc (8 đặc trưng) lấn át; lời "
      "giải được chọn chỉ có 3 cụm, rất ổn định nhưng chênh lệch rủi ro giữa các cụm nhỏ hơn nhiều (khoảng 32 điểm phần "
      "trăm). Khi cân bằng các nhóm đặc trưng (B) hoặc bỏ đặc trưng trùng (C), K-Means tách được nhóm gói dài hạn và nhóm "
      "trả thủ công, chênh lệch rủi ro tăng lên khoảng 77 điểm phần trăm, đổi lại độ ổn định thấp hơn A. Thêm kênh đăng "
      "ký và phương thức thanh toán (D) cho nhiều cụm hơn nhưng kém tách và kém ổn định nhất.")
    P("Cần nói rõ: nhóm chọn cách C cho phân cụm chính sau khi đã xem kết quả của phần thử nghiệm này. Lý do là cách C "
      "loại bỏ trùng lặp giữa các đặc trưng (một nguyên tắc chung khi phân cụm) và vẫn ổn định khi lấy mẫu lại, nhưng "
      "đây vẫn là lựa chọn sau quan sát, nên báo cáo trình bày đủ cả bốn cách. Lựa chọn này cũng phụ thuộc vào mẫu: trên "
      "mẫu 100.000 khách của bảng trên, cách C với 7 cụm chỉ có ARI 0,75, trong khi trên mẫu chính 200.000 khách là "
      "0,996. Các cách cho kết quả khá khác nhau (ARI so với A chỉ 0,2–0,3), nên cách chia nhóm phụ thuộc đáng kể vào "
      "việc chọn đặc trưng và chưa được xác nhận trên dữ liệu độc lập.")
para(d, "b) Quy tắc sắp xếp các giao dịch cùng ngày (Codex thực hiện)", bold=True)
osj = j("ordering_sensitivity").get("mar", {})
osc = csv("ordering_sensitivity.csv")
if osj:
    om = osc[(osc.cohort == "mar") & osc.metric.isin(["n_users_last_is_cancel_1", "churn_rate_last_is_cancel_1",
                                                       "churn_rate_last_auto_renew_0", "churn_rate_last_auto_renew_1"])]
    t = om.pivot(index="rule", columns="metric", values="value").reset_index()
    t = t[["rule", "n_users_last_is_cancel_1", "churn_rate_last_is_cancel_1", "churn_rate_last_auto_renew_0", "churn_rate_last_auto_renew_1"]]
    t["rule"] = t.rule.map({"standard": "Chuẩn (đang dùng)", "cancel_first": "Hủy được coi là trước",
                            "expiry_desc": "Ngày hết hạn giảm dần", "paid_desc": "Số tiền giảm dần"}).fillna(t.rule)
    t.columns = ["Quy tắc", "Số khách có giao dịch cuối là hủy", "Rời bỏ khi giao dịch cuối là hủy",
                 "Rời bỏ khi tắt tự gia hạn", "Rời bỏ khi bật tự gia hạn"]
    T(t, "Ảnh hưởng của quy tắc sắp xếp giao dịch cùng ngày (tháng 3)",
      fmt={"Số khách có giao dịch cuối là hủy": "{:,.0f}", "Rời bỏ khi giao dịch cuối là hủy": "{:.1%}",
           "Rời bỏ khi tắt tự gia hạn": "{:.1%}", "Rời bỏ khi bật tự gia hạn": "{:.1%}"})
    P(f"Chỉ {pct(osj.get('share_users_with_same_day_ties'))} khách có nhiều giao dịch cùng ngày ở lần giao dịch cuối, và "
      "mỗi thông tin của giao dịch cuối thay đổi ở tối đa 1,3% khách. Kết quả về tự gia hạn gần như không đổi. Riêng "
      "tỉ lệ rời bỏ của nhóm “giao dịch cuối là hủy” dao động mạnh (từ khoảng 55% đến 85%) vì nhóm này nhỏ: với quy tắc "
      "“ngày hết hạn giảm dần”, nhóm này tăng thêm khoảng 7.900 khách có giao dịch hủy và giao dịch khác cùng ngày, "
      "nhiều khả năng là hủy gói cũ rồi mua gói mới (đổi gói), và phần lớn họ ở lại. Quy tắc chuẩn coi lần mua mới là trạng thái cuối, hợp lý hơn về "
      "nghiệp vụ. Vì vậy con số 83% chỉ nên đọc là “rất cao”, chiều liên hệ không đổi nhưng độ lớn phụ thuộc quy tắc.")
para(d, "c) Bỏ các đặc trưng thay đổi theo lịch", bold=True)
dab = csv("drift_ablation.csv")
if len(dab):
    t = dab[["feature_set", "auc_in_month", "auc_out_of_time", "gap"]].copy()
    t.columns = ["Đặc trưng dùng", "ROC-AUC trong tháng 2", "ROC-AUC trên tháng 3", "Chênh lệch"]
    T(t, "Bỏ đặc trưng có PSI > 0,1: khoảng chênh giữa hai cách đánh giá có giảm không?",
      fmt={"ROC-AUC trong tháng 2": "{:.4f}", "ROC-AUC trên tháng 3": "{:.4f}", "Chênh lệch": "{:.4f}"})
    P(f"Bỏ hai đặc trưng thay đổi theo lịch chỉ làm khoảng chênh giảm từ {num(dab.gap.iloc[0])} xuống "
      f"{num(dab.gap.iloc[-1])} và ROC-AUC trên tháng 3 gần như không đổi (tăng {num(dab.oot_delta_vs_all.iloc[-1], 4)}). "
      "Các đặc trưng lệch theo lịch vì vậy chỉ giải thích một phần nhỏ khoảng chênh; nguyên nhân còn lại chưa xác định "
      "được với hai tháng dữ liệu.")

# =============================================================================================
H("7. Gợi ý cho doanh nghiệp", 1)
P("Các gợi ý dưới đây dựa trên mối liên hệ tìm được trong dữ liệu. Trước khi áp dụng rộng, mỗi gợi ý nên được thử "
  "nghiệm có nhóm đối chứng (A/B test).")
bullets(d, [
    f"Khuyến khích bật tự gia hạn. Khách tắt tự gia hạn rời đi {pct(g('auto_renew', 'off'), 0)}, khách bật chỉ "
    f"{pct(g('auto_renew', 'on'))}. Nên thử trước với nhóm “nghe nhiều nhưng trả thủ công” (khoảng "
    f"{k(seg.n[seg.name.str.startswith('Nghe nhiều nhưng')].sum() / 1000)} nghìn khách, rời bỏ "
    f"{pct(seg.churn_rate[seg.name.str.startswith('Nghe nhiều nhưng')].mean(), 0)}), ví dụ tặng ưu đãi nhỏ khi bật tự gia hạn.",
    f"Chăm sóc riêng khách mua gói dài hạn ({k(seg.loc[risk_seg, 'n'] / 1000)} nghìn khách, rời bỏ "
    f"{pct(seg.loc[risk_seg, 'churn_rate'], 0)}). Có thể nhắc gia hạn trước ngày hết hạn kèm ưu đãi, hoặc mời chuyển sang "
    "gói tháng tự gia hạn. Nếu vẫn không cải thiện, nên xem lại hiệu quả của loại gói này.",
    "Cảnh báo theo sự kiện: khi khách tắt tự gia hạn, chuyển sang trả thủ công, đổi gói hoặc để gói lỡ hạn, hệ thống "
    "nên báo cho bộ phận chăm sóc khách hàng ngay, không đợi đến ngày hết hạn.",
    f"Dùng 5 luật đầu làm bộ lọc đơn giản: gắn cờ khoảng {pct(cov5.share_users_flagged, 0) if cov5 is not None else '7%'} khách "
    f"mà bao phủ khoảng {pct(cov5.share_churners_covered, 0) if cov5 is not None else '44%'} số khách sắp rời đi, dễ giải "
    "thích cho bộ phận kinh doanh hơn mô hình.",
    f"Tìm hiểu nhóm khách không có nhật ký nghe trong 60 ngày ({k(seg.loc[NOLOG, 'n'] / 1000)} nghìn khách). Họ vẫn có giao "
    "dịch và hiện ít rời bỏ, nhưng dữ liệu không cho biết họ thật sự không dùng hay chỉ không được ghi nhận. Nếu đúng là "
    "không dùng, có thể gợi ý nội dung phù hợp để họ quay lại nghe trước khi họ để ý đến khoản phí.",
    "Xem lại kênh đăng ký 3 và 4 (tỉ lệ rời bỏ 17–23%) so với kênh 7 (4,5%) trước khi quyết định đầu tư thêm cho kênh nào.",
    f"Dùng mô hình để chọn khách cần liên hệ mỗi tháng: liên hệ 10% khách rủi ro nhất tiếp cận khoảng "
    f"{pct(lg.get('recall@10%'), 0)} số khách sắp rời đi. Nên theo dõi PSI của các đặc trưng và cân nhắc huấn luyện lại "
    "mô hình khi dữ liệu thay đổi; không dùng trực tiếp con số xác suất khi tỉ lệ rời bỏ chung thay đổi.",
    "Không dùng việc nghe ít đi làm tín hiệu cảnh báo chính: trong 8 tuần trước mốc quan sát, khách rời bỏ không hề "
    "nghe ít đi so với khách ở lại.",
])

# =============================================================================================
H("8. Kỹ thuật nâng cao (phần cộng điểm)", 1)
P("Ngoài các kỹ thuật chính, nhóm dùng thêm: tạo đặc trưng theo thời gian (xu hướng nghe, số ngày ngưng nghe, độ đều "
  "đặn theo tuần); xử lý dữ liệu lớn bằng cách đọc dữ liệu nén theo luồng; phát hiện bất thường; so sánh các nhóm con; "
  "giảm chiều dữ liệu bằng PCA và UMAP; đo độ thay đổi đặc trưng giữa các tháng (PSI); và các kiểm tra như dữ liệu giữ "
  "riêng, lặp lại trên tháng khác, đo rò rỉ dữ liệu.")
FIG(F / "bonus_trajectory.png", "Số ngày nghe và số giờ nghe mỗi tuần trong 8 tuần trước mốc quan sát")
P("Khách rời bỏ không nghe ít đi trước khi rời đi; số ngày nghe của họ còn nhỉnh hơn khách ở lại và tăng nhẹ ở những "
  "tuần cuối. Nhóm có mức nghe tăng mạnh lại có tỉ lệ rời bỏ cao nhất (12,7%), có thể là khách mới hoặc khách quay lại "
  "dùng thử rồi rời đi.")
tr = csv("bonus_trend.csv")
t = tr.rename(columns={tr.columns[0]: "g"})
T(pd.DataFrame({"Xu hướng nghe": t.g.map(vi.TREND), "Số khách": t.n, "Tỉ lệ rời bỏ": t.churn_rate}),
  "Tỉ lệ rời bỏ theo xu hướng nghe (khách có nhật ký nghe)", fmt={"Số khách": "{:,.0f}", "Tỉ lệ rời bỏ": "{:.1%}"})
an = csv("bonus_anomaly.csv")
T(pd.DataFrame({"Nhóm": an.iloc[:, 0].map({"anomaly": "bất thường", "normal": "bình thường"}), "Số khách": an.n,
                "Tỉ lệ rời bỏ": an.churn_rate}),
  "Phát hiện bất thường (Isolation Forest): 2% khách khác thường nhất", fmt={"Số khách": "{:,.0f}", "Tỉ lệ rời bỏ": "{:.1%}"})
P(f"Isolation Forest không biết ai rời đi, chỉ tìm những khách khác thường nhất. 2% khách bị đánh dấu có tỉ lệ rời bỏ "
  f"{pct(an.churn_rate.iloc[0], 0)}. Phần lớn họ chỉ có một lần mua, trả thủ công cho gói khoảng 200 ngày, tức cùng kiểu "
  "khách với nhóm gói dài hạn.")
sub = csv("bonus_subsets.csv")
t = sub[~sub.subset.str.startswith("segment")][["subset", "n", "churn_rate", "roc_auc"]].copy()
t["subset"] = t.subset.map(lambda x: vi.SUBSETS.get(x, x))
t.columns = ["Nhóm con", "Số khách", "Tỉ lệ rời bỏ", "ROC-AUC trong nhóm"]
T(t, "Mô hình dự báo hoạt động thế nào trong từng nhóm con", fmt={"Số khách": "{:,.0f}", "Tỉ lệ rời bỏ": "{:.1%}", "ROC-AUC trong nhóm": "{:.3f}"})
P("Mô hình phân biệt tốt nhất ở khách mới dưới 1 năm và kém hơn ở khách đang bật tự gia hạn, vì với nhóm này dữ liệu "
  "trước mốc quan sát ít cho biết ai sẽ hủy.")
FIG(F / "umap_churn.png", "UMAP: chiếu dữ liệu xuống 2 chiều; trái tô màu theo rời bỏ, phải theo nhóm khách")
FIG(F / "bonus_pca_scree.png", "PCA: phần phương sai mà mỗi thành phần chính giải thích", width=4.5)
P("Với PCA, ba thành phần chính đầu tiên giải thích khoảng 67% tổng phương sai; chúng lần lượt liên quan chủ yếu đến "
  "mức độ nghe nhạc, số lượng giao dịch trong lịch sử, và giá cùng thời hạn gói. UMAP cho thấy khách rời bỏ tập trung ở "
  "các vùng nhỏ ứng với nhóm gói dài hạn, còn khối lớn trộn lẫn, khớp với việc thói quen nghe không tách được khách rời bỏ.")

# =============================================================================================
H("9. Kết luận, hạn chế và hướng phát triển", 1)
P("Kết luận: ở KKBox, rủi ro khách rời bỏ gắn chặt nhất với cách khách trả tiền (tự gia hạn hay trả tay, gói tháng hay "
  "gói dài), còn thói quen nghe nhạc chỉ bổ sung. Bốn kỹ thuật cho kết quả nhất quán; luật kết hợp và mẫu tuần tự đều "
  "đúng trên dữ liệu giữ riêng và lặp lại ở tháng 2. Phân cụm cho 7 nhóm có ý nghĩa và ổn định khi lấy mẫu lại, dù các "
  "nhóm tách nhau yếu và cách chia phụ thuộc vào việc chọn đặc trưng; nhóm gói dài hạn là phát hiện rõ nhất.")
P(f"Về phương pháp, đồ án phát hiện và đo được lỗi rò rỉ dữ liệu của phiên bản trước (trên cùng phép chia, ROC-AUC "
  f"tăng ảo từ {num(clean_auc)} lên {num(audit.get('v1 recipe (unfiltered *_v2 files)', {}).get('roc_auc'))}; khi kiểm tra đúng "
  f"cách trên tháng sau, mô hình sạch đạt {num(lg.get('roc_auc'))}), phát hiện và sửa các bản ghi lỗi (gói 0 ngày, ngày "
  "hỏng), bỏ cụm nhân tạo, và xây dựng quy trình có thể chạy lại từ đầu với cùng kết quả.")
P("Hạn chế:")
bullets(d, [
    "Danh sách khách mỗi tháng do KKBox chọn sẵn; chỉ có hai tháng có nhãn.",
    "Không có dữ liệu về lý do khách tắt tự gia hạn, giá của đối thủ hay khiếu nại; mã phương thức thanh toán không có tên.",
    "Các luật và nhóm khách cho thấy mối liên hệ, chưa chứng minh nguyên nhân.",
    "Các cụm tách nhau yếu (silhouette dưới 0,25); cách chia phụ thuộc vào việc chọn đặc trưng và lặp lại ở tháng khác chỉ ở mức khá.",
    "Tuổi và giới tính thiếu nhiều; một số đặc trưng bị giới hạn bởi ngày bắt đầu của dữ liệu.",
    "Xác suất dự báo bị lệch khi tỉ lệ rời bỏ thay đổi; giá trị kinh doanh dựa trên giả định.",
    "Các giải thích ở mục 6.5 là giả thuyết; chưa đo được thời gian từ tín hiệu cảnh báo đến ngày hết hạn, và chưa có "
    "thử nghiệm A/B để biết biện pháp nào thật sự giữ chân được khách.",
])
P("Hướng phát triển: kiểm tra trên nhiều tháng hơn; thử nghiệm A/B cho từng biện pháp giữ chân; đo khoảng thời gian từ "
  "tín hiệu cảnh báo đến ngày hết hạn; thử phân cụm có trọng số theo nhóm biến hoặc k-prototypes cho cả biến phân loại; "
  "bổ sung dữ liệu về lý do khách tắt tự gia hạn.")

H("Phụ lục: cách chạy lại", 1)
bullets(d, [
    "Cài thư viện: pip install -r requirements.txt",
    "Lấy dữ liệu: tải file “Download All” trên trang Kaggle của cuộc thi, rồi chạy python data_prep/00_download.py <đường dẫn file zip>",
    "Chạy toàn bộ: python run_all.py --skip-download (khoảng 50 phút trên máy 16 GB RAM)",
    "Hướng dẫn chi tiết: file HUONG_DAN_CHAY.md trong repo.",
])

d.save(OUT)
print(f"report written: {OUT} ({n_tab[0]} tables, {n_fig[0]} figures)")
