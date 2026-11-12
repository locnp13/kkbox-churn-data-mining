"""Vietnamese wording for codes that appear in result tables (rules, sequences, features, models)."""
import re

ITEMS = {
    "auto_renew=on": "bật tự gia hạn", "auto_renew=off": "tắt tự gia hạn", "no_tx": "không có giao dịch",
    "last_tx=cancel": "giao dịch cuối là hủy", "cancelled_before": "từng hủy", "never_cancelled": "chưa từng hủy",
    "plan<=7d": "gói ≤ 7 ngày", "plan=30d": "gói 8–30 ngày", "plan>30d": "gói > 30 ngày", "plan=none": "không có gói",
    "paid=free": "trả 0 đồng", "paid=discount": "được giảm giá", "paid=full": "trả đủ giá",
    "tenure<1y": "thâm niên < 1 năm", "tenure1-3y": "thâm niên 1–3 năm", "tenure3-6y": "thâm niên 3–6 năm",
    "tenure>6y": "thâm niên > 6 năm", "gender=male": "nam", "gender=female": "nữ",
    "gender=unknown": "không khai giới tính", "no_logs": "không có nhật ký nghe",
    "active=low": "ít ngày nghe", "active=mid": "số ngày nghe trung bình", "active=high": "nhiều ngày nghe",
    "idle<=3d": "nghe gần đây (≤ 3 ngày)", "idle4-14d": "ngưng nghe 4–14 ngày", "idle>14d": "ngưng nghe > 14 ngày",
    "trend=falling": "nghe giảm dần", "trend=flat": "nghe ổn định", "trend=rising": "nghe tăng dần",
    "skip=high": "hay bỏ qua bài", "expired_at_T": "đã hết hạn tại mốc quan sát",
    "expires<=31d": "còn hạn ≤ 31 ngày", "expires>31d": "còn hạn > 31 ngày",
}
TOKENS = {
    "PAY_AUTO": "trả tự động", "PAY_MANUAL": "trả thủ công", "CANCEL": "hủy", "FREE": "miễn phí",
    "DISC": "giảm giá", "PLAN_30": "gói 30 ngày", "PLAN_LONG": "gói dài", "PLAN_SHORT": "gói ngắn",
    "PLAN_CHANGE": "đổi gói", "METHOD_CHANGE": "đổi cách trả tiền", "GAP": "lỡ hạn > 30 ngày",
}
FEATURES = {
    "auto_renew_rate": "tỉ lệ giao dịch tự gia hạn", "last_auto_renew": "giao dịch cuối có tự gia hạn",
    "last_list_price": "giá niêm yết gói cuối", "last_paid": "số tiền trả ở giao dịch cuối",
    "n_tx_180d": "số giao dịch 180 ngày gần nhất", "days_since_last_tx": "số ngày từ giao dịch cuối",
    "n_tx": "tổng số giao dịch", "last_plan_days": "độ dài gói cuối (ngày)", "age": "tuổi",
    "n_payment_methods": "số cách thanh toán đã dùng", "n_free_tx": "số giao dịch 0 đồng",
    "n_plan_changes": "số lần đổi gói", "last_is_cancel": "giao dịch cuối là hủy",
    "last_payment_method": "cách thanh toán cuối", "days_since_first_tx": "số ngày từ giao dịch đầu",
    "days_since_last_listen": "số ngày từ lần nghe cuối", "active_days_last14": "số ngày nghe trong 14 ngày cuối",
    "weekly_cv": "độ thất thường khi nghe", "days_to_expire": "số ngày còn lại tới hạn",
    "total_paid": "tổng tiền đã trả", "tenure_days": "thâm niên (ngày)", "active_days": "số ngày có nghe",
    "total_hours": "tổng số giờ nghe", "discount_rate": "tỉ lệ giao dịch được giảm giá", "n_cancel": "số lần hủy",
    "complete_ratio": "tỉ lệ nghe hết bài", "skip_ratio": "tỉ lệ bỏ qua bài", "listen_trend": "xu hướng nghe",
    "avg_unique_songs": "số bài khác nhau mỗi ngày", "mins_per_active_day": "phút nghe mỗi ngày có nghe",
    "total_songs": "tổng số bài đã nghe", "has_logs": "có nhật ký nghe",
}
GROUPS = {"plan": "loại gói cuối", "auto_renew": "tự gia hạn ở giao dịch cuối", "cancelled_last": "giao dịch cuối là hủy",
          "expire": "số ngày còn hạn", "registered_via": "kênh đăng ký", "gender": "giới tính",
          "trend": "xu hướng nghe", "tenure_y": "thâm niên", "idle": "số ngày ngưng nghe"}
MODELS = {"Baseline (prior)": "Đoán theo tỉ lệ chung (mốc so sánh)", "Logistic regression": "Hồi quy logistic",
          "Decision tree (depth 6)": "Cây quyết định (độ sâu 6)", "Random forest": "Rừng ngẫu nhiên",
          "LightGBM": "LightGBM"}
AUDIT = {"clean (cutoff T)": "Dữ liệu sạch (chỉ trước mốc)",
         "clean + March listening logs": "Sạch + nhật ký nghe tháng 3",
         "clean + March transactions": "Sạch + giao dịch tháng 3",
         "v1 recipe (unfiltered *_v2 files)": "Cách làm của phiên bản 1",
         "listening only, clean": "Chỉ hành vi nghe, sạch",
         "listening only, March logs (v1)": "Chỉ hành vi nghe, có tháng 3"}
CONFIGS = {"A. baseline (03_classification)": "A. Mô hình ban đầu",
           "B. + payment-history features": "B. Thêm 13 đặc trưng thanh toán",
           "C. tuned hyper-parameters": "C. Tinh chỉnh tham số",
           "D. features + tuning": "D. Cả hai"}
ABLATION = {"profile only": "Chỉ hồ sơ", "listening only": "Chỉ hành vi nghe", "subscription only": "Chỉ thuê bao",
            "subscription + profile": "Thuê bao + hồ sơ", "all features": "Tất cả"}
CALIB = {"original (feb prior)": "Ban đầu", "EM prior-shift corrected": "Hiệu chỉnh bằng EM",
         "oracle: true mar prior (not deployable)": "Nếu biết trước tỉ lệ thật (tham khảo)"}
SUBSETS = {"all users": "Tất cả", "registered_via 7": "Kênh 7", "registered_via 9": "Kênh 9",
           "registered_via 3/4": "Kênh 3 và 4", "tenure < 1y": "Thâm niên < 1 năm", "tenure >= 4y": "Thâm niên ≥ 4 năm",
           "auto-renew on": "Đang bật tự gia hạn", "auto-renew off": "Đã tắt tự gia hạn",
           "no listening logs": "Không có nhật ký nghe", "with listening logs": "Có nhật ký nghe"}
TREND = {"falling fast": "giảm mạnh", "falling": "giảm", "flat": "ổn định", "rising": "tăng", "rising fast": "tăng mạnh"}


def item(code):
    if code.startswith("via="):
        return f"kênh đăng ký {code[4:]}"
    if code.startswith("segment="):
        return f"nhóm {code[8:]}"
    return ITEMS.get(code, code)


def rule(text):
    return " + ".join(item(x.strip()) for x in text.split("&"))


def pattern(text):
    steps = re.findall(r"\{([^}]*)\}", text)
    return " → ".join(" + ".join(TOKENS.get(t.strip(), t.strip()) for t in s.split(",")) for s in steps)


def feature(code):
    return FEATURES.get(code, code)
