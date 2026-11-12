# DAM501 Final Project — Khám phá hành vi người dùng và các mẫu rời bỏ dịch vụ nhạc thuê bao (KKBox)

Final project môn Data Mining (DAM501): áp dụng quy trình Knowledge Discovery from Data trên dữ liệu churn thật của KKBox để tìm pattern và insight phục vụ quyết định giữ chân khách hàng. Trọng tâm là **khám phá mẫu**; phân lớp chỉ là kỹ thuật bổ trợ.

> Nhánh `Khanh`: phiên bản 2 của pipeline. Phiên bản 1 (nhánh `main`) dùng dữ liệu của chính tháng nhãn nên bị rò rỉ thông tin (AUC 0,977, luật confidence 97–99%). Phiên bản này cắt mọi đặc trưng tại một mốc quan sát, đánh giá ngoài thời gian và đo lại mức rò rỉ (xem `report/leakage_audit.csv`).

**Tài liệu:** [Hướng dẫn chạy code từng bước](HUONG_DAN_CHAY.md) · [Những gì đã thay đổi so với bản cũ](THAY_DOI.md) · [Đánh giá phương pháp và tổng hợp cải tiến](DANH_GIA_PHUONG_PHAP.md) · [Báo cáo](report/report.docx) · Slide thuyết trình: https://claude.ai/artifact/WLTuVPaYLubUEq5Q6mXBCt (16 trang, có ghi chú lời nói; tải về PowerPoint/PDF từ trang đó)

## Kết quả chính

| | Kết quả (cohort tháng 3/2017: 970.960 người dùng, churn 9,0%) |
|---|---|
| Phân cụm (K-Means 7 cụm trên 13 đặc trưng + 1 nhóm quy tắc "không có nhật ký nghe"; silhouette 0,21, ARI 0,996) | 8 nhóm, tỉ lệ rời bỏ từ 3,2% (trung thành, tự gia hạn – 36% khách) tới 22% (nghe nhiều nhưng trả thủ công) và 84% (gói dài hạn trả thủ công – 2,2%) |
| Luật kết hợp (FP-Growth, kiểm định + FDR, kiểm chứng holdout, tương quan lift 0,999) | gói > 30 ngày → churn 83% (lift 9,2); tắt tự gia hạn → 42%; + ít nghe + kênh 9 → 55%; kênh 7 & mới < 1 năm → 2,2%; 5 luật đầu gắn cờ 5,1% khách, bao phủ 36% khách rời bỏ |
| Mẫu tuần tự (PrefixSpan) | trả thủ công → gói dài → gói dài → 70% rời bỏ (lift 7,8); đổi gói thật → 33%; trả tự động liên tục → 4,9% |
| Phân lớp (bổ trợ, train 02/2017 → test 03/2017) | LightGBM ROC-AUC 0,870, PR-AUC 0,60, top 10% bắt 61% người churn (lift 6,1) |
| Leakage audit | dữ liệu tháng nhãn thổi phồng AUC thêm 0,05–0,07; hành vi nghe đơn lẻ chỉ đạt AUC 0,57 |
| Độ ổn định (áp kết quả tháng 3 lên tháng 2) | thứ tự rủi ro các nhóm gần như giữ nguyên (Spearman 0,88; phân cụm lại ARI 0,72); 33/34 luật và 30/30 mẫu tuần tự giữ hướng |
| Hiệu chuẩn và giá trị kinh doanh | dự báo trung bình 6,4% so với thực tế 9,0%, EM không nhãn không sửa được; liên hệ 8–16% người rủi ro nhất cho giá trị ròng dương (giả định minh họa), chọn ngẫu nhiên thì lỗ |
| Cải tiến mô hình (đặc trưng thanh toán + Optuna) | ROC-AUC 0,870 → 0,874 (+0,004), recall top 10% không đổi |
| Đánh giá 7 khía cạnh | hợp lệ, chất lượng, độ tin cậy, phân biệt rủi ro, độ phủ, giá trị kinh doanh, khả năng giải thích (`report/metrics_summary.csv`) |
| Bonus | Isolation Forest: 2% bất thường churn 69%; quỹ đạo nghe 8 tuần; so sánh tập con; PCA/UMAP |

Kết luận: churn ở KKBox là **quyết định thanh toán** (tự gia hạn hay thủ công, gói tháng hay gói dài), hành vi nghe chỉ bổ sung. Báo cáo đầy đủ: `report/report.docx`.

## Dataset

[WSDM – KKBox's Churn Prediction Challenge](https://www.kaggle.com/c/kkbox-churn-prediction-challenge) (Kaggle). Dùng `train` (nhãn 02/2017), `train_v2` (nhãn 03/2017), `members_v3`, `transactions` + `transactions_v2`, `user_logs` (~28 GB, đọc dạng luồng) và `user_logs_v2` (chỉ cho audit). Dữ liệu thô nằm trong `data/`, không commit.

## Thiết kế chống rò rỉ

- Cohort `feb` (hết hạn 02/2017): cutoff **31/01/2017**; cohort `mar` (hết hạn 03/2017): cutoff **28/02/2017**. Mọi giao dịch/log đều có ngày ≤ cutoff (`data/manifest.json` kiểm tra tự động).
- Đặc trưng nghe dùng cửa sổ 60 ngày trước cutoff; 8 tuần đếm lùi từ cutoff.
- Phân lớp: train trên `feb`, test trên `mar` (ngoài thời gian); chọn mô hình bằng CV trên `feb`.
- Luật/mẫu tuần tự: khai phá trên 70%, đo lại trên 30% holdout; p-value hiệu chỉnh Benjamini–Hochberg.
- Giữ tỉ lệ churn thật (không lấy mẫu cân bằng) để confidence/lift có nghĩa tuyệt đối.

## Kỹ thuật

| Kỹ thuật | File | Vai trò |
|---|---|---|
| EDA (Wilson CI, χ²/Cramér's V, Mann–Whitney) | `analysis/01_eda.py` | mô tả, chọn kỹ thuật |
| K-Means + Ward/GMM/HDBSCAN, bootstrap ARI | `mining/01_clustering.py` | phân khúc (unsupervised) |
| FP-Growth, productive rules, holdout | `mining/02_association_rules.py` | tổ hợp đặc điểm → churn |
| Logistic/Tree/RF/LightGBM, SHAP, ablation | `mining/03_classification.py` | bổ trợ, xếp hạng rủi ro |
| Quỹ đạo theo tuần, Isolation Forest, tập con, PCA/UMAP | `mining/04_bonus.py` | bonus |
| PrefixSpan (bitmask NumPy) trên chuỗi giao dịch | `data_prep/02_build_sequences.py`, `mining/05_sequential.py` | mẫu tuần tự |
| Leakage audit, so sánh thuật toán | `data_prep/03_build_leakage_audit.py`, `mining/06_validation.py` | đánh giá |
| Khung đánh giá 7 khía cạnh, xếp hạng yếu tố, PSI, độ phủ luật | `mining/10_metrics_summary.py` | đánh giá |
| Phân tích độ nhạy: cách chọn đặc trưng phân cụm, thứ tự giao dịch cùng ngày (Codex), bỏ đặc trưng lệch theo lịch | `mining/11_clustering_variants.py`, `mining/12_ordering_sensitivity.py`, `mining/13_drift_ablation.py` | đánh giá |
| Độ ổn định theo thời gian (tháng 3 → tháng 2) | `mining/07_stability.py` | đánh giá |
| Hiệu chỉnh xác suất (EM), đường lợi nhuận | `mining/08_business_value.py` | đánh giá, kinh doanh |
| Thí nghiệm 2×2: đặc trưng thanh toán, Optuna | `mining/09_model_improvement.py` | cải tiến mô hình |

## Cách chạy

Yêu cầu Python 3.11+ và `pip install -r requirements.txt`.

```
python data_prep/00_download.py              # Kaggle CLI (cần token) ...
python data_prep/00_download.py <đường dẫn>/kkbox-churn-prediction-challenge.zip   # ... hoặc file "Download All"
python run_all.py --skip-download            # toàn bộ pipeline (~45 phút trên máy 16 GB RAM)
python run_all.py --from 01_clustering       # chạy lại từ một bước
```

Log từng bước ở `report/logs/`, kết quả ở `report/*.csv`, `report/metrics/*.json`, hình ở `report/figures/`; `report/make_diagrams.py` vẽ sơ đồ quy trình và trục thời gian; `report/build_report.py` dựng `report/report.docx` từ các kết quả này. Xuất PDF: mở `report.docx` trong Word → Save As → PDF (hoặc `report/export_pdf.ps1` nếu Word chạy được ở chế độ ẩn).

Kiểm thử nhanh không cần dữ liệu thật:

```
python tests/make_synthetic_data.py data_synth
DATA_DIR=data_synth REPORT_DIR=report_synth python run_all.py --skip-download
```

## Cấu trúc repo

```
data/          dữ liệu thô và bảng đặc trưng (không commit)
data_prep/     tải dữ liệu, đặc trưng theo cohort, chuỗi giao dịch, dữ liệu audit
analysis/      EDA
mining/        clustering, rules, classification, bonus, sequential, validation
report/        kết quả (csv, metrics, figures), report.docx, bộ dựng báo cáo
tests/         sinh dữ liệu giả cùng schema để kiểm thử
```

## Phân công

| Thành viên | Vai trò | Phần đề (trọng số) | Công việc |
|---|---|---|---|
| Lộc Nguyễn Phúc | Data engineering, trưởng nhóm kỹ thuật | 2 (5%), 3 (10%) | Tải dữ liệu, aggregate log, làm sạch, bảng đặc trưng chung |
| Khánh Phạm Duy Văn | Business và báo cáo | 1 (15%), 7 (10%) | Bối cảnh, bài toán, giả định/hạn chế, insight và khuyến nghị, ghép báo cáo; phiên bản 2 của pipeline (nhánh `Khanh`) |
| Bình Chu Đức | EDA | 4 (15%) | Phân phối, so sánh churn theo nhóm, xu hướng, hình vẽ |
| Quân Đoàn Đàm | Clustering | 5, 6 (clustering) | K-Means, Hierarchical/DBSCAN, chọn k, mô tả cụm |
| Hạnh Nguyễn Thị Hồng | Association rules | 5, 6 (rules) | Rời rạc hóa, FP-Growth, lọc và diễn giải rule, độ nhạy |
| Tùng Trần | Classification | 5, 6 (classification) | Decision Tree/Random Forest, feature importance, leakage |
| Phùng Quý Quốc | Sequential, bonus, validation, slide | 5, 8, 6, slide | Mẫu tuần tự, bonus, so sánh phương pháp, slide |

## Tiêu chí chấm điểm

| Phần | Trọng số |
|---|---|
| 1. Problem Definition & Business Understanding | 15% |
| 2. Dataset Selection & Understanding | 5% |
| 3. Pre-processing & Transformation | 10% |
| 4. EDA & Descriptive Mining | 15% |
| 5. Data Mining Methods & Pattern Discovery (≥ 2 kỹ thuật, ≥ 1 unsupervised/pattern-based) | 30% |
| 6. Evaluation & Interpretation | 15% |
| 7. Business Insights & Recommendations | 10% |
| 8. Advanced Techniques (bonus) | +10% |
