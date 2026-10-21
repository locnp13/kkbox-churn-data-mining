# DAM501 Final Project — Khám phá hành vi người dùng và các mẫu rời bỏ dịch vụ nhạc đăng ký (KKBox)

Final project môn Data Mining (DAM501): áp dụng quy trình Knowledge Discovery from Data để tìm pattern và insight phục vụ quyết định kinh doanh trên dữ liệu churn thật của KKBox.

## Mục tiêu

Hiểu hành vi người dùng, phát hiện cấu trúc ẩn và các tổ hợp đặc điểm dẫn đến rời bỏ dịch vụ, từ đó đề xuất chiến lược giữ chân theo từng nhóm khách hàng. Churn classification chỉ là kỹ thuật bổ trợ; trọng tâm là khám phá pattern.

## Dataset

[WSDM – KKBox's Churn Prediction Challenge](https://www.kaggle.com/c/kkbox-churn-prediction-challenge) (Kaggle). Dữ liệu thật của dịch vụ nghe nhạc đăng ký KKBox (Đài Loan), gồm thông tin thành viên, giao dịch đăng ký, log nghe nhạc và nhãn churn, nối với nhau qua `msno`.

Do giới hạn ổ đĩa, nhóm **không dùng `user_logs.csv` gốc** (~28 GB). Dùng `train_v2`, `members_v3`, `transactions_v2` (hoặc `transactions`) và `user_logs_v2`, rồi lấy mẫu khoảng 100–200 nghìn người dùng (stratified theo churn). Dữ liệu thô đặt trong `data/` và không đưa lên repo.

## Kỹ thuật dự kiến

| Kỹ thuật | Vai trò |
|---|---|
| Clustering (K-Means, Hierarchical/DBSCAN) | Chính: phân khúc người dùng, so churn rate theo cụm |
| Association rules (FP-Growth) | Chính: tổ hợp đặc điểm dẫn đến churn, đo support/confidence/lift |
| Sequential / temporal pattern | Chuỗi hành vi trước khi rời dịch vụ |
| Classification (Decision Tree, Random Forest) | Bổ trợ: giải thích yếu tố quan trọng |
| PCA/UMAP, feature engineering | Bonus (+10%) |

## Phân công

| Thành viên | Vai trò | Phần đề (trọng số) | Công việc |
|---|---|---|---|
| Lộc Nguyễn Phúc | Data engineering, trưởng nhóm kỹ thuật | 2 (5%), 3 (10%) | Tải và giải nén dữ liệu, aggregate log, lấy mẫu, làm sạch, xuất bảng feature chung |
| Khánh Phạm Duy Văn | Business và báo cáo | 1 (15%), 7 (10%) | Bối cảnh, bài toán, assumptions/limitations, tổng hợp insight và khuyến nghị, ghép báo cáo và slide |
| Bình Chu Đức | EDA | 4 (15%) | Phân phối, so sánh churn theo nhóm, xu hướng, hình vẽ, kết luận dẫn tới chọn kỹ thuật |
| Quân Đoàn Đàm | Clustering | 5, 6 (phần clustering) | K-Means, Hierarchical/DBSCAN, chọn k, mô tả cụm, churn rate theo cụm |
| Hạnh Nguyễn Thị Hồng | Association rules, sequential | 5, 6 (phần rules) | Rời rạc hóa feature, FP-Growth, lọc và diễn giải rule, sequential pattern nếu còn thời gian |
| Tùng Trần | Classification và bonus | 5, 6 (phần classification), 8 (+10%) | Decision Tree/Random Forest, feature importance, kiểm tra leakage, PCA/UMAP, feature engineering |

Phần 6 (Evaluation, 15%): mỗi người tự viết phần đánh giá cho kỹ thuật của mình; Khánh ghép và viết phần so sánh điểm mạnh, yếu giữa các phương pháp.

## Cấu trúc repo (dự kiến)

```
data/          # dữ liệu thô, không commit
data_prep/     # tiền xử lý, bảng feature chung (chỉ Lộc sửa)
mining/        # clustering, association rules, classification
analysis/      # EDA, đánh giá, hình vẽ
report/        # báo cáo và slide
```

Mọi notebook mining đọc bảng feature từ `data_prep/` để số liệu thống nhất giữa các thành viên.

## Sản phẩm phải nộp

1. Báo cáo (PDF hoặc Word) gồm 7 phần: Introduction & Business Understanding, Dataset Description, Data Pre-processing, Exploratory & Descriptive Analysis, Data Mining Methods & Results, Evaluation & Interpretation, Business Insights & Recommendations.
2. Code sạch, có tài liệu, tái lập được, tách rõ data preparation, mining, analysis.
3. Slide thuyết trình ngắn: bài toán kinh doanh, pattern và insight chính, kỹ thuật dùng, ý nghĩa kinh doanh.

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

## Cách chạy

Yêu cầu: Python 3.13, `pip install duckdb py7zr pandas scikit-learn mlxtend matplotlib pyarrow`, và đã chấp nhận rules của competition trên Kaggle (token ở `~/.kaggle/`).

```
python3 data_prep/00_download.py          # tải và giải nén 4 file cần dùng vào data/
python3 data_prep/01_build_features.py    # tạo data/features.parquet (biến N_PER_CLASS chỉnh cỡ mẫu)
python3 analysis/01_eda.py                # EDA, hình vào report/figures/
python3 mining/01_clustering.py           # K-Means, profile cụm
python3 mining/02_association_rules.py    # FP-Growth, rule dẫn tới churn
python3 mining/03_classification.py       # Decision Tree / Random Forest
```

Trạng thái: các script mining mới chỉ được kiểm thử trên dữ liệu giả lập; chưa chạy trên dữ liệu KKBox thật (đang chờ chấp nhận rules trên Kaggle).
