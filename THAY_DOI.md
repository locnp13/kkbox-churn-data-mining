# Những gì đã thay đổi so với bản cũ (nhánh `main`)

Nhánh `Khanh` sửa và bổ sung khoảng 3.500 dòng code trên 27 file. Bảng dưới đây tóm tắt theo từng phần.

## 1. Sửa lỗi quan trọng nhất: rò rỉ dữ liệu

**Lỗi ở bản cũ:** `data_prep/01_build_features.py` lấy toàn bộ giao dịch và nhật ký nghe của tháng 3/2017, đúng tháng cần dự báo. Ví dụ "khách không có giao dịch tháng 3" gần như chính là "khách đã rời bỏ", nên mô hình đạt ROC-AUC 0,977 và các luật đạt 97–99% là do nhìn thấy đáp án.

**Cách sửa:**
- Mỗi nhóm khách có một **mốc quan sát** (`data_prep/config.py`): khách hết hạn tháng 2 → mốc 31/01/2017; khách hết hạn tháng 3 → mốc 28/02/2017. Mọi đặc trưng chỉ dùng dữ liệu trước mốc.
- Mô hình **học trên tháng 2, kiểm tra trên tháng 3**, giống cách dùng thật.
- Thêm `data/manifest.json` tự kiểm tra không có giao dịch nào sau mốc.
- Thêm `mining/06_validation.py` đo lại mức thổi phồng: ROC-AUC tăng ảo 0,05–0,07 khi dùng dữ liệu tháng 3.

## 2. Thay đổi theo từng file

| File | Bản cũ | Bản mới |
|---|---|---|
| `data_prep/00_download.py` | Tải bằng Kaggle CLI, giải nén 4 file | Nhận được file zip "Download All"; file nhật ký 28 GB được đọc lọc khi giải nén (`stream_logs.py`) thay vì giải nén ra ổ đĩa |
| `data_prep/01_build_features.py` | 13 đặc trưng, dùng dữ liệu tháng 3, lấy mẫu 50/50 | 29 đặc trưng cắt theo mốc quan sát; thêm thông tin giao dịch cuối, số ngày còn hạn, xu hướng nghe, độ đều đặn theo tuần; giữ tỉ lệ rời bỏ thật (~9%); dữ liệu cả tháng 2 và tháng 3 |
| `data_prep/02_build_sequences.py` | – | **Mới** (Codex viết): chuỗi giao dịch cho mỗi khách |
| `data_prep/03_build_leakage_audit.py` | – | **Mới**: dữ liệu để đo rò rỉ |
| `analysis/01_eda.py` | 2 hình | Khoảng tin cậy 95%, kiểm định χ² và Cramér's V, so sánh phân bố, ma trận tương quan |
| `mining/01_clustering.py` | K-Means, chọn k bằng silhouette | Thêm Davies–Bouldin, Calinski–Harabasz, elbow; kiểm tra độ ổn định (ARI); so sánh với Ward, GMM, HDBSCAN; đặt tên nhóm; gán toàn bộ khách |
| `mining/02_association_rules.py` | FP-Growth, chia nhóm bằng `rank + qcut` (tạo đặc điểm giả) | Sửa cách chia nhóm; thêm kiểm định thống kê và hiệu chỉnh FDR; bỏ luật thừa; kiểm tra lại trên 30% dữ liệu giữ riêng; độ nhạy theo ngưỡng; thêm luật "rời bỏ thấp" |
| `mining/03_classification.py` | Cây quyết định, rừng ngẫu nhiên, chia ngẫu nhiên 75/25 | 5 mô hình (thêm hồi quy logistic, LightGBM, mốc so sánh); học tháng 2, kiểm tra tháng 3; CV 5 lần; PR-AUC, độ chính xác ở top 10%; SHAP; so sánh theo nhóm đặc trưng |
| `mining/04_bonus.py` | Xu hướng nghe, PCA/UMAP trên dữ liệu tháng 3 | Tính lại theo mốc quan sát; thêm quỹ đạo 8 tuần, phát hiện bất thường (Isolation Forest), so sánh nhóm con |
| `mining/05_sequential.py` | – | **Mới** (Codex viết): mẫu tuần tự PrefixSpan, kiểm định, kiểm tra trên tháng 2 |
| `mining/06_validation.py` | – (README có nhắc nhưng chưa có) | **Mới**: đo rò rỉ dữ liệu, bảng so sánh kỹ thuật |
| `mining/07_stability.py` | – | **Mới**: áp nhóm khách và luật của tháng 3 lên tháng 2 |
| `mining/08_business_value.py` | – | **Mới**: hiệu chỉnh xác suất, đường lợi nhuận |
| `mining/09_model_improvement.py` | – | **Mới**: thêm 13 đặc trưng thanh toán, tinh chỉnh tham số bằng Optuna |
| `mining/10_metrics_summary.py` | – | **Mới**: khung đánh giá 7 khía cạnh, độ phủ luật, PSI giữa 2 tháng, xếp hạng yếu tố ảnh hưởng |
| `mining/11_clustering_variants.py` | – | **Mới**: so sánh 4 cách chọn đặc trưng cho phân cụm |
| `mining/12_ordering_sensitivity.py` | – | **Mới** (Codex viết): độ nhạy của thứ tự giao dịch cùng ngày |
| `mining/13_drift_ablation.py` | – | **Mới**: bỏ đặc trưng lệch theo lịch để kiểm chứng nguyên nhân AUC giảm |
| `mining/common.py`, `mining/transforms.py` | Đọc dữ liệu, điền giá trị thiếu trước khi chia (gây rò rỉ nhẹ) | Không điền trước khi chia; hàm dùng chung cho các bước |
| `report/build_report.py` | Số liệu và tên nhóm ghi tay | Lấy số từ kết quả; viết lại lời văn dễ hiểu; tự đánh số bảng/hình; dịch luật và chuỗi sang tiếng Việt (`vi_labels.py`) |
| `report/make_diagrams.py` | – | **Mới**: sơ đồ quy trình và trục thời gian |
| `run_all.py`, `requirements.txt` | – | **Mới**: chạy toàn bộ bằng một lệnh; ghi rõ phiên bản thư viện |
| `tests/make_synthetic_data.py` | – | **Mới**: dữ liệu giả để chạy thử |
| `report/slides.pptx` | Có (số liệu cũ) | Đã xoá vì số liệu sai; slide mới sẽ làm sau |

## 3. Sửa sau khi rà soát phương pháp (vòng 11)

- Bản ghi gói 0 ngày (615 nghìn giao dịch) coi là "không rõ gói" ở `01_build_features.py`, `02_build_sequences.py` (Codex), `09_model_improvement.py`.
- Ngày hết hạn hỏng, ngày đăng ký sau mốc quan sát → thiếu; thêm cờ `has_member`; `manifest.json` ghi số bản ghi lỗi.
- `01_clustering.py`: tách khách không có nhật ký theo quy tắc; chọn k bằng silhouette + độ ổn định (không dùng nhãn); Jaccard từng cụm; bảng nhiệt trong không gian chuẩn hoá.
- `07_stability.py`: gán nhóm quy tắc cho tháng 2 giống tháng 3.
- Dữ liệu giả (`tests/`) có thêm khách không có nhật ký và bản ghi gói 0 ngày để kiểm thử các nhánh mới.

## 4. Kết quả trước và sau

| Chỉ số | Bản cũ | Bản mới |
|---|---|---|
| ROC-AUC mô hình | 0,977 (công bố, có rò rỉ, kiểm tra trong cùng tháng) | 0,870 (kiểm tra trên tháng sau). So sánh công bằng trên cùng phép chia: rò rỉ làm tăng ảo 0,920 → 0,976 |
| Luật mạnh nhất | "không có nhật ký nghe + không có gói (trong tháng 3) → rời bỏ 99,7%" (gần như chính là định nghĩa nhãn) | "gói > 30 ngày → rời bỏ 83%", kiểm tra lại trên dữ liệu riêng và trên tháng 2 |
| Số nhóm khách | 5 (dựa trên dữ liệu tháng 3, có cụm nhân tạo) | 7 cụm (13 đặc trưng, bỏ 3 đặc trưng trùng) + 1 nhóm quy tắc; ổn định khi lấy mẫu lại (ARI 0,996), tách nhau vừa phải |
| Hành vi nghe nhạc | Có vẻ quan trọng | Gần như không phân biệt được ai rời bỏ (phát hiện thật sau khi bỏ rò rỉ) |
| Kỹ thuật khai phá | 3 | 4 kỹ thuật chính + phát hiện bất thường, PCA/UMAP |

## 5. Phân công Claude – Codex

- **Codex**: review code bản cũ (tìm ra lỗi rò rỉ và lỗi `rank + qcut`), viết phần mẫu tuần tự (`02_build_sequences.py`, `05_sequential.py`), review chéo code và soát toàn bộ số liệu, văn phong trong báo cáo.
- **Claude**: thiết kế lại quy trình, viết các phần còn lại, chạy trên dữ liệu thật, viết báo cáo.
