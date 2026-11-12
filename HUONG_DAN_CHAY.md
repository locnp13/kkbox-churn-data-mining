# Hướng dẫn chạy code

Hướng dẫn này viết cho máy Windows; lệnh cho macOS/Linux ghi ở cuối. Chạy hết từ đầu mất khoảng 1–1,5 giờ (phần lớn là giải nén dữ liệu và huấn luyện mô hình).

## 1. Chuẩn bị

| Cần có | Ghi chú |
|---|---|
| Python 3.11 trở lên | Kiểm tra: `python --version` |
| Git | Để tải code về |
| RAM 16 GB | Ít hơn vẫn chạy được nhưng chậm, có thể thiếu bộ nhớ ở bước phân lớp |
| Ổ đĩa trống ~20 GB | File zip 9 GB + dữ liệu đã xử lý ~7 GB |
| Tài khoản Kaggle | Để tải dữ liệu |

## 2. Tải code và cài thư viện

Mở PowerShell:

```powershell
git clone -b Khanh https://github.com/duongvanlo20003/kkbox-churn-data-mining.git
cd kkbox-churn-data-mining
pip install -r requirements.txt
$env:PYTHONUTF8 = "1"          # để in tiếng Việt không lỗi; mỗi lần mở PowerShell mới cần đặt lại
```

## 3. Tải dữ liệu

1. Vào https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/data, đăng nhập.
2. Bấm **Join the competition** / **I Understand and Accept** (chỉ cần làm một lần).
3. Bấm **Download All**, lưu file `kkbox-churn-prediction-challenge.zip` (~9 GB). Nên tải bằng Chrome/Edge để tải tiếp được nếu mạng chập chờn.
4. Chạy lệnh sau (thay đường dẫn cho đúng):

```powershell
python data_prep/00_download.py "C:\Users\<tên máy>\Downloads\kkbox-churn-prediction-challenge.zip"
```

Lệnh này lấy các file cần dùng ra thư mục `data/`. Riêng file nhật ký nghe nhạc 28 GB không được giải nén ra ổ đĩa: chương trình đọc và lọc ngay trong lúc giải nén (khoảng 13 phút), chỉ giữ lại ~2 GB cần dùng.

Cách khác: nếu đã có token Kaggle trong `C:\Users\<tên máy>\.kaggle\`, chạy `python data_prep/00_download.py` (không cần đường dẫn) để tự tải.

## 4. Chạy toàn bộ phân tích

```powershell
python run_all.py --skip-download
```

Chương trình chạy lần lượt 19 bước dưới đây và dừng ngay nếu một bước lỗi. Nhật ký từng bước nằm ở `report/logs/<tên bước>.log`.

| Bước | File | Làm gì | Thời gian* |
|---|---|---|---|
| 1 | `data_prep/01_build_features.py` | Cắt dữ liệu theo mốc quan sát, tạo đặc trưng cho mỗi khách | ~6 phút |
| 2 | `data_prep/02_build_sequences.py` | Tạo chuỗi giao dịch | < 1 phút |
| 3 | `data_prep/03_build_leakage_audit.py` | Dữ liệu để đo rò rỉ | < 1 phút |
| 4 | `analysis/01_eda.py` | Phân tích khám phá, hình | ~1 phút |
| 5 | `mining/01_clustering.py` | Phân cụm K-Means, so sánh thuật toán | ~1 phút |
| 6 | `mining/02_association_rules.py` | Luật kết hợp FP-Growth | ~1 phút |
| 7 | `mining/03_classification.py` | Huấn luyện và đánh giá 5 mô hình | ~12 phút |
| 8 | `mining/04_bonus.py` | Quỹ đạo nghe, phát hiện bất thường, PCA/UMAP | ~1,5 phút |
| 9 | `mining/05_sequential.py` | Mẫu tuần tự PrefixSpan | < 1 phút |
| 10 | `mining/06_validation.py` | Đo rò rỉ dữ liệu, so sánh kỹ thuật | ~2 phút |
| 11 | `mining/07_stability.py` | Kiểm tra lại trên tháng 2 | < 1 phút |
| 12 | `mining/08_business_value.py` | Hiệu chỉnh xác suất, đường lợi nhuận | < 1 phút |
| 13 | `mining/09_model_improvement.py` | Thử thêm đặc trưng và tinh chỉnh tham số | ~12 phút |
| 14 | `mining/10_metrics_summary.py` | Bảng đánh giá 7 khía cạnh, xếp hạng yếu tố ảnh hưởng | < 1 phút |
| 15 | `mining/11_clustering_variants.py` | So sánh 4 cách chọn đặc trưng cho phân cụm | ~5 phút |
| 16 | `mining/12_ordering_sensitivity.py` | Độ nhạy của thứ tự giao dịch cùng ngày | < 1 phút |
| 17 | `mining/13_drift_ablation.py` | Bỏ đặc trưng lệch theo lịch, đo lại | ~3 phút |
| 18 | `report/make_diagrams.py` | Vẽ sơ đồ quy trình, trục thời gian | vài giây |
| 19 | `report/build_report.py` | Dựng `report/report.docx` | vài giây |

\* Đo trên laptop 16 GB RAM.

Chạy lại từ một bước (ví dụ sau khi sửa code phân cụm):

```powershell
python run_all.py --from 01_clustering
```

Chỉ chạy một bước: `python mining/02_association_rules.py`.

## 5. Kết quả nằm ở đâu

| Thư mục / file | Nội dung |
|---|---|
| `report/report.docx` | Báo cáo hoàn chỉnh. Xuất PDF: mở bằng Word → File → Save As → PDF |
| `report/figures/` | Toàn bộ hình (PNG) |
| `report/*.csv` | Bảng kết quả (phân khúc, luật, mẫu tuần tự, kết quả mô hình…) |
| `report/metrics/*.json` | Các chỉ số tóm tắt của từng bước |
| `report/metrics_summary.csv`, `report/factor_summary.csv` | Bảng đánh giá 7 khía cạnh; xếp hạng các yếu tố ảnh hưởng |
| `data/manifest.json` | Kiểm tra dữ liệu: số khách, tỉ lệ rời bỏ, ngày giao dịch muộn nhất (phải trước mốc quan sát) |

## 6. Chạy thử nhanh không cần dữ liệu thật (~2 phút)

Dùng để kiểm tra code chạy được trước khi tải 9 GB dữ liệu. Kết quả là dữ liệu giả nên không có ý nghĩa.

```powershell
python tests/make_synthetic_data.py data_synth
$env:DATA_DIR = "data_synth"; $env:REPORT_DIR = "report_synth"
python run_all.py --skip-download
Remove-Item Env:DATA_DIR, Env:REPORT_DIR      # trả lại cài đặt bình thường
```

## 7. Tuỳ chỉnh

Đặt biến môi trường trước khi chạy (PowerShell: `$env:TEN = "giá trị"`):

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `CLUSTER_SAMPLE` | 200000 | Số khách dùng để chọn số cụm |
| `CLUSTER_K` | tự chọn | Ép số cụm cố định |
| `SIL_TOL`, `ARI_MIN`, `CLUSTER_BOOT`, `K_MAX` | 0.75, 0.8, 5, 7 | Quy tắc chọn số cụm: k lớn nhất ≤ K_MAX có silhouette ≥ SIL_TOL × tốt nhất và độ ổn định ≥ ARI_MIN; số lần bootstrap |
| `MIN_SUPPORT` | 0.01 | Ngưỡng support của luật kết hợp |
| `SEQ_SAMPLE` | 300000 | Số khách dùng cho mẫu tuần tự |
| `N_TRIALS` | 30 | Số lần thử khi tinh chỉnh tham số |
| `CONTACT_COST`, `DISCOUNT`, `VALUE_MONTHS`, `SUCCESS` | 5, 0.2, 6, "0.1,0.2,0.3" | Giả định của đường lợi nhuận |
| `FAST` | – | Đặt `1` để phân lớp chạy nhanh hơn (ít cây, ít fold) |

Mốc quan sát và độ dài cửa sổ nghe nhạc nằm trong `data_prep/config.py`.

## 8. Lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| `UnicodeEncodeError` khi in tiếng Việt | Chạy `$env:PYTHONUTF8 = "1"` rồi chạy lại |
| `PermissionError ... report.docx` | Đang mở `report.docx` trong Word; đóng Word rồi chạy lại bước cuối |
| `FileNotFoundError: data/features_mar.parquet` | Chưa chạy bước tạo đặc trưng: `python run_all.py --skip-download` |
| Hết bộ nhớ ở bước phân lớp | Đặt `$env:RF_SAMPLE = "150000"` và `$env:FAST = "1"` |
| Tải zip bị đứt, file không mở được | Xoá file hỏng, tải lại bằng Chrome/Edge |

## macOS / Linux

Giống hệt, chỉ khác cách đặt biến môi trường: `export PYTHONUTF8=1`, và chạy thử nhanh bằng
`DATA_DIR=data_synth REPORT_DIR=report_synth python run_all.py --skip-download`.
