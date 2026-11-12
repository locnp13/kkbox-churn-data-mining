# Đánh giá phương pháp và tổng hợp cải tiến

Tài liệu này dùng để tự đánh giá đồ án theo góc nhìn người chấm: mỗi bước làm gì, vì sao, đã kiểm chứng bằng gì, và còn điểm yếu nào. Claude và Codex rà soát độc lập trên code và dữ liệu thật; các vấn đề tìm được ở vòng rà soát đã được sửa và chạy lại (vòng 11).

Ký hiệu trạng thái: **Đạt** – ổn ở mức thạc sĩ · **Đã sửa** – vấn đề phát hiện khi rà soát, đã sửa và chạy lại · **Còn hạn chế** – đã ghi rõ trong báo cáo · **Nên bổ sung** – làm bài mạnh hơn nhưng không bắt buộc.

---

## Phần A. Tổng hợp cải tiến theo từng vòng

| Vòng | Nội dung | Lý do | Kết quả chính |
|---|---|---|---|
| 1. Rà soát bản cũ | Codex đọc code bản cũ | Tìm lỗi trước khi làm tiếp | Phát hiện rò rỉ dữ liệu tháng 3, lỗi `rank + qcut` tạo đặc điểm giả, báo cáo ghi số tay |
| 2. Thiết kế lại | Mốc quan sát cho từng nhóm khách; học tháng 2, kiểm tra tháng 3 | Bỏ rò rỉ, mô phỏng đúng cách dùng thật | Bản cũ công bố ROC-AUC 0,977 (có rò rỉ). Cùng phép chia: dữ liệu sạch 0,920, cách làm cũ 0,976 (tăng ảo 0,055). Kiểm tra đúng trên tháng sau: 0,870 |
| 3. Dữ liệu lớn | Đọc file nhật ký 28 GB theo luồng | Máy không đủ ổ đĩa để giải nén | 392 triệu dòng → 40,9 triệu dòng cần dùng, 13 phút |
| 4. Khai phá | Phân cụm, luật kết hợp, mẫu tuần tự (Codex), phân lớp, phát hiện bất thường | Đề yêu cầu ≥ 2 kỹ thuật, ≥ 1 không giám sát | Luật "gói > 30 ngày → 83%"; chuỗi "trả thủ công → gói dài → gói dài → 70%" |
| 5. Kiểm chứng | Holdout 30%, kiểm định + FDR, bootstrap, đo rò rỉ | Tránh mẫu ngẫu nhiên, đo mức thổi phồng | Lift train/holdout tương quan 0,999; rò rỉ thổi AUC thêm 0,05–0,07 |
| 6. Ổn định theo thời gian | Áp kết quả tháng 3 lên tháng 2 | Mẫu thật phải lặp lại | 33/34 luật, 30/30 chuỗi giữ ý nghĩa; thứ tự rủi ro các nhóm gần như giữ nguyên |
| 7. Giá trị kinh doanh | Hiệu chỉnh xác suất (EM), đường lợi nhuận | Nối kết quả với quyết định | Nên liên hệ 8–15% khách rủi ro nhất; EM không sửa được lệch xác suất |
| 8. Cải thiện mô hình | 13 đặc trưng thanh toán + Optuna | Thử nâng độ chính xác | +0,004 ROC-AUC, không đổi ở top 10% |
| 9. Trình bày | Viết lại lời văn, sơ đồ, thuật ngữ, Việt hoá | Báo cáo dễ đọc | Có sơ đồ quy trình, trục thời gian, bảng thuật ngữ |
| 10. Rà soát phương pháp | Kiểm tra lại dữ liệu và phương pháp | Tự đánh giá trước khi nộp | Phát hiện lỗi gói 0 ngày, cụm nhân tạo, chọn k có dùng nhãn (mục B, C) |
| 11. Sửa và chạy lại | Sửa toàn bộ vấn đề "Cần sửa" của vòng 10, chạy lại trên dữ liệu thật | Kết quả phải dựa trên dữ liệu sạch và phương pháp đúng | Phân cụm còn 3 cụm + 1 nhóm quy tắc (thay vì 7); tín hiệu "đổi gói" yếu đi nhưng vẫn rõ; kết luận chính giữ nguyên |
| 12. Khung đánh giá | Đánh giá theo 7 khía cạnh của bài toán; xếp hạng và giải thích các yếu tố ảnh hưởng | Đánh giá phải trả lời "bài toán giải quyết tốt đến đâu" | Thêm độ phủ khách rời bỏ của luật, PSI (độ thay đổi giữa tháng), độ dài luật; phát hiện đặc trưng "số ngày từ giao dịch đầu" bị lệch theo lịch (PSI 0,86) |
| 13. Phân tích độ nhạy | Thử 4 cách chọn đặc trưng cho phân cụm; thử 4 quy tắc sắp xếp giao dịch cùng ngày (Codex); bỏ đặc trưng lệch theo lịch | Kiểm tra kết quả có phụ thuộc vào lựa chọn của nhóm không | Bỏ 3 đặc trưng trùng → 7 cụm ổn định khi lấy mẫu lại (ARI 0,996 trên mẫu chính) thay cho 3 cụm; lựa chọn này được làm sau khi xem kết quả và phụ thuộc vào mẫu (ARI 0,75 trên mẫu 100.000); thứ tự cùng ngày chỉ ảnh hưởng 1,3% khách nhưng làm tỉ lệ rời bỏ của nhóm "hủy" dao động 55–85%; bỏ đặc trưng lệch theo lịch chỉ thu hẹp khoảng 1/10 khoảng chênh AUC |

Chi tiết thay đổi từng file: [THAY_DOI.md](THAY_DOI.md).

---

## Phần B. Tiền xử lý và làm sạch dữ liệu

### B1. Những gì làm tốt từ đầu

| Vấn đề | Cách xử lý | Trạng thái |
|---|---|---|
| Rò rỉ dữ liệu tháng nhãn | Cắt mọi nguồn tại mốc quan sát; `manifest.json` tự kiểm tra | **Đạt** |
| Giao dịch trùng giữa 2 file | Gộp và bỏ dòng trùng | **Đạt** |
| Thứ tự giao dịch cùng ngày | Một quy tắc sắp xếp chung cho mọi bước | **Đạt** |
| Tuổi sai (âm, 0, > 100) | Ngoài 10–80 coi là thiếu (60% thiếu); không dùng tuổi trong phân cụm | **Đạt** |
| Giới tính trống | Nhóm "không rõ" | **Đạt** |
| Giờ nghe phi lý (> 24 giờ/ngày, âm) | Chặn trong [0, 24 giờ] | **Đạt** |
| Điền giá trị thiếu | Chỉ học trên phần huấn luyện (trong Pipeline) | **Đạt** |
| Mất cân bằng lớp (9% rời bỏ) | Giữ tỉ lệ thật, dùng PR-AUC, lift, recall ở top 10% thay vì accuracy | **Đạt** |
| Đuôi dài (giờ nghe, số giao dịch) | log1p trước khi chuẩn hoá cho phân cụm | **Đạt** |

### B2. Vấn đề phát hiện khi rà soát và trạng thái

| # | Vấn đề (số liệu trên dữ liệu thật) | Mức | Đã làm | Trạng thái |
|---|---|---|---|---|
| 1 | **Giao dịch gói 0 ngày**: 615.360 giao dịch (3,5%) có độ dài gói = 0 và giá niêm yết = 0 nhưng vẫn trả 149 NTD. 78% số lần "đổi gói" là đổi giả. | Cao | Coi là "không rõ gói" ở mọi bước (đặc trưng, chuỗi, thí nghiệm cải tiến); chỉ đếm đổi gói giữa gói hợp lệ; giảm giá chỉ khi có giá niêm yết | **Đã sửa** |
| 2 | **Ngày hết hạn hỏng**: 7.271 giao dịch có ngày hết hạn trước 2015 | Trung bình | Chuyển thành thiếu; giao dịch có hạn trước ngày giao dịch (73.980, phần lớn là hủy) được giữ | **Đã sửa** |
| 3 | **Ngày đăng ký sau mốc quan sát**: 1.057 khách (tháng 3) | Trung bình | Thâm niên chuyển thành thiếu | **Đã sửa** |
| 4 | **Thành phố 1**: 51% khách, 97% không khai giới tính → có thể là mã "không rõ" | Thấp | Ghi chú khi diễn giải; không dùng thành phố để kết luận | **Còn hạn chế** |
| 5 | **Thứ tự giao dịch cùng ngày** chỉ là quy ước | Thấp | Codex thử 3 quy tắc khác (vòng 13): chỉ 1,3% khách bị ảnh hưởng; kết quả về tự gia hạn không đổi; tỉ lệ rời bỏ của nhóm "giao dịch cuối là hủy" dao động 55–85% | **Đã kiểm tra** (ghi rõ trong báo cáo) |
| 6 | **11% khách không có hồ sơ** | Thấp | Thêm cờ `has_member` | **Đã sửa** |
| 7 | Ngưỡng tuổi 10–80, chặn `days_to_expire` chưa giải thích | Thấp | Bảng "lỗi dữ liệu và cách xử lý" trong báo cáo (mục 3.1) | **Đã sửa** |
| 8 | **Đặc trưng "số ngày từ giao dịch đầu" bị giới hạn bởi ngày bắt đầu dữ liệu (01/2015)**, thay đổi mạnh giữa 2 tháng (PSI 0,86) – phát hiện ở vòng 12 | Trung bình | Thí nghiệm bỏ đặc trưng (vòng 13): khoảng chênh AUC chỉ giảm 0,031 → 0,028, nên đây không phải nguyên nhân chính; nguyên nhân còn lại chưa xác định | **Đã kiểm tra** |

---

## Phần C. Phân cụm

### C1. Điểm mạnh

- Không đưa nhãn rời bỏ vào dữ liệu phân cụm, **kể cả khi chọn số cụm** (đã sửa ở vòng 11).
- Khách không có nhật ký nghe được tách bằng quy tắc thay vì để K-Means tìm lại một cụm nhân tạo.
- Chọn k bằng 5 chỉ số nội tại: elbow, silhouette, Davies–Bouldin, Calinski–Harabasz và **độ ổn định bootstrap cho từng k**.
- Đo ổn định ở hai mức: toàn bộ (ARI) và **từng cụm (Jaccard)**; mỗi lần bootstrap fit lại cả bước điền thiếu và chuẩn hoá.
- So sánh với Ward, GMM, HDBSCAN trên cùng mẫu; kiểm tra trên tháng 2 (fit lại ARI 0,98, thứ tự rủi ro giữ nguyên).
- Bảng nhiệt dùng tâm cụm trong đúng không gian chuẩn hoá mà K-Means dùng.

### C2. Vấn đề và trạng thái

| # | Vấn đề | Mức | Đã làm | Trạng thái |
|---|---|---|---|---|
| 1 | Cụm nhân tạo từ khách không có nhật ký (99% một cụm) | Cao | Tách theo quy tắc; K-Means chỉ chạy trên khách có nhật ký | **Đã sửa** |
| 2 | Lý do chọn k có dùng tỉ lệ rời bỏ | Cao | Chọn k chỉ bằng silhouette + độ ổn định | **Đã sửa** |
| 3 | Quy tắc "75% silhouette" là quy ước | Trung bình | Thêm điều kiện ổn định ARI ≥ 0,8; ghi rõ là quy ước. Không k nào đạt cả hai → dùng k có silhouette cao nhất | **Đã sửa** (minh bạch) |
| 4 | Biến nghe nhạc trùng nhau làm chiều "mức độ nghe" bị tính nhiều lần | Trung bình | Thử 4 cách (vòng 13); chọn cách bỏ biến trùng (tương quan > 0,8) cho phân cụm chính sau khi xem kết quả, báo cáo đủ 4 cách | **Đã sửa** (lựa chọn sau quan sát, ghi rõ) |
| 5 | Chưa đo ổn định từng cụm | Trung bình | Thêm Jaccard từng cụm | **Đã sửa** |
| 6 | Bỏ biến phân loại khỏi phân cụm | Thấp | Đã thử thêm kênh đăng ký và phương thức thanh toán (cách D): kém tách và kém ổn định hơn | **Đã kiểm tra** |
| 7 | So sánh thuật toán chưa tinh chỉnh tham số | Thấp | Ghi rõ là thiết lập mặc định | **Còn hạn chế** |
| 8 | Bảng nhiệt dùng z-score thang gốc | Thấp | Dùng tâm cụm trong không gian chuẩn hoá | **Đã sửa** |

### C3. Kết quả qua các vòng (trung thực)

| Chỉ số | Vòng 4 (7 cụm, có cụm nhân tạo) | Vòng 11 (16 đặc trưng, 3 cụm + 1 nhóm quy tắc) | Vòng 13 – bản cuối (13 đặc trưng, 7 cụm + 1 nhóm quy tắc) |
|---|---|---|---|
| Silhouette | 0,273 | 0,297 | 0,212 |
| Độ ổn định ARI (lấy mẫu lại) | 0,88 | 0,71 | 0,996 |
| Cụm kém ổn định nhất (Jaccard) | chưa đo | 0,68 | 0,47 (cụm "hay được giảm giá") |
| Phân cụm lại ở tháng 2 (ARI) | 0,98 | 0,98 | 0,72 |
| Chênh lệch tỉ lệ rời bỏ | 3,3% – 84% | 6,6% – 84% | 3,2% – 84% |

Bản cuối dùng 13 đặc trưng (bỏ 3 đặc trưng trùng), tìm ra 7 nhóm có ý nghĩa nghiệp vụ, trong đó có nhóm "nghe nhiều nhưng trả thủ công" (8,3% khách, 22% rời bỏ) mà bản 3 cụm không thấy. Đánh đổi: các nhóm tách nhau yếu (silhouette 0,21, dưới ngưỡng 0,25), một cụm nhỏ kém ổn định, và phân cụm lại ở tháng 2 chỉ giống ở mức khá (0,72). Cả ba điểm đều ghi rõ trong báo cáo.

---

## Phần D. Khung đánh giá 7 khía cạnh (vòng 12)

File `mining/10_metrics_summary.py` → `report/metrics_summary.csv`, trình bày ở mục 6.4 của báo cáo.

| Khía cạnh | Câu hỏi | Kết luận tóm tắt |
|---|---|---|
| 1. Tính hợp lệ | Có dùng nhầm thông tin tương lai, dữ liệu lỗi không? | **Đạt**: không có giao dịch sau mốc; lỗi đã xử lý; đo được mức thổi phồng 0,05–0,07 |
| 2. Chất lượng kỹ thuật | Từng kỹ thuật tự nó có tốt không? | Mô hình (ROC-AUC 0,870, PR-AUC 0,603), luật, chuỗi **đạt**; phân cụm **chưa đạt về độ tách** (silhouette 0,21 dưới ngưỡng 0,25; DB 1,36) |
| 3. Độ tin cậy | Có giữ trên dữ liệu chưa thấy và tháng khác? | Luật (r 0,999, 33/34 giữ hướng), chuỗi (100%) **đạt**; phân cụm ổn định khi lấy mẫu lại (ARI 0,996) nhưng một cụm nhỏ kém ổn định (Jaccard 0,47) và phân cụm lại ở tháng 2 chỉ giống ở mức khá (ARI 0,72) |
| 4. Phân biệt rủi ro | Có tách được khách rủi ro cao và thấp? | **Đạt**: lift 9,3 so với 0,21; lift top 10% = 6,1 |
| 5. Độ phủ | Phát hiện bao trùm bao nhiêu phần vấn đề? | 5 luật gắn cờ 5,1% khách, bao phủ 36% khách rời bỏ; mô hình 10% khách → 61% |
| 6. Kinh doanh và vận hành | Áp dụng thật có lợi và bền không? | Có lãi theo giả định; **chưa đạt** về hiệu chuẩn (ECE 0,026); 2 đặc trưng thay đổi đáng kể giữa tháng |
| 7. Khả năng giải thích | Người làm kinh doanh có dùng được? | **Đạt**: 8 nhóm có tên nghiệp vụ; luật trung bình 2 điều kiện; chuỗi 2–3 bước |

Các yếu tố ảnh hưởng (mục 6.5 báo cáo, `report/factor_summary.csv`): tự gia hạn > hủy ở giao dịch cuối > độ dài gói và số tiền trả > phương thức thanh toán > mức giao dịch gần đây và thời hạn còn lại > kênh đăng ký > thói quen nghe (yếu). Mỗi yếu tố có bằng chứng từ nhiều kỹ thuật và một giải thích nghiệp vụ, kèm lưu ý đây là liên hệ, chưa phải nhân quả.

---

## Phần E. Kết luận đánh giá

**Theo đánh giá của nhóm, đồ án đáp ứng đầy đủ các yêu cầu của đề**: quy trình KDD rõ ràng; chống rò rỉ dữ liệu và làm sạch có kiểm chứng; bốn kỹ thuật khai phá bổ trợ nhau; đánh giá theo câu hỏi của bài toán (7 khía cạnh) chứ không chỉ liệt kê chỉ số; kiểm chứng trên dữ liệu giữ riêng và tháng khác; có giá trị kinh doanh và tái lập được.

**Kết luận chính không đổi sau khi sửa**: rủi ro rời bỏ gắn với cách khách trả tiền (tự gia hạn, gói dài trả thủ công), thói quen nghe chỉ bổ sung.

**Giới hạn đã nêu rõ trong báo cáo**: các cụm tách nhau yếu (silhouette 0,21), cách chia phụ thuộc việc chọn đặc trưng (lựa chọn sau quan sát) và lặp lại ở tháng khác chỉ ở mức khá; xác suất dự báo lệch khi tỉ lệ rời bỏ thay đổi; chỉ có 2 tháng có nhãn; quan hệ tìm được là liên hệ, chưa phải nhân quả.

**Các mục "nên bổ sung" đã làm ở vòng 13** (mục 6.7 báo cáo): so sánh 4 cách chọn đặc trưng cho phân cụm, kiểm tra độ nhạy của thứ tự giao dịch cùng ngày, và thí nghiệm bỏ đặc trưng lệch theo lịch. Còn lại cho hướng phát triển: kiểm tra trên nhiều tháng hơn, thử nghiệm A/B, đo thời gian từ tín hiệu cảnh báo đến ngày hết hạn.
