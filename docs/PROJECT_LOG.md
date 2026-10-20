# Nhật ký dự án: dự báo giá dầu WTI bằng họ mô hình LTSF-Linear

Tài liệu này ghi lại **mọi việc đã làm, quyết định, lý do, sai sót và kết quả**, để dùng khi viết báo cáo.
Cập nhật lần cuối: 13/10/2026. Mỗi lần có thay đổi, thêm một mục vào phần tương ứng (xem mục 11).

Quy ước: **[Đã chạy]** = người dùng đã chạy trên máy và báo kết quả; **[Chưa chạy]** = mới được viết, chưa có kết quả;
**[Chưa xác minh]** = thông tin cần đối chiếu nguồn trước khi dùng trong báo cáo.

---

## 1. Tóm tắt dự án

- **Đề tài:** dự báo giá đóng cửa dầu WTI cho 5 phiên tiếp theo bằng LTSF-Linear (Linear, NLinear, DLinear), mô hình lai
  NLinear + DLinear, và mô hình lai đa biến; đánh giá walk-forward và kiểm định Diebold–Mariano.
- **Hình thức:** đồ án môn học (có mã nguồn và demo), giữ chuẩn nghiên cứu như luận văn; báo cáo theo khuôn 5 chương của đồ án cũ.
- **Câu hỏi nghiên cứu:** RQ2: lai có hơn đơn biến không? RQ1: biến phụ trợ có cải thiện mô hình lai không?
- **Kết quả chính (đã chạy):** không mô hình nào vượt naive; NLinear và lai hoà với naive; thêm biến phụ trợ không giúp;
  nhánh "mức hiện tại" gây hại. Chi tiết ở mục 6.

## 2. Dòng thời gian các giai đoạn

| GĐ | Việc đã làm | Kết quả / quyết định |
|---|---|---|
| 1 | Đọc tài liệu *Project: Forecasting with DLinear and NLinear Models* (AI VIET NAM, AIO 2026) | Hiểu 3 mô hình, dự báo trực tiếp, MSE; ghi nhận các lưu ý về tài liệu (mục 8) |
| 2 | Đánh giá ý tưởng áp dụng cho dầu với USD/CAD, lãi suất FED, Brent, WTI | Brent và WTI trùng lặp; FED yếu ở kỳ hạn ngắn; tương quan chưa phải khả năng dự báo; cần so với naive |
| 3 | Chốt mục tiêu: dự báo **giá đóng cửa** (bỏ giá mở), mục tiêu **WTI**, mục đích nghiên cứu học thuật | Bỏ giá mở vì gap qua đêm khó dự báo và định nghĩa phụ thuộc nguồn |
| 4 | Chốt phạm vi: RQ1 và RQ2; mô hình đơn biến, lai, lai + đa biến | Loại nhánh phi tuyến, dự báo phân phối, mô hình học sâu đối chứng (đưa vào hướng phát triển) |
| 5 | Đọc file đồ án cũ (Power BI, ĐH Công nghệ TP.HCM) để lấy khuôn báo cáo | 5 chương, Times New Roman, lề trái 3 cm, giãn dòng 1,5, đánh số bảng/hình theo chương |
| 6 | Chốt dữ liệu: nguồn FRED, danh sách biến, quy tắc thời điểm | Mục 4 |
| 7 | Dựng cấu trúc thư mục dự án (62 file giữ chỗ) | `wti-ltsf-forecast/` |
| 8 | Viết `data_loader`, `preprocessing`, `run_data_pipeline`, test chống rò rỉ | **[Đã chạy]** 4.649 phiên, 4.061 ngày dùng được |
| 9 | Kiểm tra kết quả dữ liệu (19 cờ biến động, `lead_lag_check`, ADF) | Dữ liệu đạt; không thấy lệch ngày; return trễ gần như không dự báo được |
| 10 | Viết `features` (biến lịch) và `dataset` (cửa sổ, scaler, walk-forward có purge) | Bảng số mẫu theo L; phát hiện hố 2020 |
| 11 | Xử lý khoảng trống dữ liệu 2011 và 2015 | Cột `gap_before`; loại cửa sổ chạy qua khoảng trống |
| 12 | Viết `models` (Linear, NLinear, DLinear, Hybrid), `baselines` | Test khớp tính tay của tài liệu **[Đã chạy]** |
| 13 | Viết `train`, `evaluate`, test | Phát hiện lỗi seed qua test, đã sửa **[Đã chạy]** |
| 14 | Viết `experiments`, `run_experiments`, nhánh "mức hiện tại" | Dry-run: 2.088 kết quả; một lần chạy thử 0,5 phút/16 cấu hình |
| 15 | Chạy toàn bộ thí nghiệm L chính và L=480 | **[Đã chạy]** Mục 6 |
| 16 | Viết `report`, `run_evaluation`; đọc DM, theo năm, theo chế độ, theo bước | Mục 6 và 7 |
| 17 | Viết demo Streamlit (`app/`), `app/README.md`, `README.md` tổng | Demo đọc dự báo đã lưu, không dự báo tương lai |
| 18 | Thảo luận khung giờ (1 giờ, 4 giờ) | Giữ khung ngày làm trục chính; khung giờ vào hướng phát triển |
| 19 | **Cải tiến vòng 1** (mục 9): khởi tạo 0, weight decay chọn trên validation, co dự báo về naive | **[Đã chạy]** L=5 và L=30, kết quả ở 9.6 |
| 20 | Sự cố chép file: `app/predictor.py` và `run_experiments.py` trên máy người dùng sai nội dung (E8, E9) | Chép lại; nhờ `pytest` phát hiện |
| 21 | Chạy `run_evaluation.py` khi chưa sao lưu bảng vòng 0 (E10) | Thêm `--exclude-improvements`; bảng vòng 0 tái tạo, số liệu **trùng khớp** bản cũ |
| 22 | Chẩn đoán vòng 1 (`diagnose_improve1.py`): weight decay được chọn và λ | Mục 9.6 |
| 23 | Chức năng **dự đoán thử một ngày bất kỳ** (`src/live_predict.py`, `predict_day.py`, tab trong app): mô hình huấn luyện đến ngày chọn rồi dự báo từ L phiên cuối | Kiểm thử trong môi trường của trợ lý (có torch): không nhìn tương lai, tái lập, cache **[chưa chạy ở máy người dùng]** |
| 24 | `check_setup.py`, `docs/SETUP_CONFIG.md`, `configs/live.yaml`; viết lại `README.md` và `app/README.md` | Hướng dẫn chạy từ đầu và danh sách việc người dùng phải tự điền (khoá FRED...) |
| 26 | Nhóm A thêm đa biến (`linear_mv`, `nlinear_mv`, `dlinear_mv`); `train_all.py` + `src/model_store.py` huấn luyện và lưu toàn bộ mô hình; `predict_saved.py`; `run_all.py` chạy trọn quy trình; app thêm tab dự báo bằng mô hình đã lưu | 84 test đạt ở môi trường trợ lý; model store và app kiểm thử bằng dữ liệu giả lập **[chưa chạy ở máy người dùng]**; các mô hình `_mv` **chưa có kết quả thí nghiệm** |
| 25 | Đóng gói toàn bộ dự án sạch (không kèm dữ liệu/kết quả sinh ra, bỏ file rỗng giữ chỗ) để chạy lại từ đầu | Kiểm tra bằng chạy thử pipeline, thí nghiệm nhỏ, đánh giá và dự đoán thử trên dữ liệu giả lập |

## 3. Sổ đăng ký quyết định thiết kế

| ID | Quyết định | Phương án đã cân nhắc | Lý do |
|---|---|---|---|
| D1 | Mục tiêu là giá đóng cửa, không phải giá mở | Dự báo cả open và close | Giá mở chứa gap qua đêm và cuối tuần, định nghĩa open/close phụ thuộc nguồn |
| D2 | Mục tiêu WTI | Brent, CFD OIL/USD | Giá tham chiếu chuẩn; Brent trở thành biến phụ trợ |
| D3 | Nguồn dữ liệu duy nhất là FRED (giá spot) | Yahoo Finance, futures trả phí | Miễn phí, sạch, tái lập được, không có vấn đề roll; hạn chế: spot không phải futures |
| D4 | Khoảng thời gian: tải từ 2008-01-01, nghiên cứu từ 2010-01-01, cắt 2026-09-30 | 2000 trở đi; kết thúc 2025 | Đệm 2 năm cho cửa sổ quá khứ; cấu trúc thị trường sau 2010; mốc cắt chung vì tỷ giá công bố trễ |
| D5 | Mô hình hoá trên log-giá, đánh giá trên thay đổi so với log-giá đóng cuối | R² trên mức giá như tài liệu | R² trên mức giá luôn cao dù không dự báo được; giá WTI từng âm |
| D6 | Loại cửa sổ 2020-03-01 đến 2020-06-30; mọi cửa sổ chạm giai đoạn này bị loại | Giữ lại, thay chuỗi | Giá âm 20/4/2020 phá phép log và chi phối kết quả |
| D7 | Quy tắc trễ: WTI, Brent, USD/CAD, USD broad cùng ngày; OVX và FED trễ 1 phiên | Mọi biến trễ 1 phiên | Dự báo thực hiện sau khi thị trường đóng ngày t, nên giá cùng ngày đã biết; trễ là thận trọng |
| D8 | Điền khuyết chỉ bằng quan sát trước đó, bỏ nếu cũ hơn 5 ngày lịch | Nội suy | Không dùng thông tin tương lai |
| D9 | Khoảng cách hơn 5 ngày lịch giữa hai phiên là khoảng trống; cửa sổ chạy qua bị loại, không nội suy | Nội suy, vá bằng nguồn khác | Không tự tạo dữ liệu hay trộn hai nguồn khác định nghĩa |
| D10 | Walk-forward expanding window, test từng năm 2017 đến 2026 | Một lần chia 70/15/15 | Phủ nhiều chế độ thị trường; chia 70/15/15 dễ phụ thuộc điểm cắt |
| D11 | Validation = 250 mẫu ngay trước test; purge nhãn train/val trước origin đầu tiên của đoạn sau | Validation là "năm trước" | Năm 2020 gần như rỗng nên "năm trước" không dùng được cho fold 2021 |
| D12 | Scaler chỉ fit trên giai đoạn train của từng fold | Fit trên toàn chuỗi | Tránh rò rỉ dữ liệu |
| D13 | L chính là 5, 30, 120; L=480 chạy bổ sung | Giữ 4 giá trị như tài liệu | L=480 mất gần hết 2012, 2015, 2016 và 2020 đến 2021; fold không so sánh được |
| D14 | Mô hình lai: trừ giá cuối từng kênh rồi tách xu hướng/phần dư bằng trung bình trượt; thứ tự không ảnh hưởng | Phân rã trước | MA(x − c) = MA(x) − c với replicate padding (có test) |
| D15 | Trung bình trượt m = 5 phải lẻ và không bị cắt xuống `seq_len − 1` | Làm như tài liệu (nội suy khi m chẵn) | m = 5 hợp lệ cả với L = 5; tránh phép nội suy xấp xỉ |
| D16 | Nhánh "mức hiện tại" cho biến phụ trợ, cắt ±10 độ lệch chuẩn | Không có nhánh này | Bước trừ giá cuối làm mất mức của biến phụ trợ; chạy cả hai bản để làm ablation |
| D17 | Mọi mô hình dùng cùng một tập mẫu (cửa sổ hợp lệ với mọi kênh) | Mỗi mô hình tự chọn mẫu | So sánh ghép cặp công bằng |
| D18 | Huấn luyện: Adam 1e-3, StepLR(20, 0,5), 50 epoch, batch 32, giữ epoch có val loss thấp nhất, 5 seed | Giữ epoch cuối | Theo tài liệu, thêm chọn epoch trên validation (không đụng test) |
| D19 | Chia batch thủ công thay DataLoader | DataLoader | Chạy nhanh hơn, ngữ nghĩa giữ nguyên |
| D20 | Diebold–Mariano loss bình phương, hiệu chỉnh HLN, phương sai HAC độ trễ h−1, dùng dự báo trung bình qua seed | Chỉ so RMSE | Cần chứng minh khác biệt có ý nghĩa; sai số chồng lấn giữa các origin liên tiếp |
| D21 | Fold 2020 bị loại khỏi bảng tổng (vẫn có trong bảng theo năm) | Gộp vào | Chỉ một phần năm, lẫn chế độ khủng hoảng |
| D22 | Demo chỉ đọc dự báo đã lưu | Huấn luyện lại khi demo; dự báo tương lai | Nhanh, không cần torch; chưa lưu trọng số mô hình |
| D23 | Vòng 1: khởi tạo trọng số 0, weight decay chọn trên validation từ lưới {0; 0,01; 0,1; 1,0}, và bản co về naive (λ trên validation); chỉ 2 so sánh chính đã đăng ký | Tinh chỉnh dựa trên test; nhiều biến thể hơn | Nhắm vào nhiễu ước lượng; tránh chọn lọc kết quả sau khi thấy dữ liệu |
| D25 | Dự đoán thử: cắt dữ liệu tại ngày chọn, huấn luyện chỉ trên mẫu nằm hoàn toàn trong dữ liệu đã cắt (validation 250 mẫu cuối, purge, scaler chỉ fit trên train), rồi dự báo từ L phiên cuối | Dùng mô hình của fold năm test; đọc kết quả có sẵn | Yêu cầu "mô hình phải dự đoán, không lấy dữ liệu có sẵn"; không rò rỉ theo thiết kế (có test) |
| D26 | L do người dùng chọn tự do (2 đến 240): huấn luyện theo yêu cầu và lưu cache theo (mô hình, L, ngày, seed, dấu vân dữ liệu) | Huấn luyện sẵn một lưới L cố định | Mô hình tuyến tính có kích thước đầu vào cố định theo L; huấn luyện chỉ vài giây |
| D27 | Naive và drift không huấn luyện, không cần torch | Huấn luyện cả hai | Không có tham số để học |
| D29 | Mô hình đa biến nhóm A dùng mọi kênh phụ (nhóm `all`), trộn kênh (L×C đầu vào) | Từng nhóm biến riêng | Giữ số cấu hình thấp (kiểm soát đa so sánh); so với bản đơn biến cùng họ |
| D30 | `train_all.py` huấn luyện mọi mô hình cho nhiều L và hai mốc cắt (mặc định 2025-12-31 và toàn bộ dữ liệu), một seed, lưu cùng scaler và manifest | Chỉ một mô hình cuối | Mốc cắt sớm cho phép dự báo ngoài mẫu; app cảnh báo trong mẫu |
| D28 | Gói giao không kèm dữ liệu/kết quả sinh ra và bỏ file rỗng giữ chỗ | Kèm kết quả cũ | Người dùng muốn chạy lại từ đầu; file rỗng (ví dụ `wti_dataset.csv`) có thể làm lỗi chương trình |
| D24 | Giữ bảng vòng 0 riêng bằng hậu tố `_round0` (tuỳ chọn `--exclude-improvements`) | Ghi đè | Báo cáo cần tách kết quả vòng 0 và vòng 1 |

## 4. Dữ liệu

| Tên | Mã FRED | Vai trò | Trễ |
|---|---|---|---|
| wti | DCOILWTICO | Mục tiêu | 0 |
| brent | DCOILBRENTEU | Spread Brent–WTI | 0 |
| usdcad | DEXCAUS | Tỷ giá | 0 |
| usd_broad | DTWEXBGS | Chỉ số USD broad | 0 |
| ovx | OVXCLS | Biến động ngụ ý dầu | 1 |
| fed_funds | DFF | Lãi suất FED (thử nghiệm) | 1 |

Cả 6 mã tải thành công qua FRED API **[Đã chạy]**. Nguồn gốc từng chuỗi (EIA, Federal Reserve Board, CBOE) **[Chưa xác minh]**,
cần ghi từ trang của chuỗi trên FRED khi viết báo cáo.

Số liệu chất lượng sau cắt `end_date = 2026-09-30` **[Đã chạy]**:

| Chỉ số | Giá trị |
|---|---|
| Số phiên giao dịch | 4.649 (2008-01-02 đến 2026-09-30) |
| Ngày dùng được cho nghiên cứu | 4.061 |
| Phiên giá không dương | 1 (20/4/2020) |
| Phiên bị loại (cửa sổ 2020) | 85 |
| Phiên đệm (2008 đến 2009) | 503 |
| Khoảng cách hơn 5 ngày lịch | 8 (7 trong 2011, 1 cuối tháng 1/2015) |
| Phiên đứng giá lặp ≥ 4 ngày | 0 |
| Cờ biến động ≥ 15%/ngày | 19 |
| Phiên kỳ vọng bị thiếu theo năm | 2011: 40; 2015: 8; 2014: 4; 2018: 3; 2009, 2013: 2; các năm còn lại 0 đến 1 |

Kiểm tra 19 cờ biến động: các ngày trong giai đoạn đệm (2008-09-22) và giai đoạn loại 2020 là sự kiện thật; 2019-09-16
(tấn công Abqaiq) là sự kiện thật; 2026-04-08 (giảm 18,41 USD) trùng thông báo ngừng bắn Mỹ–Iran (đã đối chiếu tin tức);
**2026-09-28 (tăng 14,14 USD) chưa tìm được nguồn xác nhận trực tiếp [Chưa xác minh]**. Quyết định: giữ nguyên, không loại.

Nguyên nhân thiếu phiên 2011 và 2015 **[Chưa xác minh]**: nhiều khả năng là thiếu sẵn có trong chuỗi nguồn; chưa đối chiếu với
`wti.csv` thô (dấu `.`) và Brent cùng ngày.

Kiểm định ADF (giai đoạn nghiên cứu, bỏ giai đoạn loại) **[Đã chạy]**: giá WTI không dừng (p = 0,129); log-return dừng (p ≈ 0).

Tương quan chéo `lead_lag_check` giữa lợi suất WTI tại t và biến x tại t+k **[Đã chạy]**:

| Biến | x(t−2) | x(t−1) | x(t) | x(t+1) | x(t+2) |
|---|---|---|---|---|---|
| Brent log-return | −0,007 | 0,013 | **0,755** | 0,140 | −0,001 |
| Spread diff | 0,029 | 0,027 | −0,276 | 0,211 | 0,023 |
| USD/CAD log-return | −0,024 | −0,004 | −0,326 | −0,054 | 0,009 |
| USD broad log-return | −0,023 | −0,001 | −0,209 | −0,053 | −0,000 |
| OVX diff | 0,004 | 0,014 | −0,198 | −0,029 | 0,008 |
| FED diff | −0,003 | −0,013 | −0,006 | 0,010 | −0,025 |

Diễn giải: tương quan lớn chỉ ở cùng ngày (dấu hợp lý); tương quan của biến ngày trước với lợi suất hôm nay tối đa khoảng 0,03,
nằm trong ngưỡng nhiễu (±1,96/√4061 ≈ ±0,031). Giá trị 0,14 và 0,21 ở t+1 của Brent và spread giải thích bằng việc Brent chốt
giá trước WTI và spread có quan hệ cơ học với WTI. Không có dấu hiệu lệch ngày.

**Số mẫu cửa sổ dùng được theo năm của origin (sau khi loại khoảng trống) [Đã chạy]:**

| Năm | L=5 | L=30 | L=120 | L=480 |
|---|---|---|---|---|
| 2010 | 252 | 252 | 252 | 252 |
| 2011 | 179 | 154 | 65 | 65 |
| 2012 | 252 | 252 | 251 | 0 |
| 2013 | 250 | 250 | 250 | 141 |
| 2014 | 248 | 248 | 248 | 248 |
| 2015 | 235 | 210 | 120 | 8 |
| 2016 | 252 | 252 | 252 | 4 |
| 2017 | 250 | 250 | 250 | 250 |
| 2018 | 249 | 249 | 249 | 249 |
| 2019 | 250 | 250 | 250 | 250 |
| 2020 | 158 | 133 | 43 | 35 |
| 2021 | 251 | 251 | 251 | 0 |
| 2022 | 251 | 251 | 251 | 150 |
| 2023 | 248 | 248 | 248 | 248 |
| 2024 | 250 | 250 | 250 | 250 |
| 2025 | 248 | 248 | 248 | 248 |
| 2026 | 186 | 186 | 186 | 186 |

(Bảng này lấy từ lần chạy trước khi cắt `end_date`: các số của 2026 và tổng mẫu có thể lệch vài đơn vị. Các giá trị
chốt dùng trong báo cáo lấy từ `results/tables/folds_L*.csv`.)
Tổng số mẫu sau cắt `end_date`: L=5: 3.995; L=30: 3.895; L=120: 3.535. Số fold chạy: L=5: 10; L=30: 10; L=120: 9 (bỏ 2020).

## 5. Danh sách file đã tạo hoặc sửa

| Nhóm | File |
|---|---|
| Cấu hình | `configs/config.yaml`, `walk_forward.yaml`, `experiments.yaml`, `features.yaml` |
| Dữ liệu | `src/config.py`, `data_loader.py`, `preprocessing.py`, `features.py`, `run_data_pipeline.py` |
| Mẫu và chia fold | `src/dataset.py`, `src/experiments.py` |
| Mô hình | `src/models.py`, `src/baselines.py` |
| Huấn luyện, đánh giá | `src/train.py`, `src/evaluate.py`, `src/report.py`, `src/improvements.py`, `run_experiments.py`, `run_evaluation.py`, `diagnose_improve1.py` |
| Demo, dự đoán thử | `app/app.py`, `app/predictor.py`, `app/__init__.py`, `app/README.md`, `src/live_predict.py`, `predict_day.py`, `configs/live.yaml` |
| Thiết lập | `check_setup.py`, `docs/SETUP_CONFIG.md`, `.env.example`, `requirements.txt` |
| Tài liệu | `README.md`, `docs/PROJECT_LOG.md` |
| Kiểm thử | `tests/`: `test_leakage`, `test_dataset`, `test_baselines`, `test_models`, `test_models_level`, `test_train`, `test_evaluate`, `test_experiment_setup`, `test_report`, `test_predictor`, `test_improvements`, `test_train_improve`, `test_live_predict` |

Các file rỗng giữ chỗ không dùng: `src/walk_forward.py`, `statistical_tests.py`, `plots.py`, `utils.py`, `tests/test_metrics.py`,
thư mục `notebooks/`, `docs/` (trừ file này), `report/`, `saved_models/`. `src/features.py` đã viết và kiểm thử nhưng **chưa dùng**
trong thí nghiệm.

## 6. Kết quả thí nghiệm gốc (vòng 0) [Đã chạy]

Tỉ lệ RMSE so với naive, trung bình qua 5 bước, bỏ fold 2020 (< 1 là tốt hơn naive):

| Mô hình | L=5 | L=30 | L=120 | L=480* |
|---|---|---|---|---|
| naive | 1,0000 | 1,0000 | 1,0000 | 1,0000 |
| drift | 1,0005 | 1,0004 | 1,0005 | 1,0000 |
| hybrid_B | 1,0017 | 1,0074 | 1,0455 | 1,1103 |
| nlinear | 1,0018 | 1,0101 | 1,0452 | 1,1200 |
| hybrid_C_ovx | 1,0030 | 1,0298 | 1,1329 | 1,6631 |
| hybrid_C_spread | 1,0031 | 1,0195 | 1,0870 | 1,3051 |
| hybrid_C_usd | 1,0040 | 1,0138 | 1,0745 | 1,2588 |
| hybrid_C_ovx_lvl | 1,0053 | 1,0258 | 1,1336 | 5,5269 |
| hybrid_C_fed | 1,0055 | 1,0607 | 1,2686 | 1,4663 |
| hybrid_C_spread_lvl | 1,0066 | 1,0224 | 1,1150 | 3,5530 |
| hybrid_C_all | 1,0084 | 1,1008 | 1,4103 | 2,9936 |
| dlinear | 1,0464 | 1,0643 | 1,0972 | 1,5509 |
| hybrid_C_usd_lvl | 1,0998 | 1,1267 | 1,2968 | 4,8746 |
| hybrid_C_all_lvl | 1,1524 | 1,2217 | 2,4537 | 10,4351 |
| linear | 1,2038 | 1,1959 | 1,2101 | 1,6164 |
| hybrid_C_fed_lvl | 1,3673 | 1,4882 | 2,8278 | 12,3013 |

\* L=480 chỉ có 6 năm test (2019, 2022 đến 2026), không so sánh trực tiếp với ba cột còn lại.
Năm test dùng cho L=5, 30: 2017, 2018, 2019, 2021 đến 2026; L=120 cùng tập đó.

Độ chính xác hướng đi: mọi mô hình quanh 0,47 đến 0,52 (lớn nhất tìm thấy: DLinear, τ = 3, 0,519).

**Diebold–Mariano, L=30 [Đã chạy]** (140 phép so sánh, mức 5%):
- drift, hybrid_B, nlinear so với naive: không khác biệt ở cả 5 bước. DLinear: naive tốt hơn ở 3/5 bước. Linear: naive tốt hơn ở cả 5 bước.
- hybrid_B so với Linear, DLinear: lai tốt hơn (p < 0,001 ở hầu hết các bước). hybrid_B so với NLinear: không khác biệt (4/5 bước; τ=1 NLinear tốt hơn, p ≈ 0,026).
- hybrid_C_usd so với hybrid_B: không khác biệt; spread, ovx kém hơn B ở hầu hết các bước; fed, all và các biến thể `_lvl` kém hơn rõ.
- Lưu ý đa so sánh: với 140 phép kiểm định ở mức 5%, khoảng 7 kết quả "có ý nghĩa" có thể chỉ do ngẫu nhiên.

**Theo năm, L=30, tỉ lệ so với naive [Đã chạy]** (NLinear): 2017: 1,041; 2018: 1,005; 2019: 1,024; 2021: 1,016; 2022: 1,013;
2023: 1,010; 2024: **0,993**; 2025: **0,984**; 2026: 1,008. hybrid_B: 2024: 0,988; 2025: 0,981. Theo chế độ (2023 đến 2025):
hybrid_B 0,994, NLinear 0,997. Theo bước: hybrid_B giữ 1,006 đến 1,009 ở cả 5 bước; Linear và DLinear tệ nhất ở τ=1 (1,47; 1,19)
và tiến về naive ở τ=5.

**Theo năm cho nhóm nhiễu fed [Đã chạy]:** `hybrid_C_fed_lvl` có tỉ lệ 2,77 (2017), 2,13 (2018), 1,33 (2019), về quanh 1,0 từ 2020.

## 7. Các phát hiện và diễn giải

1. **Không mô hình nào vượt naive** một cách vững (RQ chính). Bằng chứng: tỉ lệ ≥ 1 ở mọi L; DM không khác biệt với NLinear, lai, drift.
2. **RQ2:** lai tốt hơn Linear và DLinear, **không khác NLinear**: phần trừ giá cuối tạo ra cải thiện, phần phân rã không thêm giá trị.
   Giả thuyết ban đầu "lai tốt hơn vì dầu có cả dịch mức lẫn xu hướng" **không được ủng hộ**.
3. **RQ1:** mọi biến thể nhóm C không tốt hơn B (USD hoà, các nhóm khác kém hơn). Phù hợp với tương quan trễ gần 0.
4. **Nhánh mức gây hại**, nặng nhất với `fed` và `all` ở các fold đầu: mức OVX, spread, lãi suất đổi theo chế độ thị trường nên hệ số
   học ở train không chuyển sang test; FED gần như hằng số trong 2010 đến 2015 nên ngoại suy khi lãi suất tăng. Ngay cả từ 2021
   trở đi `all` vẫn kém hơn naive (1,02 đến 1,09), nên kết luận "thêm biến không giúp" không chỉ do lỗi ngoại suy.
5. **L càng dài càng tệ:** NLinear 1,0018 (L=5), 1,0101 (L=30), 1,0452 (L=120), 1,1200 (L=480).
6. **Quan sát thăm dò (chọn sau khi thấy dữ liệu, chưa kiểm định):** tỉ lệ của NLinear giảm gần như đều theo lượng dữ liệu train
   (2017 đến 2025) và dưới 1 nhẹ ở 2024 và 2025; năm 2026 (cú sốc lớn) xấu hơn. Gợi ý rằng phần lớn việc thua naive là nhiễu
   ước lượng. Đây là cơ sở cho cải tiến vòng 1.
7. Kết quả phù hợp với văn liệu kinh tế lượng về giá dầu thường báo cáo dự báo "không đổi" khó bị vượt ở kỳ hạn ngắn
   **[Chưa xác minh trích dẫn]**.

8. **Vòng 1 (điều chuẩn hoá):** thu hẹp khoảng cách tới naive ở L=30 (NLinear 1,0101 → 1,0027) nhưng **không có
   biến thể nào vượt naive** (tỉ lệ thấp nhất 1,0014 ở L=5) và các cải thiện so với bản gốc **không có ý nghĩa thống kê**.
   Chi tiết ở 9.6 và 9.7.

## 8. Sai sót, lỗi và điều chỉnh (ghi trung thực)

| # | Sự việc | Cách phát hiện | Xử lý |
|---|---|---|---|
| E1 | Nhận định ban đầu về rò rỉ do lệch giờ đóng cửa giữa các biến bị **nói quá**. Với thiết kế dự báo sau khi thị trường đóng ngày t (mục tiêu t+1 đến t+5), giá cùng ngày đã biết; trễ 1 phiên chỉ là thận trọng | Xem lại sau khi có `lead_lag_check` | Điều chỉnh nhận định; rủi ro còn lại chỉ ở ứng dụng thật (tỷ giá công bố trễ) |
| E2 | Khoảng trống dữ liệu 2011, 2015 chưa được dự tính trong bản đầu của pipeline | Bảng số mẫu: 2011 chỉ 212 mẫu giống nhau ở mọi L | Thêm `gap_before`, loại cửa sổ qua khoảng trống |
| E3 | `config.yaml` do trợ lý gửi lại đã **ghi đè** chỉnh sửa `end_date` của người dùng | Kết quả pipeline hiện `last_date` 2026-10-06 | Gửi lại file có `end_date: "2026-09-30"`; từ nay chỉ yêu cầu thêm dòng khi sửa cấu hình |
| E4 | **Lỗi seed:** `run_fold` khởi tạo mô hình trước khi đặt seed, nên cùng seed vẫn cho kết quả khác | `test_train.py` lỗi 2/8 | Thêm `set_seed(seed)` đầu `run_fold`; sau đó 8/8 đạt. Nếu không có test, 5 seed sẽ không tái lập được |
| E5 | Test `test_level_clip_bounds_extrapolation` sai: đổi cả diễn biến lẫn mức của kênh phụ | Test lỗi 1/52 | Sửa test (chỉ đổi mức); mô hình không đổi |
| E6 | Môi trường của trợ lý không cài được torch (hết dung lượng), nên các test cần torch chỉ chạy ở máy người dùng | Cài torch thất bại | Mọi mã torch chỉ được xác nhận qua test chạy ở máy người dùng |
| E7 | Lệnh `pip install` lâu, trông như treo | Do cài `torch` (nặng) | Hướng dẫn cài từng phần |
| E8 | `app/predictor.py` trên máy người dùng sai nội dung (thiếu `Forecasts` và `label`); bản của trợ lý đúng (110 dòng, 4.375 byte) | `pytest` báo `ImportError` ở 2 file test | Chép lại file; khuyến nghị dùng `Copy-Item` thay vì dán qua trình soạn thảo |
| E9 | `run_experiments.py` trên máy người dùng sai/rỗng: lệnh `--dry-run` chạy mà không in gì | Không có đầu ra, không có lỗi | Chép lại file; đối chiếu bằng `Select-String` và dung lượng |
| E10 | Chạy `run_evaluation.py` khi chưa sao lưu `results/tables`, nên bảng L=5 và L=30 bị ghi đè bằng bản gộp vòng 0 và vòng 1 | Người dùng báo | Thêm `--exclude-improvements` và `--tag`; tái tạo bảng vòng 0, so sánh thấy số liệu **trùng khớp** bản cũ (ví dụ L=30: hybrid_B 1,0074; nlinear 1,0101; linear 1,1959) |

## 9. Cải tiến vòng 1 (đã đăng ký giả thuyết trước khi chạy)

### 9.1 Chẩn đoán dẫn tới cải tiến

Kết quả vòng 0 cho thấy NLinear và lai xấp xỉ naive cộng một chút nhiễu ước lượng (tỉ lệ 1,002 đến 1,01), và tốt dần khi có thêm
dữ liệu train. Hướng cải tiến vì vậy nhắm vào **giảm phương sai ước lượng**, không thêm độ phức tạp.

### 9.2 Các thay đổi

| ID | Thay đổi | Cách làm | Mục đích |
|---|---|---|---|
| R1-A | `nlinear_zr` | Khởi tạo trọng số và bias bằng 0 (mô hình bắt đầu đúng là naive) + weight decay chọn trên validation từ lưới {0, 0,01, 0,1, 1,0} | Giảm nhiễu ước lượng của NLinear |
| R1-B | `hybrid_B_zr` | Như R1-A cho mô hình lai đơn biến | Như trên |
| R1-C | `hybrid_C_usd_zr` | Như R1-A cho mô hình lai + nhóm USD (nhóm C ít hại nhất) | Kiểm tra liệu điều chuẩn có làm biến phụ trợ hữu ích |
| R1-A', B', C' | Hậu tố `_shrink` | Dự báo = giá cuối + λ·(dự báo mô hình − giá cuối), λ trong [0, 1] ước lượng bằng bình phương tối thiểu trên validation, chung cho các bước | Kết hợp dự báo với naive |

Chạy cho L = 5 và 30, cả 10 fold, 5 seed. Mọi lựa chọn siêu tham số (weight decay, λ, epoch) chỉ dựa trên validation;
test chỉ dùng để đánh giá.

### 9.3 Giả thuyết (viết trước khi có kết quả)

- **H1 (chính):** `nlinear_zr` và `hybrid_B_zr` có tỉ lệ RMSE so với naive gần 1 hơn so với `nlinear` và `hybrid_B` tương ứng
  (giảm ít nhất một nửa mức vượt 1,000 ở L=30).
- **H2:** các bản `_shrink` có tỉ lệ không lớn hơn bản gốc và nằm trong khoảng 1,000 đến 1,003.
- **H3:** `hybrid_C_usd_zr` không khác `hybrid_B_zr` (biến USD không thêm thông tin dự báo).
- **Kỳ vọng thực tế:** **không** có biến thể nào có tỉ lệ nhỏ hơn 1 một cách có ý nghĩa thống kê. Điều chuẩn hoá chỉ có thể thu hẹp
  khoảng cách tới naive; muốn vượt naive cần thông tin mới (ví dụ đường cong kỳ hạn, dữ liệu tồn kho).

### 9.4 Quy tắc để kết quả đáng tin

- So sánh **chính** (đã đăng ký): `nlinear_zr` so với `nlinear` và `hybrid_B_zr` so với `hybrid_B`, ở L=30, gộp các fold.
  Các so sánh còn lại là thăm dò.
- Báo cáo **tất cả** biến thể đã thử, kể cả không hiệu quả (xem bộ đếm ở 9.5).
- Không chỉnh thêm biến thể dựa trên kết quả test; vòng sau phải đăng ký giả thuyết mới và cộng vào bộ đếm.
- λ có thể hơi lạc quan vì epoch tốt nhất cũng chọn trên validation (ghi vào phần hạn chế).

### 9.5 Bộ đếm số cấu hình đã thử (để kiểm soát đa so sánh)

| Vòng | Số cấu hình mô hình | Số L | Trạng thái |
|---|---|---|---|
| 0 | 14 mô hình huấn luyện + 2 baseline | 3 chính + 1 bổ sung | Đã chạy |
| 1 | 3 mô hình huấn luyện, mỗi cái 2 đầu ra (`_zr`, `_zr_shrink`) = 6 cấu hình | 2 (L = 5, 30) | **Đã chạy** (1.020 file mỗi L, gồm 300 file của vòng 1) |

### 9.6 Kết quả vòng 1 [Đã chạy]

Tỉ lệ RMSE so với naive (bỏ fold 2020; năm test 2017 đến 2019 và 2021 đến 2026):

| Mô hình | L=5 | L=30 | Bản gốc (L=5 / L=30) | DM so với bản gốc (L=30) |
|---|---|---|---|---|
| nlinear_zr | 1,0014 | 1,0027 | nlinear 1,0018 / 1,0101 | không khác biệt ở cả 5 bước (p = 0,46 đến 0,64) |
| nlinear_zr_shrink | 1,0014 | 1,0027 | | không khác biệt |
| hybrid_B_zr | 1,0026 | 1,0037 | hybrid_B 1,0017 / 1,0074 | không khác biệt ở cả 5 bước (p = 0,26 đến 0,33) |
| hybrid_B_zr_shrink | 1,0025 | 1,0037 | | không khác biệt |
| hybrid_C_usd_zr | 1,0030 | 1,0049 | hybrid_C_usd 1,0040 / 1,0138 | tốt hơn ở τ = 4 (p = 0,025) và τ = 5 (p = 0,020); không khác biệt ở τ = 1 đến 3 |
| hybrid_C_usd_zr_shrink | 1,0029 | 1,0046 | | như trên |

So với naive: mọi biến thể không khác biệt ở cả 5 bước (p nhỏ nhất 0,32; mean_diff dương, tức hơi kém naive).
Mốc tham chiếu cùng bảng: naive 1,0000; drift 1,0005 (L=5), 1,0004 (L=30).

**Chẩn đoán (50 lần mỗi mô hình và L: 10 fold × 5 seed):**

| L | Mô hình | weight decay được chọn nhiều nhất | λ trung bình | λ = 1 | λ = 0 |
|---|---|---|---|---|---|
| 5 | nlinear_zr | 0 (30/50) | 0,940 | 94% | 6% |
| 5 | hybrid_B_zr | 0 (31/50) | 0,932 | 90% | 6% |
| 5 | hybrid_C_usd_zr | 0 (23/50) | 0,929 | 86% | 6% |
| 30 | nlinear_zr | 1,0 (18/50) | 0,967 | 92% | 0% |
| 30 | hybrid_B_zr | 1,0 (17/50) | 0,990 | 94% | 0% |
| 30 | hybrid_C_usd_zr | 1,0 (18/50) | 0,968 | 90% | 2% |

λ theo năm test (trung bình qua seed): gần như bằng 1 ở mọi năm, trừ fold 2017 (ít dữ liệu train nhất): L=5: 0,40 cho cả ba mô hình;
L=30: 0,67 (nlinear), 0,91 (hybrid_B), 0,78 (C_usd).

### 9.7 Đối chiếu với giả thuyết đã đăng ký trước

| Giả thuyết | Kết luận | Bằng chứng và lưu ý |
|---|---|---|
| **H1** (chính): `_zr` gần 1 hơn bản gốc, giảm ít nhất một nửa mức vượt 1,000 ở L=30 | **Ủng hộ một phần** | Về số điểm: NLinear giảm 73% mức vượt (0,0101 → 0,0027); lai B giảm 50% (0,0074 → 0,0037, đúng ngưỡng, số đã làm tròn); C_usd giảm 64%. Nhưng **không có ý nghĩa thống kê** (DM không khác biệt, p ≥ 0,26) và ở L=5 hybrid_B_zr còn **tệ hơn** bản gốc (1,0026 so với 1,0017) |
| **H2**: `_shrink` không lớn hơn bản gốc, nằm trong 1,000 đến 1,003 | **Ủng hộ một phần, nhưng vô nghĩa** | Bản `_shrink` giống bản `_zr` đến chữ số thứ tư vì λ ≈ 1 (86% đến 94% các lần chọn λ = 1): validation báo "giữ mô hình". Chỉ 2 trong 6 bản ở L=30 (hai bản NLinear, 1,0027) nằm trong khoảng 1,000 đến 1,003; các bản lai B (1,0037) và C_usd (1,0046 đến 1,0049) vượt khoảng |
| **H3**: `hybrid_C_usd_zr` không khác `hybrid_B_zr` | **Ủng hộ theo ước lượng điểm** | 1,0049 so với 1,0037 (L=30), 1,0030 so với 1,0026 (L=5): USD không thêm giá trị. **Chưa tính DM trực tiếp cặp này** (cặp chuẩn hiện so với `hybrid_B`) |
| **Kỳ vọng**: không biến thể nào vượt naive có ý nghĩa | **Đúng** | Tỉ lệ thấp nhất 1,0014; không phép DM nào so với naive có p < 0,3 |

### 9.8 Diễn giải và lưu ý

- Điều chuẩn hoá **thu hẹp nhưng không xoá** khoảng cách tới naive: sau khi khởi tạo 0 và weight decay mạnh (được chọn nhiều nhất là 1,0 ở L=30), mô hình vẫn hơi kém naive (+0,14% đến +0,49%). Không có dấu hiệu tín hiệu tuyến tính khai thác được trong đầu vào.
- Cửa sổ dài cần điều chuẩn mạnh hơn: ở L=5 weight decay 0 được chọn nhiều nhất, ở L=30 là 1,0 (nhiều tham số hơn, cần co mạnh hơn).
- Hệ số co λ chỉ khác 1 đáng kể ở fold 2017 (ít dữ liệu train nhất), phù hợp với giả thuyết nhiễu ước lượng nhưng chỉ dựa trên một fold.
- λ có thể hơi **lạc quan** vì epoch tốt nhất cũng được chọn trên validation; đây cũng có thể là lý do λ gần 1 nhưng test vẫn không vượt naive.
- Hai kết quả "mô hình tốt hơn" (C_usd_zr so với C_usd ở τ = 4, 5; và bản `_shrink` tương ứng) **không nên được nhấn mạnh**: bảng DM có 225 phép kiểm định, khoảng 11 phép có thể dương tính giả ở mức 5%; đây là so sánh thăm dò, không phải so sánh chính đã đăng ký.
- Kết luận cho báo cáo: **tinh chỉnh điều chuẩn hoá không đủ để vượt naive**; muốn cải thiện thật sự cần thông tin mới.

### 9.9 Ứng viên cho vòng 2 (chưa quyết định, chưa làm)

| Ứng viên | Cần gì | Nhận xét |
|---|---|---|
| Khoảng dự báo (phân vị) cho giá, tỉ lệ theo độ biến động gần đây | Chỉ dữ liệu hiện có | Dự báo độ bất định thường có giá trị thực tế hơn dự báo điểm; đo bằng pinball loss và độ phủ **[chưa thử]** |
| Đường cong kỳ hạn (chênh giá hợp đồng tháng 1 và tháng 2 trở đi) | Dữ liệu futures, nguồn cần kiểm tra | Tín hiệu có cơ sở kinh tế mạnh với dầu **[Chưa xác minh khả năng có dữ liệu miễn phí]** |
| Dữ liệu tồn kho hàng tuần của EIA | Chuỗi tồn kho và quy tắc thời điểm công bố | Cần cẩn thận rò rỉ thời gian **[Chưa xác minh mã chuỗi]** |
| Biến lịch (`features.py` đã viết) | Danh sách ngày FOMC | Kỳ vọng thấp với mức giá, có thể hữu ích hơn với độ biến động |
| So sánh trực tiếp `hybrid_C_x_zr` với `hybrid_B_zr` | Thêm cặp vào `default_pairs` | Sửa nhỏ, nên làm để hoàn tất H3 |

## 10. Việc còn mở và thông tin chưa xác minh

- Năm 2018 vắng trong bảng L=480 (lẽ ra chạy được): cần xem `results/tables/folds_L480.csv`.
- 2026-09-28: chưa xác nhận sự kiện giá tăng 14,14 USD.
- Nguyên nhân thiếu phiên 2011 và 2015: chưa đối chiếu dữ liệu thô.
- File ngày FOMC (`data/raw/fomc_dates.csv`) chưa tạo; quy tắc ngày công bố EIA và ngày đáo hạn CL trong `features.py` là xấp xỉ.
- Số test: lần chạy gần nhất người dùng báo **71 test đạt** (gồm cả `test_train_improve`, `test_improvements`, `test_report`, `test_predictor`), 2 cảnh báo vô hại.
- Trích dẫn tài liệu tham khảo (Diebold–Mariano 1995; Harvey–Leybourne–Newbold 1997; văn liệu dự báo giá dầu) cần kiểm tra lại.
- Chức năng dự đoán thử chưa được người dùng chạy trên máy (mới kiểm thử ở môi trường của trợ lý, dữ liệu giả lập).
- Cặp DM trực tiếp `hybrid_C_x_zr` với `hybrid_B_zr` chưa có trong bảng (cần thêm vào `default_pairs`).
- Hình 14 và 15 trong tài liệu gốc ghi "7-day forecast" trong khi văn bản dùng T = 5: nên đối chiếu khi trích dẫn.

## 11. Cách cập nhật nhật ký và ánh xạ sang chương báo cáo

**Cập nhật:** sau mỗi thay đổi, thêm (a) một dòng ở mục 2, (b) quyết định mới ở mục 3, (c) lỗi mới ở mục 8, (d) kết quả mới ở mục 6 hoặc 9.6,
(e) cập nhật bộ đếm 9.5.

| Chương báo cáo | Lấy từ |
|---|---|
| 1. Giới thiệu | Mục 1, 3 (D1 đến D4) |
| 2. Cơ sở lý thuyết | Tài liệu AIO; D14, D15; ví dụ tính tay trong `test_models.py`; DM (D20) |
| 3. Dữ liệu và tiền xử lý | Mục 4; D3 đến D9; bảng chất lượng; E2 |
| 4. Xây dựng mô hình và kết quả | Mục 3 (D10 đến D21), 6, 7, 9; demo (D22) |
| 5. Kết luận và hướng phát triển | Mục 7, 8, 10; mục Hướng phát triển trong `README.md` |
