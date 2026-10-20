# Hướng dẫn thiết lập cấu hình (những gì bạn phải tự điền)

Phần lớn cấu hình đã có giá trị mặc định chạy được. Tài liệu này liệt kê **những thứ tôi không thể điền thay bạn**, cách
lấy từng thứ, đặt vào đâu, và cách kiểm tra. Sau khi làm xong, chạy:

```powershell
python check_setup.py            # kiểm tra cục bộ
python check_setup.py --online   # thêm: thử khoá FRED bằng một yêu cầu thật
```

Các dòng **CẦN SỬA** là mục bắt buộc chưa đạt; dòng **TUỲ CHỌN** có thể bỏ qua.

## Tóm tắt: ai điền cái gì

| # | Mục | Bắt buộc? | Ở đâu | Ai điền |
|---|---|---|---|---|
| 1 | Khoá FRED API | **Bắt buộc** (để tải dữ liệu) | `.env` | Bạn (đăng ký tài khoản FRED) |
| 2 | Python 3.10 trở lên và môi trường ảo | **Bắt buộc** | Máy của bạn | Bạn |
| 3 | Thông tin tác giả, lớp, giảng viên | Tuỳ chọn | Dòng đầu `README.md` | Bạn |
| 4 | Mốc cuối dữ liệu `end_date` | Tuỳ chọn (có mặc định) | `configs/config.yaml` | Bạn quyết định |
| 5 | Thiết bị tính toán (`cpu`, `cuda`) | Tuỳ chọn | `configs/live.yaml`, tham số `--device` | Bạn |
| 6 | Proxy mạng | Chỉ khi mạng bắt buộc | `.env` | Bạn / quản trị mạng |
| 7 | Danh sách ngày họp FOMC | Tuỳ chọn (chưa dùng) | `data/raw/fomc_dates.csv` | Bạn (từ trang Federal Reserve) |
| 8 | Cổng chạy ứng dụng | Tuỳ chọn | Tham số `--server.port` | Bạn |

Mọi mục khác (mã chuỗi FRED, quy tắc làm trễ, cửa sổ loại 2020, tham số huấn luyện...) đã được điền sẵn và **không cần
sửa** để chạy lại kết quả của đồ án.

---

## 1. Khoá FRED API (bắt buộc)

**Vì sao cần:** dữ liệu giá dầu, tỷ giá, OVX... đều tải từ FRED qua API. Khoá miễn phí.

**Cách lấy** (giao diện trang web có thể thay đổi chút ít):
1. Vào https://fred.stlouisfed.org/ và tạo tài khoản (email hoặc Google).
2. Đăng nhập, mở **My Account**, chọn **API Keys**.
3. Bấm **Request API Key**, ghi mô tả ngắn (ví dụ "đồ án môn học dự báo giá dầu"), đồng ý điều khoản.
4. Trang hiện một chuỗi **32 ký tự gồm chữ thường và số**. Đó là khoá của bạn.

**Cách đặt vào dự án:**
1. Trong thư mục gốc (cùng chỗ với `README.md`), sao chép file mẫu:
   - Windows PowerShell: `copy .env.example .env`
   - macOS / Linux: `cp .env.example .env`
2. Mở file `.env` bằng trình soạn thảo văn bản và sửa dòng:
   ```
   FRED_API_KEY=your_key_here
   ```
   thành (dán khoá của bạn, **không có dấu cách, không có dấu ngoặc**):
   ```
   FRED_API_KEY=abcdef0123456789abcdef0123456789
   ```
3. Lưu file. Tên file phải đúng là `.env` (bắt đầu bằng dấu chấm, không có đuôi `.txt`). Trên Windows, nếu trình soạn
   thảo tự thêm `.txt`, hãy đổi tên lại.

**Kiểm tra:** `python check_setup.py --online`. Kết quả mong đợi: `FRED_API_KEY: có trong file .env, đúng định dạng`
và `Kết nối FRED: khoá hợp lệ, tải thử thành công`.

**Bảo mật:** không gửi file `.env` cho người khác và không đưa lên GitHub (`.gitignore` đã loại sẵn file này). Nếu lộ
khoá, vào lại trang FRED và tạo khoá mới.

**Không muốn đăng ký?** Vẫn làm được bằng cách tải CSV thủ công từ trang từng chuỗi trên FRED (nút *Download*), đổi
tên thành `wti.csv`, `brent.csv`, `usdcad.csv`, `usd_broad.csv`, `ovx.csv`, `fed_funds.csv`, đặt vào `data/raw/`, rồi chạy
`python run_data_pipeline.py` (không kèm `--download`). Mã chuỗi tương ứng nằm trong `configs/config.yaml`.

| Lỗi thường gặp | Nguyên nhân | Cách sửa |
|---|---|---|
| `Chưa có FRED_API_KEY` | Chưa tạo `.env` hoặc đặt sai thư mục | Đặt `.env` cùng thư mục với `run_data_pipeline.py` |
| `CẦN SỬA ... sai định dạng` | Khoá thừa dấu cách/ngoặc, hoặc chép thiếu ký tự | Chép lại đủ 32 ký tự |
| `FRED từ chối khoá (HTTP 400)` | Khoá sai hoặc đã bị thu hồi | Tạo khoá mới |
| `không kết nối được` | Mất mạng, tường lửa, hoặc cần proxy | Xem mục 6 |

## 2. Python và môi trường ảo (bắt buộc)

1. Cài **Python 3.10 trở lên** từ https://www.python.org/ (đã thử trên 3.11 và 3.12). Khi cài trên Windows, tích
   *Add Python to PATH*.
2. Mở PowerShell trong thư mục dự án và tạo môi trường ảo:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
   Nếu PowerShell chặn lệnh `activate`, chạy một lần: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
3. `torch` rất nặng. Nếu máy chỉ có CPU, nên cài bản nhẹ trước:
   `pip install torch --index-url https://download.pytorch.org/whl/cpu`, rồi mới `pip install -r requirements.txt`.
4. Bước `pip install` có thể mất 5 đến 15 phút và trông như bị treo ở giai đoạn cuối; đó là bình thường.

## 3. Thông tin tác giả (tuỳ chọn)

Dòng đầu của `README.md` có các chỗ `[điền ...]` (họ tên, học phần/lớp, giảng viên hướng dẫn). Hãy điền trước khi nộp.
Việc này không ảnh hưởng đến cách chạy.

## 4. `end_date` trong `configs/config.yaml` (tuỳ chọn)

Mặc định `end_date: "2026-09-30"`: dữ liệu được cắt ở mốc chung vì tỷ giá và chỉ số USD của FRED công bố trễ vài ngày
so với giá dầu.

- **Muốn tái lập đúng kết quả của đồ án:** giữ nguyên.
- **Muốn dùng dữ liệu mới hơn** (ví dụ để dự đoán thử cho những ngày gần hiện tại): đặt `end_date: null`, rồi chạy lại
  `python run_data_pipeline.py --download`. Lưu ý: kết quả thí nghiệm sẽ khác đôi chút vì có thêm dữ liệu, và các ngày
  cuối có thể thiếu dữ liệu tỷ giá (FRED công bố trễ).
- **Quy tắc:** `end_date` phải là ngày `YYYY-MM-DD` sau `study_start` (2010-01-01), hoặc `null`.

Sau khi đổi, các kết quả đã có trong `results/` không còn khớp dữ liệu mới. Hãy chạy lại các bước từ
`run_experiments.py`, hoặc xoá `results/predictions` để tránh lẫn lộn.

## 5. Thiết bị tính toán (tuỳ chọn)

- Mặc định chạy trên **CPU**; toàn bộ thí nghiệm mất khoảng 1 đến 1,5 giờ.
- Có GPU NVIDIA và đã cài torch bản CUDA: dùng `python run_experiments.py --device auto`. Với mô hình nhỏ như ở đây,
  GPU thường **không nhanh hơn đáng kể**.
- Chức năng *Dự đoán thử* đọc `device` trong `configs/live.yaml` (`cpu`, `cuda` hoặc `auto`).

## 6. Proxy mạng (chỉ khi cần)

Nếu bạn ở mạng trường/công ty bắt buộc đi qua proxy và `check_setup.py --online` báo không kết nối được, mở `.env`,
bỏ dấu `#` ở hai dòng `HTTPS_PROXY` và `HTTP_PROXY` rồi điền địa chỉ do quản trị mạng cung cấp (dạng
`http://user:password@host:port`). Hỏi quản trị mạng nếu không biết địa chỉ này.

## 7. Danh sách ngày họp FOMC (tuỳ chọn, hiện chưa dùng)

Biến lịch FOMC (`src/features.py`) đã được viết nhưng **chưa được đưa vào thí nghiệm**, nên bạn có thể bỏ qua mục này.
Nếu muốn dùng sau này: tạo file `data/raw/fomc_dates.csv` có **một cột tên `date`** là ngày công bố quyết định lãi suất
(định dạng `YYYY-MM-DD`), mỗi ngày một dòng; lấy danh sách từ trang lịch họp FOMC của Federal Reserve
(federalreserve.gov). Tôi không điền sẵn vì danh sách phải đối chiếu từ nguồn chính thức.

## 8. Cổng chạy ứng dụng (tuỳ chọn)

Mặc định `http://localhost:8501`. Nếu cổng bận: `streamlit run app\app.py --server.port 8502`.

---

## Các file cấu hình và ý nghĩa

Bạn **không cần sửa** các file dưới đây để chạy lại đồ án; chỉ sửa khi muốn thử nghiệm khác.

| File | Nội dung chính | Sửa khi nào |
|---|---|---|
| `.env` | Khoá FRED, proxy | **Bắt buộc điền khoá** |
| `configs/config.yaml` | Mã chuỗi FRED, `study_start`, `end_date`, quy tắc làm trễ, cửa sổ loại 2020, ngưỡng khoảng trống | Đổi `end_date` (mục 4) |
| `configs/walk_forward.yaml` | T = 5, các L chính (5, 30, 120) và L bổ sung (480), năm test, số mẫu validation | Muốn thêm/bớt năm test hoặc L |
| `configs/experiments.yaml` | Số seed, epoch, learning rate, batch, độ dài trung bình trượt, các nhóm biến, lưới weight decay (`improve1`) | Muốn chạy nhanh hơn (giảm `seeds`) |
| `configs/features.yaml` | Đường dẫn file FOMC, ngưỡng "gần đáo hạn" | Chỉ khi dùng biến lịch |
| `configs/live.yaml` | Mặc định của tab huấn luyện theo yêu cầu (mô hình, L, giới hạn L, thiết bị) và mục `store` của `train_all.py` (các L, các mốc cắt, seed, thư mục `saved_models`) | Đổi mặc định hoặc giới hạn L |

**Muốn chạy nhanh để thử:** giữ nguyên cấu hình và dùng tham số dòng lệnh thay vì sửa file, ví dụ
`python run_experiments.py --windows 30 --test-years 2023 2024 --seeds 1`.
