# Dự báo giá dầu WTI 5 phiên bằng họ mô hình LTSF-Linear

So sánh mô hình đơn biến (Linear, NLinear, DLinear), mô hình lai NLinear + DLinear và mô hình lai đa biến trên giá dầu WTI,
đánh giá walk-forward và kiểm định Diebold–Mariano, kèm ứng dụng demo và chức năng dự đoán thử một ngày bất kỳ.

- **Tác giả:** NGUYỄN CÔNG HOÀNG VŨ, ĐỖ DUẪN | **Học phần / lớp:** Cấu trúc dữ liệu - 25DKHA1 
- **Nền tảng:** đề bài *Forecasting with DLinear and NLinear Models* (AI VIET NAM, AIO 2026) và bài báo Zeng và cộng sự (2023).

> **Kết quả chính:** trên WTI, dự báo 5 phiên, các mô hình tuyến tính **không vượt được naive** (giữ nguyên giá đóng cửa
> gần nhất). NLinear và mô hình lai hoà với naive; thêm biến phụ trợ không giúp. Đây là kết quả âm được kiểm định, không
> phải khuyến nghị giao dịch.

**Mục lục:** [1. Chuẩn bị](#1-chuẩn-bị-việc-bạn-phải-tự-làm) · [2. Cài đặt](#2-cài-đặt) · [3. Chạy từ đầu đến demo](#3-chạy-từ-đầu-đến-demo) ·
[4. Dự đoán thử](#4-dự-đoán-thử-một-ngày-bất-kỳ) · [5. Kết quả](#5-kết-quả-chính) · [6. Cấu trúc](#6-cấu-trúc-thư-mục) ·
[7. Dữ liệu và phương pháp](#7-dữ-liệu-và-phương-pháp) · [8. Kiểm thử](#8-kiểm-thử) · [9. Sự cố](#9-xử-lý-sự-cố) · [10. Giới hạn](#10-giới-hạn-và-hướng-phát-triển)

---

## 1. Chuẩn bị (việc bạn phải tự làm)

Chi tiết từng mục, cách lấy và cách kiểm tra: **[`docs/SETUP_CONFIG.md`](docs/SETUP_CONFIG.md)**. Tối thiểu cần:

| Việc | Bắt buộc | Ghi chú |
|---|---|---|
| Python 3.10+ | Có | Khuyến nghị dùng môi trường ảo |
| **Khoá FRED API** (miễn phí) | **Có**, để tải dữ liệu | Điền vào file `.env` (sao chép từ `.env.example`) |
| Internet | Có, ở bước tải dữ liệu và cài thư viện | |


## 2. Cài đặt

```powershell
cd wti-ltsf-forecast
python -m venv .venv
.venv\Scripts\activate                 # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt      # có thể 5-15 phút vì torch nặng; trông như treo là bình thường
copy .env.example .env                 # rồi mở .env và điền FRED_API_KEY (macOS/Linux: cp)
python check_setup.py --online         # kiểm tra môi trường, cấu hình và khoá FRED
```

`check_setup.py` in từng mục **ĐẠT / CẦN SỬA / TUỲ CHỌN** kèm cách sửa và gợi ý bước tiếp theo.

## 3. Chạy từ đầu đến cuối

### Cách nhanh: một lệnh cho tất cả

```powershell
python run_all.py --download       # dữ liệu -> thí nghiệm -> đánh giá -> huấn luyện và lưu mô hình
streamlit run app\app.py           # mở ứng dụng
```

`run_all.py` chạy lần lượt: (1) tải và xử lý dữ liệu, (2) thí nghiệm walk-forward, (3) cải tiến vòng 1, (4) đánh giá,
(5) **huấn luyện toàn bộ mô hình và lưu vào `saved_models/`**. Tổng khoảng **2-3 giờ trên CPU** (ước tính từ các lần đo trước).
Bị ngắt thì chạy lại đúng lệnh, các kết quả đã xong được giữ. Tuỳ chọn: `--quick` (chạy thử vài phút), `--supp` (thêm L=480),
`--skip-data`, `--skip-experiments`, `--skip-train`. Chỉ cần mô hình để dùng (không cần bảng so sánh):
`python run_all.py --download --skip-experiments`.

### Từng bước (nếu muốn tự chạy)

| Bước | Lệnh | Thời gian | Kết quả |
|---|---|---|---|
| 1. Dữ liệu   | `python run_data_pipeline.py --download` | 1-2 phút | `data/raw/*.csv`, `data/processed/wti_dataset.csv`, `results/tables/quality_*.csv` |
| 2. Kiểm thử | `pytest tests` | khoảng 1-2 phút | Toàn bộ test đạt |
| 3. Thí nghiệm | `python run_experiments.py` | khoảng 1,5 giờ | `results/predictions/L*/*.npz` (xem trước: `--dry-run`) |
| 3b. Cải tiến vòng 1 | `python run_experiments.py --set improve1 --windows 5 30` | khoảng 45 phút | thêm biến thể `_zr`, `_shrink` |
| 4. Đánh giá | `python run_evaluation.py` | vài giây | `results/tables/*.csv`, `results/figures/*.png` |
| 5. **Huấn luyện và lưu mô hình** | `python train_all.py` | khoảng 10 phút | `saved_models/asof_*/` (manifest + file `.pt` từng mô hình) |
| 6. Ứng dụng | `streamlit run app\app.py` | tức thì | `http://localhost:8501` |

Chạy thử nhanh (vài phút): `python run_all.py --quick` (sau khi đã có dữ liệu), hoặc `python run_experiments.py --windows 30
--test-years 2023 2024 --seeds 1` rồi `python train_all.py --quick`. Giữ bảng cũ: sao chép `results\tables` trước khi chạy lại đánh giá.

**Mỗi tab cần gì:** *Dự báo bằng mô hình đã lưu* cần bước 5 (và torch, trừ naive/drift); *Huấn luyện đến ngày chọn* chỉ cần bước 1
(và torch); *Phát lại* và *Tổng quan* cần bước 3-4. Thiếu thứ gì, ứng dụng báo cần chạy bước nào.

## 4. Mô hình đã lưu và cách tái sử dụng

### 4.1 Có những mô hình nào (mọi mô hình đều được `train_all.py` huấn luyện và lưu)

| Nhóm | Mô hình |
|---|---|
| Mốc không học | `naive`, `drift` |
| **A đơn biến** | `linear`, `nlinear`, `dlinear` (chỉ giá WTI) |
| **A đa biến** | `linear_mv`, `nlinear_mv`, `dlinear_mv` (thêm spread Brent-WTI, USD/CAD, chỉ số USD, OVX, FED; trộn kênh) |
| B lai | `hybrid_B` |
| C lai đa biến | `hybrid_C_{spread, usd, ovx, fed, all}` và bản `_lvl` (nhánh mức) |
| Điều chuẩn (vòng 1) | `nlinear_zr`, `hybrid_B_zr`, `hybrid_C_usd_zr` |

### 4.2 Cấu trúc thư mục `saved_models/`

```
saved_models/asof_20251231/        # bộ mô hình huấn luyện đến ngày cắt 2025-12-31
├── manifest.json                  # ngày cắt, L, danh sách mô hình, số mẫu, weight decay, drift...
├── nlinear__L30.pt                # trọng số + scaler + cấu hình của một mô hình
└── ...
saved_models/asof_20260930/        # bộ huấn luyện bằng toàn bộ dữ liệu ("last")
```
Cấu hình (`configs/live.yaml`, mục `store`): các L (mặc định 5, 30, 120), các mốc cắt (mặc định `2025-12-31` và `last`) và seed.
Bộ có mốc cắt **sớm hơn** cho phép đánh giá ngoài mẫu trên các ngày sau mốc; bộ `last` dùng để dự báo cho các phiên sắp tới.

### 4.3 Dùng lại mô hình đã lưu (không huấn luyện lại)

**Trong ứng dụng:** tab đầu *Dự báo bằng mô hình đã lưu*: chọn bộ mô hình, L, ngày, tích các mô hình (hoặc "Chọn tất cả"), bấm **Dự báo**.
Mỗi mô hình **chạy suy luận thật** trên L phiên đến ngày chọn; biểu đồ vẽ mọi mô hình cạnh giá thực, kèm biểu đồ thanh so sánh và bảng sai số.
Nếu cả 5 phiên mục tiêu nằm **trước hoặc đúng ngày cắt** của bộ mô hình thì dự báo là **trong mẫu** (đã dùng để huấn luyện/chọn epoch) và ứng dụng cảnh báo; từ ngày cắt trở đi là **ngoài mẫu** (gồm cả dự báo cho tương lai).

**Từ dòng lệnh:**
```powershell
python predict_saved.py --list                                    # các bộ mô hình đã lưu
python predict_saved.py --model nlinear --L 30 --date 2026-03-10  # một mô hình
python predict_saved.py --model all --L 30                        # mọi mô hình, ngày cuối của dữ liệu
python predict_saved.py --set asof_20251231 --model hybrid_B --L 120 --date 2026-05-04
```

**Từ mã Python:**
```python
from src.live_predict import load_all_configs, load_frame
from src.model_store import list_sets, predict_saved

cfg, wf, exp, live = load_all_configs()
df = load_frame(cfg)                         # data/processed/wti_dataset.csv (kèm log-giá)
man = list_sets(live)[0]                     # bộ mô hình mới nhất
r = predict_saved(df, man, exp, "nlinear", 30, "2026-03-10")
print(r["forecast"], r["in_sample"])         # giá dự báo 5 phiên (USD), có phải trong mẫu không
```

**Dự báo cho dữ liệu mới mà không huấn luyện lại:** đặt `end_date: null` trong `configs/config.yaml`, chạy lại
`python run_data_pipeline.py --download` (bước 1), rồi dùng `predict_saved.py` với bộ `last` hoặc bộ cũ trên các ngày mới.
Mô hình đã lưu giữ nguyên trọng số và scaler của lúc huấn luyện; chỉ cần dữ liệu đủ L phiên và các cột cần cho mô hình đó.
Huấn luyện lại khi dữ liệu đã thay đổi nhiều: `python train_all.py --overwrite`.

**Huấn luyện lại một phần:** `python train_all.py --names nlinear hybrid_B --windows 60 --cutoffs 2024-12-31`.
Cần L khác với L đã lưu: dùng tab *Huấn luyện đến ngày chọn* hoặc `python predict_day.py --model nlinear --L 45 --date 2025-06-02`
(huấn luyện theo yêu cầu đến đúng ngày chọn, L tự do từ 2 đến 240).

| Điều cần biết | Chi tiết |
|---|---|
| Ngày/cửa sổ không hợp lệ | Bị từ chối kèm lý do (chạm giai đoạn 2020, khoảng trống dữ liệu, thiếu biến phụ trợ, không đủ L phiên) |
| Một seed | Mô hình đã lưu huấn luyện một seed (mặc định 0); kết quả thí nghiệm là trung bình 5 seed nên có thể hơi khác |
| Dự báo cho tương lai | Chọn ngày cuối của dữ liệu: các phiên sau chưa có giá thực ("chưa có"); ngày suy ra theo lịch, xấp xỉ |
| Cache của "Huấn luyện đến ngày chọn" | `saved_models/live/`, dùng lại cho cùng cấu hình và dữ liệu |

Kết quả của đồ án cho thấy dự báo của mọi mô hình xấp xỉ giá đóng cuối; hãy luôn so với cột/đường *naive*.

## 5. Kết quả chính

Tỉ lệ RMSE so với naive, trung bình qua 5 bước, gộp các fold test 2017-2026 (bỏ 2020). **Nhỏ hơn 1 là tốt hơn naive**;
chênh lệch trong ±0,5% coi là hoà. Số liệu từ lần chạy của tác giả; chạy lại với dữ liệu tải ngày khác có thể lệch nhẹ.

| Mô hình | L=5 | L=30 | L=120 | L=480* |
|---|---|---|---|---|
| naive | 1,0000 | 1,0000 | 1,0000 | 1,0000 |
| drift | 1,0005 | 1,0004 | 1,0005 | 1,0000 |
| lai (B) | 1,0017 | 1,0074 | 1,0455 | 1,1103 |
| nlinear | 1,0018 | 1,0101 | 1,0452 | 1,1200 |
| lai + USD (C, tốt nhất nhóm C ở L=30) | 1,0040 | 1,0138 | 1,0745 | 1,2588 |
| dlinear | 1,0464 | 1,0643 | 1,0972 | 1,5509 |
| linear | 1,2038 | 1,1959 | 1,2101 | 1,6164 |
| nlinear điều chuẩn (`_zr`, vòng 1) | 1,0014 | 1,0027 | | |

\* L=480 chỉ có 6 năm test, không so sánh trực tiếp với các cột khác.

1. **Không mô hình nào vượt naive** một cách vững; NLinear, lai và drift không khác naive có ý nghĩa thống kê (DM ở L=30).
2. **Lai tốt hơn Linear và DLinear nhưng không khác NLinear**: cải thiện đến từ bước trừ giá cuối, không phải phần phân rã.
3. **Biến phụ trợ không cải thiện** mô hình lai (USD hoà, các nhóm còn lại kém hơn); tương quan trễ với lợi suất WTI gần 0.
4. **Nhánh "mức hiện tại" (`_lvl`) gây hại**, nhất là với `fed` và `all` ở các fold đầu (ngoại suy khi chế độ thị trường đổi).
5. **L càng dài càng tệ**; **điều chuẩn hoá** (khởi tạo 0, weight decay) thu hẹp khoảng cách tới naive nhưng không vượt.
6. Độ chính xác hướng đi của mọi mô hình quanh 0,47-0,52 (nằm trong nhiễu).

Nhật ký đầy đủ, giả thuyết đã đăng ký trước và các sai sót: [`docs/PROJECT_LOG.md`](docs/PROJECT_LOG.md).

## 6. Cấu trúc thư mục

```
wti-ltsf-forecast/
├── README.md, requirements.txt, .env.example, .gitignore
├── run_all.py               # chạy toàn bộ từ dữ liệu đến mô hình đã lưu bằng một lệnh
├── train_all.py             # huấn luyện TOÀN BỘ mô hình và lưu vào saved_models/
├── predict_saved.py         # dự báo bằng mô hình đã lưu (dòng lệnh)
├── check_setup.py           # kiểm tra môi trường/cấu hình, gợi ý bước tiếp theo
├── run_data_pipeline.py     # (bước 1) tải + tiền xử lý dữ liệu
├── run_experiments.py       # (bước 4) huấn luyện thí nghiệm walk-forward
├── run_evaluation.py        # (bước 5) bảng, kiểm định, biểu đồ
├── predict_day.py           # huấn luyện đến ngày chọn rồi dự báo, L tự do (dòng lệnh)
├── diagnose_improve1.py     # chẩn đoán cải tiến vòng 1
├── configs/                 # config, walk_forward, experiments, features, live (.yaml)
├── data/{raw,processed}/    # dữ liệu thô FRED; dữ liệu sạch (tự tạo)
├── src/                     # config, data_loader, preprocessing, features, dataset, models, baselines, train,
│                            # experiments, improvements, evaluate, report, live_predict, model_store
├── app/                     # demo Streamlit (app.py, predictor.py) và app/README.md
├── tests/                   # kiểm thử tự động
├── results/{predictions,tables,figures}/   # kết quả (tự tạo)
├── saved_models/            # mô hình đã lưu: asof_*/ (train_all.py) và live/ (cache huấn luyện theo yêu cầu)
├── docs/                    # SETUP_CONFIG.md, PROJECT_LOG.md
└── report/                  # chỗ để báo cáo (trống)
```

## 7. Dữ liệu và phương pháp

**Dữ liệu** (nguồn FRED; mã trong `configs/config.yaml`): `wti` (DCOILWTICO, mục tiêu), `brent` (DCOILBRENTEU), `usdcad`
(DEXCAUS), `usd_broad` (DTWEXBGS), `ovx` (OVXCLS), `fed_funds` (DFF). Tải từ 2008-01-01, nghiên cứu từ 2010-01-01, cắt ở
`end_date` (mặc định 2026-09-30). Căn chỉnh theo lịch của WTI, chỉ dùng quan sát trước hoặc đúng ngày; OVX và FED trễ 1 phiên;
cửa sổ 2020-03 đến 2020-06 (giá âm) và các cửa sổ chạy qua khoảng trống dữ liệu bị loại. Với dữ liệu mặc định: 4.649 phiên,
4.061 ngày dùng được. Giá là spot, chưa xử lý futures.

**Mô hình** (`src/models.py`): Linear (`W·x + b`; bản đa biến trộn kênh, đầu vào L×C); NLinear (trừ giá cuối); DLinear (tách xu hướng/phần dư, trung bình trượt 5);
Hybrid = trừ giá cuối + tách + hai lớp tuyến tính (1 kênh = nhóm B, nhiều kênh = nhóm C; tuỳ chọn nhánh mức).
Thứ tự "trừ mốc rồi phân rã" không ảnh hưởng vì MA(x − c) = MA(x) − c (có test). Mô hình hoá trên log-giá, đánh giá trên thay đổi so với giá cuối.

**Đánh giá** (`src/dataset.py`, `src/evaluate.py`, `src/report.py`): walk-forward expanding window, test từng năm 2017-2026;
validation 250 mẫu cuối có purge; scaler chỉ fit trên train; mọi mô hình dùng cùng tập mẫu; Adam 1e-3, StepLR, 50 epoch, 5 seed;
RMSE, MAE, độ chính xác hướng đi, tỉ lệ so với naive; Diebold–Mariano (hiệu chỉnh HLN, phương sai HAC độ trễ h−1).

**Cấu hình:** xem bảng trong `docs/SETUP_CONFIG.md` (cuối trang).

**Các file kết quả** (`results/tables/`): `quality_*`, `adf_report`, `lead_lag_check`, `missing_business_days_by_year`, `folds_L*`,
`summary_avg_L*`, `summary_by_tau_L*`, `dm_tests_L*` (`mean_diff < 0`: mô hình tốt hơn tham chiếu), `by_year_L*`, `by_regime_L*`.

**Tái lập:** seed đặt trước khi khởi tạo mô hình; lưu nguyên `data/raw/` (có `_metadata.json` ghi ngày tải) vì FRED có thể hiệu chỉnh
số liệu; ghi lại phiên bản thư viện (`pip freeze > requirements-lock.txt`).

## 8. Kiểm thử

`pytest tests` gồm: chống rò rỉ (dữ liệu, cửa sổ, purge, scaler, dự đoán thử không nhìn tương lai), đúng với tài liệu gốc
(Linear/NLinear/DLinear khớp kết quả tính tay: 104.4/103.6/105.7; 103.1/102.2/103.3; 99.933/103.967/103.0), thống kê
(Diebold–Mariano), hạ tầng thí nghiệm và demo. Test cần torch tự bỏ qua nếu chưa cài torch.

## 9. Xử lý sự cố

| Hiện tượng | Cách xử lý |
|---|---|
| `Chưa có FRED_API_KEY` / `HTTP 400` | Xem `docs/SETUP_CONFIG.md` mục 1; chạy `python check_setup.py --online` |
| `pip install` quá lâu, như treo | Do `torch` nặng; chờ, hoặc cài torch CPU trước (xem mục 2 trong `docs/SETUP_CONFIG.md`) |
| `FileNotFoundError` khi đọc dữ liệu thô | Chạy `python run_data_pipeline.py --download` |
| Demo báo "Chưa có data/processed/wti_dataset.csv" | Chạy bước 1 |
| Tab Phát lại/Tổng quan báo chưa có kết quả | Chạy bước 4 rồi bước 5 |
| Tab Dự đoán thử: "Cần cài torch" | `pip install torch --index-url https://download.pytorch.org/whl/cpu` (naive và drift vẫn chạy không cần torch) |
| Dự đoán thử từ chối ngày/L | Đọc lý do hiển thị: ngày quá sớm, cửa sổ chạm 2020 hoặc khoảng trống dữ liệu, thiếu biến phụ trợ. Chọn ngày muộn hơn hoặc L nhỏ hơn |
| Sau khi sửa file `.py`, lệnh không in gì hoặc lỗi `ImportError` | File bị chép sai/rỗng; kiểm tra kích thước, chép lại bằng `Copy-Item` thay vì dán nội dung |
| Số liệu demo cũ sau khi chạy lại | Nhấn `C` trong ứng dụng hoặc khởi động lại Streamlit |
| Pipeline báo `last_date` không như mong muốn | Kiểm tra `end_date` trong `configs/config.yaml` |
| Bị ngắt giữa `run_experiments.py` | Chạy lại đúng lệnh cũ, các kết quả đã có được bỏ qua |
| Cổng 8501 bận | `streamlit run app\app.py --server.port 8502` |

## 10. Giới hạn và hướng phát triển

**Giới hạn:** kết luận chỉ áp dụng cho WTI spot, 5 phiên, mô hình tuyến tính, test 2017-2026 (không kết luận được rằng dầu không
thể dự báo); dữ liệu nguồn thiếu nhiều phiên (đặc biệt 2011); tỷ giá FRED công bố trễ nên ngoài đời không có sẵn cùng ngày;
bảng kiểm định có nhiều phép so sánh nên một số kết quả "có ý nghĩa" có thể là ngẫu nhiên; quan sát về 2024-2025 và về lượng
dữ liệu train là thăm dò; biến lịch (`src/features.py`) đã viết nhưng chưa dùng. **Không phải khuyến nghị đầu tư.**

**Hướng phát triển:** khoảng dự báo (phân vị) theo độ biến động; dữ liệu mới (đường cong kỳ hạn futures, tồn kho EIA) với
quy tắc thời điểm chặt chẽ; biến lịch; dùng futures thay spot; thành phần phi tuyến nhỏ; khung giờ (1 giờ, 4 giờ).

**Tài liệu tham khảo:** A. Zeng et al., *Are transformers effective for time series forecasting?*, AAAI 2023 · H. Wu et al.,
*Autoformer*, NeurIPS 2021 · F. X. Diebold, R. S. Mariano, *Comparing predictive accuracy*, JBES 1995 · D. Harvey, S. Leybourne,
P. Newbold, *Testing the equality of prediction mean squared errors*, IJF 1997 (hãy kiểm tra lại thông tin trích dẫn trước
khi dùng) · Dữ liệu: FRED, Federal Reserve Bank of St. Louis (nguồn gốc từng chuỗi ghi trên trang của chuỗi).
