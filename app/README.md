# Demo dự báo giá dầu WTI 5 phiên

Ứng dụng web (Streamlit) cho đồ án *Dự báo giá dầu WTI bằng họ mô hình LTSF-Linear*. Có hai chế độ: **dự đoán thử** (mô hình
thật sự huấn luyện và chạy tại ngày bạn chọn) và **phát lại** các dự báo thí nghiệm đã lưu, so với giá thực tế.

> **Lưu ý.** Ở mức tổng thể, **không mô hình nào vượt naive** (giữ nguyên giá đóng cửa gần nhất). Một ngày dự báo chỉ là một
> mẫu. Đây là sản phẩm nghiên cứu, không phải khuyến nghị giao dịch.

## 1. Chạy ứng dụng

Từ thư mục gốc dự án (đã làm các bước chuẩn bị trong `README.md`):

```powershell
streamlit run app\app.py
```

Mở `http://localhost:8501`. Đổi cổng: `--server.port 8502`. Dừng: `Ctrl+C`.

**Cần gì để chạy:**

| Tính năng | Cần |
|---|---|
| Toàn bộ ứng dụng | `data/processed/wti_dataset.csv` (bước 1 của README) và `pip install streamlit pandas numpy matplotlib pyyaml` |
| Tab 1 (mô hình đã lưu) | `saved_models/asof_*/` (chạy `python train_all.py`) và **torch** (trừ naive/drift) |
| Tab 2 (huấn luyện đến ngày chọn), mô hình có huấn luyện | Thêm **torch** (`pip install torch --index-url https://download.pytorch.org/whl/cpu`) |
| Tab 3 và 4 | Thêm `results/predictions/` (bước 4) và `results/tables/` (bước 5) |

Thiếu thứ gì, ứng dụng báo rõ cần chạy bước nào.

## 2. Kiến trúc

```
data/processed/wti_dataset.csv ──► src/live_predict.py ──► tab "Dự đoán thử"   (huấn luyện đến ngày chọn, dự báo thật)
        │                                │ lưu/nạp mô hình
        │                                ▼
        │                         saved_models/live/*.pt
        └──────────────────────► app/predictor.py ──► tab "Phát lại"   (đọc results/predictions/*.npz)
results/tables/*.csv ───────────────────────────────► tab "Tổng quan"
```

| Thành phần | Vai trò |
|---|---|
| `app/app.py` | Giao diện: 5 tab, biểu đồ matplotlib |
| `src/model_store.py` | Huấn luyện và lưu toàn bộ mô hình, nạp và dự báo bằng mô hình đã lưu |
| `src/live_predict.py` | Cắt dữ liệu tại ngày chọn, kiểm tra cửa sổ, huấn luyện (hoặc nạp mô hình đã lưu), dự báo từ L phiên cuối. Dùng chung với `predict_day.py` |
| `app/predictor.py` | Đọc các dự báo đã lưu của thí nghiệm, trung bình qua seed, đổi log-giá về USD |
| `configs/live.yaml` | Mặc định của tab Dự đoán thử (mô hình, L, giới hạn L, thiết bị, thư mục lưu mô hình) |

## 3. Hướng dẫn sử dụng (5 tab)

### Tab 1: Dự báo bằng mô hình đã lưu (chính)

Dùng các mô hình do `train_all.py` huấn luyện sẵn và lưu trong `saved_models/`. **Không** huấn luyện lại, **không** đọc dự báo có sẵn:
mỗi mô hình chạy suy luận thật trên cửa sổ L phiên đến ngày bạn chọn.
1. Chọn **Bộ mô hình** (mỗi bộ huấn luyện đến một ngày cắt), **L** (trong các L đã huấn luyện) và **ngày dự báo**.
2. Chọn các mô hình hoặc tích **Chọn tất cả mô hình** (gồm naive, drift, nhóm A đơn biến và đa biến, B, C, điều chuẩn).
3. Bấm **Dự báo**: biểu đồ so sánh mọi mô hình với giá thực, biểu đồ thanh (dự báo t+5 so với giá hiện tại, vạch đen là thực tế)
   và bảng giá dự báo t+1..t+5 kèm sai số.
4. Banner **Trong mẫu / Một phần / Ngoài mẫu**: trong mẫu nghĩa là 5 phiên mục tiêu đã nằm trong dữ liệu huấn luyện/chọn epoch (không dùng để đánh giá); ngoài mẫu là từ ngày cắt trở đi.

### Tab 2: Huấn luyện đến ngày chọn

Chọn một mô hình, L tự do (2 đến 240), ngày và seed; ứng dụng huấn luyện mô hình chỉ bằng dữ liệu **đến đúng ngày chọn** rồi dự báo
(luôn ngoài mẫu). Mô hình được lưu cache ở `saved_models/live/`. Ngày không hợp lệ bị từ chối kèm lý do.

### Tab 3: Phát lại dự báo thí nghiệm

Vẽ dự báo ngoài mẫu đã tính trong thí nghiệm walk-forward (trung bình 5 seed) cạnh giá thực; có nút chọn ngày ngẫu nhiên.

### Tab 4: Tổng quan toàn bộ tập test

Bảng và biểu đồ tỉ lệ RMSE so với naive, bảng theo năm (fold 2020 bị loại khỏi bảng tổng).

### Tab 5: Giải thích mô hình

## 4. Cách đọc biểu đồ

- **Biểu đồ dự đoán/phát lại:** đường xám là dữ liệu quá khứ, vạch chấm là ngày dự báo, **đường đen dày là giá thực**, đường cam
  chấm là naive, đường nét đứt là mô hình. Mọi đường dự báo xuất phát từ cùng điểm (giá đóng cuối). Đường gần như nằm ngang
  nghĩa là "giá gần như không đổi", tức xấp xỉ naive (phần lớn mô hình của đề tài). Trục tung tự co giãn nên chênh lệch nhỏ
  có thể trông lớn; hãy đọc bảng sai số và so với **naive**.
- **Biểu đồ thanh (Tổng quan):** thanh bắt đầu từ vạch đen tại 1,0 (naive). Kéo sang phải là kém hơn naive, sang trái là tốt
  hơn; chênh lệch trong ±0,5% coi là hoà.
- **Bảng màu theo năm:** xanh (≈0,95) tốt hơn, đỏ (≈1,20) kém hơn; màu chỉ cho mức tương đối, không phải ý nghĩa thống kê.
- **Độ chính xác hướng đi:** 0,50 là tung đồng xu; 0,47 đến 0,52 nằm trong nhiễu.

**Vì sao so với naive:** sai số tính so với giá thực tế; naive ("giữ nguyên giá hiện tại") là mốc tối thiểu để biết sai số đó tốt
hay dở. Một mô hình không gần giá thực hơn mốc này thì không có giá trị dự báo.

## 5. Hai chế độ có thể cho số hơi khác nhau

Tab *Dự đoán thử* huấn luyện một mô hình (một seed) đến đúng ngày chọn. Tab *Phát lại* dùng mô hình của từng năm test, trung
bình 5 seed. Cả hai đều ngoài mẫu và không nhìn dữ liệu tương lai, nhưng không phải cùng một mô hình.

## 6. Quy ước dữ liệu của tab Phát lại

- File: `results/predictions/L{L}/{mô hình}__{năm test}__s{seed}.npz` với `origin`, `pred_log`, `actual_log`, `last_log` ...
- Tên mô hình: `naive`, `drift`, `linear`, `nlinear`, `dlinear`, `hybrid_B`, `hybrid_C_<nhóm>` (nhóm: `spread`, `usd`, `ovx`, `fed`,
  `all`), hậu tố `_lvl` (nhánh mức), `_zr` (khởi tạo 0 + ridge), `_shrink` (co về naive).
- Giá hiển thị là `exp(log-giá)`, USD/thùng (giá spot WTI Cushing từ FRED).

## 7. Xử lý sự cố

| Hiện tượng | Cách xử lý |
|---|---|
| "Chưa có data/processed/wti_dataset.csv" | Chạy `python run_data_pipeline.py --download` |
| Tab Phát lại/Tổng quan báo chưa có kết quả | Chạy `python run_experiments.py` rồi `python run_evaluation.py` |
| "Cần cài torch" | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| Dự đoán thử từ chối ngày/L | Đọc lý do hiển thị, chọn ngày muộn hơn hoặc L nhỏ hơn |
| `ModuleNotFoundError` | `pip install streamlit pandas numpy matplotlib pyyaml` (hoặc `pip install -r requirements.txt`) |
| `ModuleNotFoundError: app` | Chạy từ thư mục gốc, kiểm tra có `app/__init__.py` |
| Số liệu cũ sau khi chạy lại | Nhấn `C` trong ứng dụng để xoá cache |

## 8. Kiểm thử

`pytest tests/test_predictor.py tests/test_live_predict.py` kiểm tra đọc kết quả đã lưu, và dự đoán thử (dự báo không phụ thuộc
dữ liệu tương lai, tái lập theo seed, dùng lại và vô hiệu hoá mô hình đã lưu, từ chối cửa sổ không hợp lệ).
