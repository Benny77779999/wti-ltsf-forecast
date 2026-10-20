"""Demo: dự đoán giá WTI 5 phiên bằng các mô hình LTSF-Linear.

Chạy:  streamlit run app/app.py
Tab 1 "Dự đoán thử": mô hình THỰC SỰ chạy (huấn luyện bằng dữ liệu đến ngày chọn rồi dự báo), không đọc kết quả có sẵn.
Tab 2-3 đọc kết quả backtest đã lưu trong results/ (không huấn luyện lại).
"""
import importlib.util
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from app.predictor import Forecasts, future_dates, history_window, label

st.set_page_config(page_title="Dự báo WTI 5 phiên", layout="wide")


@st.cache_data
def load_df():
    return pd.read_csv(ROOT / "data" / "processed" / "wti_dataset.csv", parse_dates=["date"], index_col="date")


@st.cache_resource
def get_forecasts():
    return Forecasts(ROOT / "results" / "predictions")


@st.cache_data
def load_table(name: str):
    p = ROOT / "results" / "tables" / name
    return pd.read_csv(p) if p.exists() else None


st.title("Dự báo giá dầu WTI 5 phiên: các mô hình LTSF-Linear")
if not (ROOT / "data" / "processed" / "wti_dataset.csv").exists():
    st.error("Chưa có data/processed/wti_dataset.csv. Chạy `python run_data_pipeline.py --download` trước "
             "(xem README.md, bước 3).")
    st.stop()

df, fc = load_df(), get_forecasts()
Ls = fc.windows()

st.info("**Lưu ý trung thực:** ở mức tổng thể trên toàn bộ tập test, **không mô hình nào vượt được naive** "
        "(giữ nguyên giá đóng cửa gần nhất). Một ngày dự báo chỉ là **một mẫu**; hãy xem tab *Tổng quan* "
        "trước khi kết luận mô hình nào tốt hơn. Đây là sản phẩm nghiên cứu, không phải khuyến nghị giao dịch.")

has_torch = importlib.util.find_spec("torch") is not None
tab_saved, tab_live, tab_bt, tab_ov, tab_info = st.tabs(
    ["Dự báo bằng mô hình đã lưu", "Huấn luyện đến ngày chọn", "Phát lại dự báo thí nghiệm",
     "Tổng quan toàn bộ tập test", "Giải thích mô hình"])

# =============================================================================
# Tab 1: dự báo bằng mô hình ĐÃ LƯU (train_all.py), suy luận thật, không đọc kết quả có sẵn
# =============================================================================
with tab_saved:
    st.subheader("Dự báo bằng mô hình đã huấn luyện và lưu")
    st.caption("Chọn bộ mô hình đã lưu, số phiên quá khứ L, ngày và các mô hình muốn so sánh. Mỗi mô hình **chạy suy luận thật** "
               "trên cửa sổ L phiên đến ngày chọn (không huấn luyện lại, không đọc dự báo có sẵn).")
    try:
        from src.dataset import add_log_price
        from src.live_predict import LivePredictError, load_all_configs
        from src.model_store import available_models, list_sets, predict_saved
        cfg, wf, exp, live = load_all_configs()
        sets = list_sets(live)
        s_err = None
    except Exception as e:
        sets, s_err = [], str(e)
    if s_err:
        st.error(f"Không nạp được cấu hình: {s_err}")
    elif not sets:
        st.info("Chưa có mô hình đã lưu. Chạy `python train_all.py` (hoặc `python run_all.py`), xem README.md.")
    else:
        if not has_torch:
            st.warning("Chưa cài torch nên chỉ dùng được naive và drift. Cài: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.")
        c1, c2, c3 = st.columns([2, 1, 1.3])
        tag = c1.selectbox("Bộ mô hình", [m["tag"] for m in sets], key="sv_set",
                           format_func=lambda t: f"{t} (huấn luyện đến {next(m['cutoff'] for m in sets if m['tag'] == t)})")
        man = next(m for m in sets if m["tag"] == tag)
        L = c2.selectbox("Số phiên quá khứ đưa vào (L)", man["windows"], key="sv_L")
        day = c3.date_input("Ngày dự báo", value=df.index[-1].date(), min_value=df.index[0].date(),
                            max_value=df.index[-1].date(), key="sv_date")
        names = available_models(man, L) if has_torch else ["naive", "drift"]
        pick_all = st.checkbox("Chọn tất cả mô hình", value=False, key="sv_all")
        dflt = names if pick_all else [m for m in ("naive", "nlinear", "nlinear_mv", "hybrid_B", "hybrid_C_usd") if m in names]
        chosen = st.multiselect("Mô hình", names, default=dflt, format_func=label, key=f"sv_models_{tag}_{L}_{pick_all}")
        if st.button("Dự báo", type="primary", key="sv_run"):
            frame, res, errs = add_log_price(df), [], []
            for n in chosen:
                try:
                    res.append(predict_saved(frame, man, exp, n, int(L), str(day)))
                except LivePredictError as e:
                    errs.append(f"{label(n)}: {e}")
            st.session_state["sv_res"], st.session_state["sv_err"] = res, errs
        for e in st.session_state.get("sv_err") or []:
            st.warning(e)
        res = st.session_state.get("sv_res")
        if res:
            r0 = res[0]
            if r0["status"] == "trong mẫu":
                st.warning(f"**Trong mẫu:** 5 phiên mục tiêu sau {r0['origin'].date()} nằm trong dữ liệu mà bộ mô hình này đã dùng để huấn luyện/chọn "
                           f"epoch (đến {r0['cutoff'].date()}), nên đừng dùng để đánh giá độ chính xác. Chọn ngày từ {r0['cutoff'].date()} trở đi "
                           "hoặc bộ mô hình huấn luyện đến sớm hơn để có dự báo ngoài mẫu.")
            elif r0["status"] == "một phần":
                st.warning(f"**Một phần trong mẫu:** ngày cắt {r0['cutoff'].date()} của bộ mô hình nằm giữa 5 phiên mục tiêu; chỉ phần sau ngày cắt là ngoài mẫu.")
            else:
                st.success(f"**Ngoài mẫu:** ngày dự báo {r0['origin'].date()} từ ngày cắt {r0['cutoff'].date()} của bộ mô hình trở đi (mô hình chưa thấy các phiên mục tiêu).")
            h = len(r0["forecast"])
            fig, ax = plt.subplots(figsize=(10, 4.8))
            ax.plot(r0["input_dates"], r0["input_prices"], color="gray", label=f"Dữ liệu đưa vào ({len(r0['input_dates'])} phiên)")
            ax.axvline(r0["origin"], color="black", linestyle=":", linewidth=1)
            xs = [r0["origin"], *r0["dates"]]
            for r in res:
                ax.plot(xs, [r["last_price"], *r["forecast"]], linestyle="--", marker="o", markersize=3, linewidth=1.2, label=label(r["model"]))
            if np.isfinite(r0["actual"]).any():
                ok = np.isfinite(r0["actual"])
                ax.plot([r0["origin"], *r0["dates"][ok]], [r0["last_price"], *r0["actual"][ok]], color="black", linewidth=2.6, label="Giá thực tế")
            ax.set_ylabel("Giá WTI (USD/thùng)"); ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2); fig.autofmt_xdate(); fig.tight_layout()
            st.pyplot(fig)
            chg = pd.Series({label(r["model"]): 100 * (r["forecast"][-1] / r["last_price"] - 1) for r in res}).sort_values()
            fig2, ax2 = plt.subplots(figsize=(8, 0.32 * len(chg) + 1.2))
            ax2.barh(chg.index, chg.values)
            if np.isfinite(r0["actual"][-1]):
                ax2.axvline(100 * (r0["actual"][-1] / r0["last_price"] - 1), color="black", linewidth=2, label="Thực tế")
                ax2.legend(fontsize=8)
            ax2.axvline(0, color="gray", linewidth=1); ax2.set_xlabel(f"Dự báo t+{h} so với giá hiện tại (%)"); ax2.grid(alpha=0.3, axis="x")
            fig2.tight_layout(); st.pyplot(fig2)
            rows = []
            for r in res:
                row = {"mô hình": label(r["model"]), **{f"t+{k + 1}": float(v) for k, v in enumerate(r["forecast"])}}
                if np.isfinite(r["actual"]).any():
                    ok = np.isfinite(r["actual"])
                    row["sai số TB (%)"] = float(100 * np.mean(np.abs(r["forecast"][ok] / r["actual"][ok] - 1)))
                rows.append(row)
            tbl = pd.DataFrame(rows)
            act = {"mô hình": "Giá thực tế", **{f"t+{k + 1}": (float(v) if np.isfinite(v) else None) for k, v in enumerate(r0["actual"])}}
            tbl = pd.concat([pd.DataFrame([act]), tbl], ignore_index=True)
            fmt = {c: "{:.2f}" for c in tbl.columns if c != "mô hình"}
            st.dataframe(tbl.style.format(fmt, na_rep="chưa có"), width="stretch", hide_index=True)
            st.caption(f"Ngày dự báo {r0['origin'].date()}, giá {r0['last_price']:.2f} USD; các phiên t+1..t+{h}: {r0['dates'][0].date()} đến {r0['dates'][-1].date()}"
                       + (" (một phần suy ra theo lịch)." if r0["dates_projected"] else ".")
                       + " Hãy so với naive; ở mức tổng thể không mô hình nào vượt naive.")

# =============================================================================
# Tab 2: huấn luyện đến ngày chọn, mô hình thực sự chạy
# =============================================================================
with tab_live:
    st.subheader("Dự đoán thử một ngày bất kỳ")
    st.caption("Chọn mô hình, số phiên quá khứ đưa vào (L) và ngày dự báo. Ứng dụng huấn luyện mô hình chỉ bằng dữ "
               "liệu **đến ngày bạn chọn**, rồi đưa L phiên cuối qua mô hình để dự báo 5 phiên kế tiếp. Dữ liệu sau "
               "ngày chọn không được dùng để huấn luyện hay dự báo; giá thực tế (nếu đã có) chỉ để so sánh sau đó.")
    try:
        from src.dataset import add_log_price
        from src.live_predict import LivePredictError, load_all_configs, model_catalog, predict_day
        cfg, wf, exp, live = load_all_configs()
        catalog = model_catalog(exp)
        setup_error = None
    except Exception as e:                                            # cấu hình thiếu hoặc lỗi
        setup_error, catalog, live = str(e), {}, {}

    if setup_error:
        st.error(f"Không nạp được cấu hình: {setup_error}")
    else:
        if not has_torch:
            st.warning("Chưa cài torch nên chỉ chạy được *naive* và *drift* (không cần huấn luyện). Để dùng các mô "
                       "hình còn lại: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.")
        names = [n for n, s in catalog.items() if has_torch or s["kind"] == "baseline"]
        default_model = live.get("default_model", "nlinear")
        c1, c2, c3, c4 = st.columns([2.2, 1, 1.2, 0.8])
        model = c1.selectbox("Mô hình", names, index=names.index(default_model) if default_model in names else 0,
                             format_func=label, key="live_model")
        L = c2.number_input("Số phiên quá khứ đưa vào (L)", min_value=int(live.get("min_L", 2)),
                            max_value=int(live.get("max_L", 240)), value=int(live.get("default_L", 30)),
                            step=1, key="live_L")
        default_date = df.index[max(0, len(df) - 30)].date()
        day = c3.date_input("Ngày dự báo", value=default_date, min_value=df.index[0].date(),
                            max_value=df.index[-1].date(), key="live_date")
        seed = c4.number_input("Seed", min_value=0, max_value=9999, value=int(live.get("default_seed", 0)),
                               step=1, key="live_seed")
        use_cache = st.checkbox("Dùng lại mô hình đã huấn luyện nếu có (nhanh hơn)", value=True, key="live_cache")
        if catalog.get(model, {}).get("kind") == "tuned":
            st.caption("Mô hình này thử nhiều mức weight decay và chọn mức tốt nhất trên validation nên chậm hơn.")

        if st.button("Dự đoán", type="primary", key="live_run"):
            device = live.get("device", "cpu")
            if device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            with st.spinner("Đang huấn luyện (nếu chưa có mô hình đã lưu) và dự báo..."):
                try:
                    res = predict_day(add_log_price(df), model, int(L), str(day), seed=int(seed),
                                      exp=exp, wf=wf, cfg=cfg,
                                      cache_dir=ROOT / live.get("cache_dir", "saved_models/live"),
                                      use_cache=use_cache, device=device)
                    st.session_state["live_result"], st.session_state["live_error"] = res, None
                except LivePredictError as e:
                    st.session_state["live_result"], st.session_state["live_error"] = None, str(e)

        if st.session_state.get("live_error"):
            st.error(st.session_state["live_error"])
        r = st.session_state.get("live_result")
        if r:
            snapped = " (phiên gần nhất trước ngày bạn chọn)" if str(r["origin"].date()) != str(day) else ""
            st.success(f"Đã dự báo: **{label(r['model'])}**, L = {r['L']}, ngày dự báo **{r['origin'].date()}**{snapped}. "
                       f"{'Nạp mô hình đã lưu' if r['from_cache'] else 'Vừa huấn luyện'} trong {r['seconds']:.1f} giây.")
            h = len(r["forecast"])
            m1, m2, m3 = st.columns(3)
            m1.metric(f"Giá đóng cửa {r['origin'].date()}", f"{r['last_price']:.2f} USD")
            m2.metric(f"Dự báo t+{h} ({r['dates'][-1].date()})", f"{r['forecast'][-1]:.2f} USD",
                      f"{100 * (r['forecast'][-1] / r['last_price'] - 1):+.2f}% so với giá hiện tại")
            if np.isfinite(r["actual"][-1]):
                m3.metric(f"Thực tế t+{h}", f"{r['actual'][-1]:.2f} USD",
                          f"dự báo lệch {100 * (r['forecast'][-1] / r['actual'][-1] - 1):+.2f}%", delta_color="off")
            else:
                m3.metric(f"Thực tế t+{h}", "chưa có")

            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.plot(r["input_dates"], r["input_prices"], color="gray", label=f"Dữ liệu đưa vào ({len(r['input_dates'])} phiên)")
            ax.axvline(r["origin"], color="black", linestyle=":", linewidth=1)
            xs = [r["origin"], *r["dates"]]
            ax.plot(xs, [r["last_price"], *r["forecast"]], linestyle="--", marker="o", linewidth=1.6, label=f"Dự báo: {label(r['model'])}")
            ax.plot(xs, [r["last_price"], *r["naive"]], linestyle=":", color="tab:orange", label="Naive (giữ giá hiện tại)")
            if np.isfinite(r["actual"]).any():
                ok = np.isfinite(r["actual"])
                ax.plot([r["origin"], *r["dates"][ok]], [r["last_price"], *r["actual"][ok]], color="black", linewidth=2.5, label="Giá thực tế")
            ax.set_ylabel("Giá WTI (USD/thùng)"); ax.grid(alpha=0.3); ax.legend(fontsize=8); fig.autofmt_xdate(); fig.tight_layout()
            st.pyplot(fig)

            rows = []
            for k in range(h):
                a = r["actual"][k]
                rows.append({"phiên": f"t+{k + 1}", "ngày": str(r["dates"][k].date()) + (" (suy ra)" if r["dates_projected"] and not np.isfinite(a) else ""),
                             "dự báo (USD)": r["forecast"][k], "naive (USD)": r["naive"][k],
                             "thực tế (USD)": a if np.isfinite(a) else None,
                             "sai số dự báo (%)": 100 * (r["forecast"][k] / a - 1) if np.isfinite(a) else None,
                             "sai số naive (%)": 100 * (r["naive"][k] / a - 1) if np.isfinite(a) else None})
            st.dataframe(pd.DataFrame(rows).style.format({c: "{:.2f}" for c in ["dự báo (USD)", "naive (USD)", "thực tế (USD)", "sai số dự báo (%)", "sai số naive (%)"]}, na_rep="chưa có"),
                         width="stretch", hide_index=True)
            if np.isfinite(r["actual"]).any():
                ok = np.isfinite(r["actual"])
                mae_m, mae_n = np.mean(np.abs(r["forecast"][ok] / r["actual"][ok] - 1)) * 100, np.mean(np.abs(r["naive"][ok] / r["actual"][ok] - 1)) * 100
                st.caption(f"Sai số tuyệt đối trung bình trên {int(ok.sum())} phiên đã có giá thực: mô hình **{mae_m:.2f}%**, naive **{mae_n:.2f}%**. "
                           "Chỉ là một mẫu; kết quả tổng thể ở tab *Tổng quan*.")
            else:
                st.caption("Chưa có giá thực tế cho các phiên này: đây là dự báo cho tương lai theo dữ liệu hiện có. "
                           "Các ngày ghi *(suy ra)* được tính theo lịch làm việc, xấp xỉ.")
            with st.expander("Chi tiết huấn luyện và dữ liệu đưa vào"):
                if r["kind"] == "baseline":
                    st.write("Mô hình không cần huấn luyện." + (f" Drift ước lượng từ dữ liệu đến ngày dự báo: {r['info'].get('drift_per_session', 0):+.5f} log-giá mỗi phiên." if "drift_per_session" in r["info"] else ""))
                else:
                    i = r["info"]
                    st.write(f"- Huấn luyện bằng **{i['n_train']}** mẫu (validation {i['n_val']}), nhãn cuối cùng ngày **{i['train_end_date']}**.\n"
                             f"- Weight decay được chọn: {i['chosen_wd']}; epoch tốt nhất: {i['best_epoch']}; seed {r['seed']}.\n"
                             f"- Kênh đầu vào: {', '.join(r['input_columns'])}.\n"
                             f"- Cửa sổ đưa vào: {r['input_dates'][0].date()} đến {r['input_dates'][-1].date()}.\n"
                             "- Không có dữ liệu sau ngày dự báo được dùng để huấn luyện hay dự báo.")

# =============================================================================
# Tab 2: phát lại dự báo đã lưu (backtest)
# =============================================================================
with tab_bt:
    if not Ls:
        st.info("Chưa có kết quả trong results/predictions. Chạy `python run_experiments.py` (xem README.md, bước 4).")
    else:
        st.caption("Phát lại các dự báo ngoài mẫu đã tính sẵn trong thí nghiệm walk-forward (trung bình qua các seed).")
        b1, b2 = st.columns(2)
        L = b1.selectbox("Độ dài cửa sổ quá khứ L (phiên)", Ls, index=min(1, len(Ls) - 1), key="bt_L")
        years = fc.years(L)
        year = b2.selectbox("Năm test", years, index=len(years) - 1, key="bt_year")
        options = list(fc.origins(L, year))
        if st.session_state.get("origin_slider") not in options:
            st.session_state["origin_slider"] = options[len(options) // 2]

        def _random_day():
            st.session_state["origin_slider"] = random.choice(options)

        st.button("Chọn ngày ngẫu nhiên", on_click=_random_day, key="bt_rand")
        origin = st.select_slider("Ngày dự báo (phiên cuối của cửa sổ)", options=options,
                                  format_func=lambda d: d.strftime("%Y-%m-%d"), key="origin_slider")
        available = fc.models(L)
        default = [m for m in ("naive", "nlinear", "hybrid_B", "hybrid_C_usd") if m in available]
        chosen = st.multiselect("Mô hình hiển thị", available, default=default, format_func=label, key="bt_models")

        hist = history_window(df, origin, 30)
        fut = future_dates(df, origin, 5)
        ref = fc.forecast_at(L, "naive", origin)
        if ref is None or len(fut) < 5:
            st.warning("Không có dữ liệu cho ngày này.")
        else:
            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.plot(hist.index, hist.values, color="gray", label="Lịch sử (30 phiên)")
            ax.axvline(origin, color="black", linestyle=":", linewidth=1)
            ax.plot([origin, *fut], [ref["last"], *ref["actual"]], color="black", linewidth=2.5, label="Giá thực tế")
            for m in chosen:
                f = fc.forecast_at(L, m, origin)
                if f is not None:
                    ax.plot([origin, *fut], [f["last"], *f["pred"]], linestyle="--", marker="o", markersize=3,
                            linewidth=1.3, label=label(m))
            ax.set_ylabel("Giá WTI (USD/thùng)")
            ax.grid(alpha=0.3); ax.legend(fontsize=8); fig.autofmt_xdate(); fig.tight_layout()
            st.pyplot(fig)
            st.caption(f"Cửa sổ L={L}, dự báo cho {fut[0].date()} đến {fut[-1].date()}. "
                       f"Giá đóng cuối đã biết tại {origin.date()}: {ref['last']:.2f} USD.")
            table = fc.forecast_table(L, origin, chosen)
            if not table.empty:
                actual_row = {"model": "Giá thực tế", **{f"t+{j + 1}": float(v) for j, v in enumerate(ref["actual"])}}
                show = pd.concat([pd.DataFrame([actual_row]), table.drop(columns="_key")], ignore_index=True)
                st.dataframe(show.style.format({c: "{:.2f}" for c in show.columns if c.startswith("t+")}
                                               | {"sai số TB (%)": "{:.2f}", "sai số t+5 (%)": "{:+.2f}"}, na_rep=""),
                             width="stretch", hide_index=True)
                best = table.loc[table["sai số TB (%)"].idxmin()]
                st.caption(f"Gần giá thực nhất tại ngày này: **{best['model']}** "
                           f"(sai số TB {best['sai số TB (%)']:.2f}%). Đây chỉ là một mẫu, không phải bằng chứng.")

# =============================================================================
# Tab 3: tổng quan
# =============================================================================
with tab_ov:
    if not Ls:
        st.info("Chưa có kết quả. Chạy run_experiments.py rồi run_evaluation.py (xem README.md, bước 4 và 5).")
    else:
        L_ov = st.selectbox("Độ dài cửa sổ L (phiên)", Ls, index=min(1, len(Ls) - 1), key="ov_L")
        st.subheader(f"Tỉ lệ RMSE so với naive (L={L_ov})")
        st.caption("Nhỏ hơn 1 là tốt hơn naive. Fold 2020 bị loại khỏi bảng tổng (một phần năm, chế độ lẫn lộn).")
        avg = load_table(f"summary_avg_L{L_ov}.csv")
        if avg is None:
            st.info("Chạy `python run_evaluation.py` để tạo bảng tổng quan.")
        else:
            avg = avg.assign(model=avg["model"].map(label))
            st.dataframe(avg.style.format({"rmse": "{:.4f}", "mae": "{:.4f}", "dir_acc": "{:.3f}",
                                           "ratio_vs_naive": "{:.4f}"}, na_rep="-"), width="stretch", hide_index=True)
            fig, ax = plt.subplots(figsize=(8, 0.35 * len(avg) + 1))
            ax.barh(avg["model"][::-1], avg["ratio_vs_naive"][::-1] - 1.0, left=1.0)
            ax.axvline(1.0, color="black", linewidth=1)
            ax.set_xlabel("RMSE / RMSE naive"); ax.grid(alpha=0.3, axis="x"); fig.tight_layout()
            st.pyplot(fig)
        by_year = load_table(f"by_year_L{L_ov}.csv")
        if by_year is not None:
            st.subheader("Theo từng năm test")
            piv = by_year.assign(model=by_year["model"].map(label)).pivot(index="model", columns="group",
                                                                          values="ratio_vs_naive")
            st.dataframe(piv.style.format("{:.3f}").background_gradient(cmap="RdYlGn_r", vmin=0.95, vmax=1.2),
                         width="stretch")

# =============================================================================
# Tab 4: giải thích
# =============================================================================
with tab_info:
    st.markdown("""
- **Naive**: dự báo giá 5 phiên tới bằng đúng giá đóng cửa gần nhất.
- **Drift**: naive cộng thêm mức tăng trung bình mỗi phiên ước lượng từ lịch sử.
- **Linear / NLinear / DLinear**: bản đơn biến chỉ dùng giá WTI; hậu tố *đa biến* thêm các biến phụ trợ (spread, USD, OVX, FED) và trộn kênh.
- **Linear**: mỗi giá dự báo là tổng có trọng số của L giá quá khứ, cộng một hằng số.
- **NLinear**: trừ giá cuối khỏi cả cửa sổ, dự báo phần chênh bằng một lớp tuyến tính, rồi cộng lại giá cuối.
- **DLinear**: tách cửa sổ thành xu hướng (trung bình trượt 5 phiên) và phần dư, mỗi phần một lớp tuyến tính.
- **Lai (B)**: trừ giá cuối rồi mới tách xu hướng/phần dư.
- **Lai đa biến (C)**: như B nhưng thêm biến phụ trợ (spread Brent-WTI, USD/CAD và chỉ số USD, OVX, lãi suất FED);
  hậu tố *nhánh mức* nghĩa là có thêm một lớp nhận mức hiện tại của các biến phụ trợ.
- **Hậu tố "khởi tạo 0 + ridge"**: trọng số khởi tạo bằng 0 (mô hình bắt đầu là naive) và weight decay chọn trên validation.
  **Hậu tố "co về naive"**: dự báo = giá cuối + λ·(dự báo mô hình − giá cuối), λ chọn trên validation.

**Ba chế độ của ứng dụng.** Tab *Dự báo bằng mô hình đã lưu* dùng mô hình đã huấn luyện sẵn bằng `train_all.py` (suy luận tức thì; trong mẫu nếu 5 phiên mục tiêu nằm trong dữ liệu huấn luyện). Tab *Huấn luyện đến ngày chọn* huấn luyện mô hình lúc bạn bấm nút và dự báo từ cửa sổ dữ liệu bạn chọn.
Tab *Phát lại dự báo đã lưu* chỉ đọc kết quả thí nghiệm đã tính trước (trung bình 5 seed). Hai tab có thể cho số hơi khác
nhau vì cách chia huấn luyện khác nhau (một mô hình huấn luyện đến đúng ngày chọn, một seed, so với mô hình của từng năm test, 5 seed).

**Cách so sánh.** Sai số tính so với giá thực tế; naive ("giữ nguyên giá hiện tại") là mốc tối thiểu để biết sai số đó tốt hay
dở. Mọi dự báo là dự báo điểm trên log-giá, đổi về USD để hiển thị. Đây là sản phẩm nghiên cứu, không phải khuyến nghị giao dịch.
""")
