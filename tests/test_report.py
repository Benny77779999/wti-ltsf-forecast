import numpy as np
import pytest

from src.report import (by_group, common_years, default_pairs, dm_table, load_predictions, pool_all,
                        summarize)

T = 5


def write(path, origin, pred, actual, last):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, origin=origin.astype("datetime64[ns]").astype("int64"),
                        pred_log=pred, actual_log=actual, last_log=last)


@pytest.fixture
def pred_dir(tmp_path):
    rng = np.random.default_rng(0)
    for year in (2017, 2018, 2020):
        n = 300
        origin = np.arange(np.datetime64(f"{year}-01-02"), np.datetime64(f"{year}-01-02") + n)
        last = rng.normal(4, 0.1, n)
        actual = last[:, None] + rng.normal(0, 0.05, (n, T)).cumsum(axis=1)
        write(tmp_path / "L30" / f"naive__{year}__s0.npz", origin, np.repeat(last[:, None], T, 1), actual, last)
        for seed in (0, 1):
            good = actual + rng.normal(0, 0.01, (n, T))              # "mô hình tốt" giả lập
            write(tmp_path / "L30" / f"hybrid_B__{year}__s{seed}.npz", origin, good, actual, last)
            write(tmp_path / "L30" / f"linear__{year}__s{seed}.npz", origin,
                  last[:, None] + 0.8 * (actual - last[:, None]), actual, last)
    return tmp_path


def test_load_and_common_years(pred_dir):
    data = load_predictions(pred_dir)
    assert set(data) == {30} and set(data[30]) == {"naive", "hybrid_B", "linear"}
    assert common_years(data[30]) == [2017, 2018, 2020]
    del data[30]["linear"][1][2018]                                  # thiếu một kết quả
    assert common_years(data[30]) == [2017, 2020]


def test_summary_ratio_and_exclusion(pred_dir):
    models = load_predictions(pred_dir)[30]
    pooled = pool_all(models, [2017, 2018])                          # loại 2020
    assert len(pooled["naive"][0]["origin"]) == 600
    by_tau, avg = summarize(pooled)
    r = avg.set_index("model")["ratio_vs_naive"]
    assert r["naive"] == pytest.approx(1.0) and r["hybrid_B"] < r["linear"] < 1.0
    assert avg.iloc[0]["model"] == "hybrid_B"                         # sắp theo tỉ lệ tăng dần
    assert set(by_tau["tau"]) == {1, 2, 3, 4, 5}


def test_dm_pairs_and_verdict(pred_dir):
    pooled = pool_all(load_predictions(pred_dir)[30], [2017, 2018])
    _, avg = summarize(pooled)
    pairs = default_pairs(list(pooled), avg)
    assert ("hybrid_B", "naive") in pairs and ("hybrid_B", "linear") in pairs
    dm = dm_table(pooled, pairs)
    row = dm[(dm.model == "hybrid_B") & (dm.reference == "naive") & (dm.tau == 3)].iloc[0]
    assert row["p_value"] < 0.001 and row["mean_diff"] < 0 and row["verdict"] == "mô hình tốt hơn"


def test_by_group_includes_excluded_year_when_asked(pred_dir):
    models = load_predictions(pred_dir)[30]
    g = by_group(pool_all(models, [2017, 2018, 2020]), {"2020": [2020], "2017-18": [2017, 2018]})
    assert set(g["group"]) == {"2020", "2017-18"}
    naive = g[(g.group == "2020") & (g.model == "naive")].iloc[0]
    assert naive["ratio_vs_naive"] == pytest.approx(1.0) and naive["n_samples"] == 300


def test_mismatched_origins_are_rejected(pred_dir):
    models = load_predictions(pred_dir)[30]
    models["linear"][0][2017]["origin"] = models["linear"][0][2017]["origin"] + np.timedelta64(1, "D")
    with pytest.raises(ValueError):
        pool_all(models, [2017])
