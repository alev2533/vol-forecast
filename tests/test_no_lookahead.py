"""Run with: python -m pytest -q
These tests protect the one thing that silently ruins a forecasting project: lookahead."""
import numpy as np
import pandas as pd

from src.features import build_frame, feature_cols, forward_mean
from src.models import train_end


def _fake_ohlc(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", periods=n)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    o = c * np.exp(rng.normal(0, 0.003, n))
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.005, n)))
    l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.005, n)))
    px = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c}, index=idx)
    vix = pd.Series(20 + rng.normal(0, 1, n), index=idx)
    return px, vix


def test_forward_mean_alignment():
    v = pd.Series(np.arange(10, dtype=float))
    fm = forward_mean(v, 3)
    assert fm.iloc[0] == np.mean([1, 2, 3])   # row 0 averages days 1..3
    assert fm.iloc[6] == np.mean([7, 8, 9])
    assert fm.iloc[7:].isna().all()             # not enough future days


def test_features_ignore_the_future():
    px, vix = _fake_ohlc()
    base = build_frame(px, vix, [5])
    cut = 200
    px2, vix2 = px.copy(), vix.copy()
    px2.iloc[cut + 1:] *= 3.0                   # wreck every price after day `cut`
    vix2.iloc[cut + 1:] *= 3.0
    shocked = build_frame(px2, vix2, [5])
    cols = feature_cols(base) + ["v_d", "v_w", "v_m"]
    pd.testing.assert_frame_equal(base[cols].iloc[: cut + 1], shocked[cols].iloc[: cut + 1])


def test_training_targets_end_before_origin():
    for h in [1, 5, 22]:
        for p in [100, 250, 1000]:
            last_train_row = train_end(p, h) - 1
            assert last_train_row + h <= p      # its target window ends by the origin
