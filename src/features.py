"""Volatility proxy, forecast targets and model features.

Timing convention (the whole project depends on this):
  - Row t holds information known at the CLOSE of day t.
  - The target at row t is the average daily variance over days t+1 ... t+h.
  - Nothing on row t may use prices from day t+1 or later, except the target.
"""
import logging

import numpy as np
import pandas as pd

from .config import VAR_FLOOR

log = logging.getLogger(__name__)


def daily_variance(df: pd.DataFrame, ticker: str = "") -> pd.Series:
    """Daily variance proxy = squared overnight return + Rogers-Satchell intraday range.

    Rogers-Satchell is unbiased for Brownian motion with or without drift and uses
    the full OHLC bar; adding the overnight gap captures close-to-open jumps that a pure
    intraday range estimator misses. This is a stand-in for 5-minute realized variance,
    which needs intraday data we do not have for free over this sample.
    """
    o, h, l, c = (np.log(df[k]) for k in ["Open", "High", "Low", "Close"])
    overnight = (o - c.shift(1)) ** 2
    rs = (h - c) * (h - o) + (l - c) * (l - o)
    var = overnight + rs
    n_floor = int((var < VAR_FLOOR).sum())
    if n_floor:
        log.info("%s: %d days floored at VAR_FLOOR=%g", ticker, n_floor, VAR_FLOOR)
    return var.clip(lower=VAR_FLOOR).rename("var")


def forward_mean(v: pd.Series, h: int) -> pd.Series:
    """Mean of v over t+1 ... t+h, stored at row t (NaN for the last h rows)."""
    return v.shift(-1).rolling(h).mean().shift(-(h - 1))


def build_frame(df: pd.DataFrame, vix: pd.Series, horizons, ticker: str = "") -> pd.DataFrame:
    """One row per trading day: proxy, returns, targets for each horizon, features."""
    out = pd.DataFrame(index=df.index)
    out["var"] = daily_variance(df, ticker)
    out["ret"] = np.log(df["Close"]).diff()

    # HAR components: daily, weekly (5d) and monthly (22d) average variance
    out["v_d"] = out["var"]
    out["v_w"] = out["var"].rolling(5).mean()
    out["v_m"] = out["var"].rolling(22).mean()

    for h in horizons:
        out[f"y_{h}"] = forward_mean(out["var"], h)

    # VIX aligned to the ETF calendar; forward-fill at most one missing day
    vix = vix.reindex(out.index).ffill(limit=1)
    out["vix"] = vix

    # ML features (all known at close of t)
    f = pd.DataFrame(index=out.index)
    f["log_v_d"] = np.log(out["v_d"])
    f["log_v_w"] = np.log(out["v_w"])
    f["log_v_m"] = np.log(out["v_m"])
    for k in range(1, 5):
        f[f"log_v_d_lag{k}"] = f["log_v_d"].shift(k)
    f["ret"] = out["ret"]
    f["ret_neg"] = out["ret"].clip(upper=0)             # leverage effect
    f["ret_w"] = out["ret"].rolling(5).sum()
    f["ret_m"] = out["ret"].rolling(22).sum()
    f["log_vix"] = np.log(vix)
    f["d_log_vix_5"] = f["log_vix"].diff(5)
    implied_daily_var = (vix / 100) ** 2 / 252
    f["log_vrp"] = np.log(implied_daily_var) - f["log_v_m"]  # implied minus realized
    f["dow"] = out.index.dayofweek
    f.columns = [f"x_{c}" for c in f.columns]

    return out.join(f)


def feature_cols(frame: pd.DataFrame) -> list:
    return [c for c in frame.columns if c.startswith("x_")]
