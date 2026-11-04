"""Download, cache and validate daily OHLC data.

Validation follows a skip-and-log pattern: rows that fail a check are dropped,
and every drop is counted and written to the log, so nothing is silently accepted.
"""
import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
OHLC = ["Open", "High", "Low", "Close"]


def _cache_path(ticker: str) -> Path:
    return RAW / f"{ticker.replace('^', '')}.csv"


def download(tickers, start, refresh=False) -> dict:
    """Return {ticker: DataFrame}. Prices are split- and dividend-adjusted
    (auto_adjust=True), so overnight gaps on ex-dividend days are not fake jumps."""
    RAW.mkdir(parents=True, exist_ok=True)
    out = {}
    for t in tickers:
        f = _cache_path(t)
        if f.exists() and not refresh:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            log.info("%s: loaded %d rows from cache", t, len(df))
        else:
            df = yf.download(t, start=start, auto_adjust=True, progress=False,
                             multi_level_index=False)
            if df.empty:
                raise RuntimeError(f"{t}: download returned no rows")
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.to_csv(f)
            log.info("%s: downloaded %d rows (%s to %s)", t, len(df),
                     df.index.min().date(), df.index.max().date())
        out[t] = df
    return out


def validate(df: pd.DataFrame, ticker: str, rel_tol: float = 1e-6) -> pd.DataFrame:
    """Drop rows that fail basic OHLC sanity checks; log what was dropped and why."""
    n0 = len(df)
    df = df.sort_index()

    dupes = df.index.duplicated(keep="first")
    if dupes.any():
        log.warning("%s: dropping %d duplicate dates", ticker, dupes.sum())
        df = df[~dupes]

    # A download during market hours includes a partial bar for today.
    today_ny = datetime.now(ZoneInfo("America/New_York")).date()
    if len(df) and df.index[-1].date() == today_ny:
        log.info("%s: dropping today's bar (%s), session may be incomplete", ticker, today_ny)
        df = df.iloc[:-1]

    px = df[OHLC]
    oc_max = px[["Open", "Close"]].max(axis=1)
    oc_min = px[["Open", "Close"]].min(axis=1)
    checks = {
        "missing_ohlc": px.isna().any(axis=1),
        "non_positive_price": (px <= 0).any(axis=1),
        "high_below_low": px["High"] < px["Low"],
        "high_below_open_close": px["High"] < oc_max * (1 - rel_tol),
        "low_above_open_close": px["Low"] > oc_min * (1 + rel_tol),
    }
    bad = pd.Series(False, index=df.index)
    for name, mask in checks.items():
        if mask.any():
            first = mask[mask].index[:3].strftime("%Y-%m-%d").tolist()
            log.warning("%s: %d rows fail %s (e.g. %s)", ticker, mask.sum(), name, first)
        bad |= mask

    df = df[~bad]
    log.info("%s: kept %d of %d rows after validation", ticker, len(df), n0)
    return df


def load_clean(tickers, start, refresh=False) -> dict:
    raw = download(tickers, start, refresh=refresh)
    return {t: validate(df, t) for t, df in raw.items()}
