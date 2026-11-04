"""Forecasting models, all run through the same expanding-window walk-forward.

Every function returns a DataFrame indexed by forecast origin (OOS dates only) with
one column per horizon h: the forecast of mean daily variance over t+1 ... t+h.

Leakage rule for training: at origin p, a training row s is usable only if its
target window ends by p, i.e. s + h <= p. So training rows are iloc[: p - h + 1].
"""
import logging
import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from lightgbm import LGBMRegressor

from .config import LGBM_PARAMS, LGBM_POOLED_PARAMS
from .features import feature_cols

log = logging.getLogger(__name__)


def blocks(index: pd.DatetimeIndex, oos_start, refit_every: int):
    """Yield (p, q) positional bounds of each refit block of forecast origins."""
    p0 = index.searchsorted(pd.Timestamp(oos_start))
    for p in range(p0, len(index), refit_every):
        yield p, min(p + refit_every, len(index))


def train_end(p: int, h: int) -> int:
    """Exclusive positional end of the training set for origin p, horizon h."""
    return p - h + 1


def _empty(frame, oos_start, horizons):
    idx = frame.index[frame.index >= pd.Timestamp(oos_start)]
    return pd.DataFrame(index=idx, columns=horizons, dtype=float)


# ---------------------------------------------------------------- HAR family
def har(frame, horizons, oos_start, refit_every, log_space=False):
    """HAR-RV (Corsi 2009). log_space=True fits log target on log components and
    converts back with Duan's smearing factor so the forecast is not biased low."""
    xcols = ["v_d", "v_w", "v_m"]
    out = _empty(frame, oos_start, horizons)
    for h in horizons:
        ycol = f"y_{h}"
        for p, q in blocks(frame.index, oos_start, refit_every):
            tr = frame.iloc[: train_end(p, h)][xcols + [ycol]].dropna()
            X, y = tr[xcols].to_numpy(), tr[ycol].to_numpy()
            if log_space:
                X, y = np.log(X), np.log(y)
            X = np.column_stack([np.ones(len(X)), X])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)

            Xo = frame.iloc[p:q][xcols].to_numpy()
            if log_space:
                Xo = np.log(Xo)
            pred = np.column_stack([np.ones(len(Xo)), Xo]) @ beta
            if log_space:
                smear = np.mean(np.exp(y - X @ beta))
                pred = np.exp(pred) * smear
            else:
                pred = np.maximum(pred, 1e-8)  # level HAR can go negative
            out.iloc[out.index.get_indexer(frame.index[p:q]), out.columns.get_loc(h)] = pred
    return out


HARX_COLS = ["x_log_v_d", "x_log_v_w", "x_log_v_m", "x_ret_neg", "x_log_vix"]


def har_x(frame, horizons, oos_start, refit_every, xcols=HARX_COLS):
    """Log-HAR plus leverage (negative return) and log VIX: a LINEAR model with the
    key information LightGBM gets. LGBM vs. HAR-X isolates nonlinearity from data."""
    out = _empty(frame, oos_start, horizons)
    for h in horizons:
        ycol = f"y_{h}"
        for p, q in blocks(frame.index, oos_start, refit_every):
            tr = frame.iloc[: train_end(p, h)][xcols + [ycol]].dropna()
            X = np.column_stack([np.ones(len(tr)), tr[xcols].to_numpy()])
            y = np.log(tr[ycol].to_numpy())
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            smear = np.mean(np.exp(y - X @ beta))
            Xo = frame.iloc[p:q][xcols].ffill().to_numpy()
            pred = np.exp(np.column_stack([np.ones(len(Xo)), Xo]) @ beta) * smear
            out.iloc[out.index.get_indexer(frame.index[p:q]), out.columns.get_loc(h)] = pred
    return out


# ---------------------------------------------------------------- no-fit baselines
def ewma(frame, horizons, oos_start, lam=0.94):
    """RiskMetrics EWMA on squared close-to-close returns; flat term structure."""
    r2 = frame["ret"].fillna(0.0).to_numpy() ** 2
    s2 = np.empty_like(r2)
    s2[0] = r2[1:23].mean()
    for t in range(1, len(r2)):
        s2[t] = lam * s2[t - 1] + (1 - lam) * r2[t]
    s2 = pd.Series(s2, index=frame.index)
    out = _empty(frame, oos_start, horizons)
    for h in horizons:
        out[h] = s2.reindex(out.index)
    return out


def vix_implied(frame, horizons, oos_start):
    """VIX converted to daily variance. VIX is a 30-calendar-day risk-neutral
    expectation, so it embeds a variance risk premium and should be biased high."""
    iv = (frame["vix"] / 100) ** 2 / 252
    out = _empty(frame, oos_start, horizons)
    for h in horizons:
        out[h] = iv.reindex(out.index)
    return out


# ---------------------------------------------------------------- GARCH family
def garch(frame, horizons, oos_start, refit_every, asym=False):
    """GARCH(1,1) or GJR-GARCH(1,1,1) with Student-t errors on close-to-close returns.

    Parameters are re-estimated each block on data strictly before the block;
    within the block, the variance filter updates daily with the fixed parameters.
    """
    r = (100 * frame["ret"]).dropna()
    H = max(horizons)
    out = _empty(frame, oos_start, horizons)
    prev_params = None
    for p, q in blocks(frame.index, oos_start, refit_every):
        start, end = frame.index[p], frame.index[q - 1]
        am = arch_model(r.loc[:end], mean="Constant", vol="GARCH",
                        p=1, o=1 if asym else 0, q=1, dist="t")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = am.fit(last_obs=start, disp="off")
            if res.convergence_flag != 0:
                raise RuntimeError(f"convergence flag {res.convergence_flag}")
            prev_params = res.params
        except Exception as e:  # skip-and-log: reuse last good parameters
            if prev_params is None:
                log.warning("GARCH block %s failed with no fallback: %s", start.date(), e)
                continue
            log.warning("GARCH block %s failed (%s); reusing previous params", start.date(), e)
            res = am.fix(prev_params, last_obs=start)
        fc = res.forecast(horizon=H, start=start, reindex=False).variance / 1e4
        fc = fc.loc[start:end]
        for h in horizons:
            out.loc[fc.index, h] = fc.iloc[:, :h].mean(axis=1).to_numpy()
    return out


# ---------------------------------------------------------------- gradient boosting
def lgbm(frame, horizons, oos_start, refit_every, params=None, return_models=False):
    """LightGBM on log target with the full feature set; Duan smearing back to levels."""
    params = params or LGBM_PARAMS
    xcols = feature_cols(frame)
    out = _empty(frame, oos_start, horizons)
    last_models = {}
    for h in horizons:
        ycol = f"y_{h}"
        for p, q in blocks(frame.index, oos_start, refit_every):
            tr = frame.iloc[: train_end(p, h)].dropna(subset=[ycol, "x_log_v_m"])
            X, y = tr[xcols], np.log(tr[ycol])
            m = LGBMRegressor(**params, random_state=0).fit(X, y)
            # TODO: in-sample residuals of a boosted model are too small, so this
            # smearing factor is biased toward 1. Estimate it on a held-out tail instead.
            smear = np.mean(np.exp(y - m.predict(X)))
            pred = np.exp(m.predict(frame.iloc[p:q][xcols])) * smear
            out.iloc[out.index.get_indexer(frame.index[p:q]), out.columns.get_loc(h)] = pred
            last_models[h] = m
    return (out, last_models) if return_models else out


# ---------------------------------------------------------------- pooled gradient boosting
def lgbm_pooled(frames, horizons, oos_start, refit_every, params=None, feature_subset=None):
    """One LightGBM model trained on ALL tickers at once, ticker as a categorical feature,
    instead of a separate model per ticker. ~10x the training rows per fit, and the model
    can learn whether patterns (e.g. the leverage effect) are stronger in some sectors.

    frames: {ticker: frame}, every frame built by features.build_frame on the SAME dates
    (true for ETFs on a shared trading calendar; the code below trusts that length and
    index match across tickers and raises if they don't, rather than silently misaligning).

    feature_subset: restrict the model to exactly these x_ columns instead of the full
    feature set (ticker is always added on top). Pass HARX_COLS to run the ablation that
    isolates nonlinearity from information: same inputs HARX gets, same pooled data, but
    a tree model instead of a linear one. If pooled LGBM's edge over HARX survives this
    restriction, the edge is the nonlinearity; if it vanishes, the edge was extra features
    (VIX, leverage) combined with more data, not the model class itself.

    Returns {ticker: forecast DataFrame}, same shape as calling lgbm() once per ticker.
    """
    params = params or LGBM_POOLED_PARAMS
    tickers = list(frames)
    ref = frames[tickers[0]]
    for t in tickers[1:]:
        if not frames[t].index.equals(ref.index):
            raise ValueError(f"{t}'s calendar does not match {tickers[0]}'s; pooling "
                             f"requires identical trading dates across tickers")
    base_cols = feature_subset or feature_cols(ref)
    xcols = base_cols + ["x_ticker"]
    outs = {t: _empty(frames[t], oos_start, horizons) for t in tickers}
    cats = pd.CategoricalDtype(categories=tickers)
    nblocks = sum(1 for _ in blocks(ref.index, oos_start, refit_every))

    for h in horizons:
        ycol = f"y_{h}"
        for i, (p, q) in enumerate(blocks(ref.index, oos_start, refit_every)):
            parts = []
            for t in tickers:
                tr = frames[t].iloc[: train_end(p, h)].dropna(subset=[ycol, "x_log_v_m"]).copy()
                tr["x_ticker"] = pd.Categorical([t] * len(tr), dtype=cats)
                parts.append(tr)
            pool = pd.concat(parts, ignore_index=True)
            X, y = pool[xcols], np.log(pool[ycol])
            m = LGBMRegressor(**params, random_state=0).fit(
                X, y, categorical_feature=["x_ticker"])
            smear = np.mean(np.exp(y - m.predict(X)))

            for t in tickers:
                Xo = frames[t].iloc[p:q][base_cols].copy()
                Xo["x_ticker"] = pd.Categorical([t] * len(Xo), dtype=cats)
                pred = np.exp(m.predict(Xo[xcols])) * smear
                outs[t].iloc[outs[t].index.get_indexer(frames[t].index[p:q]),
                            outs[t].columns.get_loc(h)] = pred
            if (i + 1) % 10 == 0 or i + 1 == nblocks:
                log.info("pooled LGBM h=%d: block %d/%d done (%d training rows)",
                         h, i + 1, nblocks, len(pool))
    return outs
