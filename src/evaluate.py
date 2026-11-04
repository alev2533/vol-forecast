"""Out-of-sample evaluation.

Losses are on variance. MSE and QLIKE both rank forecasts correctly when the true
variance is replaced by a noisy proxy (Patton 2011), which is the situation here.
QLIKE punishes under-forecasting more.

For h > 1 the targets overlap, so loss differentials are autocorrelated. The DM
tests use Newey-West SEs with lags >= h for that reason.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def mse(y, f):
    return (y - f) ** 2


def qlike(y, f):
    ratio = y / f
    return ratio - np.log(ratio) - 1


def _hac_mean_test(d: pd.Series, lags: int):
    """Diebold-Mariano: t-test of mean(d) = 0 with Newey-West SE."""
    res = sm.OLS(d.to_numpy(), np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.tvalues[0]), float(res.pvalues[0])


def mincer_zarnowitz(y, f, lags):
    """y = a + b f + e. Well calibrated means a = 0, b = 1."""
    res = sm.OLS(y.to_numpy(), sm.add_constant(f.to_numpy())).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.params[1]), float(res.rsquared)


def evaluate(frame: pd.DataFrame, forecasts: dict, horizons, bench="HAR") -> pd.DataFrame:
    """forecasts = {model: DataFrame[origin x horizon]}. One output row per (h, model)."""
    rows = []
    for h in horizons:
        y = frame[f"y_{h}"]
        F = pd.DataFrame({m: fc[h] for m, fc in forecasts.items()})
        data = pd.concat([y.rename("y"), F], axis=1).dropna()  # common sample only
        y_, lags = data["y"], max(h, 5)
        L = {m: (mse(y_, data[m]), qlike(y_, data[m])) for m in F.columns}
        b_mse, b_ql = L[bench]
        for m in F.columns:
            m_mse, m_ql = L[m]
            t_mse, p_mse = (np.nan, np.nan) if m == bench else _hac_mean_test(m_mse - b_mse, lags)
            t_ql, p_ql = (np.nan, np.nan) if m == bench else _hac_mean_test(m_ql - b_ql, lags)
            mz_b, mz_r2 = mincer_zarnowitz(y_, data[m], lags)
            rows.append(dict(
                h=h, model=m, n=len(data),
                MSE_rel=m_mse.mean() / b_mse.mean(),
                QLIKE=m_ql.mean(), QLIKE_rel=m_ql.mean() / b_ql.mean(),
                DM_t_MSE=t_mse, DM_p_MSE=p_mse, DM_t_QLIKE=t_ql, DM_p_QLIKE=p_ql,
                MZ_slope=mz_b, MZ_R2=mz_r2,
            ))
    return pd.DataFrame(rows)
