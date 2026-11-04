# Volatility forecasting: gradient boosting vs. HAR and GARCH

Can a gradient-boosted model (LightGBM) forecast the realized volatility of SPY and the SPDR sector ETFs better than the standard econometric models (HAR-RV, GARCH, GJR-GARCH, EWMA, and VIX for SPY), out of sample, at 1-day, 1-week and 1-month horizons? And if it does, where does the gain come from?

Short answer: mostly no. A linear HAR model with VIX and a leverage term (HARX) is as good or better almost everywhere. LightGBM only catches up at 1 day, and only when it's trained on all ten ETFs pooled together. The ablation in [Results](#results) shows that gain comes from the extra features, not from the trees.

## Run it

```bash
pip install -r requirements.txt
python inspect_data.py    # coverage, validation drops, proxy vs VIX
python -m pytest -q       # lookahead tests, run before trusting any results
python run.py             # SPY, horizons 1/5/22, monthly refits
python run.py --all --refit 63                       # all ten ETFs, quarterly refits (faster)
python run.py --all --refit 63 --pool --pool-only    # pooled LightGBM, skips the slow per-ticker one
python run.py --all --refit 63 --pool --pool-ablation --pool-only   # + the ablation
python run.py --bench HARX                           # DM tests against HARX instead of HAR
```

`--pool` adds a model called `LGBM-pooled`: one LightGBM trained on all the requested tickers at once (ticker as a feature) instead of a separate model per ticker. `--pool-ablation` adds `LGBM-pooled-HARXfeat`, the same pooled model but restricted to HARX's five inputs (log_v_d, log_v_w, log_v_m, ret_neg, log_vix).

Outputs go to `results/`: `metrics.csv` (one row per ticker x horizon x model), `forecasts_<TICKER>.csv` (every forecast next to the realized target) and `run.log` (validation drops, floored days, GARCH fallbacks, timings). These aren't committed, so run the code to regenerate them. Prices and forecasts are cached: `--fresh` ignores the forecast cache and `--refresh` re-downloads the data.

## Design decisions

| Decision | Choice | Why |
|---|---|---|
| Volatility proxy | Squared overnight return + Rogers-Satchell range, from daily OHLC | Proper realized variance needs 5-minute data, which isn't free back to 1999. Range estimators are much less noisy than squared returns. Intraday RV can be swapped in later. |
| Target | Mean daily variance over t+1 … t+h, h ∈ {1, 5, 22} | Next day, next week, next month (22 days is roughly the VIX horizon). |
| Scheme | Expanding window, OOS from January 2006, refit every 22 trading days by default (the results below use 63) | The OOS period covers 2008, 2011, 2015, 2020 and 2022, so it isn't one calm regime. |
| Leakage control | Training row s is used at origin p only if s + h ≤ p | With overlapping multi-day targets, training on the last h rows leaks the future. Tested in `tests/`. |
| Losses | MSE and QLIKE on variance | Both rank forecasts correctly under a noisy proxy (Patton 2011). QLIKE punishes under-forecasting more, which is the expensive mistake for risk and options. |
| Significance | Diebold-Mariano vs. the benchmark, Newey-West SE with lags ≥ h | Overlapping targets make the loss differentials autocorrelated. |
| Calibration | Mincer-Zarnowitz regression (slope 1 = unbiased scale) | A forecast can rank well and still be at the wrong level. |
| Log models | Fit in logs, back-transform with Duan smearing | exp(E[log y]) is biased low and smearing corrects it. |

**Reading the output.** `MSE_rel` and `QLIKE_rel` are a model's average loss divided by the benchmark's (HAR unless you pass `--bench`), so anything below 1 beats it. `DM_t_*` is computed on model loss minus benchmark loss, so a negative t-stat means the model is better, and |t| > 1.96 is significant at 5%.

## Models

- **HAR / logHAR**: OLS of the target on 1-, 5- and 22-day average variance (Corsi 2009). The benchmark to beat.
- **EWMA**: RiskMetrics, λ = 0.94. Nothing to estimate.
- **GARCH(1,1)-t, GJR-GARCH(1,1,1)-t**: on close-to-close returns, using `arch`. GJR adds the leverage effect.
- **VIX** (SPY only): implied variance, flat over the horizon. Should be biased high because of the variance risk premium.
- **HARX**: log-HAR plus negative returns and log VIX. It's linear but gets the key information LightGBM gets, so comparing the two separates "nonlinearity helps" from "VIX helps".
- **LGBM**: LightGBM on the log target with 15 features: HAR terms, daily lags, signed returns, VIX level and change, the implied-minus-realized spread and day of week.
- **LGBM-pooled**: the same LightGBM, but one model for all tickers (ticker as a categorical feature).
- **LGBM-pooled-HARXfeat**: LGBM-pooled restricted to HARX's 5 inputs. This is the ablation.

## Results

All ten tickers (SPY + 9 sector ETFs), forecast origins from January 2006 to October 2026 (about 5,200 days per horizon), refit quarterly. The summary tables are in `results/summary_table*.csv` and the charts are generated from them.

### Does gradient boosting beat the benchmarks?

QLIKE loss relative to HAR, averaged across the ten tickers (lower is better, HAR = 1.000):

| Model | 1-day | 1-week | 1-month |
|---|---|---|---|
| HAR | 1.000 | 1.000 | 1.000 |
| logHAR | 0.955 | 0.911 | 0.930 |
| GJR-GARCH | 0.961 | 0.970 | 1.020 |
| **HARX** (log-HAR + leverage + VIX) | **0.855** | **0.808** | **0.906** |
| LightGBM, per ticker | 0.914 | 0.890 | 1.083 |
| LightGBM, pooled across tickers | 0.849 | 0.825 | 1.020 |

![Model comparison](results/chart1_main_comparison.png)

Per-ticker LightGBM loses to HARX at every horizon. Each per-ticker model trains on about 1,700 rows at the first refit and about 7,000 by the last, which isn't a lot for a boosted model with 15 features. Pooling the ten tickers into one model (ticker as a feature, ~10x the rows) closes most of the gap and edges ahead of HARX at 1 day, but HARX stays clearly ahead at 1 week and 1 month.

### Is the 1-day edge real?

Diebold-Mariano test on QLIKE, pooled LightGBM vs. HARX, 5% level, per ticker:

![Significance by horizon](results/chart3_significance.png)

At 1 day the two are statistically tied on 8 of 10 tickers, and LightGBM is significantly better on 2 (XLK, XLP). HARX doesn't win any. At 1 week HARX wins on 3 tickers and LightGBM on 1 (XLV). At 1 month HARX wins on half the tickers and loses on none. So pooling helps mainly at the shortest horizon.

### Where does the pooled edge come from?

HARX has 5 inputs. Pooled LightGBM has 16: more variance lags, raw and multi-day returns, VIX change, the implied-minus-realized spread, day of week and ticker. The ablation gives the pooled model only HARX's 5 inputs. The information and the amount of data are then the same, and the only difference is trees vs. a linear fit:

| Model | 1-day | 1-week | 1-month |
|---|---|---|---|
| HARX (linear, 5 inputs) | 0.855 | 0.808 | 0.906 |
| LGBM-pooled (trees, 16 inputs) | 0.849 | 0.825 | 1.020 |
| LGBM-pooled-HARXfeat (trees, same 5 inputs as HARX) | 0.869 | 0.845 | 1.021 |

![Ablation](results/chart2_ablation.png)

With the same inputs, the trees do worse than HARX in 28 of 30 ticker-horizon cells, including all ten tickers at 1 week and 1 month (the two exceptions are XLK and XLP at 1 day). They're also worse than the full 16-input pooled model in 25 of 30 cells, and the five exceptions are all at 1 month. So whatever edge pooled LightGBM has over HARX comes from the extra features and lags. Given the same information, the linear model uses it better.

### Takeaway

Per-ticker LightGBM loses to a well-specified linear model. Pooling across tickers fixes most of that at 1 day but not at 1 week or 1 month, and the ablation says the 1-day gain comes from giving the trees more features, not from the trees getting more out of the same ones. Past a few days out I'd use HARX. Boosting only looks worth it for next-day forecasts, with pooled data.

## Known limitations

- The proxy is range-based, not intraday RV. Rankings should hold up (the Patton losses are robust to proxy noise), absolute loss levels less so.
- GARCH models close-to-close variance, while the proxy includes the intraday range. The Mincer-Zarnowitz slope shows any level mismatch.
- Early sector-ETF data (1999–2002) is thinly traded, so there are more zero-range days (see `run.log`). May 6, 2010 and August 24, 2015 have ETF dislocation prints that look like extreme volatility, and they're still in the sample.
- LightGBM hyperparameters are fixed, not tuned. Smearing uses in-sample residuals, which understates the correction for a flexible model (TODO in `src/models.py`).

## Next steps

- Check the HAR coefficients against Corsi (2009).
- Decide what to do with the 2010 and 2015 dislocation days (drop, winsorize or keep).
- SHAP feature importance and performance by regime (VIX terciles, 2008, 2020), to see which features drive the 1-day gain.
- Economic value: vol-target SPY at 10% with each forecast and compare Sharpe and turnover. Also try the SPY VIX-minus-forecast spread as a variance risk premium signal.
- Tune LightGBM with time-series CV inside each training window.

## References

- Corsi (2009), A Simple Approximate Long-Memory Model of Realized Volatility, *Journal of Financial Econometrics*.
- Patton (2011), Volatility Forecast Comparison Using Imperfect Volatility Proxies, *Journal of Econometrics*.
- Rogers & Satchell (1991), Estimating Variance from High, Low and Closing Prices, *Annals of Applied Probability*.
- Glosten, Jagannathan & Runkle (1993), *Journal of Finance* (GJR-GARCH).
- Diebold & Mariano (1995), Comparing Predictive Accuracy, *Journal of Business & Economic Statistics*.
