"""Project-wide settings. Change things here, not inside the modules."""

# SPDR sector ETFs that trade since Dec 1998 (XLRE 2015 and XLC 2018 are excluded:
# too little history for an out-of-sample window that includes 2008).
TICKERS = ["SPY", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB"]
VIX = "^VIX"

START = "1999-01-01"       # first download date
OOS_START = "2006-01-03"   # first forecast origin; OOS covers 2008, 2011, 2020, 2022
HORIZONS = [1, 5, 22]      # trading days ahead: day, week, month
REFIT_EVERY = 22           # re-estimate every model roughly monthly (expanding window)
ANN = 252                  # trading days per year

# Floor for the daily variance proxy (daily vol of 0.1% ~ 1.6% annualized).
# Zero-range days would otherwise give log(0). Floored days are counted and logged.
VAR_FLOOR = 1e-6

# Fixed, conservative LightGBM settings for the first pass. Tune later with
# time-series CV *inside* each training window only.
LGBM_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    verbose=-1,
)

# Pooled LightGBM (all tickers, one model) sees ~10x the rows of a per-ticker fit,
# so it can support a few more leaves per tree without overfitting as fast. Still a
# first-pass guess, not tuned — see README step 4.
LGBM_POOLED_PARAMS = dict(LGBM_PARAMS, num_leaves=31, min_child_samples=100)
