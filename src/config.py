# Settings used across the project. Change things here, not inside the modules.

# SPDR sector ETFs with history back to Dec 1998. XLRE (2015) and XLC (2018) are
# left out: not enough history for an OOS window that includes 2008.
TICKERS = ["SPY", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB"]
VIX = "^VIX"

START = "1999-01-01"
OOS_START = "2006-01-03"   # first forecast origin, so OOS includes 2008, 2011, 2020, 2022
HORIZONS = [1, 5, 22]      # trading days: day, week, month
REFIT_EVERY = 22           # refit about monthly (expanding window)
ANN = 252

# floor for the daily variance proxy (0.1% daily vol, about 1.6% annualized).
# zero-range days would give log(0) otherwise; run.log counts how many get floored
VAR_FLOOR = 1e-6

# not tuned yet, just conservative. If I tune these it has to be time-series CV
# inside each training window, otherwise it leaks.
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

# pooled model gets ~10x the rows so it can afford bigger trees. Also untuned.
LGBM_POOLED_PARAMS = dict(LGBM_PARAMS, num_leaves=31, min_child_samples=100)
