"""Look at the data before running any models.

Prints coverage, validation drops (also in results/inspect.log), the variance proxy
in annualized vol terms, and how it compares to VIX.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from src.data import load_clean
from src.features import build_frame

Path("results").mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s",
                    handlers=[logging.FileHandler("results/inspect.log", mode="w"),
                              logging.StreamHandler()])

data = load_clean(C.TICKERS + [C.VIX], C.START)
vix = data.pop(C.VIX)["Close"]

rows = []
for t, px in data.items():
    f = build_frame(px, vix, [22], ticker=t)
    ann_vol = np.sqrt(C.ANN * f["var"])
    rv22 = np.sqrt(C.ANN * f["v_m"])
    rows.append(dict(
        ticker=t, first=f.index.min().date(), last=f.index.max().date(), n=len(f),
        vol_median=ann_vol.median(), vol_p99=ann_vol.quantile(0.99), vol_max=ann_vol.max(),
        max_date=ann_vol.idxmax().date(),
        corr_rv22_vix=rv22.corr(f["vix"] / 100),
        vix_minus_rv22=(f["vix"] / 100 - rv22).mean(),
    ))

with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
    print(pd.DataFrame(rows).to_string(index=False))

spy = build_frame(data["SPY"], vix, [22], ticker="SPY")
yearly = pd.DataFrame({
    "SPY_rv_ann": np.sqrt(C.ANN * spy["var"].groupby(spy.index.year).mean()),
    "VIX_avg": (spy["vix"] / 100).groupby(spy.index.year).mean(),
})
print("\nSPY yearly: realized proxy vs VIX (both annualized, decimal)")
print(yearly.round(3).to_string())
