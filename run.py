"""End-to-end run: data -> features -> walk-forward forecasts -> evaluation.

Examples
    python run.py                         # SPY only, default horizons
    python run.py --tickers SPY XLF XLE   # a few ETFs
    python run.py --all --refit 63        # every ETF, quarterly refits (faster)
    python run.py --all --refit 63 --pool # + one LightGBM trained on all tickers together
    python run.py --all --refit 63 --pool --pool-only   # skip the slow per-ticker LGBM
"""
import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from src import config as C
from src.data import load_clean
from src.evaluate import evaluate
from src.features import build_frame
from src import models as M

RESULTS = Path(__file__).resolve().parent / "results"


def setup_logging():
    RESULTS.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(RESULTS / "run.log", mode="w"), logging.StreamHandler()],
    )


CACHE = RESULTS / "cache"


def cached(path, fresh, fn, log, label):
    """Generic pickle cache. Works for a single DataFrame (per-ticker models) or a dict
    of them (lgbm_pooled's {ticker: DataFrame}) — pd.to_pickle/read_pickle handle both."""
    if path.exists() and not fresh:
        log.info("%s loaded from cache", label)
        return pd.read_pickle(path)
    t0 = time.time()
    out = fn()
    pd.to_pickle(out, path)
    log.info("%s done in %.1fs", label, time.time() - t0)
    return out


def run_ticker(t, frame, horizons, refit, fresh=False, bench="HAR", pooled_fc=None,
               pool_only=False, ablation_fc=None):
    """Forecasts are cached per (ticker, model, refit, horizons) so you can iterate on one
    model without re-running the others. Use --fresh after changing data or features.
    pooled_fc, if given, is this ticker's forecast from the pooled LightGBM (see main());
    it is added as an extra model called "LGBM-pooled" alongside everything else.
    ablation_fc, if given, is added as "LGBM-pooled-HARXfeat" — pooled LightGBM restricted
    to exactly HARX's inputs, to test whether the pooled LGBM edge is the nonlinearity or
    the extra features."""
    log = logging.getLogger("run")
    args = (frame, horizons, C.OOS_START)
    CACHE.mkdir(parents=True, exist_ok=True)
    tag = f"r{refit}_h{'-'.join(map(str, horizons))}"
    fc = {}
    models = [
        ("HAR", lambda: M.har(*args, refit)),
        ("logHAR", lambda: M.har(*args, refit, log_space=True)),
        ("HARX", lambda: M.har_x(*args, refit)),
        ("EWMA", lambda: M.ewma(*args)),
        ("GARCH", lambda: M.garch(*args, refit)),
        ("GJR", lambda: M.garch(*args, refit, asym=True)),
    ]
    if not pool_only:
        models.append(("LGBM", lambda: M.lgbm(*args, refit)))
    if t == "SPY":  # VIX is implied vol of the S&P 500: a fair benchmark only for SPY
        models.append(("VIX", lambda: M.vix_implied(*args)))
    for name, fn in models:
        fc[name] = cached(CACHE / f"{t}_{name}_{tag}.pkl", fresh, fn, log, f"{t} {name}")
    if pooled_fc is not None:
        fc["LGBM-pooled"] = pooled_fc
    if ablation_fc is not None:
        fc["LGBM-pooled-HARXfeat"] = ablation_fc

    wide = pd.concat({m: f for m, f in fc.items()}, axis=1)
    wide.columns = [f"{m}_h{h}" for m, h in wide.columns]
    wide.join(frame[[f"y_{h}" for h in horizons]]).to_csv(RESULTS / f"forecasts_{t}.csv")

    metrics = evaluate(frame, fc, horizons, bench=bench)
    metrics.insert(0, "ticker", t)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["SPY"])
    ap.add_argument("--all", action="store_true", help="run every ticker in config")
    ap.add_argument("--horizons", nargs="+", type=int, default=C.HORIZONS)
    ap.add_argument("--refit", type=int, default=C.REFIT_EVERY)
    ap.add_argument("--refresh", action="store_true", help="re-download data")
    ap.add_argument("--fresh", action="store_true", help="ignore cached forecasts")
    ap.add_argument("--bench", default="HAR", help="model the DM tests compare against, e.g. HARX")
    ap.add_argument("--pool", action="store_true",
                    help="also fit one LightGBM across all requested tickers together "
                         "(ticker as a feature), in addition to the per-ticker LGBM")
    ap.add_argument("--pool-only", action="store_true",
                    help="with --pool, skip the slow per-ticker LGBM entirely")
    ap.add_argument("--pool-ablation", action="store_true",
                    help="also fit pooled LightGBM restricted to exactly HARX's inputs "
                         "(log_v_d, log_v_w, log_v_m, ret_neg, log_vix) — isolates whether "
                         "pooled LGBM's edge is the nonlinearity or the extra features")
    a = ap.parse_args()
    if a.pool_only and not a.pool:
        ap.error("--pool-only requires --pool")
    if a.pool_ablation and not a.pool:
        ap.error("--pool-ablation requires --pool")

    setup_logging()
    CACHE.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("run")
    tickers = C.TICKERS if a.all else a.tickers
    data = load_clean(tickers + [C.VIX], C.START, refresh=a.refresh)
    vix = data.pop(C.VIX)["Close"]
    frames = {t: build_frame(data[t], vix, a.horizons, ticker=t) for t in tickers}

    pooled = {}
    if a.pool:
        if len(tickers) < 2:
            log.warning("--pool with a single ticker is the same as the per-ticker LGBM; skipping")
        else:
            tag = f"POOLED_{'-'.join(sorted(tickers))}_r{a.refit}_h{'-'.join(map(str, a.horizons))}"
            pooled = cached(
                CACHE / f"{tag}.pkl", a.fresh,
                lambda: M.lgbm_pooled(frames, a.horizons, C.OOS_START, a.refit), log, tag)
            # cached() stores a single pickled object; lgbm_pooled returns a dict of
            # DataFrames (one per ticker), which pickles and unpickles as one object fine.

    ablation = {}
    if a.pool_ablation:
        atag = f"POOLED-ABL_{'-'.join(sorted(tickers))}_r{a.refit}_h{'-'.join(map(str, a.horizons))}"
        ablation = cached(
            CACHE / f"{atag}.pkl", a.fresh,
            lambda: M.lgbm_pooled(frames, a.horizons, C.OOS_START, a.refit,
                                  feature_subset=M.HARX_COLS),
            log, atag)

    all_metrics = pd.concat([
        run_ticker(t, frames[t], a.horizons, a.refit, a.fresh, a.bench,
                  pooled_fc=pooled.get(t), pool_only=a.pool_only,
                  ablation_fc=ablation.get(t))
        for t in tickers
    ])
    all_metrics.to_csv(RESULTS / "metrics.csv", index=False)

    show = all_metrics[["ticker", "h", "model", "n", "MSE_rel", "QLIKE_rel",
                        "DM_t_QLIKE", "DM_p_QLIKE", "MZ_slope", "MZ_R2"]]
    with pd.option_context("display.width", 140, "display.float_format", "{:.3f}".format):
        print(show.to_string(index=False))


if __name__ == "__main__":
    main()
