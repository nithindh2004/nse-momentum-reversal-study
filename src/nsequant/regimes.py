"""Market volatility regimes, classified with past data only.

At each date t:
    rv_t   = annualised std of NIFTY 500 daily returns over the last `window` sessions
    regime = LOW / MID / HIGH according to where rv_t sits relative to the
             terciles of ALL rv values observed up to t (expanding window,
             at least `min_obs` observations).

Using an expanding window (not full-sample terciles) is what keeps this free
of look-ahead: in 2011 we do not know how calm 2017 or how wild 2020 will be.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LABELS = np.array(["LOW", "MID", "HIGH"])


def realized_vol(index_close: pd.Series, window: int = 21) -> pd.Series:
    r = index_close.pct_change(fill_method=None)
    return r.rolling(window).std() * np.sqrt(252)


def expanding_tercile_regime(x: pd.Series, min_obs: int = 252) -> pd.Series:
    vals = x.to_numpy(dtype=float)
    out = np.full(len(vals), None, dtype=object)
    history: list[float] = []
    for i, v in enumerate(vals):
        if np.isnan(v):
            continue
        history.append(v)
        if len(history) >= min_obs:
            lo, hi = np.quantile(history, [1 / 3, 2 / 3])
            out[i] = LABELS[0] if v <= lo else (LABELS[2] if v > hi else LABELS[1])
    return pd.Series(out, index=x.index, name="regime")


def market_regimes(index_close: pd.DataFrame, source: str = "NIFTY500", window: int = 21) -> pd.DataFrame:
    if source == "INDIAVIX":
        level = index_close["INDIAVIX"] / 100
    else:
        level = realized_vol(index_close[source], window)
    return pd.DataFrame({"vol": level, "regime": expanding_tercile_regime(level)})
