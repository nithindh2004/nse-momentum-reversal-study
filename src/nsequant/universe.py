"""Point-in-time investable universe.

At each decision date t the universe is the `size` most liquid stocks by
median daily traded value (close x volume) over the previous `liq_window`
sessions, among stocks that
    * traded on t (have a price on t),
    * have at least `min_history` sessions of price history before t
      (needed for a 12-month momentum signal), and
    * traded on >= 90% of the last 252 sessions (no long suspensions).

Everything uses data up to and including t only. This replaces "today's NIFTY
100 members", which would leak survivorship bias into the past.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Panel


def rebalance_dates(calendar: pd.DatetimeIndex, freq: str = "M") -> pd.DatetimeIndex:
    """Last trading session of each month ("M") or week ("W")."""
    s = pd.Series(calendar, index=calendar)
    key = calendar.to_period("M" if freq == "M" else "W-FRI")
    return pd.DatetimeIndex(s.groupby(key).max().values)


def liquid_universe(panel: Panel, dates: pd.DatetimeIndex, size: int = 100,
                    liq_window: int = 126, min_history: int = 273) -> pd.DataFrame:
    px = panel.adj_close
    has_px = px.notna()
    tv = panel.traded_value.where(has_px)

    liq = tv.rolling(liq_window, min_periods=liq_window // 2).median()
    history = has_px.cumsum()                                  # sessions with a price so far
    coverage = has_px.rolling(252, min_periods=1).mean()

    eligible = has_px & (history >= min_history) & (coverage >= 0.9)
    score = liq.where(eligible).loc[dates]
    rank = score.rank(axis=1, ascending=False, method="first")
    return (rank <= size).fillna(False)


def fixed_list_universe(panel: Panel, dates: pd.DatetimeIndex, symbols: list[str],
                        min_history: int = 273) -> pd.DataFrame:
    """Survivorship-biased universe: a list chosen with hindsight (e.g. today's NIFTY 100)."""
    px = panel.adj_close
    has_px = px.notna()
    history = has_px.cumsum()
    members = pd.DataFrame(False, index=px.index, columns=px.columns)
    cols = [s for s in symbols if s in px.columns]
    members[cols] = True
    ok = members & has_px & (history >= min_history)
    return ok.loc[dates]


def universe_stats(univ: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "avg_names": univ.sum(axis=1).mean(),
        "distinct_names": int(univ.any().sum()),
        "monthly_churn": (univ.astype(int).diff().abs().sum(axis=1) / 2 / univ.sum(axis=1)).iloc[1:].mean(),
    })


__all__ = ["rebalance_dates", "liquid_universe", "fixed_list_universe", "universe_stats", "np"]
