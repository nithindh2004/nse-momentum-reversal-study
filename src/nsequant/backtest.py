"""Daily portfolio backtester with explicit timing and costs.

Timing convention (the most common source of look-ahead bias, so spelled out):

    decision at close of day d  -- signal, universe and regime use data <= d
    execution at close of d+L   -- L = exec_lag (default 1 session)
    the new weights earn returns from day d+L+1 onwards

With exec_lag=0 the strategy trades at the same close that produced the
signal, which is not achievable in practice (the close is only known after the
market shuts). It is kept only for the bias audit.

Between rebalances weights drift with prices (buy-and-hold), so turnover and
costs reflect the trades actually needed to get back to target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .costs import CostModel


@dataclass
class BacktestResult:
    daily: pd.DataFrame          # gross, cost, net, turnover, n_long, n_short
    targets: pd.DataFrame        # target weights at decision dates

    @property
    def net(self) -> pd.Series:
        return self.daily["net"]

    @property
    def gross(self) -> pd.Series:
        return self.daily["gross"]


def quantile_portfolio(signal: pd.DataFrame, universe: pd.DataFrame, q: float = 0.2,
                       side: str = "long") -> pd.DataFrame:
    """Equal-weight target weights from a cross-sectional ranking.

    side = "long"       : top q of the universe, weights sum to 1
           "short"      : bottom q, weights sum to -1
           "long_short" : both legs (gross exposure 2, net 0)
           "universe"   : every universe member equally (the benchmark)
    """
    sig = signal.reindex(index=universe.index, columns=universe.columns).where(universe)
    if side == "universe":
        w = universe.astype(float)
        return w.div(w.sum(axis=1), axis=0).fillna(0.0)
    pct = sig.rank(axis=1, pct=True)
    out = pd.DataFrame(0.0, index=sig.index, columns=sig.columns)
    if side in ("long", "long_short"):
        top = (pct > 1 - q).astype(float)
        out += top.div(top.sum(axis=1), axis=0).fillna(0.0)
    if side in ("short", "long_short"):
        bot = (pct <= q).astype(float)
        out -= bot.div(bot.sum(axis=1), axis=0).fillna(0.0)
    return out


def run_backtest(returns: pd.DataFrame, targets: pd.DataFrame, costs: CostModel,
                 exec_lag: int = 1, start: str | None = None) -> BacktestResult:
    dates = returns.index
    cols = targets.columns
    R = returns.reindex(columns=cols).fillna(0.0).to_numpy()   # missing price -> no move
    T = targets.fillna(0.0).to_numpy()
    pos = dates.get_indexer(targets.index)
    if (pos < 0).any():
        raise ValueError("decision dates must be trading days")
    live = np.abs(T).sum(axis=1) > 0                       # skip warm-up rows with no positions
    exec_at = {p + exec_lag: k for k, p in enumerate(pos) if p + exec_lag < len(dates) and live[k]}

    first = min(exec_at)
    if start is not None:
        first = max(first, dates.searchsorted(pd.Timestamp(start)))

    n = len(dates)
    gross = np.zeros(n); cost = np.zeros(n); turn = np.zeros(n)
    nl = np.zeros(n); ns = np.zeros(n)
    w = np.zeros(len(cols))
    for i in range(first, n):
        # 1) positions held overnight earn today's return
        g = float(w @ R[i])
        gross[i] = g
        if g != -1.0:
            w = w * (1 + R[i]) / (1 + g)
        # 2) rebalance at today's close if an execution is scheduled
        k = exec_at.get(i)
        if k is not None:
            tgt = T[k]
            trade = tgt - w
            buys = np.clip(trade, 0, None)      # any purchase (new long or short cover)
            sells = np.clip(-trade, 0, None)    # any sale (long exit or new short)
            d = dates[i]
            n_sold = int(((w > 0) & (trade < -1e-12)).sum())   # long positions reduced -> DP charge
            cost[i] = buys.sum() * costs.buy_rate(d) + sells.sum() * costs.sell_rate(d) + costs.dp_cost(n_sold)
            turn[i] = np.abs(trade).sum() / 2
            w = tgt.copy()
        nl[i] = (w > 0).sum(); ns[i] = (w < 0).sum()

    daily = pd.DataFrame({"gross": gross, "cost": cost, "net": gross - cost,
                          "turnover": turn, "n_long": nl, "n_short": ns}, index=dates).iloc[first:]
    return BacktestResult(daily=daily, targets=targets)
