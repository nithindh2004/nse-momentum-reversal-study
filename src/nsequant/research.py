"""Glue that turns the engine into named, comparable strategies.

`Study` holds the cleaned panel plus the point-in-time universes and the
benchmark, so every experiment is evaluated on exactly the same footing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .backtest import BacktestResult, quantile_portfolio, run_backtest
from .costs import CostModel
from .data import Panel
from .metrics import summarize
from .regimes import market_regimes
from .universe import liquid_universe, rebalance_dates

START = "2010-01-01"
RF = 0.06


@dataclass
class Study:
    panel: Panel
    size: int = 100
    costs: CostModel = field(default_factory=CostModel)

    def __post_init__(self):
        self.returns = self.panel.returns
        cal = self.panel.adj_close.index
        last_full = cal[-2]                     # need at least one session after a decision
        self.dates = {f: d[(d >= START) & (d <= last_full)] for f, d in
                      (("M", rebalance_dates(cal, "M")), ("W", rebalance_dates(cal, "W")))}
        self.universe = {f: liquid_universe(self.panel, d, self.size) for f, d in self.dates.items()}
        self.regimes = market_regimes(self.panel.index_close)
        self._bench = {}

    # ------------------------------------------------------------------ #
    def targets(self, signal: pd.DataFrame, freq: str = "M", q: float = 0.2, side: str = "long",
                universe: pd.DataFrame | None = None) -> pd.DataFrame:
        u = self.universe[freq] if universe is None else universe
        return quantile_portfolio(signal, u, q, side)

    def run(self, targets: pd.DataFrame, costs: CostModel | None = None, exec_lag: int = 1,
            returns: pd.DataFrame | None = None, start: str | None = None) -> BacktestResult:
        return run_backtest(self.returns if returns is None else returns, targets,
                            self.costs if costs is None else costs, exec_lag, start)

    def benchmark(self, freq: str = "M", costs: CostModel | None = None) -> BacktestResult:
        key = (freq, costs)
        if key not in self._bench:
            tgt = quantile_portfolio(self.universe[freq], self.universe[freq], side="universe")
            self._bench[key] = self.run(tgt, costs)
        return self._bench[key]

    def evaluate(self, res: BacktestResult, side: str, bench: BacktestResult | None = None) -> dict:
        if side == "long":
            b = (bench or self.benchmark()).net
            return summarize(res.daily, RF, b)
        return summarize(res.daily, 0.0)

    # ------------------------------------------------------------------ #
    def holding_regime(self, res: BacktestResult, exec_lag: int = 1) -> pd.Series:
        """Regime known at the decision date of the holding period each day belongs to."""
        cal = self.returns.index
        dec = res.targets.index
        pos = cal.get_indexer(dec) + exec_lag
        reg_at_dec = self.regimes["regime"].reindex(dec).to_numpy()
        lab = pd.Series(np.nan, index=cal, dtype=object)
        for k, p in enumerate(pos):
            if p + 1 < len(cal):
                lab.iloc[p + 1] = reg_at_dec[k]
        return lab.ffill().reindex(res.daily.index)
