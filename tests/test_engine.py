"""Tests for the research engine. Run with:  python -m pytest -q

They use synthetic data, so they run without the downloaded market data.
The most important ones are the look-ahead tests: we change every price AFTER
a cut-off date and check that nothing decided on or before the cut-off moves.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nsequant.backtest import quantile_portfolio, run_backtest  # noqa: E402
from nsequant.costs import NO_COSTS, CostModel  # noqa: E402
from nsequant.data import Panel, _fix_spikes, _fix_splits  # noqa: E402
from nsequant.metrics import (deflated_sharpe, max_drawdown, probabilistic_sharpe,  # noqa: E402
                              sharpe)
from nsequant.regimes import market_regimes  # noqa: E402
from nsequant.signals import momentum, reversal  # noqa: E402
from nsequant.universe import liquid_universe, rebalance_dates  # noqa: E402


def make_panel(seed: int = 0, n_days: int = 700, n_stocks: int = 40) -> Panel:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    cols = [f"S{i:02d}" for i in range(n_stocks)]
    rets = rng.normal(0.0004, 0.02, (n_days, n_stocks))
    px = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=cols)
    vol = pd.DataFrame(rng.integers(1_000, 100_000, (n_days, n_stocks)).astype(float), index=dates, columns=cols)
    idx = pd.DataFrame({"NIFTY500": px.mean(axis=1), "INDIAVIX": 15 + rng.normal(0, 1, n_days)}, index=dates)
    return Panel(px, px.copy(), vol, idx, pd.DataFrame())


def perturb_after(panel: Panel, cutoff: pd.Timestamp, seed: int = 99) -> Panel:
    rng = np.random.default_rng(seed)
    later = panel.adj_close.index > cutoff
    shock = pd.DataFrame(np.exp(rng.normal(0, 0.3, panel.adj_close.shape)),
                         index=panel.adj_close.index, columns=panel.adj_close.columns)
    adj = panel.adj_close.copy(); adj[later] = adj[later] * shock[later]
    vol = panel.volume.copy(); vol[later] = vol[later] * shock[later] * 7
    idx = panel.index_close.copy(); idx.loc[later] = idx.loc[later] * 1.5
    return Panel(adj, adj.copy(), vol, idx, pd.DataFrame())


def targets_for(panel: Panel, dates):
    univ = liquid_universe(panel, dates, size=20)
    sig = momentum(panel.adj_close, 126, 21) + reversal(panel.adj_close, 5)
    return quantile_portfolio(sig, univ, 0.2, "long_short")


# --------------------------------------------------------------------------- #
# Look-ahead tests
# --------------------------------------------------------------------------- #

def test_targets_ignore_future_prices():
    p = make_panel()
    dates = rebalance_dates(p.adj_close.index, "M")
    dates = dates[dates > p.adj_close.index[300]]
    cutoff = dates[len(dates) // 2]
    a = targets_for(p, dates)
    b = targets_for(perturb_after(p, cutoff), dates)
    past = a.index <= cutoff
    pd.testing.assert_frame_equal(a[past], b[past])
    assert not a[~past].equals(b[~past]), "sanity: future decisions should change"


def test_regimes_ignore_future():
    p = make_panel()
    cutoff = p.adj_close.index[500]
    a = market_regimes(p.index_close)
    b = market_regimes(perturb_after(p, cutoff).index_close)
    pd.testing.assert_series_equal(a["regime"].loc[:cutoff], b["regime"].loc[:cutoff])


def test_backtest_returns_ignore_future():
    p = make_panel()
    dates = rebalance_dates(p.adj_close.index, "M")
    dates = dates[dates > p.adj_close.index[300]]
    cutoff = dates[len(dates) // 2]
    q = perturb_after(p, cutoff)
    ra = run_backtest(p.returns, targets_for(p, dates), CostModel()).net
    rb = run_backtest(q.returns, targets_for(q, dates), CostModel()).net
    pd.testing.assert_series_equal(ra.loc[:cutoff], rb.loc[:cutoff])


def test_execution_lag_delays_returns():
    """With exec_lag=1, the first session after a decision earns nothing new."""
    dates = pd.bdate_range("2020-01-01", periods=5)
    rets = pd.DataFrame({"A": [0, 0.10, 0.20, 0.30, 0.0]}, index=dates)
    tgt = pd.DataFrame({"A": [1.0]}, index=[dates[0]])
    lag1 = run_backtest(rets, tgt, NO_COSTS, exec_lag=1).daily["gross"]
    lag0 = run_backtest(rets, tgt, NO_COSTS, exec_lag=0).daily["gross"]
    assert lag0.loc[dates[1]] == pytest.approx(0.10)   # bought at close of day 0
    assert lag1.loc[dates[1]] == pytest.approx(0.0)    # bought at close of day 1
    assert lag1.loc[dates[2]] == pytest.approx(0.20)


# --------------------------------------------------------------------------- #
# Accounting tests
# --------------------------------------------------------------------------- #

def test_two_asset_portfolio_and_drift():
    dates = pd.bdate_range("2020-01-01", periods=3)
    rets = pd.DataFrame({"A": [0, 0.10, 0.0], "B": [0, -0.10, 0.10]}, index=dates)
    tgt = pd.DataFrame({"A": [0.5], "B": [0.5]}, index=[dates[0]])
    g = run_backtest(rets, tgt, NO_COSTS, exec_lag=0).daily["gross"]
    assert g.loc[dates[1]] == pytest.approx(0.0)
    # after day 1 weights drift to A=0.55, B=0.45 -> day-2 return = 0.45 * 10%
    assert g.loc[dates[2]] == pytest.approx(0.045)


def test_cost_model_hand_calculation():
    cm = CostModel(slippage_bps=10, capital_inr=1e7)
    d = pd.Timestamp("2024-06-03")
    fees = (0.0000297 + 0.000001) * 1.18
    assert cm.buy_rate(d) == pytest.approx(0.001 + 0.00015 + fees + 0.001)
    assert cm.sell_rate(d) == pytest.approx(0.001 + fees + 0.001)
    assert cm.stt(pd.Timestamp("2011-05-02")) == 0.00125          # pre July-2012 STT
    assert cm.dp_cost(10) == pytest.approx(10 * 15.34 / 1e7)

    dates = pd.bdate_range("2024-06-03", periods=2)
    rets = pd.DataFrame({"A": [0.0, 0.0]}, index=dates)
    tgt = pd.DataFrame({"A": [1.0]}, index=[dates[0]])
    c = run_backtest(rets, tgt, cm, exec_lag=0).daily["cost"]
    assert c.loc[dates[0]] == pytest.approx(cm.buy_rate(d))   # buying 100% once


# --------------------------------------------------------------------------- #
# Data cleaning tests
# --------------------------------------------------------------------------- #

def test_spike_fixed_but_real_crash_kept():
    idx = pd.bdate_range("2020-01-01", periods=8)
    glitch = pd.Series([100, 101, 40, 102, 103, 104, 105, 106.0], index=idx)
    fixed = _fix_spikes(glitch, "X", log := [])
    assert fixed.iloc[2] == 101 and len(log) == 1
    crash = pd.Series([100, 101, 50, 55, 60, 70, 80, 99.0], index=idx)  # slow recovery = real
    assert _fix_spikes(crash, "Y", []).equals(crash)


def test_unadjusted_split_detected():
    idx = pd.bdate_range("2020-01-01", periods=4)
    px = pd.Series([1000, 1010, 202, 205.0], index=idx)       # 5:1 split not adjusted
    adj, close = _fix_splits(px, px.copy(), "Z", log := [])
    assert adj.pct_change().abs().max() < 0.02 and log[0]["rule"] == "unadjusted_split"


# --------------------------------------------------------------------------- #
# Metric tests
# --------------------------------------------------------------------------- #

def test_max_drawdown_known_path():
    r = pd.Series([0.10, -0.50, 0.20])
    assert max_drawdown(r) == pytest.approx(-0.50)


def test_sharpe_and_deflation():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0008, 0.01, 2500))
    assert sharpe(r) == pytest.approx(r.mean() / r.std() * math.sqrt(252))
    psr = probabilistic_sharpe(r)
    trials = list(rng.normal(0, 0.03, 50))
    assert 0 < deflated_sharpe(r, trials) < psr <= 1
