"""Performance and statistical-validation metrics.

Conventions: daily simple returns, 252 sessions per year. Risk-free rate is an
ASSUMPTION of 6.0% p.a. -- roughly the 2010-2026 average 91-day T-bill yield
(RBI auction cut-offs ranged ~3.3% in 2020-21 to ~9% in 2011-13). It is only
used for long-only Sharpe ratios; long-short spreads are self-financing.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

ANN = 252
RF_ANNUAL = 0.06


def rf_daily(rf_annual: float = RF_ANNUAL) -> float:
    return (1 + rf_annual) ** (1 / ANN) - 1


def cagr(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    return float((1 + r).prod() ** (ANN / len(r)) - 1)


def ann_vol(r: pd.Series) -> float:
    return float(r.std() * math.sqrt(ANN))


def sharpe(r: pd.Series, rf_annual: float = 0.0) -> float:
    ex = r.dropna() - (rf_daily(rf_annual) if rf_annual else 0.0)
    sd = ex.std()
    return float(ex.mean() / sd * math.sqrt(ANN)) if sd > 0 else np.nan


def sortino(r: pd.Series, rf_annual: float = 0.0) -> float:
    ex = r.dropna() - (rf_daily(rf_annual) if rf_annual else 0.0)
    dd = np.sqrt((np.minimum(ex, 0) ** 2).mean())
    return float(ex.mean() / dd * math.sqrt(ANN)) if dd > 0 else np.nan


def drawdown(r: pd.Series) -> pd.Series:
    w = (1 + r.fillna(0)).cumprod()
    return w / w.cummax() - 1


def max_drawdown(r: pd.Series) -> float:
    return float(drawdown(r).min())


def monthly(r: pd.Series) -> pd.Series:
    return (1 + r).groupby(r.index.to_period("M")).prod() - 1


def profit_factor(r: pd.Series) -> float:
    """Sum of winning months / |sum of losing months| (monthly P&L)."""
    m = monthly(r)
    loss = -m[m < 0].sum()
    return float(m[m > 0].sum() / loss) if loss > 0 else np.inf


def hit_rate(r: pd.Series) -> float:
    m = monthly(r)
    return float((m > 0).mean())


def newey_west_t(x: pd.Series, lags: int | None = None) -> float:
    """t-statistic of the mean with Newey-West (HAC) standard errors."""
    x = x.dropna().to_numpy()
    n = len(x)
    if lags is None:
        lags = int(4 * (n / 100) ** (2 / 9))
    e = x - x.mean()
    s = e @ e / n
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        s += 2 * w * (e[L:] @ e[:-L]) / n
    return float(x.mean() / math.sqrt(s / n))


# --------------------------------------------------------------------------- #
# Overfitting-aware Sharpe statistics (Bailey & Lopez de Prado)
# --------------------------------------------------------------------------- #

def probabilistic_sharpe(r: pd.Series, sr_benchmark: float = 0.0) -> float:
    """P(true Sharpe > benchmark) given sample length, skew and kurtosis.

    Sharpe ratios here are per-period (daily), as in the original paper.
    """
    x = r.dropna()
    n = len(x)
    sr = x.mean() / x.std()
    g3 = stats.skew(x)
    g4 = stats.kurtosis(x, fisher=False)
    denom = math.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return float(stats.norm.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / denom))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """Expected maximum of n_trials Sharpe ratios under the null of zero skill."""
    if n_trials <= 1:
        return 0.0
    em = 0.5772156649
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * math.e))
    return float(math.sqrt(var_sr) * ((1 - em) * z1 + em * z2))


def deflated_sharpe(r: pd.Series, trial_sharpes_daily: list[float]) -> float:
    """Deflated Sharpe Ratio: PSR against the best Sharpe expected by luck
    after trying len(trial_sharpes_daily) variants."""
    sr0 = expected_max_sharpe(len(trial_sharpes_daily), float(np.var(trial_sharpes_daily, ddof=1)))
    return probabilistic_sharpe(r, sr0)


def block_bootstrap_sharpe_ci(r: pd.Series, n_boot: int = 2000, block: int = 21,
                              alpha: float = 0.05, rf_annual: float = 0.0, seed: int = 7):
    """Stationary-style circular block bootstrap CI for the annualised Sharpe.

    Blocks of ~1 month keep the autocorrelation / volatility clustering that an
    i.i.d. bootstrap would destroy.
    """
    x = (r.dropna() - (rf_daily(rf_annual) if rf_annual else 0.0)).to_numpy()
    n = len(x)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        s = x[idx]
        out[b] = s.mean() / s.std() * math.sqrt(ANN)
    lo, hi = np.quantile(out, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Summary table
# --------------------------------------------------------------------------- #

def summarize(daily: pd.DataFrame, rf_annual: float = 0.0, benchmark: pd.Series | None = None) -> dict:
    """Headline statistics for one backtest's daily frame (gross, cost, net, turnover)."""
    net, gross = daily["net"], daily["gross"]
    years = len(net) / ANN
    out = {
        "CAGR": cagr(net),
        "Vol": ann_vol(net),
        "Sharpe": sharpe(net, rf_annual),
        "Sharpe_gross": sharpe(gross, rf_annual),
        "Sortino": sortino(net, rf_annual),
        "MaxDD": max_drawdown(net),
        "ProfitFactor": profit_factor(net),
        "HitRate_m": hit_rate(net),
        "t_NW": newey_west_t(net - (rf_daily(rf_annual) if rf_annual else 0.0)),
        "Turnover_ann": daily["turnover"].sum() / years,
        "CostDrag_ann": daily["cost"].sum() / years,
    }
    # break-even one-way cost: the extra cost per unit of one-way turnover that
    # would wipe out the gross excess return
    gross_ex = (gross.mean() - (rf_daily(rf_annual) if rf_annual else 0.0)) * ANN
    if benchmark is not None:
        b = benchmark.reindex(net.index)
        active = net - b
        out["Active_ret"] = float(active.mean() * ANN)
        out["IR"] = sharpe(active)
        out["t_active_NW"] = newey_west_t(active)
        gross_ex = (gross - b).mean() * ANN
    turn = out["Turnover_ann"]
    out["Breakeven_bps_oneway"] = float(gross_ex / (2 * turn) * 1e4) if turn > 0 else np.nan
    return out
