"""Cross-sectional signals. Higher value = more attractive to buy.

All signals at date t use prices up to and including t only. Prices are
forward-filled over short gaps (<= 5 sessions) so a one-day data hole does not
drop a stock from the ranking.
"""

from __future__ import annotations

import pandas as pd

from .data import STALE_FFILL_LIMIT


def _px(adj_close: pd.DataFrame) -> pd.DataFrame:
    return adj_close.ffill(limit=STALE_FFILL_LIMIT)


def momentum(adj_close: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """Total return from t-lookback to t-skip (the classic '12-1' when 252/21).

    Skipping the most recent month separates medium-term momentum from the
    short-term reversal effect, which works in the opposite direction.
    """
    px = _px(adj_close)
    return px.shift(skip) / px.shift(lookback) - 1


def reversal(adj_close: pd.DataFrame, lookback: int = 21) -> pd.DataFrame:
    """Negative of the recent return: recent losers score high."""
    px = _px(adj_close)
    return -(px / px.shift(lookback) - 1)


def volatility(returns: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Trailing daily-return volatility (used as an ML feature)."""
    return returns.rolling(window, min_periods=window // 2).std()


def dist_from_high(adj_close: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Price relative to its trailing 52-week high (<= 0)."""
    px = _px(adj_close)
    return px / px.rolling(window, min_periods=window // 2).max() - 1
