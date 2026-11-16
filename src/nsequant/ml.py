"""Walk-forward machine-learning ranker.

Question: does combining the simple signals (plus a few standard features)
with a model beat the best single signal OUT OF SAMPLE, after costs?

Design choices that keep it honest:
  * features are cross-sectionally rank-normalised per decision date, so the
    model learns relative attractiveness, not market-level drift;
  * target = next-period forward return, rank-normalised the same way;
  * the model is retrained once a year on an expanding window and only on
    decision dates whose forward return was fully known before the training
    cut-off (an embargo of one holding period) -- no label leakage;
  * predictions for year Y come from a model that never saw year Y.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from .signals import dist_from_high, momentum, reversal, volatility

FEATURES = ["mom_12_1", "mom_6_1", "rev_1m", "rev_1w", "vol_3m", "dist_52w_high", "liquidity"]


def feature_panel(adj_close: pd.DataFrame, returns: pd.DataFrame, traded_value: pd.DataFrame) -> dict:
    return {
        "mom_12_1": momentum(adj_close, 252, 21),
        "mom_6_1": momentum(adj_close, 126, 21),
        "rev_1m": reversal(adj_close, 21),
        "rev_1w": reversal(adj_close, 5),
        "vol_3m": volatility(returns, 63),
        "dist_52w_high": dist_from_high(adj_close, 252),
        "liquidity": np.log(traded_value.rolling(126, min_periods=63).median()),
    }


def build_dataset(feats: dict, universe: pd.DataFrame, returns: pd.DataFrame, exec_lag: int = 1) -> pd.DataFrame:
    """Long table: (date, symbol) -> rank-normalised features + forward-return target."""
    dates = universe.index
    cal = returns.index
    pos = cal.get_indexer(dates)
    log_r = np.log1p(returns.fillna(0.0))
    cum = log_r.cumsum()
    rows = []
    for k, d in enumerate(dates):
        members = universe.columns[universe.loc[d]]
        if len(members) < 20:
            continue
        x = pd.DataFrame({f: feats[f].loc[d, members] for f in FEATURES})
        x = x.rank(pct=True) - 0.5
        # forward return: from execution close (d+lag) to the next execution close
        start = pos[k] + exec_lag
        end = (pos[k + 1] + exec_lag) if k + 1 < len(dates) else None
        if end is None or end >= len(cal):
            y = pd.Series(np.nan, index=members)
            label_known = pd.NaT
        else:
            y = cum.iloc[end][members] - cum.iloc[start][members]
            label_known = cal[end]
        x["target"] = y.rank(pct=True) - 0.5 if y.notna().any() else np.nan
        x["date"], x["label_known"] = d, label_known
        rows.append(x.reset_index(names="symbol"))
    df = pd.concat(rows, ignore_index=True)
    df[FEATURES] = df[FEATURES].fillna(0.0)          # missing feature -> neutral rank
    return df


def walk_forward_predict(df: pd.DataFrame, model: str = "gbm", first_test_year: int = 2013) -> pd.DataFrame:
    """Return a (date x symbol) prediction matrix, retraining every January."""
    preds = []
    for year in range(first_test_year, df["date"].dt.year.max() + 1):
        cutoff = pd.Timestamp(f"{year}-01-01")
        train = df[(df["label_known"] < cutoff) & df["target"].notna()]
        test = df[df["date"].dt.year == year]
        if test.empty or len(train) < 2000:
            continue
        if model == "ridge":
            m = Ridge(alpha=10.0)
        else:
            m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200,
                                              min_samples_leaf=200, l2_regularization=1.0,
                                              random_state=0)
        m.fit(train[FEATURES], train["target"])
        p = test[["date", "symbol"]].copy()
        p["pred"] = m.predict(test[FEATURES])
        preds.append(p)
    out = pd.concat(preds)
    return out.pivot(index="date", columns="symbol", values="pred")


def rank_ic_by_year(pred: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
    t = df.pivot(index="date", columns="symbol", values="target").reindex_like(pred)
    ics = {}
    for d in pred.index:
        a, b = pred.loc[d], t.loc[d]
        ok = a.notna() & b.notna()
        if ok.sum() > 20:
            ics[d] = spearmanr(a[ok], b[ok]).statistic
    s = pd.Series(ics)
    return s.groupby(s.index.year).mean()


def ridge_coefficients(df: pd.DataFrame) -> pd.Series:
    """Full-sample linear coefficients -- for interpretation only, never for trading."""
    d = df[df["target"].notna()]
    m = Ridge(alpha=10.0).fit(d[FEATURES], d["target"])
    return pd.Series(m.coef_, index=FEATURES)
