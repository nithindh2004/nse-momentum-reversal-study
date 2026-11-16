"""Load, clean and store the raw Yahoo Finance NSE data.

Free vendor data for Indian stocks has three recurring defects, each handled by
an explicit, logged rule (nothing is silently changed):

0. Placeholder prices -- Yahoo repeats a stale price with ZERO volume before a
   stock's real history starts (and during suspensions). A zero-volume price is
   not a traded price, so it is set to missing.
1. Spike-and-revert glitches -- a price jumps by >35% and comes back to within
   5% of the prior level within 1-2 sessions. The bad prints are replaced by
   the last good price. (A looser rule was tested first and wrongly "fixed"
   real events such as the Feb-2023 Adani crash, so it was tightened.)
2. Unadjusted splits / bonuses -- a one-day DROP whose price ratio is close to
   a standard split/bonus ratio (2, 3, 4, 5, 10, ...) and does not revert.
   (Upward "consolidation" detection was dropped: it misfired on the genuine
   +46% PSU-bank rally of 25-Oct-2017.)
   History before the event is divided by that ratio.
3. Everything else above the extreme-move threshold is kept as a genuine
   market move (e.g. Yes Bank, March 2020) unless it appears in the manual
   corporate-action file (data/corporate_actions.csv), e.g. demergers.

Every correction is written to results/tables/data_corrections.csv.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
DB_PATH = PROCESSED / "market.sqlite"

SPIKE_THRESHOLD = 0.35      # |daily return| that triggers a spike check
REVERT_TOLERANCE = 0.05     # price must return within 5% of the pre-spike level
REVERT_WINDOW = 2           # ... within this many sessions (real crashes can also
                            # rebound, so only 1-2 session blips count as glitches)
STALE_FFILL_LIMIT = 5       # max sessions a missing price is carried forward
SPLIT_THRESHOLD = 0.40      # |daily return| that triggers a split check
SPLIT_RATIOS = np.array([1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 20])
SPLIT_TOLERANCE = 0.04      # relative distance to a standard ratio


@dataclass
class Panel:
    """Wide (date x symbol) matrices used by every downstream module."""
    adj_close: pd.DataFrame   # split + dividend adjusted -> total-return proxy
    close: pd.DataFrame       # split adjusted only -> traded value
    volume: pd.DataFrame
    index_close: pd.DataFrame  # NIFTY50/100/500, INDIAVIX
    corrections: pd.DataFrame

    @property
    def returns(self) -> pd.DataFrame:
        """Daily total returns.

        Short gaps (<= STALE_FFILL_LIMIT sessions) are bridged so the move over
        the gap lands on the next traded day; longer gaps (suspensions, pre-
        listing placeholders) give a missing return, never a fabricated one.
        """
        px = self.adj_close.ffill(limit=STALE_FFILL_LIMIT)
        r = px.pct_change(fill_method=None)
        return r.where(px.shift(1).notna())

    @property
    def traded_value(self) -> pd.DataFrame:
        return self.close * self.volume


# --------------------------------------------------------------------------- #
# Cleaning rules
# --------------------------------------------------------------------------- #

def _fix_spikes(px: pd.Series, symbol: str, log: list) -> pd.Series:
    """Rule 1: replace spike-and-revert blocks by the last good price."""
    px = px.copy()
    vals = px.to_numpy(dtype=float, copy=True)
    idx = px.index
    valid = np.where(~np.isnan(vals))[0]
    i = 1
    while i < len(valid):
        t, prev = valid[i], valid[i - 1]
        r = vals[t] / vals[prev] - 1
        if abs(r) > SPIKE_THRESHOLD:
            base = vals[prev]
            # look ahead a few sessions for a return to the pre-spike level
            for j in range(i + 1, min(i + 1 + REVERT_WINDOW, len(valid))):
                if abs(vals[valid[j]] / base - 1) < REVERT_TOLERANCE:
                    bad = valid[i:j]
                    log.append({
                        "symbol": symbol, "date": idx[t].date(), "rule": "spike_revert",
                        "detail": f"{len(bad)} bad session(s), jump {r:+.1%} reverted",
                    })
                    vals[bad] = base  # forward-fill with last good price
                    i = j
                    break
        i += 1
    return pd.Series(vals, index=idx, name=px.name)


def _fix_splits(adj: pd.Series, close: pd.Series, symbol: str, log: list):
    """Rule 2: back-adjust unadjusted splits/bonuses detected by ratio."""
    adj, close = adj.copy(), close.copy()
    r = adj.pct_change(fill_method=None)
    for t in r.index[r.abs() > SPLIT_THRESHOLD]:
        ratio = adj.shift(1).loc[t] / adj.loc[t]       # >1 for a split
        for k, direction in ((ratio, "split"),):
            rel = np.abs(SPLIT_RATIOS - k) / SPLIT_RATIOS
            if rel.min() < SPLIT_TOLERANCE:
                kk = SPLIT_RATIOS[rel.argmin()]
                factor = kk if direction == "split" else 1 / kk
                before = adj.index < t
                adj[before] /= factor
                close[before] /= factor
                log.append({
                    "symbol": symbol, "date": t.date(), "rule": f"unadjusted_{direction}",
                    "detail": f"ratio {k:.3f} ~ {kk:g}:1, history before date divided by {factor:.3g}",
                })
                break
    return adj, close


def _apply_manual_actions(adj: pd.DataFrame, close: pd.DataFrame, log: list):
    """Rule 3: corporate actions that cannot be inferred from prices (e.g. demergers).

    A demerger drops the parent's price but shareholders receive the demerged
    company's shares, so the true total return is not the price drop.
    ASSUMPTION (documented): with no price for the demerged entity, we treat the
    event-day total return as 0% by back-adjusting history by the price ratio.
    """
    path = ROOT / "data" / "corporate_actions.csv"
    if not path.exists():
        return adj, close
    actions = pd.read_csv(path, parse_dates=["date"], comment="#")
    for _, a in actions.iterrows():
        s, t = a["symbol"], a["date"]
        if s not in adj.columns or t not in adj.index:
            continue
        prev = adj[s].loc[:t].iloc[:-1].dropna()
        if prev.empty:
            continue
        factor = prev.iloc[-1] / adj[s].loc[t]
        before = adj.index < t
        adj.loc[before, s] /= factor
        close.loc[before, s] /= factor
        log.append({"symbol": s, "date": t.date(), "rule": "manual_" + a["action"],
                    "detail": f"{a['note']} (history divided by {factor:.3f})"})
    return adj, close


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def build_panel(raw_dir: Path = RAW) -> Panel:
    prices = pd.read_csv(raw_dir / "prices_daily.csv.gz", parse_dates=["date"])
    indices = pd.read_csv(raw_dir / "index_daily.csv.gz", parse_dates=["date"])

    # Trading calendar = sessions on which the NIFTY 500 index printed.
    calendar = pd.DatetimeIndex(sorted(indices.loc[indices["index"] == "NIFTY500", "date"].unique()))

    def wide(col):
        return prices.pivot(index="date", columns="symbol", values=col).reindex(calendar)

    adj, close, vol = wide("adj_close"), wide("close"), wide("volume")
    log: list[dict] = []

    # Rule 0: zero-volume prints are placeholders, not trades.
    stale = vol.fillna(0).eq(0) & adj.notna()
    for s in stale.columns[stale.any()]:
        n = int(stale[s].sum())
        log.append({"symbol": s, "date": stale.index[stale[s]][0].date(), "rule": "zero_volume_to_missing",
                    "detail": f"{n} zero-volume session(s) set to missing"})
    adj, close = adj.mask(stale), close.mask(stale)

    for s in adj.columns:
        adj[s] = _fix_spikes(adj[s], s, log)
        close[s] = _fix_spikes(close[s], s, [])     # same glitch, log once
    # Verified corporate actions go before the split heuristic, so a demerger
    # whose price ratio happens to look like a split (JSL 2015: 2.56) is
    # labelled correctly.
    adj, close = _apply_manual_actions(adj, close, log)
    for s in adj.columns:
        adj[s], close[s] = _fix_splits(adj[s], close[s], s, log)

    idx_close = indices.pivot(index="date", columns="index", values="close").reindex(calendar).ffill()
    corrections = pd.DataFrame(log, columns=["symbol", "date", "rule", "detail"])
    return Panel(adj, close, vol, idx_close, corrections)


def extreme_moves(panel: Panel, threshold: float = 0.40) -> pd.DataFrame:
    """Remaining |daily return| > threshold after cleaning -- for the audit trail."""
    r = panel.returns.stack()
    r = r[r.abs() > threshold]
    out = r.rename("return").reset_index()
    out.columns = ["date", "symbol", "return"]
    return out.sort_values("return")


def save_to_sqlite(panel: Panel, db_path: Path = DB_PATH) -> None:
    """Persist the cleaned panel in long format so it can be queried with SQL."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    long = pd.concat({
        "adj_close": panel.adj_close.stack(),
        "close": panel.close.stack(),
        "volume": panel.volume.stack(),
    }, axis=1).reset_index()
    long.columns = ["date", "symbol", "adj_close", "close", "volume"]
    long = long.dropna(subset=["adj_close"])
    long["date"] = long["date"].dt.strftime("%Y-%m-%d")
    idx = panel.index_close.stack().rename("close").reset_index()
    idx.columns = ["date", "index_name", "close"]
    idx["date"] = idx["date"].dt.strftime("%Y-%m-%d")
    with sqlite3.connect(db_path) as con:
        long.to_sql("prices", con, if_exists="replace", index=False)
        idx.to_sql("indices", con, if_exists="replace", index=False)
        panel.corrections.astype(str).to_sql("corrections", con, if_exists="replace", index=False)
        con.execute("CREATE INDEX IF NOT EXISTS ix_prices ON prices(symbol, date)")
        con.execute("CREATE INDEX IF NOT EXISTS ix_prices_date ON prices(date)")


def sql(query: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    with sqlite3.connect(db_path) as con:
        return pd.read_sql_query(query, con)
