"""
download_data.py
================
Fetch the raw daily market data for the NSE momentum / reversal study.

Run this on your own machine (it needs internet access to niftyindices.com /
nseindia.com and Yahoo Finance). It is the ONLY step that touches the network;
everything downstream reads the files it writes, so results are reproducible.

Usage
-----
    python download_data.py              # full download (approx. 5-15 min)
    python download_data.py --quick      # 10-ticker smoke test (< 1 min)
    python download_data.py --start 2009-01-01 --end 2025-12-31

Outputs (written to ./data/raw/)
--------------------------------
    nifty500_current.csv      today's NIFTY 500 list (symbol, company, industry, ISIN)
    nifty100_current.csv      today's NIFTY 100 list (used ONLY for the survivorship-bias audit)
    prices_daily.csv.gz       long format: date, symbol, open, high, low, close, adj_close, volume
    index_daily.csv.gz        long format: date, index, open, high, low, close, adj_close, volume
    download_report.json      coverage stats, failed tickers, library versions, file hashes

Design notes
------------
* Survivorship bias: a universe made only of TODAY's index members silently drops
  every stock that crashed, was delisted or merged. To reduce this, the ticker
  pool = current NIFTY 500 + a hand-curated list of former large caps that have
  since been delisted, merged or demoted (FORMER_LARGE_CAPS below). The study
  later picks a point-in-time universe from this pool each month by trailing
  traded value. Free data sources often lack delisted stocks, so the report
  records exactly which of them could not be fetched -- that residual bias is
  stated openly in the write-up rather than hidden.
* Prices are fetched with auto_adjust=False so we keep both the raw close
  (for traded value = close x volume) and the split/dividend-adjusted close
  (for returns).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

try:
    import pandas as pd
    import requests
    import yfinance as yf
except ImportError as exc:  # friendly message instead of a bare traceback
    sys.exit(
        f"Missing package: {exc.name}. Install the requirements first:\n"
        "    python -m pip install -r requirements.txt"
    )

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

CONSTITUENT_URLS = {
    "nifty500": [
        "https://niftyindices.com/IndexConstituent/ind_nifty500list.csv",
        "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
    ],
    "nifty100": [
        "https://niftyindices.com/IndexConstituent/ind_nifty100list.csv",
        "https://archives.nseindia.com/content/indices/ind_nifty100list.csv",
    ],
}

# Yahoo Finance symbols for the market indices we need.
INDEX_TICKERS = {
    "NIFTY50": "^NSEI",
    "NIFTY100": "^CNX100",
    "NIFTY500": "^CRSLDX",
    "INDIAVIX": "^INDIAVIX",
}

# Former large/mid caps that left the index universe after 2009 through
# delisting, merger, insolvency or demotion. ASSUMPTION: compiled from public
# knowledge of NSE corporate events; it is best-effort, not exhaustive. Symbols
# that Yahoo no longer serves are logged in download_report.json.
FORMER_LARGE_CAPS = [
    # merged / delisted
    "HDFC", "CAIRN", "RANBAXY", "MINDTREE", "IDFC", "HEXAWARE", "GRUH",
    "CAPF", "ABIRLANUVO", "STER", "SESAGOA", "ESSAROIL", "LTI", "TATAMTRDVR",
    "ORIENTBANK", "ALBK", "ANDHRABANK", "CORPBANK", "SYNDIBANK", "VIJAYABANK",
    "DENABANK", "BHUSANSTL", "DHFL", "JETAIRWAYS", "INFRATEL", "CROMPGREAV",
    # collapsed / demoted but still (or recently) listed
    "RCOM", "JPASSOCIAT", "UNITECH", "RELINFRA", "RPOWER", "RELCAPITAL",
    "SUZLON", "IDEA", "YESBANK", "ZEEL", "GMRINFRA", "LITL", "PUNJLLOYD",
    "IFCI", "IBULHSGFIN", "JPPOWER", "HDIL", "ADANIPOWER", "IBREALEST",
]

QUICK_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "SBIN", "ITC", "LT", "AXISBANK", "HDFC",  # HDFC = delisted 2023, tests failure path
]

BATCH_SIZE = 40          # tickers per Yahoo request
PAUSE_BETWEEN_BATCHES = 2.0  # seconds; be polite to avoid rate limits
MAX_RETRIES = 3

HEADERS = {  # niftyindices / NSE reject requests without a browser user agent
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
    ),
    "Accept": "text/csv,text/plain,*/*",
}

COLUMN_MAP = {
    "Open": "open", "High": "high", "Low": "low", "Close": "close",
    "Adj Close": "adj_close", "Volume": "volume",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def fetch_constituents(name: str, out_dir: Path) -> pd.DataFrame | None:
    """Download an index constituent CSV from niftyindices/NSE, with fallbacks.

    If a manually downloaded copy already exists in out_dir, it is used as-is.
    """
    target = out_dir / f"{name}_current.csv"
    if target.exists():
        log(f"Using existing {target.name}")
        return pd.read_csv(target)

    for url in CONSTITUENT_URLS[name]:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            df.columns = [c.strip() for c in df.columns]
            if "Symbol" not in df.columns or len(df) < 50:
                raise ValueError(f"unexpected content ({len(df)} rows)")
            df.to_csv(target, index=False)
            log(f"Downloaded {name} constituents: {len(df)} stocks ({url})")
            return df
        except Exception as exc:  # noqa: BLE001 -- try the next mirror
            log(f"  could not fetch {url}: {exc}")
    return None


def to_long(raw: pd.DataFrame, yahoo_to_name: dict[str, str], key: str) -> pd.DataFrame:
    """Convert yfinance's wide (ticker x field) frame into tidy long format."""
    frames = []
    if raw is None or raw.empty:
        return pd.DataFrame()
    multi = isinstance(raw.columns, pd.MultiIndex)
    for ysym, name in yahoo_to_name.items():
        if multi:
            if ysym not in raw.columns.get_level_values(0):
                continue
            sub = raw[ysym].copy()
        else:  # single ticker, flat columns
            sub = raw.copy()
        sub = sub.rename(columns=COLUMN_MAP)
        keep = [c for c in COLUMN_MAP.values() if c in sub.columns]
        sub = sub[keep].dropna(how="all")
        if sub.empty or "close" not in sub.columns or sub["close"].notna().sum() == 0:
            continue
        sub.index = pd.to_datetime(sub.index).tz_localize(None).normalize()
        sub.index.name = "date"
        sub.insert(0, key, name)
        frames.append(sub.reset_index())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def download_yahoo(yahoo_to_name: dict[str, str], start: str, end: str, key: str) -> pd.DataFrame:
    """Download in batches with retries; returns long-format frame."""
    symbols = list(yahoo_to_name)
    out = []
    n_batches = (len(symbols) + BATCH_SIZE - 1) // BATCH_SIZE
    for b in range(n_batches):
        batch = symbols[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
        mapping = {s: yahoo_to_name[s] for s in batch}
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw = yf.download(
                    batch, start=start, end=end, interval="1d",
                    auto_adjust=False, actions=False, group_by="ticker",
                    threads=True, progress=False, multi_level_index=True,
                )
                part = to_long(raw, mapping, key)
                out.append(part)
                got = part[key].nunique() if not part.empty else 0
                log(f"Batch {b + 1}/{n_batches}: {got}/{len(batch)} tickers returned data")
                break
            except Exception as exc:  # noqa: BLE001 -- network / rate-limit errors
                wait = 10 * attempt
                log(f"Batch {b + 1} attempt {attempt} failed ({exc}); retrying in {wait}s")
                time.sleep(wait)
        time.sleep(PAUSE_BETWEEN_BATCHES)
    out = [o for o in out if not o.empty]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def retry_missing(df: pd.DataFrame, yahoo_to_name: dict[str, str], start: str, end: str, key: str) -> pd.DataFrame:
    """Second pass: retry tickers that came back empty, one at a time."""
    have = set(df[key].unique()) if not df.empty else set()
    missing = {y: n for y, n in yahoo_to_name.items() if n not in have}
    if not missing:
        return df
    log(f"Retrying {len(missing)} missing tickers individually...")
    extra = []
    for ysym, name in missing.items():
        try:
            raw = yf.download(
                ysym, start=start, end=end, interval="1d", auto_adjust=False,
                actions=False, group_by="ticker", threads=False, progress=False,
                multi_level_index=True,
            )
            part = to_long(raw, {ysym: name}, key)
            if not part.empty:
                extra.append(part)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5)
    if extra:
        log(f"  recovered {len(extra)} of {len(missing)}")
        df = pd.concat([df, *extra], ignore_index=True)
    return df


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2009-01-01", help="first date (needs ~1y before the 2010 test start for lookbacks)")
    ap.add_argument("--end", default=date.today().isoformat(), help="last date (exclusive)")
    ap.add_argument("--out", default="data/raw", help="output folder")
    ap.add_argument("--quick", action="store_true", help="10-ticker smoke test")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Writing to {out_dir.resolve()}")

    # 1. Build the ticker pool ------------------------------------------------
    pool_source: dict[str, str] = {}
    if args.quick:
        for s in QUICK_SYMBOLS:
            pool_source[s] = "quick"
    else:
        n500 = fetch_constituents("nifty500", out_dir)
        n100 = fetch_constituents("nifty100", out_dir)
        if n500 is None or n100 is None:
            log(
                "ERROR: could not download the constituent lists.\n"
                "  Fix: open https://www.niftyindices.com/indices/equity/broad-based-indices/nifty-500\n"
                "  and .../nifty-100, click 'Index Constituent' download, save the files as\n"
                f"  {out_dir / 'nifty500_current.csv'} and {out_dir / 'nifty100_current.csv'},\n"
                "  then run this script again."
            )
            return 1
        for s in n500["Symbol"].astype(str).str.strip():
            pool_source[s] = "nifty500_current"
        for s in n100["Symbol"].astype(str).str.strip():
            pool_source.setdefault(s, "nifty100_current")
        for s in FORMER_LARGE_CAPS:
            pool_source.setdefault(s, "former_large_cap")
    log(f"Ticker pool: {len(pool_source)} symbols")

    # 2. Stock prices -----------------------------------------------------------
    yahoo_map = {f"{s}.NS": s for s in pool_source}
    prices = download_yahoo(yahoo_map, args.start, args.end, key="symbol")
    prices = retry_missing(prices, yahoo_map, args.start, args.end, key="symbol")
    if prices.empty:
        log("ERROR: no price data downloaded. Check your internet connection / Yahoo access.")
        return 1
    prices = prices.drop_duplicates(["date", "symbol"]).sort_values(["symbol", "date"])
    prices_path = out_dir / "prices_daily.csv.gz"
    prices.to_csv(prices_path, index=False, compression="gzip")
    log(f"Saved {len(prices):,} price rows for {prices['symbol'].nunique()} symbols -> {prices_path.name}")

    # 3. Indices ------------------------------------------------------------------
    idx_map = {y: n for n, y in INDEX_TICKERS.items()}
    indices = download_yahoo(idx_map, args.start, args.end, key="index")
    indices = retry_missing(indices, idx_map, args.start, args.end, key="index")
    idx_path = out_dir / "index_daily.csv.gz"
    if not indices.empty:
        indices = indices.drop_duplicates(["date", "index"]).sort_values(["index", "date"])
        indices.to_csv(idx_path, index=False, compression="gzip")
        log(f"Saved {len(indices):,} index rows for {sorted(indices['index'].unique())} -> {idx_path.name}")

    # 4. Coverage report ------------------------------------------------------------
    got = set(prices["symbol"].unique())
    by_source: dict[str, dict] = {}
    for sym, src in pool_source.items():
        d = by_source.setdefault(src, {"requested": 0, "downloaded": 0, "missing": []})
        d["requested"] += 1
        if sym in got:
            d["downloaded"] += 1
        else:
            d["missing"].append(sym)
    first_dates = prices.groupby("symbol")["date"].min()
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "args": vars(args),
        "date_range_found": [str(prices["date"].min().date()), str(prices["date"].max().date())],
        "symbols_requested": len(pool_source),
        "symbols_downloaded": len(got),
        "symbols_with_data_since_2010": int((first_dates <= "2010-01-31").sum()),
        "coverage_by_source": by_source,
        "indices_downloaded": sorted(indices["index"].unique()) if not indices.empty else [],
        "indices_missing": sorted(set(INDEX_TICKERS) - set(indices["index"].unique() if not indices.empty else [])),
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "yfinance": yf.__version__,
            "os": platform.platform(),
        },
        "sha256": {p.name: sha256(p) for p in out_dir.glob("*") if p.suffix in {".gz", ".csv"}},
    }
    rep_path = out_dir / "download_report.json"
    rep_path.write_text(json.dumps(report, indent=2, default=str))

    # 5. Human-readable summary -----------------------------------------------------
    print("\n" + "=" * 64)
    print("DOWNLOAD SUMMARY")
    print("=" * 64)
    print(f"Date range      : {report['date_range_found'][0]} -> {report['date_range_found'][1]}")
    print(f"Symbols         : {len(got)} / {len(pool_source)} downloaded")
    for src, d in by_source.items():
        print(f"  {src:<20}: {d['downloaded']:>4} / {d['requested']:<4}  missing: {len(d['missing'])}")
    print(f"Indices         : {report['indices_downloaded']}  (missing: {report['indices_missing']})")
    print(f"Report          : {rep_path}")
    print("=" * 64)
    print("Done. Missing delisted tickers are expected -- they are logged, not an error.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
