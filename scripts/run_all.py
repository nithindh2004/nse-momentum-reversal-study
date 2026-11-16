"""Reproduce every number in the README and research note.

    python scripts/run_all.py            # ~3-5 minutes on a laptop

Reads data/raw/ (from download_data.py), writes results/tables/*.csv,
results/results.json and results/figures/*.png.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nsequant import ml  # noqa: E402
from nsequant.costs import NO_COSTS, CostModel  # noqa: E402
from nsequant.data import Panel, build_panel, extreme_moves, save_to_sqlite, sql  # noqa: E402
from nsequant.metrics import (ANN, block_bootstrap_sharpe_ci, deflated_sharpe,  # noqa: E402
                              newey_west_t, probabilistic_sharpe, sharpe, summarize)
from nsequant.regimes import market_regimes  # noqa: E402
from nsequant.research import RF, Study  # noqa: E402
from nsequant.signals import momentum, reversal  # noqa: E402
from nsequant.universe import fixed_list_universe, liquid_universe, universe_stats  # noqa: E402

TABLES = ROOT / "results" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)
OUT: dict = {}
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def save(df: pd.DataFrame, name: str, index=True):
    df.to_csv(TABLES / f"{name}.csv", index=index, float_format="%.4f")


def fmt_row(d: dict) -> dict:
    return {k: (round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v) for k, v in d.items()}


# =========================================================================== #
# 0. Data
# =========================================================================== #
log("Building cleaned panel")
panel = build_panel()
save_to_sqlite(panel)
save(panel.corrections[panel.corrections.rule != "zero_volume_to_missing"], "data_corrections", index=False)
save(extreme_moves(panel), "remaining_extreme_moves", index=False)
report = json.loads((ROOT / "data" / "raw" / "download_report.json").read_text())

S = Study(panel)
R = S.returns
uM = S.universe["M"]
OUT["data"] = {
    "period": [str(S.dates["M"][0].date()), str(R.index[-1].date())],
    "symbols_downloaded": report["symbols_downloaded"],
    "former_large_caps_requested": report["coverage_by_source"]["former_large_cap"]["requested"],
    "former_large_caps_found": report["coverage_by_source"]["former_large_cap"]["downloaded"],
    "zero_volume_symbols": int((panel.corrections.rule == "zero_volume_to_missing").sum()),
    "other_corrections": int((panel.corrections.rule != "zero_volume_to_missing").sum()),
    "universe": fmt_row(universe_stats(uM).to_dict()),
    # SQL check on the stored data (shows the SQLite layer is queryable)
    "sql_rows_by_year": sql("SELECT substr(date,1,4) AS yr, COUNT(*) AS rows, "
                            "COUNT(DISTINCT symbol) AS symbols FROM prices GROUP BY yr").to_dict("records"),
}
log(f"Universe: {OUT['data']['universe']}")

# =========================================================================== #
# 1. Core signals (pre-registered defaults, chosen BEFORE looking at results)
# =========================================================================== #
log("Core signals")
SIG = {
    "MOM_12_1": (momentum(panel.adj_close, 252, 21), "M"),
    "REV_1M": (reversal(panel.adj_close, 21), "M"),
    "REV_1W": (reversal(panel.adj_close, 5), "W"),
}
bench = S.benchmark("M")
core, core_res = [], {}
for name, (sig, freq) in SIG.items():
    for side in ("long", "long_short"):
        res = S.run(S.targets(sig, freq, 0.2, side))
        core_res[(name, side)] = res
        core.append({"strategy": name, "side": side, **fmt_row(S.evaluate(res, side, bench))})
bsum = summarize(bench.daily, RF)
bsum["Breakeven_bps_oneway"] = np.nan          # not meaningful for the benchmark itself
core.append({"strategy": "EW_UNIVERSE", "side": "benchmark", **fmt_row(bsum)})
core_df = pd.DataFrame(core)
save(core_df, "core_signals", index=False)
OUT["core"] = core_df.to_dict("records")
print(core_df[["strategy", "side", "CAGR", "Sharpe", "Sharpe_gross", "MaxDD", "Turnover_ann",
               "CostDrag_ann", "Active_ret", "IR", "t_active_NW", "t_NW", "Breakeven_bps_oneway"]].to_string())

# NIFTY 50 price index for reference (price-only, excludes dividends)
nifty = panel.index_close["NIFTY50"].pct_change(fill_method=None).reindex(bench.net.index).fillna(0)
OUT["nifty50_price_cagr"] = float((1 + nifty).prod() ** (ANN / len(nifty)) - 1)

# Calendar-year returns
yr = pd.DataFrame({
    "MOM_12_1 long": core_res[("MOM_12_1", "long")].net,
    "REV_1M long": core_res[("REV_1M", "long")].net,
    "EW universe": bench.net,
}).groupby(lambda d: d.year).apply(lambda x: (1 + x).prod() - 1)
yr["MOM active"] = yr["MOM_12_1 long"] - yr["EW universe"]
yr["REV_1M active"] = yr["REV_1M long"] - yr["EW universe"]
save(yr, "calendar_year_returns")

# =========================================================================== #
# 2. Volatility regimes
# =========================================================================== #
log("Regime analysis")
reg = S.regimes
reg_share = reg["regime"].dropna().groupby(reg["regime"].dropna().index.year).value_counts(normalize=True).unstack()
save(reg_share, "regime_share_by_year")


def regime_table(res, label, side, regimes_override=None):
    lab = S.holding_regime(res) if regimes_override is None else regimes_override.reindex(res.daily.index)
    r = res.daily["gross"] if side == "long_short" else res.net - bench.net.reindex(res.net.index)
    rows = []
    for g in ("LOW", "MID", "HIGH"):
        x = r[lab == g]
        rows.append({"strategy": label, "measure": "LS gross" if side == "long_short" else "LO active net",
                     "regime": g, "days": len(x), "ann_mean": x.mean() * ANN,
                     "t_NW": newey_west_t(x) if len(x) > 50 else np.nan,
                     "Sharpe": sharpe(x)})
    return rows


rows = []
for name in SIG:
    for side in ("long_short", "long"):
        rows += regime_table(core_res[(name, side)], name, side)
regime_df = pd.DataFrame(rows)
save(regime_df, "regime_conditional", index=False)
OUT["regime"] = regime_df.round(4).to_dict("records")
print(regime_df.round(3).to_string())

# Robustness: India VIX instead of realised volatility
reg_vix = market_regimes(panel.index_close, "INDIAVIX")
rows = []
for name in ("MOM_12_1", "REV_1M"):
    res = core_res[(name, "long_short")]
    dec = res.targets.index
    vix_lab = reg_vix["regime"].reindex(dec)
    cal = R.index
    lab = pd.Series(None, index=cal, dtype=object)
    for d, g in vix_lab.items():
        p = cal.get_loc(d) + 2
        if p < len(cal):
            lab.iloc[p] = g
    rows += regime_table(res, name + " (VIX regimes)", "long_short", lab.ffill())
save(pd.DataFrame(rows), "regime_conditional_vix", index=False)
OUT["regime_vix"] = pd.DataFrame(rows).round(4).to_dict("records")

# Regime-switching strategy (walk-forward: choices use only past outcomes)
log("Regime switch strategy")
cand = {
    "MOM": core_res[("MOM_12_1", "long")],
    "REV": core_res[("REV_1M", "long")],
}
active = {k: v.net - bench.net.reindex(v.net.index) for k, v in cand.items()}
hold_reg = S.holding_regime(cand["MOM"])            # same decision dates for both
dec_dates = uM.index
switch_targets = []
choices = []
bench_t = bench.targets
for d in dec_dates:
    g = reg["regime"].get(d)
    best, best_val = "MOM", None          # default before enough history
    if g is not None:
        vals = {}
        for k, a in active.items():
            past = a[(a.index <= d) & (hold_reg.reindex(a.index) == g)]
            if len(past) >= 252:
                vals[k] = past.mean()
        vals["BENCH"] = 0.0
        if len(vals) > 1:
            best = max(vals, key=vals.get)
            best_val = vals[best]
    choices.append({"date": d, "regime": g, "choice": best})
    src = bench_t if best == "BENCH" else cand[best].targets
    switch_targets.append(src.loc[d])
switch_t = pd.DataFrame(switch_targets, index=dec_dates)
switch_res = S.run(switch_t)
choice_df = pd.DataFrame(choices)
save(choice_df, "regime_switch_choices", index=False)

# In-sample (hindsight) version: per-regime choice from FULL-sample results
full_best = {}
for g in ("LOW", "MID", "HIGH"):
    vals = {k: a[hold_reg.reindex(a.index) == g].mean() for k, a in active.items()}
    vals["BENCH"] = 0.0
    full_best[g] = max(vals, key=vals.get)
hind = []
for d in dec_dates:
    g = reg["regime"].get(d)
    best = full_best.get(g, "MOM") if g is not None else "MOM"
    hind.append((bench_t if best == "BENCH" else cand[best].targets).loc[d])
hind_res = S.run(pd.DataFrame(hind, index=dec_dates))
OUT["regime_switch"] = {
    "walk_forward": fmt_row(S.evaluate(switch_res, "long", bench)),
    "hindsight": fmt_row(S.evaluate(hind_res, "long", bench)),
    "hindsight_rule": full_best,
    "static_mom": fmt_row(S.evaluate(cand["MOM"], "long", bench)),
    "choice_counts": choice_df["choice"].value_counts().to_dict(),
}
print(json.dumps(OUT["regime_switch"], indent=1, default=str)[:1500])

# =========================================================================== #
# 3. Parameter grid, walk-forward selection, deflated Sharpe
# =========================================================================== #
log("Parameter grid")
grid = []
grid_res = {}
for lb in (126, 189, 252):
    for skip in (0, 21):
        sig = momentum(panel.adj_close, lb, skip)
        for q in (0.1, 0.2, 0.3):
            res = S.run(S.targets(sig, "M", q, "long"))
            key = f"MOM lb={lb} skip={skip} q={q}"
            grid_res[key] = res
            grid.append({"family": "MOM", "lookback": lb, "skip": skip, "q": q, "key": key,
                         **fmt_row(S.evaluate(res, "long", bench))})
for lb in (5, 10, 21):
    sig = reversal(panel.adj_close, lb)
    for q in (0.1, 0.2, 0.3):
        res = S.run(S.targets(sig, "M", q, "long"))
        key = f"REV lb={lb} q={q}"
        grid_res[key] = res
        grid.append({"family": "REV", "lookback": lb, "skip": 0, "q": q, "key": key,
                     **fmt_row(S.evaluate(res, "long", bench))})
grid_df = pd.DataFrame(grid)
save(grid_df, "parameter_grid", index=False)

act = {k: v.net - bench.net.reindex(v.net.index) for k, v in grid_res.items()}
keys = list(act)
first_oos = 2013
oos_targets, picks = [], []
for year in range(first_oos, R.index[-1].year + 1):
    cutoff = pd.Timestamp(f"{year}-01-01")
    ir = {}
    for k in keys:
        past = act[k][act[k].index < cutoff]
        ir[k] = past.mean() / past.std()
    best = max(ir, key=ir.get)
    picks.append({"year": year, "pick": best, "trailing_IR_daily": ir[best] * np.sqrt(ANN)})
    dd = dec_dates[dec_dates.year == year]
    oos_targets.append(grid_res[best].targets.loc[dd])
wf_res = S.run(pd.concat(oos_targets))
oos_start = f"{first_oos}-01-01"


def oos(res):
    d = res.daily[res.daily.index >= oos_start]
    return d


bench_oos = bench.net[bench.net.index >= oos_start]
in_sample_best = max(keys, key=lambda k: act[k].mean() / act[k].std())
default_key = "MOM lb=252 skip=21 q=0.2"
trial_srs = [act[k].mean() / act[k].std() for k in keys]
OUT["walk_forward"] = {
    "picks": picks,
    "in_sample_best": in_sample_best,
    "n_trials": len(keys),
    "oos_period_start": oos_start,
    "wf_oos": fmt_row(summarize(oos(wf_res), RF, bench_oos)),
    "default_oos": fmt_row(summarize(oos(grid_res[default_key]), RF, bench_oos)),
    "in_sample_best_oos_period": fmt_row(summarize(oos(grid_res[in_sample_best]), RF, bench_oos)),
    "in_sample_best_full": fmt_row(summarize(grid_res[in_sample_best].daily, RF, bench.net)),
    "default_full": fmt_row(summarize(grid_res[default_key].daily, RF, bench.net)),
    "PSR_in_sample_best_active": probabilistic_sharpe(act[in_sample_best]),
    "DSR_in_sample_best_active": deflated_sharpe(act[in_sample_best], trial_srs),
    "PSR_default_active": probabilistic_sharpe(act[default_key]),
    "DSR_default_active": deflated_sharpe(act[default_key], trial_srs),
}
print(json.dumps({k: v for k, v in OUT["walk_forward"].items() if k != "picks"}, indent=1, default=str))
save(pd.DataFrame(picks), "walk_forward_picks", index=False)

# =========================================================================== #
# 4. Cost sensitivity
# =========================================================================== #
log("Cost sensitivity")
rows = []
spec = {"MOM_12_1": ("MOM_12_1", "long"), "REV_1M": ("REV_1M", "long"), "REV_1W": ("REV_1W", "long")}
for slip in (0, 5, 10, 20, 30, 50):
    cm = CostModel(slippage_bps=slip)
    b = S.benchmark("M", cm)
    for name, key in spec.items():
        res = S.run(core_res[key].targets, cm)
        s = S.evaluate(res, "long", b)
        rows.append({"strategy": name, "slippage_bps": slip, "round_trip_bps": cm.round_trip_bps(),
                     "Active_ret": s["Active_ret"], "IR": s["IR"], "Sharpe": s["Sharpe"], "CAGR": s["CAGR"]})
cm_small = CostModel(capital_inr=1_000_000)
b_small = S.benchmark("M", cm_small)
for name, key in spec.items():
    res = S.run(core_res[key].targets, cm_small)
    s = S.evaluate(res, "long", b_small)
    rows.append({"strategy": name, "slippage_bps": 10, "round_trip_bps": cm_small.round_trip_bps(),
                 "Active_ret": s["Active_ret"], "IR": s["IR"], "Sharpe": s["Sharpe"], "CAGR": s["CAGR"],
                 "note": "capital Rs 10 lakh"})
cost_df = pd.DataFrame(rows)
save(cost_df, "cost_sensitivity", index=False)
OUT["cost_sensitivity"] = cost_df.round(4).to_dict("records")

# =========================================================================== #
# 5. Bias audit
# =========================================================================== #
log("Bias audit")
mom = SIG["MOM_12_1"][0]
n100 = pd.read_csv(ROOT / "data" / "raw" / "nifty100_current.csv")["Symbol"].str.strip().tolist()
u_surv = fixed_list_universe(panel, S.dates["M"], n100)
raw_prices = pd.read_csv(ROOT / "data" / "raw" / "prices_daily.csv.gz", parse_dates=["date"])
raw_adj = raw_prices.pivot(index="date", columns="symbol", values="adj_close").reindex(panel.adj_close.index)
raw_panel = Panel(raw_adj, raw_prices.pivot(index="date", columns="symbol", values="close").reindex(raw_adj.index),
                  raw_prices.pivot(index="date", columns="symbol", values="volume").reindex(raw_adj.index),
                  panel.index_close, pd.DataFrame())
raw_R = raw_adj.pct_change(fill_method=None)
raw_univ = liquid_universe(raw_panel, S.dates["M"])
mom_raw = momentum(raw_adj, 252, 21)
best_lb, best_skip, best_q = (grid_df.set_index("key").loc[in_sample_best, ["lookback", "skip", "q"]])
mom_best = momentum(panel.adj_close, int(best_lb), int(best_skip))
peek = mom.shift(-1)          # off-by-one bug: signal "knows" tomorrow's close

variants = {
    "1. Clean baseline": dict(sig=mom, univ=None, cost=None, lag=1, q=0.2),
    "2. Same-close execution (lag 0)": dict(sig=mom, univ=None, cost=None, lag=0, q=0.2),
    "3. Off-by-one look-ahead (signal uses t+1)": dict(sig=peek, univ=None, cost=None, lag=0, q=0.2),
    "4. Survivorship: today's NIFTY 100 as universe": dict(sig=mom, univ=u_surv, cost=None, lag=1, q=0.2),
    "5. No transaction costs": dict(sig=mom, univ=None, cost=NO_COSTS, lag=1, q=0.2),
    "6. In-sample best parameters": dict(sig=mom_best, univ=None, cost=None, lag=1, q=float(best_q)),
    "7. Uncleaned vendor data": dict(sig=mom_raw, univ=raw_univ, cost=None, lag=1, q=0.2, R=raw_R),
    "8. All of 3-6 combined": dict(sig=mom_best.shift(-1), univ=u_surv, cost=NO_COSTS, lag=0, q=float(best_q)),
}
rows = []
for name, v in variants.items():
    out = {}
    for side in ("long", "long_short"):
        u = v["univ"] if v["univ"] is not None else uM
        t = S.targets(v["sig"], "M", v["q"], side, universe=u)
        res = S.run(t, v["cost"], v["lag"], returns=v.get("R"))
        out[side] = res
    lo, ls = out["long"], out["long_short"]
    rows.append({"variant": name,
                 "LO_CAGR": lo.daily["net"].pipe(lambda r: (1 + r).prod() ** (ANN / len(r)) - 1),
                 "LO_Sharpe": sharpe(lo.net, RF),
                 "LO_active_vs_clean_EW": (lo.net - bench.net.reindex(lo.net.index)).mean() * ANN,
                 "LS_Sharpe": sharpe(ls.net)})
# Same off-by-one bug, applied to signals of different speeds (long-short, gross)
speed_rows = []
for name, (sig, freq) in SIG.items():
    clean = S.run(S.targets(sig, freq, 0.2, "long_short"))
    bug = S.run(S.targets(sig.shift(-1), freq, 0.2, "long_short"), exec_lag=0)
    speed_rows.append({"signal": name, "LS_gross_Sharpe_clean": sharpe(clean.gross),
                       "LS_gross_Sharpe_with_bug": sharpe(bug.gross),
                       "LS_gross_CAGR_with_bug": (1 + bug.gross).prod() ** (ANN / len(bug.gross)) - 1})
speed = pd.DataFrame(speed_rows)
save(speed, "lookahead_by_signal_speed", index=False)
OUT["lookahead_by_speed"] = speed.round(4).to_dict("records")
print(speed.round(3).to_string())
surv_bench = S.run(S.targets(u_surv.astype(float), "M", side="universe", universe=u_surv))
audit = pd.DataFrame(rows)
base = audit.iloc[0]
audit["LO_Sharpe_inflation"] = audit["LO_Sharpe"] - base["LO_Sharpe"]
audit["LS_Sharpe_inflation"] = audit["LS_Sharpe"] - base["LS_Sharpe"]
save(audit, "bias_audit", index=False)
OUT["bias_audit"] = audit.round(4).to_dict("records")
OUT["survivorship_benchmark"] = {"EW_point_in_time_CAGR": float((1 + bench.net).prod() ** (ANN / len(bench.net)) - 1),
                                 "EW_todays_nifty100_CAGR": float((1 + surv_bench.net).prod() ** (ANN / len(surv_bench.net)) - 1)}
print(audit.round(3).to_string())
print(OUT["survivorship_benchmark"])

# =========================================================================== #
# 6. Statistical confidence for headline strategies
# =========================================================================== #
log("Bootstrap / PSR")
rows = []
for label, r, rf in (("MOM_12_1 long (net, vs rf)", core_res[("MOM_12_1", "long")].net, RF),
                     ("MOM_12_1 active vs EW (net)", act[default_key], 0.0),
                     ("MOM_12_1 long-short (net)", core_res[("MOM_12_1", "long_short")].net, 0.0),
                     ("REV_1M long-short (gross)", core_res[("REV_1M", "long_short")].gross, 0.0),
                     ("REV_1W long-short (gross)", core_res[("REV_1W", "long_short")].gross, 0.0),
                     ("REV_1W long-short (net)", core_res[("REV_1W", "long_short")].net, 0.0),
                     ("EW universe (net, vs rf)", bench.net, RF)):
    lo, hi = block_bootstrap_sharpe_ci(r, rf_annual=rf)
    ex = r - ((1 + rf) ** (1 / ANN) - 1 if rf else 0)
    rows.append({"series": label, "Sharpe": sharpe(r, rf), "CI95_lo": lo, "CI95_hi": hi,
                 "PSR_vs_0": probabilistic_sharpe(ex)})
conf = pd.DataFrame(rows)
save(conf, "confidence", index=False)
OUT["confidence"] = conf.round(4).to_dict("records")
print(conf.round(3).to_string())

# =========================================================================== #
# 7. Walk-forward ML ranker
# =========================================================================== #
log("ML ranker")
feats = ml.feature_panel(panel.adj_close, R, panel.traded_value)
ds = ml.build_dataset(feats, uM, R)
ml_rows = []
ml_res = {}
for model in ("ridge", "gbm"):
    pred = ml.walk_forward_predict(ds, model)
    t = S.targets(pred.reindex(index=uM.index), "M", 0.2, "long")
    res = S.run(t)
    ml_res[model] = res
    ic = ml.rank_ic_by_year(pred, ds)
    ml_rows.append({"model": model, "mean_rank_IC": ic.mean(), "IC_years_positive": f"{(ic > 0).sum()}/{len(ic)}",
                    **fmt_row(summarize(oos(res), RF, bench_oos))})
ml_rows.append({"model": "MOM_12_1 (single signal, same period)",
                **fmt_row(summarize(oos(core_res[("MOM_12_1", "long")]), RF, bench_oos))})
ml_df = pd.DataFrame(ml_rows)
save(ml_df, "ml_ranker", index=False)
coefs = ml.ridge_coefficients(ds)
save(coefs.to_frame("ridge_coef_full_sample"), "ml_ridge_coefficients")
OUT["ml"] = ml_df.round(4).to_dict("records")
OUT["ml_coefs"] = coefs.round(4).to_dict()
print(ml_df.round(3).to_string()); print(coefs.round(4))

# =========================================================================== #
# Save everything for figures / write-up
# =========================================================================== #
(ROOT / "results" / "results.json").write_text(json.dumps(OUT, indent=2, default=str))
series = {
    "bench": bench.net, "nifty50_price": nifty,
    **{f"{k[0]}|{k[1]}": v.net for k, v in core_res.items()},
    **{f"{k[0]}|{k[1]}|gross": v.gross for k, v in core_res.items()},
    "switch_wf": switch_res.net, "wf_params": wf_res.net,
    "ml_gbm": ml_res["gbm"].net, "ml_ridge": ml_res["ridge"].net,
}
pickle.dump({"series": series, "regimes": reg, "grid": grid_df, "cost": cost_df, "audit": audit,
             "regime_df": regime_df, "yr": yr, "speed": speed}, open(ROOT / "results" / "cache.pkl", "wb"))
log("Done")
