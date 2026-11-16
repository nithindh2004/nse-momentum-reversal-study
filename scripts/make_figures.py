"""Render the figures in results/figures/ from results/cache.pkl.

    python scripts/make_figures.py

Colour roles (validated categorical palette, light mode):
    series 1 blue #2a78d6, series 2 orange #eb6834, series 3 aqua #1baf7a,
    benchmark / reference = neutral grey; diverging = blue <-> red, grey midpoint.
Every multi-series chart is direct-labelled so identity never relies on colour.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
C = pickle.load(open(ROOT / "results" / "cache.pkl", "rb"))

SURFACE, INK, INK2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, RED, NEUTRAL = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#8a897f"
DIVERGING = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
})


def pct(x, d=0):
    return f"{x * 100:.{d}f}%"


def note(fig, text):
    fig.text(0.01, -0.02, text, ha="left", va="top", fontsize=8, color=INK2)


# --------------------------------------------------------------------------- #
# 1. Growth of Rs 1 (log scale)
# --------------------------------------------------------------------------- #
s = C["series"]
curves = [
    ("12-1 momentum (top 20%)", s["MOM_12_1|long"], BLUE, 2.0),
    ("1-month reversal (top 20%)", s["REV_1M|long"], ORANGE, 2.0),
    ("Equal-weight universe (benchmark)", s["bench"], NEUTRAL, 1.6),
]
fig, ax = plt.subplots(figsize=(9, 4.6))
ends = []
for label, r, col, lw in curves:
    w = (1 + r).cumprod()
    cagr = w.iloc[-1] ** (252 / len(w)) - 1
    ax.plot(w.index, w, color=col, lw=lw)
    ends.append((w.iloc[-1], f"{label}\n{pct(cagr, 1)} a year", col))
ax.set_yscale("log")
ax.set_yticks([1, 2, 4, 8, 16])
ax.set_yticklabels(["Rs 1", "Rs 2", "Rs 4", "Rs 8", "Rs 16"])
ax.set_ylabel("Value of Rs 1 invested (log scale)")
ends.sort()
last_y = 0
for y, txt, col in ends:
    y_pos = max(y, last_y * 1.35) if last_y else y
    ax.annotate(txt, (s["bench"].index[-1], y_pos), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=8.5, color=INK)
    ax.plot([s["bench"].index[-1]], [y], "o", ms=5, color=col)
    last_y = y_pos
ax.set_xlim(right=s["bench"].index[-1] + pd.Timedelta(days=5))
ax.set_title("Momentum beat the benchmark by ~5 pts a year; 1-month reversal lagged it after costs")
note(fig, "NSE top-100 stocks by liquidity (point-in-time), monthly rebalance, net of STT, stamp duty, "
          "exchange/SEBI fees, GST, DP charges and 10 bps slippage. Mar-2010 to Nov-2026.")
fig.savefig(FIG / "01_growth_of_1.png")
plt.close(fig)

# --------------------------------------------------------------------------- #
# 2. Regime-conditional returns (long-short, gross)
# --------------------------------------------------------------------------- #
rd = C["regime_df"]
rd = rd[rd["measure"] == "LS gross"]
regimes = ["LOW", "MID", "HIGH"]
strats = [("MOM_12_1", "12-1 momentum", BLUE), ("REV_1M", "1-month reversal", ORANGE),
          ("REV_1W", "1-week reversal", AQUA)]
fig, ax = plt.subplots(figsize=(9, 4.4))
width = 0.26
x = np.arange(len(regimes))
for i, (key, label, col) in enumerate(strats):
    vals = [rd[(rd.strategy == key) & (rd.regime == g)]["ann_mean"].iloc[0] for g in regimes]
    tst = [rd[(rd.strategy == key) & (rd.regime == g)]["t_NW"].iloc[0] for g in regimes]
    bars = ax.bar(x + (i - 1) * width, np.array(vals) * 100, width * 0.92, color=col, label=label)
    for b, v, t in zip(bars, vals, tst):
        ax.annotate(f"{v * 100:+.0f}%\nt={t:.1f}", (b.get_x() + b.get_width() / 2, v * 100),
                    xytext=(0, 3 if v >= 0 else -3), textcoords="offset points",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5, color=INK2)
ax.axhline(0, color=AXIS, lw=1)
ax.set_xticks(x, ["Calm market\n(low volatility)", "Normal", "Stressed market\n(high volatility)"])
ax.set_ylabel("Long-short return, % a year (before costs)")
ax.set_ylim(-20, 40)
ax.legend(loc="upper left", ncol=3, fontsize=9)
ax.set_title("Momentum pays in calm markets; reversal pays when volatility is high")
note(fig, "Regime = NIFTY 500 21-day realised volatility vs. terciles of its OWN PAST history (no look-ahead), "
          "measured at each rebalance date. t = Newey-West t-statistic.")
fig.savefig(FIG / "02_regime_returns.png")
plt.close(fig)

# --------------------------------------------------------------------------- #
# 3. Bias audit
# --------------------------------------------------------------------------- #
au = C["audit"].copy()
au["label"] = au["variant"].str.replace(r"^\d\. ", "", regex=True)
au = au.iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 4.6))
cols = [ORANGE if v.startswith("1.") else BLUE for v in au["variant"]]
bars = ax.barh(au["label"], au["LO_Sharpe"], color=cols, height=0.62)
base = au[au["variant"].str.startswith("1.")]["LO_Sharpe"].iloc[0]
ax.axvline(base, color=ORANGE, lw=1, ls="--")
for b, v, c in zip(bars, au["LO_Sharpe"], au["LO_CAGR"]):
    ax.annotate(f"Sharpe {v:.2f}  |  CAGR {pct(c, 1)}", (v, b.get_y() + b.get_height() / 2),
                xytext=(5, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK2)
ax.set_xlim(0, max(au["LO_Sharpe"]) * 1.45)
ax.set_xlabel("Sharpe ratio of the long-only momentum portfolio (risk-free 6%)")
ax.grid(axis="y", visible=False)
ax.set_title("Same strategy, careless backtest: Sharpe 0.56 becomes 1.29")
note(fig, "Orange = the honest baseline. Each bar changes ONE thing; the last bar combines look-ahead, survivorship, "
          "no costs and in-sample parameters.\nSurvivorship alone adds more Sharpe than the other four single errors added together.")
fig.savefig(FIG / "03_bias_audit.png")
plt.close(fig)

# --------------------------------------------------------------------------- #
# 4. Cost sensitivity
# --------------------------------------------------------------------------- #
cs = C["cost"]
cs = cs[cs["note"].isna()] if "note" in cs else cs
fig, ax = plt.subplots(figsize=(9, 4.4))
for key, label, col in (("MOM_12_1", "12-1 momentum", BLUE), ("REV_1M", "1-month reversal", ORANGE),
                        ("REV_1W", "1-week reversal", AQUA)):
    d = cs[cs.strategy == key]
    ax.plot(d["round_trip_bps"], d["Active_ret"] * 100, color=col, lw=2, marker="o", ms=5)
    ax.annotate(label, (d["round_trip_bps"].iloc[-1], d["Active_ret"].iloc[-1] * 100),
                xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
ax.axhline(0, color=AXIS, lw=1)
stat = cs["round_trip_bps"].min()
ax.axvspan(0, stat, color="#f0efec", zorder=0)
ax.text(stat / 2, -40, "Below the\nstatutory\nminimum", ha="center", fontsize=8, color=INK2)
ax.axvline(42.2, color=INK2, lw=1, ls="--")
ax.text(43.5, -40, "Base case\n(10 bps slippage)", fontsize=8, color=INK2)
ax.set_xlim(0, 150)
ax.set_xlabel("Round-trip trading cost, basis points (STT + stamp + fees + 2 x slippage)")
ax.set_ylabel("Return vs benchmark, % a year")
ax.set_title("Weekly reversal loses money even before any slippage: STT alone kills it")
note(fig, "Long-only top-20% portfolios vs equal-weight universe, net of costs. "
          "Statutory costs alone are ~22 bps per round trip on delivery trades.")
fig.savefig(FIG / "04_cost_sensitivity.png")
plt.close(fig)

# --------------------------------------------------------------------------- #
# 5. Parameter heatmap (momentum)
# --------------------------------------------------------------------------- #
g = C["grid"]
g = g[g.family == "MOM"]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
norm = TwoSlopeNorm(vcenter=0, vmin=-0.8, vmax=0.8)
for ax, skip in zip(axes, (21, 0)):
    m = g[g.skip == skip].pivot(index="lookback", columns="q", values="IR").sort_index(ascending=False)
    ax.imshow(m.values, cmap=DIVERGING, norm=norm, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10,
                    color="white" if abs(v) > 0.55 else INK)
    ax.set_xticks(range(m.shape[1]), [f"top {int(q * 100)}%" for q in m.columns])
    ax.set_yticks(range(m.shape[0]), [f"{int(lb / 21)} months" for lb in m.index])
    ax.set_title("Skip last month" if skip else "No skip", fontsize=10)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
axes[0].set_ylabel("Look-back window")
fig.suptitle("12-month momentum is robust to its settings; 6-month momentum adds little (IR vs benchmark, net)",
             x=0.01, ha="left", fontweight="bold", fontsize=12, y=1.04)
note(fig, "27 variants were tested in total (18 momentum + 9 reversal); the Deflated Sharpe Ratio in the "
          "README accounts for that search.")
fig.savefig(FIG / "05_parameter_heatmap.png")
plt.close(fig)

# --------------------------------------------------------------------------- #
# 6. Calendar-year active return of momentum
# --------------------------------------------------------------------------- #
yr = C["yr"]
fig, ax = plt.subplots(figsize=(9, 3.8))
v = yr["MOM active"] * 100
ax.bar(v.index.astype(str), v, color=[BLUE if x >= 0 else RED for x in v], width=0.7)
for i, (yy, x) in enumerate(v.items()):
    ax.annotate(f"{x:+.1f}" if abs(x) < 1 else f"{x:+.0f}", (i, x), xytext=(0, 3 if x >= 0 else -3), textcoords="offset points",
                ha="center", va="bottom" if x >= 0 else "top", fontsize=8, color=INK2)
ax.axhline(0, color=AXIS, lw=1)
ax.set_ylabel("Momentum minus benchmark, % pts")
ax.set_ylim(v.min() - 4, v.max() + 4)
ax.grid(axis="x", visible=False)
won = int((v > 0).sum())
ax.set_title(f"Momentum beat the benchmark in {won} of {len(v)} calendar years, with a 3-year losing run in 2020-22")
note(fig, "2010 (from March) and 2026 (to 13-Nov) are partial years. Blue = beat the benchmark, red = lagged it; values are also labelled.")
fig.savefig(FIG / "06_calendar_year_active.png")
plt.close(fig)

print("figures written to", FIG)
