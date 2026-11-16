# Research Note: Momentum, Reversal and Regimes in NSE Large Caps

*Methodology and discussion behind the README results. All figures are from `scripts/run_all.py`.
Sample: decisions from Jan-2010, returns Mar-2010 to 13-Nov-2026 (about 16.7 years).*

---

## 1. Hypotheses (written before the backtests were run)

- **H1, momentum:** stocks with the highest returns over the past 12 months, excluding the most recent month, outperform over the next month (Jegadeesh & Titman, 1993).
- **H2, reversal:** stocks with the lowest returns over the past week or month outperform over the next week or month. The usual explanation is a reward for supplying liquidity to forced or over-reacting sellers (Jegadeesh, 1990; Lehmann, 1990).
- **H3, regimes:** momentum suffers in high-volatility markets (the "momentum crash" literature, e.g. Daniel & Moskowitz, 2016), while reversal profits *increase*, because liquidity providers are paid more when volatility is high (Nagel, 2012).
- **H4, implementability:** after Indian delivery costs, only slow signals survive.

The default specifications (12-1 lookback, top/bottom 20%, monthly rebalancing) were fixed in
advance. Everything else is labelled as a variant and counted as a trial.

## 2. Method

### 2.1 Data
Daily OHLCV for 504 NSE symbols from Yahoo Finance (2009–2026). The pool is the current NIFTY 500
plus 37 hand-listed former large caps; 5 of those could be retrieved. Cleaning is rule-based and
logged (README, "Data quality"). The cleaned panel is stored in SQLite (`data/processed/market.sqlite`,
tables `prices`, `indices`, `corrections`).

### 2.2 Universe
At each month-end *t*:

    eligible_i,t = traded on t  AND  ≥273 sessions of history  AND  traded on ≥90% of the last 252 sessions
    liquidity_i,t = median over the last 126 sessions of (close × volume)
    universe_t   = top 100 eligible stocks by liquidity

### 2.3 Signals and portfolios
- `MOM(L, s)` = P(t−s) / P(t−L) − 1, with defaults L = 252, s = 21
- `REV(L)` = −[P(t) / P(t−L) − 1], with L = 21 (monthly) or L = 5 (weekly)

Stocks are ranked within the universe. Long-only holds the top quintile at equal weight.
Long-short adds −1/n weights on the bottom quintile.

### 2.4 Backtest mechanics
- Decision at close *t*; trade at close *t+1*. Returns accrue from *t+2*.
- Weights drift between rebalances, so turnover is the actual trade needed, not an approximation.
- Cost of each trade = buys × (STT + stamp + fees·1.18 + slippage) + sells × (STT + fees·1.18 + slippage) + ₹15.34 × (number of long positions reduced) / capital.
- STT is date-aware: 0.125% before 1-Jul-2012 and 0.1% after (Union Budget 2012-13).
- Capital is ₹1 crore by default; ₹10 lakh is shown as a sensitivity case (the DP charge matters more for small accounts).

### 2.5 Regimes
rv_t = σ(NIFTY 500 daily returns over the last 21 sessions)·√252. The regime at *t* compares rv_t
with the 1/3 and 2/3 quantiles of {rv_s : s ≤ t}, using at least 252 observations. Each holding
period takes the regime known at its decision date, so a trader could have used it.

### 2.6 Statistics
- **Newey-West t-statistics** (automatic lag) on daily returns, which allow for autocorrelation from overlapping information.
- **Circular block bootstrap** (21-day blocks, 2,000 draws) for Sharpe confidence intervals. It keeps volatility clustering.
- **Probabilistic Sharpe Ratio** (Bailey & López de Prado, 2012): the probability that the true Sharpe exceeds 0, adjusting for sample length, skewness and kurtosis.
- **Deflated Sharpe Ratio** (Bailey & López de Prado, 2014): the PSR measured against the Sharpe ratio expected from the *best of N* skill-less trials, with N = 27 and the variance taken from the trials themselves.

## 3. Results by hypothesis

**H1, partly supported.** Long-only momentum beats the point-in-time equal-weight benchmark by
4.8 pts/yr net (IR 0.46, NW t = 1.74). It has a positive active return in 11 of 17 calendar years.
The 95% bootstrap CI for the IR, [-0.09, 0.98], includes zero, so this is suggestive evidence,
not proof.

The long-short version is weaker (net Sharpe 0.22). In this sample, buying winners did more work
than shorting losers. That matters for implementation, because in India the long side is the
easy one to trade.

Twelve-month look-backs were robust across holding fractions (IR 0.41–0.70). Six-month
momentum with a skip month had no edge (IR ≈ −0.05). Skipping the last month did *not* help
here (IR 0.46 with the skip, 0.49 without), which is consistent with the weak 1-month reversal
in calm markets.

**H2, supported before costs, rejected after costs.** Weekly reversal has the highest gross
long-short Sharpe of any signal (0.55, CI 0.10–1.02). It turns over its book about 83 times a
year, so its break-even cost is 5.5 bps per side. STT alone is 10 bps per side on delivery
trades. One-month reversal is weaker gross (Sharpe 0.27) and also negative net.

**H3, supported in-sample, not exploitable out of sample.** The regime split shows the predicted
mirror image:

| LS gross, % a year | Calm | Normal | Stressed |
|---|---|---|---|
| 12-1 momentum | +13.5 (t 1.8) | +9.2 | −10.0 (t −0.7) |
| 1-month reversal | −4.5 | +12.7 (t 1.6) | +13.6 (t 1.1) |
| 1-week reversal | +2.2 | +7.6 | **+29.2 (t 2.7)** |

India VIX regimes give the same ordering more strongly (momentum +22.0%, t = 3.6, in low-VIX
periods; 1-month reversal +29.1% in high-VIX periods).

The walk-forward switching rule sends each month to whichever of momentum, reversal or the
benchmark had the best past active return in the current regime. It earned an IR of 0.47 vs 0.46
for static momentum. The same rule built with hindsight shows 0.61. The gap between those two
numbers is the lesson: a pattern that is clear in a full-sample table can still be too episodic
to trade. The rule needs a year of past days in a regime before it trusts it, and high-volatility
regimes arrive rarely and in bursts.

**H4, supported.** Only momentum, the slowest signal at 3.3x turnover a year, has a meaningful
cost cushion: a break-even of 93 bps per side against about 21 bps modelled.

## 4. Can a model do better than one signal?

Seven cross-sectionally ranked features feed the models: 12-1 and 6-1 momentum, 1-month and
1-week reversal, 3-month volatility, distance from the 52-week high, and liquidity. They predict
the next month's cross-sectional return rank. The models are ridge regression and a shallow
gradient-boosting regressor (`HistGradientBoostingRegressor`, depth 3, 200 trees, minimum leaf
size 200), retrained every January on an expanding window. The window only includes labels
fully known before the cut-off: an embargo of one holding period.

| Out of sample 2013–2026 | Rank IC | Years IC > 0 | Turnover | Active (net) |
|---|---|---|---|---|
| Ridge | 0.034 | 9 / 14 | 6.1x | −1.6% |
| Gradient boosting | 0.038 | 11 / 14 | 7.2x | −0.4% |
| 12-1 momentum alone | – | – | 3.3x | **+4.1%** |

Both models learned something real. Their positive IC lines up with the full-sample ridge
coefficients: momentum and 52-week-high proximity positive, volatility negative. But they lean
on the fast reversal features, which roughly doubles turnover, so the predictive gain is spent
on STT. A cost-aware objective, for example penalising changes in predictions, is the obvious
next experiment. The negative result is reported as it is.

## 5. What the bias audit teaches

1. **Survivorship is the largest single error.** Using today's NIFTY 100 as the historical universe raises the equal-weight benchmark from 12.4% to 22.1% CAGR. Today's index is full of stocks that were mid-caps in 2010 and are large now *because* they went up.
2. **Parameter search is second.** Picking the best of 27 variants adds 0.16 of Sharpe that the Deflated Sharpe Ratio then takes away (DSR 0.49).
3. **Look-ahead bias depends on the signal's speed.** One day of leaked information barely changes a 252-day signal. It completely changes a 5-day one. Look-ahead bias isn't a constant markup: it can turn a profitable signal into an apparently disastrous one.
4. **Cleaning the vendor data changed little for this universe** (Sharpe −0.03), because most glitches sat in stocks too illiquid to enter the top 100. That is itself a useful result: liquidity screens are also data-quality screens.

## 6. Limitations and threats to validity

See the README's "Limitations" section. The most important one is residual survivorship: 32 of 37
known delisted or merged large caps are missing from free data. The remaining caveats are the
assumed slippage, the small number of high-volatility episodes, and the use of a single market
and a single 16-year sample.

## References

- Bailey, D. & López de Prado, M. (2012). *The Sharpe Ratio Efficient Frontier.* Journal of Risk.
- Bailey, D. & López de Prado, M. (2014). *The Deflated Sharpe Ratio.* Journal of Portfolio Management.
- Daniel, K. & Moskowitz, T. (2016). *Momentum Crashes.* Journal of Financial Economics.
- Jegadeesh, N. (1990). *Evidence of Predictable Behavior of Security Returns.* Journal of Finance.
- Jegadeesh, N. & Titman, S. (1993). *Returns to Buying Winners and Selling Losers.* Journal of Finance.
- Lehmann, B. (1990). *Fads, Martingales, and Market Efficiency.* Quarterly Journal of Economics.
- Nagel, S. (2012). *Evaporating Liquidity.* Review of Financial Studies.
