"""Indian cash-equity (delivery) transaction-cost model.

Statutory rates (verified Nov-2026 against broker charge pages and exchange
circulars; see README "Cost model" for links):

    STT                 0.1% buy + 0.1% sell  (0.125% before 1-Jul-2012)
    Stamp duty          0.015% on buy          (uniform national rate since 1-Jul-2020;
                                                applied to the whole sample)
    NSE transaction     0.00297% of turnover
    SEBI turnover fee   0.0001% (Rs 10 per crore)
    GST                 18% on (brokerage + exchange + SEBI fees)
    Brokerage           0 (discount brokers charge nothing for delivery)
    DP charge           Rs 15.34 per scrip sold (Zerodha/CDSL)

Modelling ASSUMPTIONS (labelled, varied in the sensitivity analysis):
    * slippage / market impact of `slippage_bps` per side (default 10 bps) for
      top-100-by-liquidity stocks;
    * the fixed DP charge is converted to a rate using `capital_inr`
      (default Rs 1 crore) -- a smaller account pays proportionally more
      (Rs 10 lakh is shown as a sensitivity case).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

STT_CUT_DATE = pd.Timestamp("2012-07-01")


@dataclass(frozen=True)
class CostModel:
    slippage_bps: float = 10.0
    capital_inr: float = 10_000_000.0
    stt_post_2012: float = 0.001
    stt_pre_2012: float = 0.00125
    stamp_buy: float = 0.00015
    exchange: float = 0.0000297
    sebi: float = 0.000001
    gst: float = 0.18
    brokerage: float = 0.0
    dp_charge_inr: float = 15.34
    enabled: bool = True

    def stt(self, date: pd.Timestamp) -> float:
        return self.stt_pre_2012 if date < STT_CUT_DATE else self.stt_post_2012

    def _fees(self) -> float:
        fees = self.brokerage + self.exchange + self.sebi
        return fees * (1 + self.gst)

    def buy_rate(self, date: pd.Timestamp) -> float:
        """Cost per rupee bought, as a fraction of the trade value."""
        if not self.enabled:
            return 0.0
        return self.stt(date) + self.stamp_buy + self._fees() + self.slippage_bps / 1e4

    def sell_rate(self, date: pd.Timestamp) -> float:
        """Cost per rupee sold (excluding the per-scrip DP charge)."""
        if not self.enabled:
            return 0.0
        return self.stt(date) + self._fees() + self.slippage_bps / 1e4

    def dp_cost(self, n_scrips_sold: int) -> float:
        """Fixed DP charges as a fraction of portfolio capital."""
        if not self.enabled:
            return 0.0
        return n_scrips_sold * self.dp_charge_inr / self.capital_inr

    def round_trip_bps(self, date: pd.Timestamp = pd.Timestamp("2026-01-01")) -> float:
        return (self.buy_rate(date) + self.sell_rate(date)) * 1e4


NO_COSTS = CostModel(enabled=False)
