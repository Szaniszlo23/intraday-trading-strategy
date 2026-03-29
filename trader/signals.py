"""
Signal generation for the intraday momentum strategy.

Logic
-----
  +1 (long)  if close > upper_band AND close > VWAP [AND RSI > rsi_long]
  -1 (short) if close < lower_band AND close < VWAP [AND RSI < rsi_short]
   0 (flat)  otherwise

Bands:
  upper_band = max(open, prev_close) * (1 + band_mult * sigma_open)
  lower_band = min(open, prev_close) * (1 - band_mult * sigma_open)

Signals are evaluated at trade_freq-minute intervals and forward-filled.
A 1-bar execution delay is applied (signal at t → trade at t+1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.config import StrategyConfig


class SignalGenerator:
    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config or StrategyConfig()

    def generate(self, day_df: pd.DataFrame, prev_close: float) -> pd.Series:
        """
        Produce a minute-level exposure series for a single trading day.

        Parameters
        ----------
        day_df    : intraday bars for the day (close, vwap, sigma_open,
                    min_from_open, rsi optional)
        prev_close: last close of the previous trading day

        Returns
        -------
        pd.Series[float] with values in {-1, 0, +1}, indexed like day_df.
        """
        cfg = self.config
        open_price = day_df["close"].iloc[0]
        close_prices = day_df["close"]
        vwap = day_df["vwap"]
        sigma = day_df["sigma_open"]

        upper_band = max(open_price, prev_close) * (1 + cfg.band_mult * sigma)
        lower_band = min(open_price, prev_close) * (1 - cfg.band_mult * sigma)

        raw = pd.Series(0.0, index=day_df.index)
        long_cond = (close_prices > upper_band) & (close_prices > vwap)
        short_cond = (close_prices < lower_band) & (close_prices < vwap)

        if cfg.rsi_filter and "rsi" in day_df.columns:
            rsi = day_df["rsi"]
            long_cond = long_cond & (rsi > cfg.rsi_long)
            short_cond = short_cond & (rsi < cfg.rsi_short)

        raw[long_cond] = 1.0
        raw[short_cond] = -1.0

        # Sample at trade_freq intervals only
        trade_mask = (day_df["min_from_open"] % cfg.trade_freq == 0)
        sampled = pd.Series(np.nan, index=day_df.index)
        sampled[trade_mask] = raw[trade_mask]

        exposure = self._forward_fill(sampled)
        return exposure.shift(1).fillna(0.0)

    @staticmethod
    def _forward_fill(sampled: pd.Series) -> pd.Series:
        """Forward-fill signal; a sampled 0 resets the carry (flat)."""
        carry = np.nan
        result = []
        for val in sampled:
            if not np.isnan(val):
                carry = val if val != 0 else np.nan
            result.append(carry)
        return pd.Series(result, index=sampled.index).fillna(0.0)
