"""
Signal generation for the intraday momentum strategy.

SignalGenerator  — Baseline strategy
--------------------------------------
  +1 (long)  if close > upper_band AND close > VWAP
  -1 (short) if close < lower_band AND close < VWAP
   0 (flat)  otherwise

  Bands:
    upper_band = max(open, prev_close) * (1 + band_mult * sigma_open)
    lower_band = min(open, prev_close) * (1 - band_mult * sigma_open)

  Signals are evaluated at trade_freq-minute intervals and forward-filled.
  A 1-bar execution delay is applied (signal at t → trade at t+1).

EnhancedSignalGenerator  — Improved strategy
----------------------------------------------
  Same boundary formula, but:
  - Watches every 1-minute bar (no 30-min clock)
  - Dual-bar confirmation: bar N close beyond boundary AND bar N+1 open
    beyond the same boundary → enter at open of bar N+1
  - VWAP trailing stop for exits:
      long  stop = max(upper_band[i], VWAP[i])
      short stop = min(lower_band[i], VWAP[i])
  - No new entries after 15:30; force-flat handled by EOD close at 16:00
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.config import StrategyConfig

_ENTRY_CUTOFF_MINUTES = 15 * 60 + 30  # 15:30 in minutes since midnight


class SignalGenerator:
    """Baseline noise-band + VWAP strategy (30-min clock, 14-day sigma)."""

    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config or StrategyConfig()

    def generate(self, day_df: pd.DataFrame, prev_close: float) -> pd.Series:
        """
        Produce a minute-level exposure series for a single trading day.

        Parameters
        ----------
        day_df    : intraday bars for the day (close, vwap, sigma_open,
                    min_from_open)
        prev_close: last close of the previous trading day

        Returns
        -------
        pd.Series[float] with values in {-1, 0, +1}, indexed like day_df.
        """
        cfg = self.config
        open_price = day_df["open"].iloc[0]
        close_prices = day_df["close"]
        vwap = day_df["vwap"]
        sigma = day_df["sigma_open"]

        upper_band = max(open_price, prev_close) * (1 + cfg.band_mult * sigma)
        lower_band = min(open_price, prev_close) * (1 - cfg.band_mult * sigma)

        raw = pd.Series(0.0, index=day_df.index)
        long_cond = (close_prices > upper_band) & (close_prices > vwap)
        short_cond = (close_prices < lower_band) & (close_prices < vwap)

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


class EnhancedSignalGenerator:
    """
    Improved noise-band + VWAP strategy with 1-min monitoring and dual-bar
    confirmation entry, VWAP trailing stop exits, and a 15:30 entry cutoff.

    Uses 90-day sigma (set vol_window=90 in StrategyConfig / Preprocessor).
    """

    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config or StrategyConfig()

    def generate(self, day_df: pd.DataFrame, prev_close: float) -> pd.Series:
        """
        Produce a minute-level exposure series for a single trading day.

        Entry logic (dual-bar confirmation)
        ------------------------------------
        Bar N:   close crosses beyond boundary (and beyond VWAP)
                 → set pending signal
        Bar N+1: open is also beyond the same boundary
                 → enter at bar N+1's open (exposure recorded for bar N+1)

        Exit logic (VWAP trailing stop)
        --------------------------------
        long  stop = max(upper_band[i], VWAP[i])  — exit long when close < stop
        short stop = min(lower_band[i], VWAP[i])  — exit short when close > stop

        No new entries after 15:30; existing positions run until stop or EOD.

        Parameters
        ----------
        day_df    : intraday session bars with columns close, open, high, low,
                    vwap, sigma_open.  Index must be a DatetimeIndex.
        prev_close: last close of the previous trading day.

        Returns
        -------
        pd.Series[float] with values in {-1, 0, +1}, indexed like day_df.
        """
        cfg = self.config
        open_price = day_df["open"].iloc[0]
        sigma_arr = day_df["sigma_open"].values
        close_arr = day_df["close"].values
        open_arr  = day_df["open"].values
        vwap_arr  = day_df["vwap"].values
        n = len(day_df)

        # Per-bar boundary arrays
        upper_band = max(open_price, prev_close) * (1 + cfg.band_mult * sigma_arr)
        lower_band = min(open_price, prev_close) * (1 - cfg.band_mult * sigma_arr)

        exposure = np.zeros(n)
        current_pos = 0.0
        pending_long  = False
        pending_short = False

        for i in range(n):
            ts = day_df.index[i]
            past_cutoff = (ts.hour * 60 + ts.minute) >= _ENTRY_CUTOFF_MINUTES

            long_stop  = max(upper_band[i], vwap_arr[i])
            short_stop = min(lower_band[i], vwap_arr[i])

            # Track whether we were already holding BEFORE this bar's checks
            was_long  = (current_pos ==  1.0)
            was_short = (current_pos == -1.0)

            # ---- Confirm pending entry at this bar's open ----
            entered_this_bar = False
            if current_pos == 0.0 and not past_cutoff:
                if pending_long and open_arr[i] > upper_band[i]:
                    current_pos = 1.0
                    entered_this_bar = True
                elif pending_short and open_arr[i] < lower_band[i]:
                    current_pos = -1.0
                    entered_this_bar = True
            pending_long  = False
            pending_short = False

            # ---- VWAP trailing stop exit ----
            stopped = False
            if current_pos == 1.0 and close_arr[i] < long_stop:
                current_pos = 0.0
                stopped = True
            elif current_pos == -1.0 and close_arr[i] > short_stop:
                current_pos = 0.0
                stopped = True

            # If an EXISTING position (held from a prior bar) was stopped out,
            # record exposure=1 so _compute_pnl captures the (negative) return
            # of the bar that triggered the stop.  If we entered AND stopped on
            # the same bar, exposure=0 is correct (no prior holding period).
            if stopped and (was_long or was_short) and not entered_this_bar:
                exposure[i] = 1.0 if was_long else -1.0
            else:
                exposure[i] = current_pos

            # ---- Pending entry on next bar (only if flat and before cutoff) ----
            if current_pos == 0.0 and not past_cutoff:
                if close_arr[i] > upper_band[i] and close_arr[i] > vwap_arr[i]:
                    pending_long = True
                elif close_arr[i] < lower_band[i] and close_arr[i] < vwap_arr[i]:
                    pending_short = True

        return pd.Series(exposure, index=day_df.index)
