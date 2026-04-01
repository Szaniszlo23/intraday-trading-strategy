"""
Technical indicators used by the intraday momentum strategy.

All methods are static — call them directly on pandas Series/DataFrames
without instantiating an object.

Baseline indicators
--------------------
- vwap       : cumulative intraday volume-weighted average price
- move_open  : absolute % move from the day's opening price
- daily_vol  : rolling daily return volatility (spy_dvol feature)

Enhanced strategy helpers
--------------------------
- vol_regime_factor   : 5-day realized vol → sizing multiplier [0.70, 1.30]
- premarket_return    : SPY return from 04:00 to 09:29
- order_imbalance     : tick-rule imbalance from first 30 session bars [-1, 1]
- flow_composite_mult : combine premarket trend + imbalance → multiplier [0.80, 1.20]
- opening_range_ratio : today's first-15-min range / yesterday's full range (day-type filter)
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


class Indicators:

    # ------------------------------------------------------------------
    # VWAP — reset each day
    # ------------------------------------------------------------------

    @staticmethod
    def vwap(df: pd.DataFrame) -> pd.Series:
        """
        Cumulative intraday VWAP, reset each trading day.

        Uses the HLC3 typical price (high + low + close) / 3 as the price proxy.
        Requires columns: high, low, close, volume, day.
        """
        hlc = (df["high"] + df["low"] + df["close"]) / 3
        cum_vol_x_hlc = df.groupby("day").apply(
            lambda g: (g["volume"] * hlc.loc[g.index]).cumsum(),
            include_groups=False,
        ).reset_index(level=0, drop=True)
        cum_vol = df.groupby("day")["volume"].cumsum()
        return cum_vol_x_hlc / cum_vol

    # ------------------------------------------------------------------
    # move_open — noise-band scaling input
    # ------------------------------------------------------------------

    @staticmethod
    def move_open(df: pd.DataFrame) -> pd.Series:
        """
        Absolute % move from the day's first open price.

        Used as the raw input to the sigma_open rolling mean.
        """
        day_open = df.groupby("day")["open"].transform("first")
        return (df["close"] / day_open - 1).abs()

    # ------------------------------------------------------------------
    # Daily volatility — vol-targeting sizing input
    # ------------------------------------------------------------------

    @staticmethod
    def daily_vol(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Rolling std of daily close-to-close returns over `window` days,
        mapped back to every intraday bar (same value for all bars in a day).

        Shifted by 1 day so today's bar uses only yesterday's completed data.
        """
        daily_close = df.groupby("day")["close"].last()
        daily_ret = daily_close.pct_change()
        rolling_vol = daily_ret.rolling(window=window, min_periods=window - 1).std().shift(1)
        rolling_vol.index = pd.to_datetime(rolling_vol.index)
        day_dt = pd.to_datetime(df["day"])
        return day_dt.map(rolling_vol)

    # ------------------------------------------------------------------
    # Enhanced strategy helpers
    # ------------------------------------------------------------------

    @staticmethod
    def vol_regime_factor(daily_returns: pd.Series) -> float:
        """
        Sizing multiplier based on the last 5 days of realized volatility.

        Maps 5-day annualized vol to:
          < 10%  → 0.70  (low-vol / choppy — reduce size)
          > 25%  → 1.30  (high-vol / trending — increase size)
          else   → linear interpolation between 0.70 and 1.30

        Returns 1.0 (neutral) if fewer than 5 observations.
        """
        recent = daily_returns.dropna().tail(5)
        if len(recent) < 5:
            return 1.0
        vol_5d = float(recent.std() * math.sqrt(252))
        if vol_5d < 0.10:
            return 0.70
        if vol_5d > 0.25:
            return 1.30
        return 0.70 + (vol_5d - 0.10) / (0.25 - 0.10) * (1.30 - 0.70)

    @staticmethod
    def premarket_return(pm_bars: pd.DataFrame) -> float:
        """
        SPY pre-market simple return from 04:00 to 09:29 ET.

        Returns 0.0 if bars are empty (e.g. holiday, missing data).
        """
        if pm_bars.empty:
            return 0.0
        return float(pm_bars["close"].iloc[-1] / pm_bars["open"].iloc[0] - 1)

    @staticmethod
    def order_imbalance(first_30: pd.DataFrame) -> float:
        """
        Tick-rule order imbalance from the first 30 regular session bars.

        Up bars (close > open) → buyer pressure; down bars → seller pressure.
        Returns 0.0 on news-driven days where the 30-min range exceeds 0.5%
        (the signal is unreliable when there's a large directional gap).

        Returns a float in [-1, 1]: positive = buyer pressure.
        """
        if first_30.empty:
            return 0.0
        session_range = (
            (first_30["high"].max() - first_30["low"].min())
            / first_30["open"].iloc[0]
        )
        if session_range > 0.005:
            return 0.0   # news-driven day — treat as neutral
        n = len(first_30)
        up_bars   = int((first_30["close"] > first_30["open"]).sum())
        down_bars = int((first_30["close"] < first_30["open"]).sum())
        return (up_bars - down_bars) / n

    @staticmethod
    def flow_composite_mult(pm_ret: float, imbalance: float) -> float:
        """
        Combine pre-market trend and order imbalance into a single sizing multiplier.

        Formula:
            composite = 0.4 × tanh(pm_ret × 100) + 0.6 × imbalance
            mult      = 0.80 + (composite + 1) / 2 × 0.40  → maps [-1,1] to [0.80, 1.20]

        The tanh squashes the pre-market return to [-1, 1] before blending.
        """
        pm_signal = math.tanh(pm_ret * 100)
        composite = 0.4 * pm_signal + 0.6 * imbalance
        return 0.80 + (composite + 1.0) / 2.0 * 0.40

    @staticmethod
    def opening_range_ratio(cur_df: pd.DataFrame, prev_df: pd.DataFrame, n_bars: int = 15) -> float:
        """
        Ratio of today's opening range (first n_bars minutes) to yesterday's full session range.

        Used by the optional day-type filter in EnhancedBacktester.
        A high ratio (e.g. > 0.30) suggests a directional/trending day.
        Returns 0.0 if either DataFrame is empty or previous range is zero.
        """
        if cur_df.empty or prev_df.empty:
            return 0.0
        opening_bars = cur_df.iloc[:n_bars]
        or_range  = opening_bars["high"].max() - opening_bars["low"].min()
        prev_range = prev_df["high"].max() - prev_df["low"].min()
        if prev_range == 0:
            return 0.0
        return float(or_range / prev_range)
