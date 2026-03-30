"""
Technical indicators used by the intraday momentum strategy.

All methods are static — call them directly on pandas Series/DataFrames
without instantiating an object.

Indicators (Baseline)
---------------------
- VWAP         : cumulative intraday volume-weighted average price
- move_open    : absolute % move from the day's opening price
- sigma_open   : per-minute rolling mean of move_open (configurable window), lagged 1 day
- spy_dvol     : rolling daily return volatility

Enhanced Strategy Helpers
--------------------------
- vol_regime_factor   : 5-day realized vol → sizing multiplier [0.70, 1.30]
- premarket_return    : SPY return from 04:00 to 09:29
- order_imbalance     : tick-rule imbalance from first 30 session bars [-1, 1]
- flow_composite_mult : combine premarket trend + imbalance → multiplier [0.80, 1.20]
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


class Indicators:

    # ------------------------------------------------------------------
    # VWAP
    # ------------------------------------------------------------------

    @staticmethod
    def vwap(df: pd.DataFrame) -> pd.Series:
        """
        Cumulative intraday VWAP, reset each trading day.

        Parameters
        ----------
        df : DataFrame with columns high, low, close, volume and a 'day' column.

        Returns
        -------
        pd.Series aligned to df.index.
        """
        hlc = (df["high"] + df["low"] + df["close"]) / 3
        cum_vol_x_hlc = df.groupby("day").apply(
            lambda g: (g["volume"] * hlc.loc[g.index]).cumsum(),
            include_groups=False,
        ).reset_index(level=0, drop=True)
        cum_vol = df.groupby("day")["volume"].cumsum()
        return cum_vol_x_hlc / cum_vol

    # ------------------------------------------------------------------
    # move_open
    # ------------------------------------------------------------------

    @staticmethod
    def move_open(df: pd.DataFrame) -> pd.Series:
        """Absolute % move from the day's first open price."""
        day_open = df.groupby("day")["open"].transform("first")
        return (df["close"] / day_open - 1).abs()

    # ------------------------------------------------------------------
    # sigma_open  (noise-band scale factor from the paper)
    # ------------------------------------------------------------------

    @staticmethod
    def sigma_open(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Per-minute rolling mean of move_open over `window` trading days,
        lagged by 1 period (yesterday's estimate → today's band width).

        Requires 'move_open' and 'minute_of_day' columns to be present.
        """
        rolling_mean = df.groupby("minute_of_day")["move_open"].transform(
            lambda x: x.rolling(window=window, min_periods=window - 1).mean()
        )
        return df.groupby("minute_of_day")[rolling_mean.name if hasattr(rolling_mean, "name") else "move_open"].transform(
            lambda x: x.shift(1)
        ) if False else (  # workaround: use intermediate series
            df.assign(_rm=rolling_mean)
            .groupby("minute_of_day")["_rm"]
            .transform(lambda x: x.shift(1))
        )

    # ------------------------------------------------------------------
    # Daily volatility
    # ------------------------------------------------------------------

    @staticmethod
    def daily_vol(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Rolling std of daily returns over `window` days, mapped back to every
        intraday bar.

        Returns a Series aligned to df.index (same value for all bars in a day).
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
        Compute the vol-regime sizing multiplier from the last 5 daily returns.

        Maps 5-day realized annualized vol to a multiplier:
          < 10%  → 0.70  (choppy/low-vol — reduce size)
          > 25%  → 1.30  (trending/high-vol — increase size)
          else   → linear interpolation between 0.70 and 1.30

        Parameters
        ----------
        daily_returns : recent daily simple returns (uses last 5)

        Returns
        -------
        float in [0.70, 1.30] — returns 1.0 if fewer than 5 observations.
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
        SPY pre-market return from 04:00 to 09:29.

        Parameters
        ----------
        pm_bars : DataFrame of pre-market 1-min bars (04:00–09:29),
                  must have 'open' and 'close' columns.

        Returns
        -------
        float — simple return; 0.0 if bars are empty.
        """
        if pm_bars.empty:
            return 0.0
        return float(pm_bars["close"].iloc[-1] / pm_bars["open"].iloc[0] - 1)

    @staticmethod
    def order_imbalance(first_30: pd.DataFrame) -> float:
        """
        Tick-rule order imbalance from the first 30 regular session bars.

        Up bars (close > open) are attributed to buyers; down bars to sellers.
        Returns neutral (0.0) on news-driven days where the session range
        exceeds 0.5% in the first 30 minutes — the signal is unreliable there.

        Parameters
        ----------
        first_30 : first 30 bars of the regular session (09:30 onward),
                   must have 'open', 'high', 'low', 'close' columns.

        Returns
        -------
        float in [-1, 1] — positive = buyer pressure, negative = seller pressure.
        """
        if first_30.empty:
            return 0.0
        session_range = (
            (first_30["high"].max() - first_30["low"].min())
            / first_30["open"].iloc[0]
        )
        if session_range > 0.005:
            return 0.0  # news-driven day — neutral
        n = len(first_30)
        up_bars = int((first_30["close"] > first_30["open"]).sum())
        down_bars = int((first_30["close"] < first_30["open"]).sum())
        return (up_bars - down_bars) / n

    @staticmethod
    def flow_composite_mult(pm_ret: float, imbalance: float) -> float:
        """
        Combine pre-market trend and order imbalance into a sizing multiplier.

        Weights: 0.4 × pre-market trend signal + 0.6 × order imbalance.
        The composite (range ~[-1, 1]) is mapped linearly to [0.80, 1.20].

        Parameters
        ----------
        pm_ret    : pre-market simple return (from premarket_return())
        imbalance : tick-rule imbalance (from order_imbalance())

        Returns
        -------
        float in [0.80, 1.20]
        """
        pm_signal = math.tanh(pm_ret * 100)          # squash to [-1, 1]
        composite = 0.4 * pm_signal + 0.6 * imbalance
        # Map [-1, 1] → [0.80, 1.20]
        return 0.80 + (composite + 1.0) / 2.0 * 0.40
