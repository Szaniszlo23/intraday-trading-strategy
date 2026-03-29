"""
Technical indicators used by the intraday momentum strategy.

All methods are static — call them directly on pandas Series/DataFrames
without instantiating an object.

Indicators
----------
- VWAP         : cumulative intraday volume-weighted average price
- move_open    : absolute % move from the day's opening price
- sigma_open   : per-minute rolling mean of move_open (14-day), lagged 1 day
- spy_dvol     : 14-day rolling daily return volatility
- rsi          : Wilder (EMA-smoothed) RSI on 1-minute closes
"""

from __future__ import annotations

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
        14-day rolling std of daily returns, mapped back to every intraday bar.

        Returns a Series aligned to df.index (same value for all bars in a day).
        """
        daily_close = df.groupby("day")["close"].last()
        daily_ret = daily_close.pct_change()
        rolling_vol = daily_ret.rolling(window=window, min_periods=window - 1).std().shift(1)
        rolling_vol.index = pd.to_datetime(rolling_vol.index)
        day_dt = pd.to_datetime(df["day"])
        return day_dt.map(rolling_vol)

    # ------------------------------------------------------------------
    # RSI  (Wilder / EMA-smoothed)
    # ------------------------------------------------------------------

    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """
        Wilder RSI on a price series.

        Uses EMA smoothing with alpha = 1/period, which is equivalent
        to Wilder's original smoothing method.

        Returns values in [0, 100].
        """
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
