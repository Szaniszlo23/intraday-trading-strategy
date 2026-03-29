"""
Feature engineering for the intraday momentum strategy.

Orchestrates all indicator computations (delegated to indicators.py)
and returns the enriched intraday frame + a daily summary frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.config import StrategyConfig
from data.indicators import Indicators


class Preprocessor:
    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config or StrategyConfig()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute all features from raw 1-minute OHLCV data.

        Parameters
        ----------
        df:
            1-minute bars with columns: open, high, low, close, volume.
            Indexed by a datetime index (caldt).  A 'day' column is added
            automatically if missing.

        Returns
        -------
        df_intraday : enriched with vwap, move_open, spy_dvol, sigma_open,
                      min_from_open, minute_of_day, rsi
        df_daily    : daily OHLCV + return, indexed by date (DatetimeIndex)
        """
        df = df.copy()
        df = self._ensure_day_column(df)
        df = self._add_session_columns(df)
        df["vwap"] = Indicators.vwap(df)
        df["move_open"] = Indicators.move_open(df)
        df["spy_dvol"] = Indicators.daily_vol(df, window=self.config.vol_window)
        df["sigma_open"] = self._sigma_open(df)
        df["rsi"] = Indicators.rsi(df["close"], period=self.config.rsi_period)

        df_daily = self._build_daily(df)
        return df, df_daily

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_day_column(df: pd.DataFrame) -> pd.DataFrame:
        if "day" not in df.columns:
            df["day"] = df.index.date
        df["day"] = pd.to_datetime(df["day"]).dt.date
        return df

    @staticmethod
    def _add_session_columns(df: pd.DataFrame) -> pd.DataFrame:
        session_start = pd.Timedelta(hours=9, minutes=30)
        df["min_from_open"] = (
            (df.index - df.index.normalize()) - session_start
        ) / pd.Timedelta(minutes=1) + 1
        df["minute_of_day"] = df["min_from_open"].round().astype(int)
        return df

    def _sigma_open(self, df: pd.DataFrame) -> pd.Series:
        """Rolling per-minute mean of move_open, lagged 1 day."""
        w = self.config.vol_window
        rolling_mean = df.groupby("minute_of_day")["move_open"].transform(
            lambda x: x.rolling(window=w, min_periods=w - 1).mean()
        )
        df = df.copy()
        df["_rm"] = rolling_mean
        return df.groupby("minute_of_day")["_rm"].transform(lambda x: x.shift(1))

    @staticmethod
    def _build_daily(df: pd.DataFrame) -> pd.DataFrame:
        df_daily = df.groupby("day").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        df_daily.index = pd.to_datetime(df_daily.index)
        df_daily["ret"] = df_daily["close"].pct_change()
        return df_daily
