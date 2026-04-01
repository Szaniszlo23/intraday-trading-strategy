"""
Backtesting engine for the intraday momentum strategy.

Backtester         — Baseline strategy (30-min clock, 14-day sigma)
EnhancedBacktester — V6 strategy (1-min bars, dual-bar confirmation,
                     VWAP stop, Layer 2 sizing multiplier)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analysis.metrics import PerformanceMetrics, PerformanceResult
from config.config import StrategyConfig
from data.indicators import Indicators
from trader.signals import EnhancedSignalGenerator, SignalGenerator
from trader.sizing import EnhancedPositionSizer, PositionSizer


def _avg_trade_duration(exp_arr: np.ndarray) -> float:
    """
    Average number of bars held per trade across a single day.

    A trade starts when exposure changes from 0 → non-zero and ends
    when it returns to 0 (or the day closes with a position still open).
    Returns nan if no trades were taken.
    """
    durations = []
    current_dur = 0
    for val in exp_arr:
        if val != 0.0:
            current_dur += 1
        else:
            if current_dur > 0:
                durations.append(current_dur)
                current_dur = 0
    if current_dur > 0:   # position still open at EOD
        durations.append(current_dur)
    return float(np.mean(durations)) if durations else float("nan")


@dataclass
class BacktestResult:
    """Container returned by both Backtester and EnhancedBacktester."""
    daily: pd.DataFrame       # index=date, cols: ret, AUM, ret_spy, trades, avg_dur, avg_lev, long_pnl, short_pnl
    metrics: PerformanceResult

    @property
    def returns(self) -> pd.Series:
        """Daily net returns (NaN days dropped)."""
        return self.daily["ret"].dropna()

    @property
    def aum(self) -> pd.Series:
        """Daily AUM series."""
        return self.daily["AUM"]


class Backtester:
    """Baseline strategy backtester (30-min clock, 14-day sigma)."""

    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config or StrategyConfig()
        self._signal_gen = SignalGenerator(self.config)
        self._sizer      = PositionSizer(self.config)

    def run(self, df: pd.DataFrame, df_daily: pd.DataFrame) -> BacktestResult:
        """
        Run the backtest over the supplied preprocessed data.

        Parameters
        ----------
        df       : 1-minute bars with features from Preprocessor.transform()
        df_daily : daily OHLCV + 'ret' column (from Preprocessor)

        Returns
        -------
        BacktestResult with daily P&L DataFrame and aggregated metrics.
        """
        cfg      = self.config
        all_days = np.array(sorted(df["day"].unique()))
        daily_groups = df.groupby("day")

        result = pd.DataFrame(
            index=all_days,
            columns=["ret", "AUM", "ret_spy", "trades", "avg_dur", "avg_lev",
                     "long_pnl", "short_pnl"],
            dtype=float,
        )
        result["AUM"] = cfg.aum_0
        prev_aum = cfg.aum_0

        for d in range(1, len(all_days)):
            current_day = all_days[d]
            prev_day    = all_days[d - 1]

            if prev_day not in daily_groups.groups or current_day not in daily_groups.groups:
                continue

            prev_df = daily_groups.get_group(prev_day)
            cur_df  = daily_groups.get_group(current_day)

            # Skip days before sigma_open has warmed up (first vol_window days)
            if cur_df["sigma_open"].isna().all():
                continue

            prev_close = prev_df["close"].iloc[-1]
            open_price = cur_df["open"].iloc[0] if "open" in cur_df.columns else cur_df["close"].iloc[0]
            daily_vol  = cur_df["spy_dvol"].iloc[0]

            exposure = self._signal_gen.generate(cur_df, prev_close)
            shares   = self._sizer.shares(prev_aum, open_price, daily_vol)
            leverage = (shares * open_price) / prev_aum if prev_aum > 0 else 0.0

            net_pnl, _, trades_count, avg_dur, long_pnl, short_pnl = self._compute_pnl(
                cur_df["close"], exposure, shares
            )

            result.loc[current_day, "AUM"]      = prev_aum + net_pnl
            result.loc[current_day, "ret"]       = net_pnl / prev_aum
            result.loc[current_day, "trades"]    = trades_count
            result.loc[current_day, "avg_dur"]   = avg_dur
            result.loc[current_day, "avg_lev"]   = leverage if trades_count > 0 else 0.0
            result.loc[current_day, "long_pnl"]  = long_pnl
            result.loc[current_day, "short_pnl"] = short_pnl
            prev_aum = prev_aum + net_pnl

            today_dt = pd.Timestamp(current_day)
            if today_dt in df_daily.index:
                result.loc[current_day, "ret_spy"] = df_daily.loc[today_dt, "ret"]

        result.index = pd.to_datetime(result.index)
        metrics = PerformanceMetrics.compute(result["ret"], result["ret_spy"])
        return BacktestResult(daily=result, metrics=metrics)

    def _compute_pnl(
        self,
        close_prices: pd.Series,
        exposure: pd.Series,
        shares: int,
    ) -> tuple[float, float, int, float, float, float]:
        """
        Compute daily net P&L, commission, trade count, avg duration,
        and gross long/short P&L contributions.

        Returns (net_pnl, commission_paid, trades_count, avg_duration,
                 long_gross, short_gross)
        """
        cfg       = self.config
        exp_arr   = exposure.values
        close_arr = close_prices.values

        # Each direction change (0→1, 1→-1, etc.) counts as one order
        trades_count = int(np.sum(np.abs(np.diff(np.append(exp_arr, 0)))))
        change_1m    = np.diff(close_arr, prepend=close_arr[0])

        # Split gross P&L into long and short contributions
        long_mask   = exp_arr > 0
        short_mask  = exp_arr < 0
        long_gross  = float(np.sum(exp_arr[long_mask]  * change_1m[long_mask])  * shares)
        short_gross = float(np.sum(exp_arr[short_mask] * change_1m[short_mask]) * shares)
        gross_pnl   = long_gross + short_gross

        # Commission: per-share rate with a per-order minimum
        commission_paid = float(trades_count * max(cfg.min_comm, cfg.commission * shares))
        # Slippage: fixed per-share cost on every fill (one-way)
        slippage_paid   = float(trades_count * cfg.slippage * shares)

        avg_duration = _avg_trade_duration(exp_arr)
        return (
            gross_pnl - commission_paid - slippage_paid,
            commission_paid,
            trades_count,
            avg_duration,
            long_gross,
            short_gross,
        )

    # ------------------------------------------------------------------
    # Train / test split helper
    # ------------------------------------------------------------------

    @staticmethod
    def split(
        df: pd.DataFrame,
        df_daily: pd.DataFrame,
        train_ratio: float = 0.70,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split intraday and daily frames into train and test subsets by date."""
        all_days  = sorted(df["day"].unique())
        split_idx = int(len(all_days) * train_ratio)
        train_days = set(all_days[:split_idx])
        test_days  = set(all_days[split_idx:])

        df_train = df[df["day"].isin(train_days)].copy()
        df_test  = df[df["day"].isin(test_days)].copy()

        train_dt = pd.to_datetime(list(train_days))
        test_dt  = pd.to_datetime(list(test_days))

        return (
            df_train,
            df_daily[df_daily.index.isin(train_dt)],
            df_test,
            df_daily[df_daily.index.isin(test_dt)],
        )


class EnhancedBacktester:
    """
    V6 enhanced strategy backtester.

    Per-day it computes Layer 2 sizing inputs:
      - Vol-regime factor   : 5-day realized vol → multiplier [0.70, 1.30]
      - Pre-market return   : 04:00–09:29 extended bars → trend signal
      - Order imbalance     : previous day's first 30 bars → tick-rule score
      - Flow composite mult : blend of the above two → multiplier [0.80, 1.20]

    Then sizes via EnhancedPositionSizer and generates signals via
    EnhancedSignalGenerator (dual-bar confirmation + VWAP stop).

    raw_extended must include pre-market bars (fetch with include_extended=True).
    """

    def __init__(
        self,
        config: StrategyConfig | None = None,
        signal_gen: EnhancedSignalGenerator | None = None,
        use_layer2: bool = True,
        use_day_filter: bool = False,
        day_filter_threshold: float = 0.30,
    ) -> None:
        self.config               = config or StrategyConfig()
        self.use_layer2           = use_layer2
        self.use_day_filter       = use_day_filter
        self.day_filter_threshold = day_filter_threshold
        self._signal_gen = signal_gen or EnhancedSignalGenerator(self.config)
        # Use EnhancedPositionSizer when Layer 2 is active, baseline sizer otherwise
        self._sizer: EnhancedPositionSizer | PositionSizer = (
            EnhancedPositionSizer() if use_layer2 else PositionSizer(self.config)
        )

    def run(
        self,
        df: pd.DataFrame,
        df_daily: pd.DataFrame,
        raw_extended: pd.DataFrame,
    ) -> BacktestResult:
        """
        Run the enhanced backtest.

        Parameters
        ----------
        df           : preprocessed session bars (Preprocessor.transform() output,
                       vol_window=90, regular session 09:30–16:00)
        df_daily     : daily OHLCV + 'ret' column (from Preprocessor)
        raw_extended : raw 1-min bars including pre-market (04:00+), needed for
                       pre-market return computation in the flow composite

        Returns
        -------
        BacktestResult with daily P&L DataFrame and aggregated metrics.
        """
        cfg      = self.config
        all_days = np.array(sorted(df["day"].unique()))
        daily_groups = df.groupby("day")

        result = pd.DataFrame(
            index=all_days,
            columns=["ret", "AUM", "ret_spy", "trades", "avg_dur", "avg_lev",
                     "long_pnl", "short_pnl"],
            dtype=float,
        )
        result["AUM"] = cfg.aum_0
        prev_aum = cfg.aum_0

        # Pre-group extended bars by day for O(1) per-day lookup
        raw_extended = raw_extended.copy()
        if "day" not in raw_extended.columns:
            raw_extended["day"] = raw_extended.index.date
        raw_extended["day"] = pd.to_datetime(raw_extended["day"]).dt.date
        raw_groups = raw_extended.groupby("day")

        daily_ret_series = df_daily["ret"].dropna()

        for d in range(1, len(all_days)):
            current_day = all_days[d]
            prev_day    = all_days[d - 1]

            if prev_day not in daily_groups.groups or current_day not in daily_groups.groups:
                continue

            prev_df = daily_groups.get_group(prev_day)
            cur_df  = daily_groups.get_group(current_day)

            # Skip days before sigma_open has warmed up
            if cur_df["sigma_open"].isna().all():
                continue

            prev_close = prev_df["close"].iloc[-1]
            open_price = cur_df["open"].iloc[0]
            daily_vol  = cur_df["spy_dvol"].iloc[0]

            # ---- Signal generation ----
            exposure = self._signal_gen.generate(cur_df, prev_close)

            # ---- Optional day-type filter ----
            # Zero out exposure on choppy days where the opening range is small
            # relative to the previous day's range. Applied after signal generation
            # so the filter doesn't interfere with the signal logic itself.
            if self.use_day_filter:
                or_ratio = Indicators.opening_range_ratio(cur_df, prev_df)
                if or_ratio < self.day_filter_threshold:
                    exposure = pd.Series(0.0, index=cur_df.index)

            # ---- Layer 2 sizing ----
            if self.use_layer2:
                # Vol-regime factor: uses only returns prior to today (lag-safe)
                prior_rets = daily_ret_series[
                    daily_ret_series.index < pd.Timestamp(current_day)
                ]
                vol_regime = Indicators.vol_regime_factor(prior_rets)

                # Pre-market return: today's 04:00–09:29 bars
                pm_ret    = 0.0
                imbalance = 0.0
                if current_day in raw_groups.groups:
                    raw_day = raw_groups.get_group(current_day)
                    pm_bars = raw_day.between_time("04:00", "09:29")
                    pm_ret  = Indicators.premarket_return(pm_bars)

                # Order imbalance: uses the PREVIOUS day's first 30 bars.
                # Sizing is decided before today's session opens, so we can
                # only observe what happened yesterday — not today 09:30–09:59.
                if prev_day in raw_groups.groups:
                    raw_prev      = raw_groups.get_group(prev_day)
                    first_30_prev = raw_prev.between_time("09:30", "09:59").iloc[:30]
                    imbalance     = Indicators.order_imbalance(first_30_prev)

                flow_mult = Indicators.flow_composite_mult(pm_ret, imbalance)

                shares = self._sizer.shares(
                    prev_aum, open_price, daily_vol, vol_regime, flow_mult
                )
            else:
                shares = self._sizer.shares(prev_aum, open_price, daily_vol)

            leverage = (shares * open_price) / prev_aum if prev_aum > 0 else 0.0

            # ---- P&L ----
            net_pnl, _, trades_count, avg_dur, long_pnl, short_pnl = self._compute_pnl(
                cur_df["close"], exposure, shares
            )
            result.loc[current_day, "AUM"]      = prev_aum + net_pnl
            result.loc[current_day, "ret"]       = net_pnl / prev_aum
            result.loc[current_day, "trades"]    = trades_count
            result.loc[current_day, "avg_dur"]   = avg_dur
            result.loc[current_day, "avg_lev"]   = leverage if trades_count > 0 else 0.0
            result.loc[current_day, "long_pnl"]  = long_pnl
            result.loc[current_day, "short_pnl"] = short_pnl
            prev_aum = prev_aum + net_pnl

            today_dt = pd.Timestamp(current_day)
            if today_dt in df_daily.index:
                result.loc[current_day, "ret_spy"] = df_daily.loc[today_dt, "ret"]

        result.index = pd.to_datetime(result.index)
        metrics = PerformanceMetrics.compute(result["ret"], result["ret_spy"])
        return BacktestResult(daily=result, metrics=metrics)

    def _compute_pnl(
        self,
        close_prices: pd.Series,
        exposure: pd.Series,
        shares: int,
    ) -> tuple[float, float, int, float, float, float]:
        """
        Compute daily net P&L, commission, trade count, avg duration,
        and gross long/short P&L contributions.

        Returns (net_pnl, commission_paid, trades_count, avg_duration,
                 long_gross, short_gross)
        """
        cfg       = self.config
        exp_arr   = exposure.values
        close_arr = close_prices.values

        trades_count = int(np.sum(np.abs(np.diff(np.append(exp_arr, 0)))))
        change_1m    = np.diff(close_arr, prepend=close_arr[0])

        # Split gross P&L into long and short contributions
        long_mask   = exp_arr > 0
        short_mask  = exp_arr < 0
        long_gross  = float(np.sum(exp_arr[long_mask]  * change_1m[long_mask])  * shares)
        short_gross = float(np.sum(exp_arr[short_mask] * change_1m[short_mask]) * shares)
        gross_pnl   = long_gross + short_gross

        # Commission: per-share rate with a per-order minimum
        commission_paid = float(trades_count * max(cfg.min_comm, cfg.commission * shares))
        # Slippage: fixed per-share cost on every fill (one-way)
        slippage_paid   = float(trades_count * cfg.slippage * shares)

        avg_duration = _avg_trade_duration(exp_arr)
        return (
            gross_pnl - commission_paid - slippage_paid,
            commission_paid,
            trades_count,
            avg_duration,
            long_gross,
            short_gross,
        )
