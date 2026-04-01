"""
Backtesting engine for the intraday momentum strategy.

Backtester        — Baseline strategy backtester
EnhancedBacktester — Enhanced strategy backtester (dual-bar confirmation,
                     VWAP trailing stop, Layer 2 sizing multiplier)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analysis.metrics import PerformanceMetrics, PerformanceResult
from config.config import StrategyConfig
from data.indicators import Indicators
from trader.signals import EnhancedSignalGenerator, SignalGenerator
from trader.sizing import EnhancedPositionSizer, PositionSizer  # noqa: F401 (PositionSizer used in EnhancedBacktester)


def _avg_trade_duration(exp_arr: np.ndarray) -> float:
    """
    Return the average number of bars a trade is held, across all trades in
    the day.  A trade starts when exposure changes from 0 → non-zero and ends
    when it returns to 0 (or the day closes with exposure still open).
    Returns float('nan') if no trades were taken.
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
    if current_dur > 0:          # position still open at end of day
        durations.append(current_dur)
    return float(np.mean(durations)) if durations else float("nan")


@dataclass
class BacktestResult:
    daily: pd.DataFrame        # index=date, cols: ret, AUM, ret_spy
    metrics: PerformanceResult

    @property
    def returns(self) -> pd.Series:
        return self.daily["ret"].dropna()

    @property
    def aum(self) -> pd.Series:
        return self.daily["AUM"]


class Backtester:
    """Baseline strategy backtester (unchanged)."""

    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config or StrategyConfig()
        self._signal_gen = SignalGenerator(self.config)
        self._sizer = PositionSizer(self.config)

    def run(self, df: pd.DataFrame, df_daily: pd.DataFrame) -> BacktestResult:
        """
        Run the backtest over the supplied preprocessed data.

        Parameters
        ----------
        df       : 1-minute bars with features from Preprocessor.transform()
        df_daily : daily OHLCV + 'ret' column

        Returns
        -------
        BacktestResult with daily P&L DataFrame and aggregated metrics.
        """
        cfg = self.config
        all_days = np.array(sorted(df["day"].unique()))
        daily_groups = df.groupby("day")

        result = pd.DataFrame(
            index=all_days,
            columns=["ret", "AUM", "ret_spy", "trades", "avg_dur", "avg_lev"],
            dtype=float,
        )
        result["AUM"] = cfg.aum_0
        prev_aum = cfg.aum_0

        for d in range(1, len(all_days)):
            current_day = all_days[d]
            prev_day = all_days[d - 1]

            if prev_day not in daily_groups.groups or current_day not in daily_groups.groups:
                continue

            prev_df = daily_groups.get_group(prev_day)
            cur_df = daily_groups.get_group(current_day)

            if cur_df["sigma_open"].isna().all():
                continue

            prev_close = prev_df["close"].iloc[-1]
            open_price = cur_df["open"].iloc[0] if "open" in cur_df.columns else cur_df["close"].iloc[0]
            daily_vol = cur_df["spy_dvol"].iloc[0]

            exposure = self._signal_gen.generate(cur_df, prev_close)
            shares = self._sizer.shares(prev_aum, open_price, daily_vol)
            leverage = (shares * open_price) / prev_aum if prev_aum > 0 else 0.0
            net_pnl, _, trades_count, avg_dur = self._compute_pnl(cur_df["close"], exposure, shares)

            result.loc[current_day, "AUM"]     = prev_aum + net_pnl
            result.loc[current_day, "ret"]     = net_pnl / prev_aum
            result.loc[current_day, "trades"]  = trades_count
            result.loc[current_day, "avg_dur"] = avg_dur
            result.loc[current_day, "avg_lev"] = leverage if trades_count > 0 else 0.0
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
    ) -> tuple[float, float, int, float]:
        cfg = self.config
        exp_arr = exposure.values
        close_arr = close_prices.values

        trades_count = int(np.sum(np.abs(np.diff(np.append(exp_arr, 0)))))
        change_1m = np.diff(close_arr, prepend=close_arr[0])
        gross_pnl = float(np.sum(exp_arr * change_1m) * shares)
        commission_paid = float(
            trades_count * max(cfg.min_comm, cfg.commission * shares)
        )
        # Slippage: paid on every order (each direction change = one order)
        # $0.001/share one-way, applied to every fill
        slippage_paid = float(trades_count * cfg.slippage * shares)
        avg_duration = _avg_trade_duration(exp_arr)
        return gross_pnl - commission_paid - slippage_paid, commission_paid, trades_count, avg_duration

    # ------------------------------------------------------------------
    # Train / test split helper
    # ------------------------------------------------------------------

    @staticmethod
    def split(
        df: pd.DataFrame,
        df_daily: pd.DataFrame,
        train_ratio: float = 0.70,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split intraday and daily frames into train and test subsets."""
        all_days = sorted(df["day"].unique())
        split_idx = int(len(all_days) * train_ratio)
        train_days = set(all_days[:split_idx])
        test_days = set(all_days[split_idx:])

        df_train = df[df["day"].isin(train_days)].copy()
        df_test = df[df["day"].isin(test_days)].copy()

        train_dt = pd.to_datetime(list(train_days))
        test_dt = pd.to_datetime(list(test_days))

        return (
            df_train,
            df_daily[df_daily.index.isin(train_dt)],
            df_test,
            df_daily[df_daily.index.isin(test_dt)],
        )


class EnhancedBacktester:
    """
    Enhanced strategy backtester.

    Per-day it computes:
      - Vol-regime factor    : Indicators.vol_regime_factor() on prior 5 daily rets
      - Pre-market return    : Indicators.premarket_return() from extended-hours bars
      - Order imbalance      : Indicators.order_imbalance() from first 30 session bars
      - Flow composite mult  : Indicators.flow_composite_mult()

    Then sizes via EnhancedPositionSizer and generates signals via
    EnhancedSignalGenerator.

    raw_extended must include pre-market bars (include_extended=True when fetching).
    """

    def __init__(
        self,
        config: StrategyConfig | None = None,
        signal_gen: EnhancedSignalGenerator | None = None,
        use_layer2: bool = True,
        use_day_filter: bool = False,
        day_filter_threshold: float = 0.30,
    ) -> None:
        self.config = config or StrategyConfig()
        self.use_layer2 = use_layer2
        self.use_day_filter = use_day_filter
        self.day_filter_threshold = day_filter_threshold
        self._signal_gen = signal_gen or EnhancedSignalGenerator(self.config)
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
                       vol_window=90, regular session 09:30-16:00)
        df_daily     : daily OHLCV + 'ret' column (from Preprocessor)
        raw_extended : raw 1-min bars including pre-market (04:00+), used to
                       extract premarket returns for the flow composite

        Returns
        -------
        BacktestResult with daily P&L DataFrame and aggregated metrics.
        """
        cfg = self.config
        all_days = np.array(sorted(df["day"].unique()))
        daily_groups = df.groupby("day")

        result = pd.DataFrame(
            index=all_days,
            columns=["ret", "AUM", "ret_spy", "trades", "avg_dur", "avg_lev"],
            dtype=float,
        )
        result["AUM"] = cfg.aum_0
        prev_aum = cfg.aum_0

        # Pre-group raw extended bars by day for O(1) per-day lookup
        raw_extended = raw_extended.copy()
        if "day" not in raw_extended.columns:
            raw_extended["day"] = raw_extended.index.date
        raw_extended["day"] = pd.to_datetime(raw_extended["day"]).dt.date
        raw_groups = raw_extended.groupby("day")

        daily_ret_series = df_daily["ret"].dropna()

        for d in range(1, len(all_days)):
            current_day = all_days[d]
            prev_day = all_days[d - 1]

            if prev_day not in daily_groups.groups or current_day not in daily_groups.groups:
                continue

            prev_df = daily_groups.get_group(prev_day)
            cur_df  = daily_groups.get_group(current_day)

            if cur_df["sigma_open"].isna().all():
                continue

            prev_close = prev_df["close"].iloc[-1]
            open_price = cur_df["open"].iloc[0]
            daily_vol  = cur_df["spy_dvol"].iloc[0]

            # ---- Signals ----
            exposure = self._signal_gen.generate(cur_df, prev_close)

            # ---- Day-type filter ----
            # Skip choppy days where the opening range (first 15 min) is small
            # relative to the previous day's full range.  Applied after signal
            # generation so the filter is orthogonal to the signal logic.
            if self.use_day_filter:
                or_ratio = Indicators.opening_range_ratio(cur_df, prev_df)
                if or_ratio < self.day_filter_threshold:
                    exposure = pd.Series(0.0, index=cur_df.index)

            # ---- Sizing ----
            if self.use_layer2:
                # Layer 2: vol-regime factor
                prior_rets = daily_ret_series[
                    daily_ret_series.index < pd.Timestamp(current_day)
                ]
                vol_regime = Indicators.vol_regime_factor(prior_rets)

                # Layer 2: flow composite
                pm_ret    = 0.0
                imbalance = 0.0
                if current_day in raw_groups.groups:
                    raw_day = raw_groups.get_group(current_day)
                    pm_bars = raw_day.between_time("04:00", "09:29")
                    pm_ret  = Indicators.premarket_return(pm_bars)
                # Order imbalance uses the PREVIOUS day's first-30 session bars to
                # avoid intra-day look-ahead (sizing is decided before the session,
                # so we can only know what happened yesterday, not today 09:30-09:59).
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
            net_pnl, _, trades_count, avg_dur = self._compute_pnl(cur_df["close"], exposure, shares)
            result.loc[current_day, "AUM"]     = prev_aum + net_pnl
            result.loc[current_day, "ret"]     = net_pnl / prev_aum
            result.loc[current_day, "trades"]  = trades_count
            result.loc[current_day, "avg_dur"] = avg_dur
            result.loc[current_day, "avg_lev"] = leverage if trades_count > 0 else 0.0
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
    ) -> tuple[float, float, int, float]:
        cfg = self.config
        exp_arr   = exposure.values
        close_arr = close_prices.values

        trades_count    = int(np.sum(np.abs(np.diff(np.append(exp_arr, 0)))))
        change_1m       = np.diff(close_arr, prepend=close_arr[0])
        gross_pnl       = float(np.sum(exp_arr * change_1m) * shares)
        commission_paid = float(
            trades_count * max(cfg.min_comm, cfg.commission * shares)
        )
        # Slippage: paid on every order (each direction change = one order)
        # $0.001/share one-way, applied to every fill
        slippage_paid   = float(trades_count * cfg.slippage * shares)
        avg_duration    = _avg_trade_duration(exp_arr)
        return gross_pnl - commission_paid - slippage_paid, commission_paid, trades_count, avg_duration
