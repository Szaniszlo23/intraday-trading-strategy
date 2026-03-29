"""
Backtesting engine for the intraday momentum strategy.

The Backtester class orchestrates:
  1. Iterating over trading days
  2. Generating signals via SignalGenerator
  3. Sizing positions via PositionSizer
  4. Computing P&L including realistic commissions
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analysis.metrics import PerformanceMetrics, PerformanceResult
from config.config import StrategyConfig
from trader.signals import SignalGenerator
from trader.sizing import PositionSizer


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
            index=all_days, columns=["ret", "AUM", "ret_spy"], dtype=float
        )
        result["AUM"] = cfg.aum_0

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
            prev_aum = result.loc[prev_day, "AUM"]

            exposure = self._signal_gen.generate(cur_df, prev_close)
            shares = self._sizer.shares(prev_aum, open_price, daily_vol)
            net_pnl, _ = self._compute_pnl(cur_df["close"], exposure, shares)

            result.loc[current_day, "AUM"] = prev_aum + net_pnl
            result.loc[current_day, "ret"] = net_pnl / prev_aum

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
    ) -> tuple[float, float]:
        cfg = self.config
        exp_arr = exposure.values
        close_arr = close_prices.values

        trades_count = np.sum(np.abs(np.diff(np.append(exp_arr, 0))))
        change_1m = np.diff(close_arr, prepend=close_arr[0])
        gross_pnl = float(np.sum(exp_arr * change_1m) * shares)
        commission_paid = float(
            trades_count * max(cfg.min_comm, cfg.commission * shares)
        )
        return gross_pnl - commission_paid, commission_paid

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
