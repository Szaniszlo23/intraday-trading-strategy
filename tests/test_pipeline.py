"""
Smoke tests for the intraday momentum pipeline.

Uses synthetic data — no Alpaca credentials or CSV files required.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from analysis.metrics import PerformanceMetrics
from config.config import StrategyConfig
from data.preprocess import Preprocessor
from trader.backtest import Backtester
from trader.signals import SignalGenerator
from trader.sizing import PositionSizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_synthetic_df(n_days: int = 30, bars_per_day: int = 390) -> pd.DataFrame:
    """Generate synthetic 1-minute SPY-like data (no weekends)."""
    np.random.seed(42)
    base_date = date(2024, 3, 4)
    timestamps = []
    for d in range(n_days * 2):          # overshoot to fill n_days after skipping weekends
        day = base_date + timedelta(days=d)
        if day.weekday() >= 5:
            continue
        if len(timestamps) // bars_per_day >= n_days:
            break
        open_time = pd.Timestamp(f"{day} 09:30:00")
        for m in range(bars_per_day):
            timestamps.append(open_time + pd.Timedelta(minutes=m))

    n = len(timestamps)
    prices = 500 + np.cumsum(np.random.randn(n) * 0.05)

    df = pd.DataFrame(
        {
            "open": prices - 0.02,
            "high": prices + 0.05,
            "low": prices - 0.05,
            "close": prices,
            "volume": np.random.randint(100_000, 500_000, n).astype(float),
        },
        index=pd.DatetimeIndex(timestamps, name="caldt"),
    )
    df["day"] = df.index.date
    return df


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

class TestPreprocessor:
    def setup_method(self):
        self.cfg = StrategyConfig()
        self.raw = _make_synthetic_df(n_days=20)
        self.df, self.df_daily = Preprocessor(self.cfg).transform(self.raw)

    def test_required_columns(self):
        for col in ["vwap", "move_open", "spy_dvol", "sigma_open", "rsi",
                    "min_from_open", "minute_of_day"]:
            assert col in self.df.columns

    def test_vwap_positive(self):
        assert (self.df["vwap"].dropna() > 0).all()

    def test_vwap_within_hlc(self):
        for _, grp in self.df.groupby("day"):
            v = grp["vwap"].dropna()
            if len(v) == 0:
                continue
            assert v.max() <= grp["high"].max() * 1.01
            assert v.min() >= grp["low"].min() * 0.99

    def test_rsi_in_range(self):
        rsi = self.df["rsi"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_min_from_open_starts_at_1(self):
        first_bars = self.df.groupby("day").first()
        assert (first_bars["min_from_open"] == 1.0).all()


# ---------------------------------------------------------------------------
# Signal generator
# ---------------------------------------------------------------------------

class TestSignalGenerator:
    def setup_method(self):
        self.cfg = StrategyConfig(rsi_filter=False)
        raw = _make_synthetic_df(n_days=20)
        self.df, _ = Preprocessor(self.cfg).transform(raw)

    def _day_pair(self, idx: int = 15):
        days = sorted(self.df["day"].unique())
        day_df = self.df[self.df["day"] == days[idx]]
        prev_close = self.df[self.df["day"] == days[idx - 1]]["close"].iloc[-1]
        return day_df, prev_close

    def test_exposure_values(self):
        day_df, prev_close = self._day_pair()
        exp = SignalGenerator(self.cfg).generate(day_df, prev_close)
        assert set(exp.unique()).issubset({-1.0, 0.0, 1.0})

    def test_exposure_length(self):
        day_df, prev_close = self._day_pair()
        exp = SignalGenerator(self.cfg).generate(day_df, prev_close)
        assert len(exp) == len(day_df)

    def test_rsi_filter_fewer_or_equal_trades(self):
        day_df, prev_close = self._day_pair()
        base = SignalGenerator(StrategyConfig(rsi_filter=False)).generate(day_df, prev_close)
        rsi  = SignalGenerator(StrategyConfig(rsi_filter=True)).generate(day_df, prev_close)
        assert (rsi.diff().abs() > 0).sum() <= (base.diff().abs() > 0).sum() + 1


# ---------------------------------------------------------------------------
# Position sizer
# ---------------------------------------------------------------------------

class TestPositionSizer:
    def test_full_notional(self):
        s = PositionSizer(StrategyConfig(sizing_type="full_notional"))
        assert s.shares(100_000, 500, 0.01) == 200

    def test_vol_target_capped(self):
        s = PositionSizer(StrategyConfig(sizing_type="vol_target", target_vol=0.02, max_leverage=4.0))
        assert s.shares(100_000, 500, 0.001) == round(100_000 / 500 * 4)

    def test_vol_target_nan(self):
        s = PositionSizer(StrategyConfig(sizing_type="vol_target", max_leverage=4.0))
        assert s.shares(100_000, 500, float("nan")) == round(100_000 / 500 * 4)

    def test_vol_target_normal(self):
        s = PositionSizer(StrategyConfig(sizing_type="vol_target", target_vol=0.02, max_leverage=4.0))
        assert s.shares(100_000, 500, 0.02) == round(100_000 / 500 * 1.0)


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class TestBacktester:
    def setup_method(self):
        raw = _make_synthetic_df(n_days=30)
        self.df, self.df_daily = Preprocessor(StrategyConfig(rsi_filter=False)).transform(raw)

    def test_smoke_run(self):
        result = Backtester(StrategyConfig(rsi_filter=False)).run(self.df, self.df_daily)
        assert result.daily is not None and len(result.daily) > 0

    def test_returns_finite(self):
        result = Backtester(StrategyConfig(rsi_filter=False)).run(self.df, self.df_daily)
        assert result.returns.dropna().apply(math.isfinite).all()

    def test_aum_positive(self):
        result = Backtester(StrategyConfig(rsi_filter=False)).run(self.df, self.df_daily)
        assert (result.aum.dropna() > 0).all()

    def test_split_sizes(self):
        df_tr, _, df_te, _ = Backtester.split(self.df, self.df_daily, 0.7)
        n = self.df["day"].nunique()
        assert df_tr["day"].nunique() == round(n * 0.7)
        assert df_te["day"].nunique() == n - round(n * 0.7)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestPerformanceMetrics:
    def test_sharpe_manual(self):
        np.random.seed(0)
        r = pd.Series(np.random.randn(252) * 0.01 + 0.0005)
        result = PerformanceMetrics.compute(r)
        expected = r.mean() / r.std() * np.sqrt(252)
        assert abs(result.sharpe_ratio - expected) < 1e-6

    def test_hit_ratio_all_positive(self):
        result = PerformanceMetrics.compute(pd.Series([0.01] * 100))
        assert result.hit_ratio == pytest.approx(100.0)

    def test_equity_curve_start(self):
        curve = PerformanceMetrics.equity_curve(pd.Series([0.01, -0.01, 0.02]), 100_000)
        assert curve.iloc[0] == pytest.approx(101_000)
