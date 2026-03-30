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
from data.indicators import Indicators
from data.preprocess import Preprocessor
from trader.backtest import Backtester, EnhancedBacktester
from trader.signals import EnhancedSignalGenerator, SignalGenerator
from trader.sizing import EnhancedPositionSizer, PositionSizer


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


def _make_synthetic_with_premarket(n_days: int = 30) -> pd.DataFrame:
    """
    Like _make_synthetic_df but also includes 30 pre-market bars (09:00-09:29)
    per day so EnhancedBacktester can extract premarket_return.
    """
    np.random.seed(7)
    base_date = date(2024, 3, 4)
    timestamps = []
    for d in range(n_days * 2):
        day = base_date + timedelta(days=d)
        if day.weekday() >= 5:
            continue
        if len(timestamps) // 420 >= n_days:  # 30 pm + 390 session
            break
        # 30 pre-market bars 09:00–09:29
        pm_start = pd.Timestamp(f"{day} 09:00:00")
        for m in range(30):
            timestamps.append(pm_start + pd.Timedelta(minutes=m))
        # 390 regular session bars 09:30–16:00
        open_time = pd.Timestamp(f"{day} 09:30:00")
        for m in range(390):
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
        for col in ["vwap", "move_open", "spy_dvol", "sigma_open",
                    "min_from_open", "minute_of_day"]:
            assert col in self.df.columns

    def test_no_rsi_column(self):
        assert "rsi" not in self.df.columns

    def test_vwap_positive(self):
        assert (self.df["vwap"].dropna() > 0).all()

    def test_vwap_within_hlc(self):
        for _, grp in self.df.groupby("day"):
            v = grp["vwap"].dropna()
            if len(v) == 0:
                continue
            assert v.max() <= grp["high"].max() * 1.01
            assert v.min() >= grp["low"].min() * 0.99

    def test_min_from_open_starts_at_1(self):
        first_bars = self.df.groupby("day").first()
        assert (first_bars["min_from_open"] == 1.0).all()


# ---------------------------------------------------------------------------
# Baseline signal generator
# ---------------------------------------------------------------------------

class TestSignalGenerator:
    def setup_method(self):
        self.cfg = StrategyConfig()
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


# ---------------------------------------------------------------------------
# Enhanced signal generator
# ---------------------------------------------------------------------------

class TestEnhancedSignalGenerator:
    def setup_method(self):
        self.cfg = StrategyConfig(vol_window=20)  # small window for test data
        raw = _make_synthetic_df(n_days=30)
        self.df, _ = Preprocessor(self.cfg).transform(raw)

    def _day_pair(self, idx: int = 20):
        days = sorted(self.df["day"].unique())
        day_df = self.df[self.df["day"] == days[idx]]
        prev_close = self.df[self.df["day"] == days[idx - 1]]["close"].iloc[-1]
        return day_df, prev_close

    def test_exposure_values(self):
        day_df, prev_close = self._day_pair()
        exp = EnhancedSignalGenerator(self.cfg).generate(day_df, prev_close)
        assert set(exp.unique()).issubset({-1.0, 0.0, 1.0})

    def test_exposure_length(self):
        day_df, prev_close = self._day_pair()
        exp = EnhancedSignalGenerator(self.cfg).generate(day_df, prev_close)
        assert len(exp) == len(day_df)

    def test_no_new_entry_after_cutoff(self):
        """No entry (position change from 0) should occur at or after 15:30."""
        day_df, prev_close = self._day_pair()
        exp = EnhancedSignalGenerator(self.cfg).generate(day_df, prev_close)
        cutoff_mask = (day_df.index.hour * 60 + day_df.index.minute) >= (15 * 60 + 30)
        cutoff_bars = exp[cutoff_mask]
        if len(cutoff_bars) == 0:
            return
        pre_cutoff_last = exp[~cutoff_mask].iloc[-1] if (~cutoff_mask).any() else 0.0
        for i, (ts, val) in enumerate(cutoff_bars.items()):
            prev_val = pre_cutoff_last if i == 0 else cutoff_bars.iloc[i - 1]
            if prev_val == 0.0 and val != 0.0:
                pytest.fail(f"New entry found after 15:30 at {ts}")


# ---------------------------------------------------------------------------
# Position sizer (baseline)
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
# Enhanced position sizer
# ---------------------------------------------------------------------------

class TestEnhancedPositionSizer:
    def test_multiplier_clipped_low(self):
        s = EnhancedPositionSizer()
        # vol_regime=0.1, flow_mult=0.1 → M=0.01 → clipped to 0.25
        shares = s.shares(100_000, 500, 0.02, vol_regime=0.1, flow_mult=0.1)
        expected = math.floor(100_000 * min(4.0, 0.02 / 0.02) * 0.25 / 500)
        assert shares == expected

    def test_multiplier_clipped_high(self):
        s = EnhancedPositionSizer()
        # vol_regime=5.0, flow_mult=5.0 → M=25 → clipped to 2.0
        shares = s.shares(100_000, 500, 0.02, vol_regime=5.0, flow_mult=5.0)
        expected = math.floor(100_000 * min(4.0, 0.02 / 0.02) * 2.0 / 500)
        assert shares == expected

    def test_nan_vol_uses_max_leverage(self):
        s = EnhancedPositionSizer()
        shares = s.shares(100_000, 500, float("nan"), vol_regime=1.0, flow_mult=1.0)
        expected = math.floor(100_000 * 4.0 * 1.0 / 500)
        assert shares == expected


# ---------------------------------------------------------------------------
# Enhanced indicator helpers
# ---------------------------------------------------------------------------

class TestIndicatorHelpers:
    def test_vol_regime_low_vol(self):
        # Very small daily returns → low annualized vol → 0.70 factor
        rets = pd.Series([0.0001] * 10)
        assert Indicators.vol_regime_factor(rets) == pytest.approx(0.70)

    def test_vol_regime_high_vol(self):
        # Large daily returns → high annualized vol → 1.30 factor
        rets = pd.Series([0.02, -0.02, 0.03, -0.03, 0.025])
        factor = Indicators.vol_regime_factor(rets)
        assert factor == pytest.approx(1.30)

    def test_vol_regime_interpolation(self):
        factor = Indicators.vol_regime_factor(pd.Series([0.0] * 5))
        assert 0.70 <= factor <= 1.30

    def test_vol_regime_insufficient_data(self):
        assert Indicators.vol_regime_factor(pd.Series([0.01, 0.02])) == 1.0

    def test_premarket_return_empty(self):
        empty = pd.DataFrame(columns=["open", "close"])
        assert Indicators.premarket_return(empty) == 0.0

    def test_premarket_return_simple(self):
        pm = pd.DataFrame({"open": [100.0, 101.0], "close": [101.0, 102.0]})
        result = Indicators.premarket_return(pm)
        assert result == pytest.approx(102.0 / 100.0 - 1)

    def test_order_imbalance_empty(self):
        empty = pd.DataFrame(columns=["open", "high", "low", "close"])
        assert Indicators.order_imbalance(empty) == 0.0

    def test_order_imbalance_neutral_on_news_day(self):
        # Session range > 0.5% → neutral
        bars = pd.DataFrame({
            "open":  [100.0] * 30,
            "high":  [101.0] * 30,
            "low":   [99.0] * 30,
            "close": [100.5] * 30,
        })
        assert Indicators.order_imbalance(bars) == 0.0

    def test_order_imbalance_all_up(self):
        # All bars close > open, tight range
        bars = pd.DataFrame({
            "open":  [100.0] * 30,
            "high":  [100.1] * 30,
            "low":   [99.99] * 30,
            "close": [100.05] * 30,
        })
        assert Indicators.order_imbalance(bars) == pytest.approx(1.0)

    def test_flow_composite_mult_neutral(self):
        # pm_ret=0, imbalance=0 → composite=0 → mult=1.0
        mult = Indicators.flow_composite_mult(0.0, 0.0)
        assert mult == pytest.approx(1.0)

    def test_flow_composite_mult_range(self):
        for pm_ret in [-0.01, 0.0, 0.01]:
            for imb in [-1.0, 0.0, 1.0]:
                mult = Indicators.flow_composite_mult(pm_ret, imb)
                assert 0.80 <= mult <= 1.20


# ---------------------------------------------------------------------------
# Backtester (baseline)
# ---------------------------------------------------------------------------

class TestBacktester:
    def setup_method(self):
        raw = _make_synthetic_df(n_days=30)
        self.df, self.df_daily = Preprocessor(StrategyConfig()).transform(raw)

    def test_smoke_run(self):
        result = Backtester(StrategyConfig()).run(self.df, self.df_daily)
        assert result.daily is not None and len(result.daily) > 0

    def test_returns_finite(self):
        result = Backtester(StrategyConfig()).run(self.df, self.df_daily)
        assert result.returns.dropna().apply(math.isfinite).all()

    def test_aum_positive(self):
        result = Backtester(StrategyConfig()).run(self.df, self.df_daily)
        assert (result.aum.dropna() > 0).all()

    def test_split_sizes(self):
        df_tr, _, df_te, _ = Backtester.split(self.df, self.df_daily, 0.7)
        n = self.df["day"].nunique()
        assert df_tr["day"].nunique() == round(n * 0.7)
        assert df_te["day"].nunique() == n - round(n * 0.7)


# ---------------------------------------------------------------------------
# EnhancedBacktester
# ---------------------------------------------------------------------------

class TestEnhancedBacktester:
    def setup_method(self):
        self.raw_extended = _make_synthetic_with_premarket(n_days=30)
        raw_session = self.raw_extended.between_time("09:30", "16:00")
        cfg = StrategyConfig(vol_window=20)
        self.df, self.df_daily = Preprocessor(cfg).transform(raw_session)
        self.cfg = cfg

    def test_smoke_run(self):
        result = EnhancedBacktester(self.cfg).run(
            self.df, self.df_daily, self.raw_extended
        )
        assert result.daily is not None and len(result.daily) > 0

    def test_returns_finite(self):
        result = EnhancedBacktester(self.cfg).run(
            self.df, self.df_daily, self.raw_extended
        )
        assert result.returns.dropna().apply(math.isfinite).all()

    def test_aum_positive(self):
        result = EnhancedBacktester(self.cfg).run(
            self.df, self.df_daily, self.raw_extended
        )
        assert (result.aum.dropna() > 0).all()


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
