"""
Performance metrics for backtesting and live evaluation.

All metrics are stateless class methods — call on any returns series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from dataclasses import dataclass

TRADING_DAYS = 252


@dataclass
class PerformanceResult:
    total_return: float
    annualized_return: float
    annualized_vol: float
    sharpe_ratio: float
    hit_ratio: float
    max_drawdown: float
    alpha: float
    beta: float

    def __str__(self) -> str:
        lines = [
            "=" * 45,
            f"{'PERFORMANCE SUMMARY':^45}",
            "=" * 45,
            f"  {'Total Return (%):':<30} {self.total_return:>8.1f}",
            f"  {'Annualized Return (%):':<30} {self.annualized_return:>8.1f}",
            f"  {'Annualized Volatility (%):':<30} {self.annualized_vol:>8.1f}",
            f"  {'Sharpe Ratio:':<30} {self.sharpe_ratio:>8.2f}",
            f"  {'Hit Ratio (%):':<30} {self.hit_ratio:>8.1f}",
            f"  {'Maximum Drawdown (%):':<30} {self.max_drawdown:>8.1f}",
            f"  {'Alpha (% ann.):':<30} {self.alpha:>8.2f}",
            f"  {'Beta:':<30} {self.beta:>8.2f}",
            "=" * 45,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "Total Return (%)": round(self.total_return, 1),
            "Annualized Return (%)": round(self.annualized_return, 1),
            "Annualized Volatility (%)": round(self.annualized_vol, 1),
            "Sharpe Ratio": round(self.sharpe_ratio, 2),
            "Hit Ratio (%)": round(self.hit_ratio, 1),
            "Maximum Drawdown (%)": round(self.max_drawdown, 1),
            "Alpha (%)": round(self.alpha, 2),
            "Beta": round(self.beta, 2),
        }


class PerformanceMetrics:

    @staticmethod
    def compute(
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series | None = None,
    ) -> PerformanceResult:
        """
        Compute all standard performance metrics.

        Parameters
        ----------
        strategy_returns  : daily net returns (not cumulative)
        benchmark_returns : daily benchmark returns (e.g. SPY buy-and-hold)
                            required for alpha/beta; defaults to NaN if None.
        """
        ret = strategy_returns.dropna()
        n = len(ret)

        total_return = (np.prod(1 + ret) - 1) * 100
        annualized_return = (np.prod(1 + ret) ** (TRADING_DAYS / n) - 1) * 100
        annualized_vol = ret.std() * np.sqrt(TRADING_DAYS) * 100
        sharpe_ratio = (
            ret.mean() / ret.std() * np.sqrt(TRADING_DAYS)
            if ret.std() > 0 else np.nan
        )
        hit_ratio = (ret > 0).sum() / (ret.abs() > 0).sum() * 100

        cum = (1 + ret).cumprod()
        rolling_max = cum.expanding().max()
        max_drawdown = ((cum - rolling_max) / rolling_max).min() * -100

        alpha, beta = np.nan, np.nan
        if benchmark_returns is not None:
            bm = benchmark_returns.dropna()
            common = ret.index.intersection(bm.index)
            if len(common) > 10:
                Y = ret.loc[common]
                X = sm.add_constant(bm.loc[common])
                model = sm.OLS(Y, X).fit()
                alpha = model.params.iloc[0] * TRADING_DAYS * 100
                beta = model.params.iloc[1]

        return PerformanceResult(
            total_return=total_return,
            annualized_return=annualized_return,
            annualized_vol=annualized_vol,
            sharpe_ratio=sharpe_ratio,
            hit_ratio=hit_ratio,
            max_drawdown=max_drawdown,
            alpha=alpha,
            beta=beta,
        )

    @staticmethod
    def compute_yearly(
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series | None = None,
    ) -> dict[int, PerformanceResult]:
        """
        Compute performance metrics broken down by calendar year.

        Parameters
        ----------
        strategy_returns  : daily net returns with a DatetimeIndex
        benchmark_returns : daily benchmark returns (optional)

        Returns
        -------
        dict mapping year (int) → PerformanceResult
        """
        strategy_returns = strategy_returns.dropna()
        results: dict[int, PerformanceResult] = {}
        for year, group in strategy_returns.groupby(strategy_returns.index.year):
            bm_year: pd.Series | None = None
            if benchmark_returns is not None:
                bm_year = benchmark_returns[benchmark_returns.index.year == year]
            results[int(year)] = PerformanceMetrics.compute(group, bm_year)
        return results

    @staticmethod
    def equity_curve(returns: pd.Series, aum_0: float = 100_000.0) -> pd.Series:
        """Convert daily returns to a dollar equity curve."""
        return aum_0 * (1 + returns.fillna(0)).cumprod()
