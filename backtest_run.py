"""
Backtesting entry point.

Usage
-----
    # Run with local CSV data (no API keys needed):
    python backtest_run.py --csv data/raw/spy_intraday.csv

    # Run with Alpaca API data:
    python backtest_run.py --start 2024-01-01 --end 2026-01-01

    # Include grid-search parameter optimisation:
    python backtest_run.py --csv data/raw/spy_intraday.csv --optimize

Output
------
  Plots saved to analysis/plots/
  Performance tables printed to stdout
"""

from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from analysis.metrics import PerformanceMetrics
from config.config import AppConfig, StrategyConfig
from data.fetch import AlpacaFetcher
from data.preprocess import Preprocessor
from trader.backtest import Backtester

PLOT_DIR = Path("analysis/plots")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Intraday Momentum Backtester")
    parser.add_argument("--csv", type=str, help="Path to local CSV data file")
    parser.add_argument("--start", type=str, default="2024-01-01")
    parser.add_argument("--end", type=str, default="2026-01-01")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--optimize", action="store_true",
                        help="Run grid-search parameter optimisation")
    return parser.parse_args()


def load_data(args: argparse.Namespace, app_cfg: AppConfig) -> pd.DataFrame:
    if args.csv:
        print(f"Loading data from CSV: {args.csv}")
        return AlpacaFetcher.from_csv(args.csv)

    print(f"Fetching data from Alpaca ({args.start} – {args.end}) ...")
    fetcher = AlpacaFetcher(app_cfg.alpaca)
    df = fetcher.get_historical_bars(app_cfg.strategy.symbol, args.start, args.end)
    if df.empty:
        print("ERROR: No data returned. Check your Alpaca credentials and date range.")
        sys.exit(1)
    return df


def plot_comparison(base_result, rsi_result, aum_0: float, save_path: Path) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 7))

    spy_aum = aum_0 * (1 + base_result.daily["ret_spy"]).cumprod(skipna=True)
    ax.plot(base_result.daily.index, base_result.aum,
            label="Base Strategy", color="#2C3E50", lw=2)
    ax.plot(rsi_result.daily.index, rsi_result.aum,
            label="RSI Filter Strategy", color="#3498DB", lw=2)
    ax.plot(base_result.daily.index, spy_aum,
            label="SPY Buy & Hold", color="#E74C3C", lw=2, ls="--")

    paper_date = pd.Timestamp("2025-04-10")
    ax.axvline(paper_date, color="#758473", lw=1.5)
    ax.text(paper_date, ax.get_ylim()[1] * 0.97, "Paper Published",
            color="#758473", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45, ha="right")
    ax.grid(True, ls="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Intraday Momentum — Base vs RSI Filter", fontweight="bold", fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved → {save_path}")
    plt.show()


def plot_train_test(train_res, test_res, best_params: dict, aum_0: float, save_path: Path) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, result, title, color in zip(
        axes,
        [train_res, test_res],
        ["Train (In-Sample)", "Test (Out-of-Sample)"],
        ["#2C3E50", "#27AE60"],
    ):
        spy_aum = aum_0 * (1 + result.daily["ret_spy"]).cumprod(skipna=True)
        ax.plot(result.daily.index, result.aum, label="Strategy", color=color, lw=2.5)
        ax.plot(result.daily.index, spy_aum, label="SPY", color="#E74C3C", lw=2)
        ax.set_title(title, fontweight="bold")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.grid(True, ls="--", alpha=0.3)
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.suptitle(f"Best params: {best_params}", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved → {save_path}")
    plt.show()


def run_grid_search(df, df_daily, base_cfg: StrategyConfig, train_ratio: float = 0.70):
    param_grid = {
        "band_mult": [0.5, 1.0, 1.5],
        "trade_freq": [15, 30, 60],
        "target_vol": [0.01, 0.02, 0.03],
        "max_leverage": [2, 4, 6],
    }
    keys = list(param_grid.keys())
    combos = list(product(*param_grid.values()))

    df_train, dd_train, df_test, dd_test = Backtester.split(df, df_daily, train_ratio)

    print(f"\nGrid search: {len(combos)} combinations...")
    best_sharpe, best_params = -np.inf, {}

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        cfg = StrategyConfig(**{**base_cfg.__dict__, **params})
        try:
            result = Backtester(cfg).run(df_train, dd_train)
            s = result.metrics.sharpe_ratio
            if s > best_sharpe:
                best_sharpe, best_params = s, params
        except Exception as e:
            print(f"  Combo {i+1} failed: {e}")
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(combos)} done...")

    print(f"\nBest train params (Sharpe={best_sharpe:.3f}): {best_params}")
    cfg_best = StrategyConfig(**{**base_cfg.__dict__, **best_params})
    train_result = Backtester(cfg_best).run(df_train, dd_train)
    test_result = Backtester(cfg_best).run(df_test, dd_test)

    print("\n[TRAIN]")
    print(train_result.metrics)
    print("\n[TEST — out-of-sample]")
    print(test_result.metrics)

    return best_params, train_result, test_result


def main() -> None:
    args = parse_args()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(args.config)
    app_cfg = AppConfig.from_yaml(cfg_path) if cfg_path.exists() else AppConfig.default()

    raw = load_data(args, app_cfg)
    print("Preprocessing features...")
    df, df_daily = Preprocessor(app_cfg.strategy).transform(raw)

    # Base strategy (no RSI filter)
    base_cfg = StrategyConfig(**{**app_cfg.strategy.__dict__, "rsi_filter": False})
    print("\nRunning base strategy...")
    base_result = Backtester(base_cfg).run(df, df_daily)
    print("\n[BASE STRATEGY]")
    print(base_result.metrics)

    # RSI-filtered strategy
    rsi_cfg = StrategyConfig(**{**app_cfg.strategy.__dict__, "rsi_filter": True})
    print("\nRunning RSI-filtered strategy...")
    rsi_result = Backtester(rsi_cfg).run(df, df_daily)
    print("\n[RSI FILTER STRATEGY]")
    print(rsi_result.metrics)

    plot_comparison(base_result, rsi_result, app_cfg.strategy.aum_0,
                    PLOT_DIR / "strategy_comparison.png")

    if args.optimize:
        best_params, train_res, test_res = run_grid_search(df, df_daily, app_cfg.strategy)
        plot_train_test(train_res, test_res, best_params, app_cfg.strategy.aum_0,
                        PLOT_DIR / "train_test_split.png")


if __name__ == "__main__":
    main()
