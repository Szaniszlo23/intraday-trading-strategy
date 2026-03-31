"""
Backtesting entry point.

Usage
-----
    # Run with local CSV data (no API keys needed):
    python backtest_run.py --csv data/raw/spy_intraday.csv

    # Run with Alpaca API data:
    python backtest_run.py --start 2024-01-01 --end 2026-01-01

    # Include grid-search parameter optimisation (baseline only):
    python backtest_run.py --csv data/raw/spy_intraday.csv --optimize

Output
------
  Plots saved to analysis/plots/
  Performance tables printed to stdout

Strategies compared
-------------------
  Baseline  — noise-band + VWAP, 30-min clock, 14-day sigma
  Enhanced  — same boundary, 1-min every-bar monitoring with dual-bar
              confirmation, VWAP trailing stop, 90-day sigma, Layer 2
              sizing multiplier (vol-regime × flow composite)
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
from trader.backtest import Backtester, EnhancedBacktester
from trader.signals import EnhancedSignalGenerator

PLOT_DIR = Path("analysis/plots")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Intraday Momentum Backtester")
    parser.add_argument("--csv", type=str, help="Path to local CSV data file")
    parser.add_argument("--start", type=str, default="2016-01-01")
    parser.add_argument("--end", type=str, default="2026-01-01")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--optimize", action="store_true",
                        help="Run grid-search parameter optimisation (baseline)")
    parser.add_argument("--save", type=str, help="Save fetched data to CSV path")
    parser.add_argument("--variants", action="store_true",
                        help="Isolate each enhancement one at a time (6 variants)")
    parser.add_argument("--yearly", action="store_true",
                        help="Print year-by-year performance breakdown")

    return parser.parse_args()


def load_data(args: argparse.Namespace, app_cfg: AppConfig) -> pd.DataFrame:
    """Load raw 1-min bars (including extended hours for flow composite)."""
    if args.csv:
        print(f"Loading data from CSV: {args.csv}")
        return AlpacaFetcher.from_csv(args.csv)

    print(f"Fetching data from Alpaca ({args.start} – {args.end}) ...")
    fetcher = AlpacaFetcher(app_cfg.alpaca)
    df = fetcher.get_historical_bars(
        app_cfg.strategy.symbol, args.start, args.end, include_extended=True
    )
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.save)
        print(f"Data saved to {args.save}")
    
    if df.empty:
        print("ERROR: No data returned. Check your Alpaca credentials and date range.")
        sys.exit(1)
    return df


def plot_comparison(
    base_result, enh_result, aum_0: float, save_path: Path
) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 7))

    spy_aum = aum_0 * (1 + base_result.daily["ret_spy"]).cumprod(skipna=True)
    ax.plot(base_result.daily.index, base_result.aum,
            label="Baseline Strategy", color="#2C3E50", lw=2)
    ax.plot(enh_result.daily.index, enh_result.aum,
            label="Enhanced Strategy", color="#27AE60", lw=2)
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
    ax.set_title("Intraday Momentum — Baseline vs Enhanced", fontweight="bold", fontsize=13)
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


def print_yearly_breakdown(name: str, result) -> None:
    """Print a year-by-year performance table for a BacktestResult."""
    yearly = PerformanceMetrics.compute_yearly(result.returns, result.daily["ret_spy"])
    trades_by_year = result.daily["trades"].groupby(result.daily.index.year).sum()
    avg_dur_by_year = result.daily["avg_dur"].groupby(result.daily.index.year).mean()

    header = f"\n{'─' * 84}\n  {name} — Year-by-Year Breakdown\n{'─' * 84}"
    print(header)
    print(f"  {'Year':<6} {'Ann Ret %':>9} {'Sharpe':>7} {'MaxDD %':>8} {'Hit %':>7} {'N Trades':>9} {'Avg Hold (min)':>14}")
    print(f"  {'─'*6} {'─'*9} {'─'*7} {'─'*8} {'─'*7} {'─'*9} {'─'*14}")
    for year in sorted(yearly):
        m = yearly[year]
        n_trades = int(trades_by_year.get(year, 0))
        avg_dur  = avg_dur_by_year.get(year, float("nan"))
        avg_dur_str = f"{avg_dur:>14.1f}" if not np.isnan(avg_dur) else f"{'—':>14}"
        print(
            f"  {year:<6} {m.annualized_return:>9.1f} {m.sharpe_ratio:>7.2f} "
            f"{m.max_drawdown:>8.1f} {m.hit_ratio:>7.1f} {n_trades:>9} {avg_dur_str}"
        )
    print(f"{'─' * 84}")


def run_variants(
    raw: "pd.DataFrame",
    raw_session: "pd.DataFrame",
    app_cfg: AppConfig,
) -> None:
    """
    Run 6 cumulative variants, adding one enhancement at a time:

      V1  Baseline          — SignalGenerator, 14d sigma, 30-min clock
      V2  +90d sigma        — SignalGenerator, 90d sigma, 30-min clock
      V3  +Every bar        — EnhancedSignalGenerator, no dual-bar, no VWAP stop
      V4  +Dual-bar confirm — EnhancedSignalGenerator, dual-bar, no VWAP stop
      V5  +VWAP stop        — EnhancedSignalGenerator, dual-bar, VWAP stop
      V6  Full enhanced     — above + Layer 2 sizing multiplier
    """
    base_cfg = app_cfg.strategy

    # Pre-process once per sigma window (two passes)
    print("\nPreprocessing 14d sigma features...")
    cfg_14 = StrategyConfig(**{**base_cfg.__dict__, "vol_window": 14})
    df_14, dd_14 = Preprocessor(cfg_14).transform(raw_session)

    print("Preprocessing 90d sigma features...")
    cfg_90 = StrategyConfig(**{**base_cfg.__dict__, "vol_window": 90})
    df_90, dd_90 = Preprocessor(cfg_90).transform(raw_session)

    variants = [
        ("V1  Baseline",           lambda: Backtester(cfg_14).run(df_14, dd_14)),
        ("V2  +90d sigma",         lambda: Backtester(cfg_90).run(df_90, dd_90)),
        ("V3  +Every bar",         lambda: EnhancedBacktester(
            cfg_90,
            signal_gen=EnhancedSignalGenerator(cfg_90, use_dual_bar=False, use_vwap_stop=False),
            use_layer2=False,
        ).run(df_90, dd_90, raw)),
        ("V4  +Dual-bar",          lambda: EnhancedBacktester(
            cfg_90,
            signal_gen=EnhancedSignalGenerator(cfg_90, use_dual_bar=True, use_vwap_stop=False),
            use_layer2=False,
        ).run(df_90, dd_90, raw)),
        ("V5  +VWAP stop",         lambda: EnhancedBacktester(
            cfg_90,
            signal_gen=EnhancedSignalGenerator(cfg_90, use_dual_bar=True, use_vwap_stop=True),
            use_layer2=False,
        ).run(df_90, dd_90, raw)),
        ("V6  Full enhanced",      lambda: EnhancedBacktester(
            cfg_90,
            signal_gen=EnhancedSignalGenerator(cfg_90, use_dual_bar=True, use_vwap_stop=True),
            use_layer2=True,
        ).run(df_90, dd_90, raw)),
    ]

    print("\n" + "═" * 104)
    print(f"  {'VARIANT ISOLATION ANALYSIS':^102}")
    print("═" * 104)
    print(
        f"  {'Variant':<22} {'Tot Ret%':>8} {'Ann Ret%':>9} {'Sharpe':>7} "
        f"{'MaxDD%':>7} {'Hit%':>6} {'Alpha':>7} {'Beta':>6} {'Trades/day':>10} {'Avg Hold(min)':>13}"
    )
    print(f"  {'─'*22} {'─'*8} {'─'*9} {'─'*7} {'─'*7} {'─'*6} {'─'*7} {'─'*6} {'─'*10} {'─'*13}")

    results = {}
    for name, run_fn in variants:
        print(f"  Running {name.strip()} ...", end="\r")
        result = run_fn()
        results[name] = result
        m = result.metrics
        n_days   = len(result.returns.dropna())
        trades_d = result.daily["trades"].sum() / max(n_days, 1)
        avg_hold = result.daily["avg_dur"].mean()
        avg_hold_str = f"{avg_hold:>13.1f}" if not np.isnan(avg_hold) else f"{'—':>13}"
        print(
            f"  {name:<22} {m.total_return:>8.1f} {m.annualized_return:>9.1f} "
            f"{m.sharpe_ratio:>7.2f} {m.max_drawdown:>7.1f} {m.hit_ratio:>6.1f} "
            f"{m.alpha:>7.2f} {m.beta:>6.2f} {trades_d:>10.2f} {avg_hold_str}"
        )

    print("═" * 104)

    # Plot all 6 equity curves on one chart
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 7))
    colours = ["#2C3E50", "#8E44AD", "#2980B9", "#27AE60", "#F39C12", "#E74C3C"]
    for (name, _), colour in zip(variants, colours):
        res = results[name]
        ax.plot(res.daily.index, res.aum, label=name.strip(), lw=1.8, color=colour)

    # SPY buy-and-hold overlay
    v1_res = results[variants[0][0]]
    spy_aum = base_cfg.aum_0 * (1 + v1_res.daily["ret_spy"]).cumprod(skipna=True)
    ax.plot(v1_res.daily.index, spy_aum, label="SPY Buy & Hold",
            color="gray", lw=1.5, ls="--")

    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha="right")
    ax.grid(True, ls="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Enhancement Isolation — V1 (Baseline) → V6 (Full Enhanced)",
                 fontweight="bold", fontsize=13)
    ax.legend(fontsize=9)
    plt.tight_layout()
    save_path = PLOT_DIR / "variant_isolation.png"
    plt.savefig(save_path, dpi=150)
    print(f"\n  Chart saved → {save_path}")
    plt.show()


def run_grid_search(df, df_daily, base_cfg: StrategyConfig, train_ratio: float = 0.70):
    param_grid = {
        "band_mult":    [0.5, 1.0, 1.5],
        "trade_freq":   [15, 30, 60],
        "target_vol":   [0.01, 0.02, 0.03],
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
    test_result  = Backtester(cfg_best).run(df_test,  dd_test)

    print("\n[TRAIN]")
    print(train_result.metrics)
    print("\n[TEST — out-of-sample]")
    print(test_result.metrics)

    return best_params, train_result, test_result


def main() -> None:
    args = parse_args()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(args.config)
    app_cfg = AppConfig.load(cfg_path)

    # raw includes extended-hours bars (pre-market) for the flow composite
    raw = load_data(args, app_cfg)

    # Session-only slice for preprocessing (indicators expect 09:30-16:00)
    raw_session = raw.between_time("09:30", "16:00")

    # ---- Baseline (14-day sigma, 30-min clock) ----
    print("Preprocessing baseline features (vol_window=14)...")
    base_strategy_cfg = StrategyConfig(**{**app_cfg.strategy.__dict__, "vol_window": 14})
    df_base, df_daily_base = Preprocessor(base_strategy_cfg).transform(raw_session)

    print("\nRunning Baseline strategy...")
    base_result = Backtester(base_strategy_cfg).run(df_base, df_daily_base)
    print("\n[BASELINE STRATEGY]")
    print(base_result.metrics)

    # ---- Enhanced (90-day sigma, 1-min dual-bar, Layer 2 sizing) ----
    print("\nPreprocessing enhanced features (vol_window=90)...")
    enh_strategy_cfg = StrategyConfig(**{**app_cfg.strategy.__dict__, "vol_window": 90})
    df_enh, df_daily_enh = Preprocessor(enh_strategy_cfg).transform(raw_session)

    print("\nRunning Enhanced strategy...")
    enh_result = EnhancedBacktester(enh_strategy_cfg).run(df_enh, df_daily_enh, raw)
    print("\n[ENHANCED STRATEGY]")
    print(enh_result.metrics)

    plot_comparison(base_result, enh_result, app_cfg.strategy.aum_0,
                    PLOT_DIR / "strategy_comparison.png")

    if args.yearly:
        print_yearly_breakdown("Baseline", base_result)
        print_yearly_breakdown("Enhanced", enh_result)

    if args.variants:
        run_variants(raw, raw_session, app_cfg)

    if args.optimize:
        best_params, train_res, test_res = run_grid_search(
            df_base, df_daily_base, base_strategy_cfg
        )
        plot_train_test(train_res, test_res, best_params, app_cfg.strategy.aum_0,
                        PLOT_DIR / "train_test_split.png")


if __name__ == "__main__":
    main()
