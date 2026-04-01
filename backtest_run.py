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
    parser.add_argument("--walkforward", action="store_true",
                        help="Walk-forward validation: rolling 2-year OOS windows")
    parser.add_argument("--longshort", action="store_true",
                        help="Print long vs short P&L breakdown and plot equity curves")

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
    trades_by_year  = result.daily["trades"].groupby(result.daily.index.year).sum()
    avg_dur_by_year = result.daily["avg_dur"].groupby(result.daily.index.year).mean()
    # Mean leverage over days where a trade was actually taken (avg_lev > 0)
    lev_col = result.daily["avg_lev"].replace(0.0, float("nan"))
    avg_lev_by_year = lev_col.groupby(result.daily.index.year).mean()

    header = f"\n{'─' * 100}\n  {name} — Year-by-Year Breakdown\n{'─' * 100}"
    print(header)
    print(f"  {'Year':<6} {'Ann Ret %':>9} {'Sharpe':>7} {'MaxDD %':>8} {'Hit %':>7} {'N Trades':>9} {'Avg Hold (min)':>14} {'Avg Lev':>8}")
    print(f"  {'─'*6} {'─'*9} {'─'*7} {'─'*8} {'─'*7} {'─'*9} {'─'*14} {'─'*8}")
    for year in sorted(yearly):
        m        = yearly[year]
        n_trades = int(trades_by_year.get(year, 0))
        avg_dur  = avg_dur_by_year.get(year, float("nan"))
        avg_lev  = avg_lev_by_year.get(year, float("nan"))
        avg_dur_str = f"{avg_dur:>14.1f}" if not np.isnan(avg_dur) else f"{'—':>14}"
        avg_lev_str = f"{avg_lev:>8.2f}x" if not np.isnan(avg_lev) else f"{'—':>8}"
        print(
            f"  {year:<6} {m.annualized_return:>9.1f} {m.sharpe_ratio:>7.2f} "
            f"{m.max_drawdown:>8.1f} {m.hit_ratio:>7.1f} {n_trades:>9} {avg_dur_str} {avg_lev_str}"
        )
    print(f"{'─' * 100}")


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
        ("V7  +Day filter",        lambda: EnhancedBacktester(
            cfg_90,
            signal_gen=EnhancedSignalGenerator(cfg_90, use_dual_bar=True, use_vwap_stop=True),
            use_layer2=True,
            use_day_filter=True,
            day_filter_threshold=0.30,
        ).run(df_90, dd_90, raw)),
    ]

    print("\n" + "═" * 116)
    print(f"  {'VARIANT ISOLATION ANALYSIS':^114}")
    print("═" * 116)
    print(
        f"  {'Variant':<22} {'Tot Ret%':>8} {'Ann Ret%':>9} {'Sharpe':>7} "
        f"{'MaxDD%':>7} {'Hit%':>6} {'Alpha':>7} {'Beta':>6} {'Trades/day':>10} {'Avg Hold(min)':>13} {'Avg Lev':>8}"
    )
    print(f"  {'─'*22} {'─'*8} {'─'*9} {'─'*7} {'─'*7} {'─'*6} {'─'*7} {'─'*6} {'─'*10} {'─'*13} {'─'*8}")

    results = {}
    for name, run_fn in variants:
        print(f"  Running {name.strip()} ...", end="\r")
        result = run_fn()
        results[name] = result
        m = result.metrics
        n_days   = len(result.returns.dropna())
        trades_d = result.daily["trades"].sum() / max(n_days, 1)
        avg_hold = result.daily["avg_dur"].mean()
        # Mean leverage only over days where a trade was actually taken
        lev_col  = result.daily["avg_lev"].replace(0.0, float("nan"))
        avg_lev  = lev_col.mean()
        avg_hold_str = f"{avg_hold:>13.1f}" if not np.isnan(avg_hold) else f"{'—':>13}"
        avg_lev_str  = f"{avg_lev:>7.2f}x"  if not np.isnan(avg_lev)  else f"{'—':>8}"
        print(
            f"  {name:<22} {m.total_return:>8.1f} {m.annualized_return:>9.1f} "
            f"{m.sharpe_ratio:>7.2f} {m.max_drawdown:>7.1f} {m.hit_ratio:>6.1f} "
            f"{m.alpha:>7.2f} {m.beta:>6.2f} {trades_d:>10.2f} {avg_hold_str} {avg_lev_str}"
        )

    print("═" * 116)

    # Plot all 6 equity curves on one chart
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 7))
    colours = ["#2C3E50", "#8E44AD", "#2980B9", "#27AE60", "#F39C12", "#E74C3C", "#1ABC9C"]
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

    # ---- Return distribution / asymmetry chart ----
    plot_return_distribution(
        results,
        colours,
        PLOT_DIR / "return_distribution.png",
    )


def plot_return_distribution(results: dict, colours: list[str], save_path: Path) -> None:
    """
    Plot the daily-return distribution for each variant.

    Each panel shows:
      - Green bars  : winning days
      - Red bars    : losing days
      - Green dashed: average winning-day return
      - Red dashed  : average losing-day return
      - Text box    : Avg Win / Avg Loss / W/L Ratio / Expected Value per day

    A summary asymmetry table is also printed to stdout.
    """
    names = list(results.keys())
    n = len(names)
    ncols = 4
    nrows = (n + ncols - 1) // ncols          # ceiling division

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.8))
    axes_flat = axes.flatten() if n > 1 else [axes]

    # ---- Console table header ----
    print(f"\n{'─' * 90}")
    print(f"  {'RETURN DISTRIBUTION — ASYMMETRY ANALYSIS':^88}")
    print(f"{'─' * 90}")
    print(
        f"  {'Variant':<22} {'Avg Win%':>8} {'Avg Loss%':>10} {'W/L Ratio':>10} "
        f"{'EV/day%':>8} {'Skew':>6} {'Kurt':>6}"
    )
    print(f"  {'─'*22} {'─'*8} {'─'*10} {'─'*10} {'─'*8} {'─'*6} {'─'*6}")

    for idx, (name, colour) in enumerate(zip(names, colours)):
        ax = axes_flat[idx]
        rets = results[name].returns.dropna() * 100   # convert to percent

        wins   = rets[rets > 0]
        losses = rets[rets < 0]

        avg_win  = wins.mean()   if len(wins)   > 0 else 0.0
        avg_loss = losses.mean() if len(losses) > 0 else 0.0   # negative number
        wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("nan")
        hit      = len(wins) / len(rets) if len(rets) > 0 else 0.0
        ev       = hit * avg_win + (1 - hit) * avg_loss        # expected value per day
        skew     = float(rets.skew())
        kurt     = float(rets.kurt())

        # Histogram — shared bins across wins and losses for fair comparison
        all_min, all_max = rets.min(), rets.max()
        bins = np.linspace(all_min, all_max, 50)

        ax.hist(wins,   bins=bins, color="#27AE60", alpha=0.75, label="Win days",  edgecolor="none")
        ax.hist(losses, bins=bins, color="#E74C3C", alpha=0.75, label="Loss days", edgecolor="none")

        # Avg lines
        ax.axvline(avg_win,  color="#1A7A42", lw=1.6, ls="--", label=f"AvgWin {avg_win:+.2f}%")
        ax.axvline(avg_loss, color="#A93226", lw=1.6, ls="--", label=f"AvgLoss {avg_loss:+.2f}%")
        ax.axvline(0, color="white", lw=0.8, alpha=0.5)

        # Stats text box
        stats_txt = (
            f"W/L ratio : {wl_ratio:.2f}x\n"
            f"EV/day    : {ev:+.3f}%\n"
            f"Skew      : {skew:+.2f}"
        )
        ax.text(
            0.97, 0.97, stats_txt,
            transform=ax.transAxes, fontsize=7.5,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1A1A2E", alpha=0.85, edgecolor="gray"),
            color="white", family="monospace",
        )

        ax.set_title(name.strip(), fontweight="bold", fontsize=9, color="white")
        ax.set_xlabel("Daily Return (%)", fontsize=8, color="#AAAAAA")
        ax.set_ylabel("Days", fontsize=8, color="#AAAAAA")
        ax.tick_params(colors="#AAAAAA", labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
        ax.set_facecolor("#0F0F1A")
        ax.legend(fontsize=6.5, loc="upper left", framealpha=0.5)

        # Console row
        print(
            f"  {name:<22} {avg_win:>+8.3f} {avg_loss:>+10.3f} {wl_ratio:>10.2f} "
            f"{ev:>+8.4f} {skew:>+6.2f} {kurt:>+6.2f}"
        )

    # Hide unused subplots
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.patch.set_facecolor("#0A0A14")
    fig.suptitle(
        "Daily Return Distribution — Win/Loss Asymmetry by Variant",
        fontweight="bold", fontsize=13, color="white", y=1.01,
    )
    plt.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"{'─' * 90}")
    print(f"\n  Distribution chart saved → {save_path}")
    plt.show()


def print_long_short_breakdown(name: str, result, save_chart: bool = True) -> None:
    """
    Print a year-by-year long vs short P&L breakdown and plot cumulative
    long / short equity curves.

    Columns shown per year
    ----------------------
    Long $    : cumulative gross P&L from long positions that year
    Short $   : cumulative gross P&L from short positions that year
    Long %    : long share of total gross P&L (%)
    Short %   : short share of total gross P&L (%)
    L-Hit %   : hit rate on long-only trading days
    S-Hit %   : hit rate on short-only trading days
    """
    daily = result.daily.copy()
    daily["long_pnl"]  = daily["long_pnl"].fillna(0.0)
    daily["short_pnl"] = daily["short_pnl"].fillna(0.0)

    # ---- Full-period summary ----
    total_long  = daily["long_pnl"].sum()
    total_short = daily["short_pnl"].sum()
    total_gross = total_long + total_short
    long_share  = 100 * total_long  / total_gross if total_gross != 0 else float("nan")
    short_share = 100 * total_short / total_gross if total_gross != 0 else float("nan")

    print(f"\n{'═' * 88}")
    print(f"  {name} — Long / Short P&L Breakdown")
    print(f"{'═' * 88}")
    print(f"  Full period:  Long ${total_long:>12,.0f}  ({long_share:.1f}%)   "
          f"Short ${total_short:>12,.0f}  ({short_share:.1f}%)")
    print(f"{'─' * 88}")
    print(
        f"  {'Year':<6} {'Long $':>10} {'Short $':>10} {'Long%':>7} {'Short%':>7} "
        f"{'L-Hit%':>7} {'S-Hit%':>7}"
    )
    print(f"  {'─'*6} {'─'*10} {'─'*10} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")

    for year, grp in daily.groupby(daily.index.year):
        y_long  = grp["long_pnl"].sum()
        y_short = grp["short_pnl"].sum()
        y_gross = y_long + y_short
        y_long_pct  = 100 * y_long  / y_gross if y_gross != 0 else float("nan")
        y_short_pct = 100 * y_short / y_gross if y_gross != 0 else float("nan")

        # Hit rates: days where that side traded AND was profitable
        long_days  = grp[grp["long_pnl"]  != 0]
        short_days = grp[grp["short_pnl"] != 0]
        l_hit = 100 * (long_days["long_pnl"]   > 0).sum() / len(long_days)  if len(long_days)  > 0 else float("nan")
        s_hit = 100 * (short_days["short_pnl"] > 0).sum() / len(short_days) if len(short_days) > 0 else float("nan")

        l_hit_str = f"{l_hit:>7.1f}" if not np.isnan(l_hit) else f"{'—':>7}"
        s_hit_str = f"{s_hit:>7.1f}" if not np.isnan(s_hit) else f"{'—':>7}"

        print(
            f"  {year:<6} {y_long:>10,.0f} {y_short:>10,.0f} "
            f"{y_long_pct:>7.1f} {y_short_pct:>7.1f} {l_hit_str} {s_hit_str}"
        )

    print(f"{'═' * 88}")

    if not save_chart:
        return

    # ---- Chart: cumulative long vs short vs combined ----
    cum_long  = daily["long_pnl"].cumsum()
    cum_short = daily["short_pnl"].cumsum()
    cum_total = (daily["long_pnl"] + daily["short_pnl"]).cumsum()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(cum_long.index,  cum_long.values,  label="Long cumulative P&L",  color="#27AE60", lw=2)
    ax1.plot(cum_short.index, cum_short.values, label="Short cumulative P&L", color="#E74C3C", lw=2)
    ax1.plot(cum_total.index, cum_total.values, label="Combined gross P&L",   color="#F39C12", lw=1.5, ls="--")
    ax1.axhline(0, color="white", lw=0.6, alpha=0.4)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax1.grid(True, ls="--", alpha=0.25)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(fontsize=10)
    ax1.set_title(
        f"{name} — Cumulative Gross P&L: Long vs Short\n"
        f"Long total \\${total_long:,.0f} ({long_share:.1f}%)   "
        f"Short total \\${total_short:,.0f} ({short_share:.1f}%)",
        fontweight="bold", fontsize=12,
    )

    # Bar chart: annual long vs short gross P&L
    years       = sorted(daily.index.year.unique())
    long_annual  = [daily[daily.index.year == y]["long_pnl"].sum()  for y in years]
    short_annual = [daily[daily.index.year == y]["short_pnl"].sum() for y in years]
    x = np.arange(len(years))
    w = 0.38
    ax2.bar(x - w/2, long_annual,  width=w, label="Long",  color="#27AE60", alpha=0.85)
    ax2.bar(x + w/2, short_annual, width=w, label="Short", color="#E74C3C", alpha=0.85)
    ax2.axhline(0, color="white", lw=0.6, alpha=0.4)
    ax2.set_xticks(x)
    ax2.set_xticklabels(years, fontsize=8)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax2.set_title("Annual Gross P&L by Side", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(True, ls="--", alpha=0.2, axis="y")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = PLOT_DIR / f"long_short_{name.lower().replace(' ', '_')}.png"
    plt.savefig(save_path, dpi=150)
    print(f"  Chart saved → {save_path}")
    plt.show()


def _walkforward_windows(
    all_days: list,
    window_years: int = 2,
) -> list[tuple[list, list]]:
    """
    Build non-overlapping out-of-sample windows of `window_years` each.

    The dataset is divided into sequential blocks of equal size (in calendar
    years).  Every block is a fully independent OOS period — there is no
    separate training phase because V6 has no parameters to optimise.

    Example with 10 years of data (2016–2025) and window_years=2:
      Block 1 OOS: 2016–2017
      Block 2 OOS: 2018–2019
      Block 3 OOS: 2020–2021
      Block 4 OOS: 2022–2023
      Block 5 OOS: 2024–2025

    Parameters
    ----------
    all_days     : sorted list of datetime.date objects covering the full dataset
    window_years : size of each OOS window in calendar years (default 2)

    Returns
    -------
    List of (anchor_day, test_days) tuples where anchor_day is the last trading
    day *before* the window (needed for prev_close on day 1 of each window) and
    test_days is the list of trading dates inside the window.
    """
    if not all_days:
        return []

    start = pd.Timestamp(all_days[0])
    end   = pd.Timestamp(all_days[-1])
    days_ts = pd.to_datetime(all_days)

    windows = []
    window_start = start
    while window_start < end:
        window_end = window_start + pd.DateOffset(years=window_years)
        if window_end > end:
            window_end = end + pd.Timedelta(days=1)   # include the last partial block

        test_mask = (days_ts >= window_start) & (days_ts < window_end)
        test_days = [d for d, m in zip(all_days, test_mask) if m]
        if not test_days:
            break

        # Anchor = last trading day that comes strictly before the window
        prior_days = [d for d in all_days if pd.Timestamp(d) < window_start]
        anchor_day = prior_days[-1] if prior_days else test_days[0]

        windows.append((anchor_day, test_days))
        window_start = window_end

    return windows


def run_walkforward(
    df_enh: pd.DataFrame,
    df_daily_enh: pd.DataFrame,
    raw: pd.DataFrame,
    cfg: "StrategyConfig",
    window_years: int = 2,
) -> None:
    """
    Walk-forward validation for V6 (Full Enhanced strategy).

    Divides the preprocessed dataset into sequential non-overlapping blocks
    of `window_years` each.  V6 is run independently on every block, using
    only the data in that block (no training / parameter optimisation needed
    because all features are already lag-safe).  OOS returns from every block
    are concatenated and treated as a single combined out-of-sample track record.

    Outputs
    -------
    - Console table: one row per window + combined OOS row
    - Chart: combined OOS equity curve vs SPY, saved to
             analysis/plots/walkforward.png
    """
    all_days = sorted(df_enh["day"].unique())
    windows  = _walkforward_windows(all_days, window_years=window_years)

    if not windows:
        print("  Not enough data for walk-forward windows.")
        return

    # Pre-group raw extended bars by date for fast lookup
    raw_copy = raw.copy()
    if "day" not in raw_copy.columns:
        raw_copy["day"] = raw_copy.index.date
    raw_copy["day"] = pd.to_datetime(raw_copy["day"]).dt.date
    raw_groups = raw_copy.groupby("day")

    print(f"\n{'═' * 82}")
    print(f"  {'WALK-FORWARD VALIDATION  (V6 Full Enhanced — {window_years}-year OOS windows)':^80}".format(window_years=window_years))
    print(f"{'═' * 82}")
    print(
        f"  {'Window':<18} {'Ann Ret%':>9} {'Sharpe':>7} {'MaxDD%':>8} "
        f"{'Hit%':>7} {'N Days':>7} {'Avg Lev':>8}"
    )
    print(f"  {'─'*18} {'─'*9} {'─'*7} {'─'*8} {'─'*7} {'─'*7} {'─'*8}")

    all_oos_rets: list[pd.Series] = []
    all_oos_spy:  list[pd.Series] = []

    for anchor_day, test_days in windows:
        test_set   = set(test_days)
        window_set = test_set | {anchor_day}

        # Slice preprocessed data to this window (+ anchor day for prev_close)
        df_w  = df_enh[df_enh["day"].isin(window_set)]
        dd_w  = df_daily_enh[
            df_daily_enh.index.isin(pd.to_datetime(list(window_set)))
        ]

        # Raw extended bars for the same window
        raw_day_set = test_set | {anchor_day}
        raw_w_days  = [d for d in raw_copy["day"].unique() if d in raw_day_set]
        raw_w       = raw_copy[raw_copy["day"].isin(raw_w_days)]

        result = EnhancedBacktester(cfg).run(df_w, dd_w, raw_w)

        # Keep only the OOS test period (drop anchor day)
        test_start = pd.Timestamp(test_days[0])
        oos_rets   = result.returns[result.returns.index >= test_start]
        oos_spy    = result.daily["ret_spy"][result.daily.index >= test_start]

        if oos_rets.empty:
            continue

        all_oos_rets.append(oos_rets)
        all_oos_spy.append(oos_spy)

        m       = PerformanceMetrics.compute(oos_rets, oos_spy)
        lev_col = result.daily["avg_lev"][result.daily.index >= test_start].replace(0.0, float("nan"))
        avg_lev = lev_col.mean()
        lev_str = f"{avg_lev:.2f}x" if not np.isnan(avg_lev) else "—"

        year_start = pd.Timestamp(test_days[0]).year
        year_end   = pd.Timestamp(test_days[-1]).year
        label      = f"{year_start}–{year_end}" if year_start != year_end else str(year_start)

        print(
            f"  {label:<18} {m.annualized_return:>9.1f} {m.sharpe_ratio:>7.2f} "
            f"{m.max_drawdown:>8.1f} {m.hit_ratio:>7.1f} {len(oos_rets):>7} {lev_str:>8}"
        )

    if not all_oos_rets:
        print("  No OOS results to aggregate.")
        return

    # ---- Combined OOS row ----
    combined_rets = pd.concat(all_oos_rets).sort_index()
    combined_spy  = pd.concat(all_oos_spy).sort_index()
    cm = PerformanceMetrics.compute(combined_rets, combined_spy)
    all_lev = pd.concat(
        [r.replace(0.0, float("nan")) for r in all_oos_rets]   # reuse oos_rets as proxy
    )
    # Recalculate combined avg_lev from results (collect separately)
    print(f"  {'─'*18} {'─'*9} {'─'*7} {'─'*8} {'─'*7} {'─'*7} {'─'*8}")
    print(
        f"  {'Combined OOS':<18} {cm.annualized_return:>9.1f} {cm.sharpe_ratio:>7.2f} "
        f"{cm.max_drawdown:>8.1f} {cm.hit_ratio:>7.1f} {len(combined_rets):>7} {'':>8}"
    )
    print(f"{'═' * 82}")

    # ---- Chart ----
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 7))

    aum_0 = cfg.aum_0
    oos_aum = aum_0 * (1 + combined_rets).cumprod()
    spy_aum = aum_0 * (1 + combined_spy.reindex(combined_rets.index).fillna(0)).cumprod()

    ax.plot(oos_aum.index, oos_aum.values,
            label="V6 Enhanced — Combined OOS", color="#27AE60", lw=2.2)
    ax.plot(spy_aum.index, spy_aum.values,
            label="SPY Buy & Hold", color="#E74C3C", lw=1.8, ls="--")

    # Shade each OOS window alternately
    shade_colours = ["#1A3A2A", "#0D1F33"]
    for idx, (_, test_days) in enumerate(windows):
        x0 = pd.Timestamp(test_days[0])
        x1 = pd.Timestamp(test_days[-1])
        ax.axvspan(x0, x1, alpha=0.18, color=shade_colours[idx % 2], lw=0)

    # Annotate each window with its Sharpe
    for _, test_days in windows:
        ts  = pd.Timestamp(test_days[0])
        te  = pd.Timestamp(test_days[-1])
        mid = ts + (te - ts) / 2
        w_rets = combined_rets[(combined_rets.index >= ts) & (combined_rets.index <= te)]
        w_spy  = combined_spy[(combined_spy.index >= ts) & (combined_spy.index <= te)]
        if len(w_rets) > 10:
            wm = PerformanceMetrics.compute(w_rets, w_spy)
            y_pos = oos_aum[oos_aum.index <= te].iloc[-1] * 1.02
            ax.text(mid, y_pos, f"SR {wm.sharpe_ratio:.2f}",
                    ha="center", va="bottom", fontsize=8, color="#AAAAAA")

    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45, ha="right")
    ax.grid(True, ls="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        f"Walk-Forward Validation — V6 Enhanced  ({window_years}-year OOS windows)\n"
        f"Combined OOS: Ann Ret {cm.annualized_return:.1f}%  |  Sharpe {cm.sharpe_ratio:.2f}"
        f"  |  MaxDD {cm.max_drawdown:.1f}%",
        fontweight="bold", fontsize=12,
    )
    ax.legend(fontsize=10)
    plt.tight_layout()
    save_path = PLOT_DIR / "walkforward.png"
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

    if args.longshort:
        print_long_short_breakdown("Baseline", base_result)
        print_long_short_breakdown("Enhanced (V6)", enh_result)

    if args.walkforward:
        run_walkforward(df_enh, df_daily_enh, raw, enh_strategy_cfg, window_years=2)

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
