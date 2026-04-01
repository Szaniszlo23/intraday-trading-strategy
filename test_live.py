"""
Live trader replay test.

Feeds yesterday's 1-minute bars through LiveTraderV6.on_bar() one by one
and prints every signal change.  No orders are submitted — AlpacaTrader is
monkey-patched to log instead of trade.

Usage
-----
    python test_live.py                   # replay most recent trading day
    python test_live.py --date 2025-03-20 # replay a specific date
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta

import pandas as pd

from config.config import AppConfig
from main import LiveTraderV6

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("replay_test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live trader replay test")
    parser.add_argument("--date",   type=str, default=None,
                        help="Date to replay (YYYY-MM-DD). Defaults to most recent trading day.")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    return parser.parse_args()


def main() -> None:
    args    = parse_args()
    app_cfg = AppConfig.load(args.config)

    # ── Patch out real order execution ───────────────────────────────────────
    orders_log: list[dict] = []

    def fake_rebalance(symbol: str, target_qty: int) -> None:
        orders_log.append({"target_qty": target_qty, "time": datetime.now()})
        logger.info("  [DRY-RUN] rebalance(%s, %d) — no real order sent",
                    symbol, target_qty)

    def fake_close_all() -> list:
        orders_log.append({"target_qty": 0, "time": datetime.now()})
        logger.info("  [DRY-RUN] close_all() — no real order sent")
        return []

    def fake_get_account() -> dict:
        return {"equity": app_cfg.strategy.aum_0,
                "buying_power": app_cfg.strategy.aum_0,
                "cash": app_cfg.strategy.aum_0,
                "portfolio_value": app_cfg.strategy.aum_0,
                "paper": True}

    def fake_get_position(symbol: str) -> int:
        return 0

    # ── Bootstrap ─────────────────────────────────────────────────────────────
    logger.info("Bootstrapping V6 live trader ...")
    live = LiveTraderV6(app_cfg)

    # Patch AFTER bootstrap (bootstrap uses real account for logging)
    live.trader.rebalance    = fake_rebalance
    live.trader.close_all    = fake_close_all
    live.trader.get_account  = fake_get_account
    live.trader.get_position = fake_get_position

    # ── Pick replay date ──────────────────────────────────────────────────────
    all_days = sorted(live._hist_df["day"].unique())
    if args.date:
        replay_day = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        replay_day = all_days[-1]   # most recent day in history

    logger.info("Replaying session: %s", replay_day)

    # ── Fetch the replay day's bars ───────────────────────────────────────────
    replay_bars = live._hist_df[live._hist_df["day"] == replay_day].copy()
    if replay_bars.empty:
        logger.error("No bars found for %s — try a different date.", replay_day)
        return

    # Reset live state so it treats replay_day as a fresh session
    live._today_bars    = []
    live._last_exposure = 0.0
    live._session_ready = False

    # prev_close for replay = close of the day before replay_day
    earlier_days = [d for d in all_days if d < replay_day]
    if earlier_days:
        prev_df          = live._hist_df[live._hist_df["day"] == earlier_days[-1]]
        live._prev_close = float(prev_df["close"].iloc[-1])
    logger.info("prev_close for replay: %.2f", live._prev_close or 0.0)

    # ── Feed bars ─────────────────────────────────────────────────────────────
    for ts, row in replay_bars.iterrows():
        bar = {
            "timestamp": ts,
            "open":      row["open"],
            "high":      row["high"],
            "low":       row["low"],
            "close":     row["close"],
            "volume":    row["volume"],
        }
        live.on_bar(bar)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"  Replay complete — {replay_day}")
    print(f"  Bars processed : {len(replay_bars)}")
    print(f"  Orders fired   : {len(orders_log)}")
    if orders_log:
        print(f"  Order log:")
        for o in orders_log:
            direction = "LONG" if o["target_qty"] > 0 else ("SHORT" if o["target_qty"] < 0 else "FLAT")
            print(f"    {o['time'].strftime('%H:%M:%S')}  {direction:5s}  qty={o['target_qty']}")
    print(f"{'─' * 60}\n")


if __name__ == "__main__":
    main()
