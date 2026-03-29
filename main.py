"""
Live trading entry point.

Connects to Alpaca via WebSocket, generates signals on each incoming
1-minute bar, and submits orders when the desired exposure changes.

Usage
-----
    python main.py                          # uses config/config.yaml
    python main.py --config config/config.yaml

Safety
------
  Defaults to paper trading (config.alpaca.paper = true).
  Closes all positions at 15:59 ET regardless of signal.
"""

from __future__ import annotations

import argparse
import logging
import math
import signal
import sys
from datetime import datetime, time
from pathlib import Path

import pandas as pd

from config.config import AppConfig
from data.fetch import AlpacaFetcher
from data.preprocess import Preprocessor
from trader.signals import SignalGenerator
from trader.sizing import PositionSizer
from trader.trading_model import AlpacaTrader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("live_trader")


class LiveTrader:
    def __init__(self, app_cfg: AppConfig) -> None:
        self.cfg = app_cfg
        self.strategy_cfg = app_cfg.strategy
        self.symbol = self.strategy_cfg.symbol

        self.fetcher = AlpacaFetcher(app_cfg.alpaca)
        self.trader = AlpacaTrader(app_cfg.alpaca)
        self.preprocessor = Preprocessor(self.strategy_cfg)
        self.signal_gen = SignalGenerator(self.strategy_cfg)
        self.sizer = PositionSizer(self.strategy_cfg)

        self._buffer: list[dict] = []
        self._last_exposure: float = 0.0

        account = self.trader.get_account()
        logger.info(
            "Account ready: equity=$%.2f  buying_power=$%.2f  paper=%s",
            account["equity"], account["buying_power"], account["paper"],
        )

    def run(self) -> None:
        logger.info("Starting live trader for %s ...", self.symbol)
        self.fetcher.stream_bars(self.symbol, self.on_bar)

    def on_bar(self, bar: dict) -> None:
        ts: datetime = bar["timestamp"]
        if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
            ts = ts.astimezone(tz=None).replace(tzinfo=None)

        et_time = ts.time()

        if not (time(9, 30) <= et_time <= time(15, 59)):
            return

        # EOD: close all
        if et_time >= time(15, 59):
            if self._last_exposure != 0:
                logger.info("EOD — closing all positions")
                self.trader.close_all()
                self._last_exposure = 0.0
            return

        self._buffer.append({
            "open": bar["open"], "high": bar["high"],
            "low": bar["low"], "close": bar["close"],
            "volume": bar["volume"], "caldt": ts,
        })

        if len(self._buffer) < 30:
            logger.debug("Warming up (%d bars)", len(self._buffer))
            return

        df_buf = pd.DataFrame(self._buffer)
        df_buf["caldt"] = pd.to_datetime(df_buf["caldt"])
        df_buf = df_buf.set_index("caldt")
        df_buf["day"] = df_buf.index.date

        try:
            df_feat, df_daily = self.preprocessor.transform(df_buf)
        except Exception as e:
            logger.warning("Preprocessing failed: %s", e)
            return

        today = df_feat["day"].iloc[-1]
        today_df = df_feat[df_feat["day"] == today]

        if today_df["sigma_open"].isna().all() or len(today_df) < 2:
            return

        prev_days = df_feat[df_feat["day"] < today]["day"].unique()
        if len(prev_days) == 0:
            return

        prev_close = df_feat[df_feat["day"] == prev_days[-1]]["close"].iloc[-1]
        exposure_series = self.signal_gen.generate(today_df, prev_close)
        desired_exposure = float(exposure_series.iloc[-1])

        if desired_exposure == self._last_exposure:
            return

        equity = self.trader.get_account()["equity"]
        open_px = float(today_df["close"].iloc[0])
        daily_vol_val = today_df["spy_dvol"].iloc[0]
        daily_vol = float(daily_vol_val) if (
            daily_vol_val is not None
            and not (isinstance(daily_vol_val, float) and math.isnan(daily_vol_val))
        ) else None

        shares = self.sizer.shares(equity, open_px, daily_vol)
        target_qty = int(desired_exposure * shares)

        logger.info(
            "Signal: %.0f → %.0f  |  target=%d shares",
            self._last_exposure, desired_exposure, target_qty,
        )
        self.trader.rebalance(self.symbol, target_qty)
        self._last_exposure = desired_exposure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live Intraday Momentum Trader")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error(
            "config not found at %s. Copy config/config.yaml and fill in your keys.",
            cfg_path,
        )
        sys.exit(1)

    app_cfg = AppConfig.from_yaml(cfg_path)
    live = LiveTrader(app_cfg)

    def _shutdown(sig, frame):
        logger.info("Shutdown signal — closing all positions...")
        live.trader.close_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    live.run()


if __name__ == "__main__":
    main()
