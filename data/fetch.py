"""
Market data fetcher backed by the Alpaca API.

Provides:
  AlpacaFetcher.get_historical_bars()  — historical 1-minute OHLCV
  AlpacaFetcher.stream_bars()          — real-time WebSocket stream
  AlpacaFetcher.from_csv()             — CSV fallback (offline / backtesting)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from config.config import AlpacaConfig

logger = logging.getLogger(__name__)

_MARKET_OPEN = "09:30"
_MARKET_CLOSE = "15:59"


class AlpacaFetcher:
    def __init__(self, config: AlpacaConfig | None = None) -> None:
        self.config = config or AlpacaConfig()

    # ------------------------------------------------------------------
    # Historical data
    # ------------------------------------------------------------------

    def get_historical_bars(
        self,
        symbol: str,
        start: str | datetime,
        end: str | datetime,
    ) -> pd.DataFrame:
        """
        Fetch 1-minute OHLCV bars from Alpaca.

        Returns a DataFrame with columns: open, high, low, close, volume, day
        indexed by caldt (tz-naive Eastern Time, market hours only).
        """
        from alpaca.data.historical.stock import (
            StockBarsRequest,
            StockHistoricalDataClient,
        )
        from alpaca.data.timeframe import TimeFrame

        client = StockHistoricalDataClient(
            self.config.api_key, self.config.secret_key
        )

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=_to_utc(start),
            end=_to_utc(end),
            adjustment="raw",
        )

        bars = client.get_stock_bars(request)
        df = bars.df

        if df.empty:
            logger.warning("Alpaca returned no data for %s %s–%s", symbol, start, end)
            return pd.DataFrame()

        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index(level=0, drop=True)

        df.index = df.index.tz_convert("America/New_York").tz_localize(None)
        df.index.name = "caldt"

        df = df[["open", "high", "low", "close", "volume"]]
        df = df.between_time(_MARKET_OPEN, _MARKET_CLOSE)
        df["day"] = df.index.date

        return df

    # ------------------------------------------------------------------
    # CSV fallback
    # ------------------------------------------------------------------

    @staticmethod
    def from_csv(path: str | Path) -> pd.DataFrame:
        """Load historical data from a local CSV file."""
        df = pd.read_csv(path)

        if "caldt" in df.columns:
            df["caldt"] = pd.to_datetime(df["caldt"])
            df = df.set_index("caldt")
        else:
            df.index = pd.to_datetime(df.index)
            df.index.name = "caldt"

        if "day" not in df.columns:
            df["day"] = df.index.date

        df["day"] = pd.to_datetime(df["day"]).dt.date
        return df

    # ------------------------------------------------------------------
    # Real-time WebSocket streaming
    # ------------------------------------------------------------------

    def stream_bars(
        self,
        symbol: str,
        callback: Callable[[dict], None],
    ) -> None:
        """
        Subscribe to real-time 1-minute bars via Alpaca WebSocket.

        Calls callback(bar_dict) on each new bar.  Blocking — run in a
        thread or via asyncio for non-blocking use.
        """
        from alpaca.data.live import StockDataStream

        stream = StockDataStream(self.config.api_key, self.config.secret_key)

        async def _handler(bar):
            callback({
                "symbol": bar.symbol,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "timestamp": bar.timestamp,
            })

        stream.subscribe_bars(_handler, symbol)
        logger.info("Starting Alpaca WebSocket stream for %s ...", symbol)
        stream.run()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _to_utc(dt: str | datetime) -> datetime:
    if isinstance(dt, str):
        dt = pd.Timestamp(dt).to_pydatetime()
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
