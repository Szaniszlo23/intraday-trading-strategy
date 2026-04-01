"""
Market data fetcher backed by the Alpaca API.

Public interface
----------------
  AlpacaFetcher.get_historical_bars() — 1-min OHLCV over a date range,
                                        includes pre-market by default
  AlpacaFetcher.stream_bars()         — real-time WebSocket (non-blocking)
  AlpacaFetcher.stop_stream()         — graceful WebSocket shutdown
  AlpacaFetcher.from_csv()            — CSV fallback for offline backtesting

Design notes
------------
- The REST client is created once in __init__ and reused across all calls.
- get_historical_bars() does NOT filter to market hours by default.
  Pass include_extended=False or call df.between_time("09:30", "16:00") yourself.
- All timestamps are returned as tz-naive Eastern Time for consistent
  comparison with the strategy's time-of-day constants.
- stream_bars() is non-blocking: it starts the WebSocket in a background
  thread and returns immediately. The callback receives a pd.Series with
  the same column names as the historical DataFrame.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import AlpacaConfig

logger = logging.getLogger(__name__)

# Guaranteed column order for every returned DataFrame
_COLUMNS = ["open", "high", "low", "close", "volume"]

# Session boundaries used by include_extended=False filter
_SESSION_OPEN  = "09:30"
_SESSION_CLOSE = "16:00"

# Pre-market window (used by callers that need pre-market bars)
_PREMARKET_OPEN  = "04:00"
_PREMARKET_CLOSE = "09:29"


class AlpacaFetcher:
    """Single entry point for all Alpaca market data."""

    def __init__(self, config: AlpacaConfig | None = None) -> None:
        self.config = config or AlpacaConfig()
        self._stream        = None   # StockDataStream instance (live only)
        self._stream_thread = None   # background thread running the stream
        self._hist_client   = None   # REST client — created once, reused

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_hist_client(self):
        """Return the REST client, creating it on first call."""
        if self._hist_client is None:
            from alpaca.data.historical.stock import StockHistoricalDataClient
            self._hist_client = StockHistoricalDataClient(
                self.config.api_key,
                self.config.secret_key,
            )
        return self._hist_client

    @staticmethod
    def _normalise(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Shared post-processing for all bar DataFrames:
          - flatten MultiIndex if present
          - convert index to tz-naive Eastern Time
          - keep only the five OHLCV columns
          - add a 'day' column (datetime.date) for grouping
        """
        if df.empty:
            return pd.DataFrame(columns=_COLUMNS)

        # Flatten the symbol level that Alpaca adds to the index
        if isinstance(df.index, pd.MultiIndex):
            df = df.loc[symbol] if symbol in df.index.get_level_values(0) \
                 else df.reset_index(level=0, drop=True)

        # Convert to Eastern Time and strip tzinfo for naive comparisons
        if df.index.tz is not None:
            df.index = df.index.tz_convert("America/New_York").tz_localize(None)
        df.index.name = "caldt"

        df = df[_COLUMNS].copy()
        df["day"] = df.index.normalize().date  # type: ignore[attr-defined]

        return df

    # ------------------------------------------------------------------
    # Historical bars — 1-minute resolution
    # ------------------------------------------------------------------

    def get_historical_bars(
        self,
        symbol: str,
        start: str | datetime,
        end: str | datetime,
        include_extended: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch 1-minute OHLCV bars from Alpaca for the given date range.

        Parameters
        ----------
        symbol           : ticker, e.g. "SPY"
        start, end       : date range — strings like "2019-01-01" or datetime objects
        include_extended : True (default) returns all sessions including pre-market
                           (04:00–09:29). Pass False to get 09:30–16:00 only.

        Returns
        -------
        pd.DataFrame
            Columns: open, high, low, close, volume, day
            Index  : caldt (tz-naive Eastern, 1-min frequency)
        """
        from alpaca.data.historical.stock import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        client = self._get_hist_client()

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=_to_utc(start),
            end=_to_utc(end),
            adjustment="raw",
            feed="iex",   # IEX feed — available on the free Alpaca tier
        )

        logger.info(
            "Fetching 1-min bars: %s  %s → %s  extended=%s",
            symbol, start, end, include_extended,
        )

        bars = client.get_stock_bars(request)
        df   = bars.df

        if df.empty:
            logger.warning(
                "Alpaca returned no data for %s %s–%s", symbol, start, end
            )
            return pd.DataFrame(columns=_COLUMNS)

        df = self._normalise(df, symbol)

        if not include_extended:
            df = df.between_time(_SESSION_OPEN, _SESSION_CLOSE)

        return df

    # ------------------------------------------------------------------
    # Real-time WebSocket streaming
    # ------------------------------------------------------------------

    def stream_bars(
        self,
        symbol: str,
        callback: Callable[[pd.Series], None],
    ) -> None:
        """
        Subscribe to real-time 1-minute bars via Alpaca WebSocket.

        Non-blocking: launches the stream in a background daemon thread.
        Call stop_stream() to shut down gracefully.

        The callback receives a pd.Series with fields:
            open, high, low, close, volume, symbol, caldt
        (same column names as the historical DataFrame rows)
        """
        from alpaca.data.live import StockDataStream

        self._stream = StockDataStream(
            self.config.api_key,
            self.config.secret_key,
        )

        async def _handler(bar):
            # Convert Alpaca bar object to a pd.Series matching the historical format
            ts = pd.Timestamp(bar.timestamp)
            if ts.tz is not None:
                ts = ts.tz_convert("America/New_York").tz_localize(None)

            series = pd.Series({
                "open":   bar.open,
                "high":   bar.high,
                "low":    bar.low,
                "close":  bar.close,
                "volume": bar.volume,
                "symbol": bar.symbol,
                "caldt":  ts,
            }, name=ts)

            callback(series)

        self._stream.subscribe_bars(_handler, symbol)

        logger.info("Starting Alpaca WebSocket stream for %s ...", symbol)

        self._stream_thread = threading.Thread(
            target=self._stream.run,
            daemon=True,
            name=f"alpaca-stream-{symbol}",
        )
        self._stream_thread.start()

    def stop_stream(self) -> None:
        """
        Gracefully stop the WebSocket stream.

        Call at EOD or on KeyboardInterrupt to cleanly shut down the
        background thread before exiting.
        """
        if self._stream is not None:
            try:
                self._stream.stop()
                logger.info("Alpaca WebSocket stream stopped.")
            except Exception as exc:
                logger.warning("Error stopping stream: %s", exc)
            finally:
                self._stream = None

        if self._stream_thread is not None:
            self._stream_thread.join(timeout=5)
            self._stream_thread = None

    # ------------------------------------------------------------------
    # CSV fallback — offline backtesting
    # ------------------------------------------------------------------

    @staticmethod
    def from_csv(path: str | Path) -> pd.DataFrame:
        """
        Load historical data from a local CSV file.

        The CSV should have been saved from get_historical_bars() output.
        Useful for fully offline backtesting without any API calls.
        """
        df = pd.read_csv(path)

        if "caldt" in df.columns:
            df["caldt"] = pd.to_datetime(df["caldt"])
            df = df.set_index("caldt")
        else:
            df.index = pd.to_datetime(df.index)
            df.index.name = "caldt"

        if "day" not in df.columns:
            df["day"] = df.index.normalize().date  # type: ignore[attr-defined]

        df["day"] = pd.to_datetime(df["day"]).dt.date
        return df


# ------------------------------------------------------------------
# Module-level helper
# ------------------------------------------------------------------

def _to_utc(dt: str | datetime) -> datetime:
    """Coerce any date string or datetime to a UTC-aware datetime."""
    if isinstance(dt, str):
        dt = pd.Timestamp(dt).to_pydatetime()
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
