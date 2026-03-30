"""
Market data fetcher backed by the Alpaca API.

Provides:
  AlpacaFetcher.get_historical_bars()  — historical 1-min OHLCV, full day
                                         including pre-market (04:00–16:00)
  AlpacaFetcher.get_daily_bars()       — daily OHLCV for vol scaler
  AlpacaFetcher.get_premarket_bars()   — 04:00–09:29 bars for flow composite
  AlpacaFetcher.stream_bars()          — real-time WebSocket, non-blocking
  AlpacaFetcher.stop_stream()          — graceful shutdown
  AlpacaFetcher.from_csv()             — CSV fallback (offline / backtesting)

Design notes
------------
- The historical client is instantiated once in __init__ and reused across
  all REST calls — not re-created on every fetch.
- get_historical_bars() does NOT filter to market hours. The caller is
  responsible for filtering. This keeps pre-market bars available for the
  flow composite and backtest pre-market derivation.
- All timestamps are returned as tz-naive Eastern Time for consistency
  with the strategy's time-of-day comparisons.
- stream_bars() is non-blocking. It launches the WebSocket in a background
  thread and returns immediately. Call stop_stream() to shut down.
- The WebSocket callback receives a pd.Series with the same column names
  as the historical DataFrame — no format mismatch with the strategy core.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

import sys
from pathlib import Path

# Make project root importable regardless of how this file is invoked
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import AlpacaConfig

logger = logging.getLogger(__name__)

# Column order guaranteed by every method
_COLUMNS = ["open", "high", "low", "close", "volume"]

# Market session boundaries — inclusive
_SESSION_OPEN  = "09:30"
_SESSION_CLOSE = "16:00"   # FIX: was 15:59, missed the EOD force-close bar

# Pre-market window
_PREMARKET_OPEN  = "04:00"
_PREMARKET_CLOSE = "09:29"


class AlpacaFetcher:
    """
    Single entry point for all Alpaca market data.

    Parameters
    ----------
    config : AlpacaConfig, optional
        API credentials and settings. Reads from environment if not provided.
    """

    def __init__(self, config: AlpacaConfig | None = None) -> None:
        self.config = config or AlpacaConfig()
        self._stream        = None          # StockDataStream instance
        self._stream_thread = None          # background thread for stream
        self._hist_client   = None          # lazily initialised REST client

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_hist_client(self):
        """
        Return the historical data client, creating it once if needed.
        FIX: was instantiated inside the method on every call — now cached.
        """
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
          - add a 'day' column (datetime.date) for easy grouping
        """
        if df.empty:
            return pd.DataFrame(columns=_COLUMNS)

        # Flatten symbol level from MultiIndex
        if isinstance(df.index, pd.MultiIndex):
            df = df.loc[symbol] if symbol in df.index.get_level_values(0) \
                 else df.reset_index(level=0, drop=True)

        # tz-aware → Eastern → strip tz so comparisons stay simple
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

        FIX 1: Does NOT filter to market hours — all sessions included
                by default so pre-market bars are available for:
                  - flow composite (pre-market trend factor)
                  - backtest pre-market derivation
                Callers that want regular-session-only data should call
                  df = df.between_time("09:30", "16:00")
                themselves.

        FIX 2: MARKET_CLOSE corrected to 16:00 (was 15:59).

        Parameters
        ----------
        symbol : str
            e.g. "SPY"
        start, end : str | datetime
            Date range. Strings like "2019-01-01" are accepted.
        include_extended : bool
            True  → includes pre-market (04:00–09:29) and post-market bars.
            False → regular session only (09:30–16:00).
            Default True for backtest and noise area lookback fetches.

        Returns
        -------
        pd.DataFrame
            Columns: open, high, low, close, volume, day
            Index: caldt (tz-naive Eastern, 1-min frequency)
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

        # Optionally restrict to regular session
        if not include_extended:
            df = df.between_time(_SESSION_OPEN, _SESSION_CLOSE)

        return df

    # ------------------------------------------------------------------
    # Daily bars — for vol scaler
    # ------------------------------------------------------------------

    def get_daily_bars(
        self,
        symbol: str,
        lookback_days: int = 60,
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV bars for the volatility scaler.

        FIX 5: Was completely missing — vol scaler had no data source.

        Returns the last `lookback_days` complete trading days, with
        a tz-naive date index (not datetime).

        Parameters
        ----------
        symbol : str
            e.g. "SPY"
        lookback_days : int
            Number of trading days to fetch (default 60 for safety margin).

        Returns
        -------
        pd.DataFrame
            Columns: open, high, low, close, volume
            Index: date (tz-naive, daily frequency)
        """
        from alpaca.data.historical.stock import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from datetime import timedelta

        client = self._get_hist_client()

        end   = datetime.now(tz=timezone.utc)
        # Buffer ×2 to account for weekends and holidays
        start = end - timedelta(days=lookback_days * 2)

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            adjustment="raw",
        )

        logger.info("Fetching daily bars: %s  last %d days", symbol, lookback_days)

        bars = client.get_stock_bars(request)
        df   = bars.df

        if df.empty:
            logger.warning("Alpaca returned no daily data for %s", symbol)
            return pd.DataFrame(columns=_COLUMNS)

        # Flatten MultiIndex if present
        if isinstance(df.index, pd.MultiIndex):
            df = df.loc[symbol] if symbol in df.index.get_level_values(0) \
                 else df.reset_index(level=0, drop=True)

        if df.index.tz is not None:
            df.index = df.index.tz_convert("America/New_York").tz_localize(None)

        df.index.name = "date"
        df = df[_COLUMNS].copy()

        # Return only the requested number of days
        return df.tail(lookback_days)

    # ------------------------------------------------------------------
    # Pre-market bars — for flow composite
    # ------------------------------------------------------------------

    def get_premarket_bars(
        self,
        symbol: str,
        trading_date: datetime | None = None,
    ) -> pd.DataFrame:
        """
        Fetch pre-market bars (04:00–09:29) for a given trading date.

        FIX: Was completely missing — flow composite had no data source.

        Called at 09:29 each morning before the session open.
        Filters the full day's extended-hours bars to the pre-market window.

        Parameters
        ----------
        symbol : str
            e.g. "SPY"
        trading_date : datetime, optional
            The session date. Defaults to today.

        Returns
        -------
        pd.DataFrame
            Subset of 1-min bars between 04:00 and 09:29 Eastern.
            Columns: open, high, low, close, volume, day
        """
        from datetime import date, timedelta

        if trading_date is None:
            trading_date = datetime.now()

        # Fetch the full day including extended hours
        start = datetime(
            trading_date.year, trading_date.month, trading_date.day, 4, 0
        )
        end = datetime(
            trading_date.year, trading_date.month, trading_date.day, 9, 30
        )

        df = self.get_historical_bars(
            symbol=symbol,
            start=start,
            end=end,
            include_extended=False,
        )

        if df.empty:
            return df

        # Filter to strict pre-market window
        df = df.between_time(_PREMARKET_OPEN, _PREMARKET_CLOSE)
        return df

    # ------------------------------------------------------------------
    # Convenience: derive daily returns from 1-min bars
    # ------------------------------------------------------------------

    @staticmethod
    def daily_returns_from_bars(bars: pd.DataFrame) -> pd.Series:
        """
        Derive daily close-to-close returns from a 1-minute bar DataFrame.

        Useful in backtesting where daily bars are not fetched separately —
        the same intraday DataFrame can produce vol scaler inputs.

        Takes the close of the last bar in each regular session
        (between_time 09:30–16:00) as the daily close.

        Returns
        -------
        pd.Series
            Daily simple returns, indexed by date.
        """
        regular = bars.between_time(_SESSION_OPEN, _SESSION_CLOSE)
        daily_close = (
            regular["close"]
            .groupby(regular["day"])
            .last()
        )
        return daily_close.pct_change().dropna()

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

        FIX 2: Non-blocking — launches stream in a background thread.
                Call stop_stream() to shut down gracefully.

        FIX 6: callback receives a pd.Series with the same column names
                as the historical DataFrame, not a plain dict.
                Series fields: open, high, low, close, volume, symbol, caldt

        Parameters
        ----------
        symbol : str
            e.g. "SPY"
        callback : Callable[[pd.Series], None]
            Called with a pd.Series on each new 1-minute bar close.
        """
        from alpaca.data.live import StockDataStream

        self._stream = StockDataStream(
            self.config.api_key,
            self.config.secret_key,
        )

        async def _handler(bar):
            # Convert to pd.Series — same format as historical DataFrame rows
            ts = pd.Timestamp(bar.timestamp)
            if ts.tz is not None:
                ts = ts.tz_convert("America/New_York").tz_localize(None)

            series = pd.Series({
                "open":    bar.open,
                "high":    bar.high,
                "low":     bar.low,
                "close":   bar.close,
                "volume":  bar.volume,
                "symbol":  bar.symbol,
                "caldt":   ts,
            }, name=ts)

            callback(series)

        self._stream.subscribe_bars(_handler, symbol)

        logger.info("Starting Alpaca WebSocket stream for %s ...", symbol)

        # FIX 2: run in background thread — does not block the caller
        self._stream_thread = threading.Thread(
            target=self._stream.run,
            daemon=True,
            name=f"alpaca-stream-{symbol}",
        )
        self._stream_thread.start()

    def stop_stream(self) -> None:
        """
        Gracefully stop the WebSocket stream.

        FIX: Was completely missing — no way to stop the stream.
        Call at 16:00 EOD or on KeyboardInterrupt.
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
    # CSV fallback — unchanged, works correctly
    # ------------------------------------------------------------------

    @staticmethod
    def from_csv(path: str | Path) -> pd.DataFrame:
        """
        Load historical data from a local CSV file.

        Useful for fully offline backtesting without any API calls.
        The CSV should have been saved from get_historical_bars() output.
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
# Helpers
# ------------------------------------------------------------------

def _to_utc(dt: str | datetime) -> datetime:
    """Coerce any date/string input to a UTC-aware datetime."""
    if isinstance(dt, str):
        dt = pd.Timestamp(dt).to_pydatetime()
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt