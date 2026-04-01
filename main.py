"""
Live trading entry point — V6 Enhanced Strategy.

Architecture
------------
  Startup  : fetch 120 days of historical 1-min bars, preprocess with
             vol_window=90 to warm up sigma_open and spy_dvol.  Derive
             today's per-minute sigma lookup from the rolling mean.
  Session  : on first bar (09:30) compute Layer-2 sizing once for the day
             (vol_regime, premarket_return, order_imbalance → flow_composite,
             → shares).  On every subsequent bar run EnhancedSignalGenerator
             on all today's bars, take the last exposure value, rebalance.
  EOD      : force-flat at 15:59 ET regardless of signal.
  Safety   : SIGINT / SIGTERM close all positions and exit.

Usage
-----
    python main.py                          # V6 enhanced (default)
    python main.py --baseline               # original baseline strategy
    python main.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import math
import signal
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from config.config import AppConfig, StrategyConfig
from data.fetch import AlpacaFetcher
from data.indicators import Indicators
from data.preprocess import Preprocessor
from trader.signals import EnhancedSignalGenerator, SignalGenerator
from trader.sizing import EnhancedPositionSizer, PositionSizer
from trader.trading_model import AlpacaTrader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("live_trader")

# ── Constants ────────────────────────────────────────────────────────────────
_HISTORY_DAYS  = 120              # warm-up window for sigma_open (90-day rolling)
_SESSION_OPEN  = time(9, 30)
_ENTRY_CUTOFF  = time(15, 30)
_EOD_CLOSE     = time(15, 59)
_ET            = ZoneInfo("America/New_York")   # US Eastern — all time comparisons use this


# ── V6 Live Trader ────────────────────────────────────────────────────────────

class LiveTraderV6:
    """
    Live implementation of the V6 Enhanced strategy.

    Layer 2 sizing (vol_regime × flow_composite) is computed ONCE per session
    at the first bar of the day (09:30) and held fixed for the entire session,
    mirroring how the backtest computes pre-session sizing.

    Signal generation runs on all accumulated today-bars after each new bar
    arrives, and the last exposure value drives the order.
    """

    def __init__(self, app_cfg: AppConfig) -> None:
        self.cfg    = app_cfg.strategy
        self.symbol = self.cfg.symbol

        # V6 always uses vol_window=90
        self._strat_cfg = StrategyConfig(
            **{**vars(self.cfg), "vol_window": 90}
        )

        # Core components
        self.fetcher     = AlpacaFetcher(app_cfg.alpaca)
        self.trader      = AlpacaTrader(app_cfg.alpaca)
        self.preprocessor = Preprocessor(self._strat_cfg)
        self.signal_gen  = EnhancedSignalGenerator(
            self._strat_cfg, use_dual_bar=True, use_vwap_stop=True
        )
        self.sizer = EnhancedPositionSizer()

        # ── Per-session state ──────────────────────────────────────────────
        self._today_bars:    list[dict] = []
        self._last_exposure: float      = 0.0
        self._prev_close:    float | None = None

        # Layer 2 — computed once at session open
        self._vol_regime:      float = 1.0
        self._flow_mult:       float = 1.0
        self._session_shares:  int   = 0
        self._session_ready:   bool  = False

        # ── Historical data ────────────────────────────────────────────────
        self._hist_df:         pd.DataFrame = pd.DataFrame()
        self._hist_daily:      pd.DataFrame = pd.DataFrame()
        self._raw_extended:    pd.DataFrame = pd.DataFrame()

        # Per-minute sigma lookup for today (recomputed daily)
        # minute_of_day (int) → sigma_open (float)
        self._sigma_lookup: dict[int, float] = {}
        self._dvol_today:   float = float("nan")

        # ── Bootstrap ─────────────────────────────────────────────────────
        account = self.trader.get_account()
        logger.info(
            "Account ready — equity=$%.2f  buying_power=$%.2f  paper=%s",
            account["equity"], account["buying_power"], account["paper"],
        )
        self._bootstrap()

    # ── Startup ───────────────────────────────────────────────────────────────

    def _bootstrap(self) -> None:
        """
        Fetch HISTORY_DAYS of 1-min bars (including pre-market) and preprocess.
        Derives the per-minute sigma lookup for today's session.
        """
        end   = datetime.now(_ET)
        start = end - timedelta(days=_HISTORY_DAYS + 15)   # extra buffer for holidays

        logger.info("Bootstrapping %d days of history ...", _HISTORY_DAYS)
        raw = self.fetcher.get_historical_bars(
            self.symbol, start, end, include_extended=True
        )
        if raw.empty:
            logger.error("Bootstrap failed — no data returned from Alpaca.")
            sys.exit(1)

        # Store full extended-hours data for flow composite lookups
        self._raw_extended = raw.copy()
        if "day" not in self._raw_extended.columns:
            self._raw_extended["day"] = self._raw_extended.index.date
        self._raw_extended["day"] = pd.to_datetime(
            self._raw_extended["day"]
        ).dt.date

        # Preprocess session-only bars
        raw_session = raw.between_time("09:30", "16:00")
        self._hist_df, self._hist_daily = self.preprocessor.transform(raw_session)

        # Previous close
        all_days = sorted(self._hist_df["day"].unique())
        if all_days:
            last_day_df  = self._hist_df[self._hist_df["day"] == all_days[-1]]
            self._prev_close = float(last_day_df["close"].iloc[-1])
            self._dvol_today = float(last_day_df["spy_dvol"].iloc[0])

        # Build per-minute sigma lookup for today
        # sigma(today, minute m) = mean of move_open at minute m over last 90 days
        self._sigma_lookup = self._compute_sigma_lookup()

        logger.info(
            "History loaded — %d trading days  prev_close=%.2f  dvol=%.4f",
            len(all_days), self._prev_close or 0.0, self._dvol_today,
        )

    def _compute_sigma_lookup(self) -> dict[int, float]:
        """
        Compute what sigma_open would be for today at each minute of the session.

        For each minute-of-day m:
            sigma[m] = mean of move_open[last 90 days, minute m]

        This is the correctly lag-safe value: uses only data up to and including
        yesterday, exactly what the backtest's shift(1) produces for today.
        """
        if self._hist_df.empty:
            return {}
        lookup = {}
        for minute, grp in self._hist_df.groupby("minute_of_day"):
            recent = grp["move_open"].dropna().tail(self._strat_cfg.vol_window)
            if len(recent) > 0:
                lookup[int(minute)] = float(recent.mean())
        return lookup

    # ── Session open ──────────────────────────────────────────────────────────

    def _backfill_today(self) -> float | None:
        """
        If we joined after 09:30 ET, fetch all bars from 09:30 up to now via
        REST and prepend them to _today_bars so VWAP and move_open are correct.

        Returns the actual 09:30 open price, or None if back-fill failed.
        """
        now_et = datetime.now(_ET)
        today  = now_et.date()

        session_open_dt = datetime(
            today.year, today.month, today.day, 9, 30,
            tzinfo=_ET,
        )

        # Nothing to back-fill if we're right at the open
        if now_et <= session_open_dt + timedelta(minutes=1):
            return None

        logger.info(
            "Joined after session open — back-filling bars from 09:30 to %s ET ...",
            now_et.strftime("%H:%M"),
        )
        try:
            missed = self.fetcher.get_historical_bars(
                self.symbol,
                start=session_open_dt,
                end=now_et,
                include_extended=False,
            )
        except Exception as exc:
            logger.error("Back-fill fetch failed: %s", exc)
            return None

        if missed.empty:
            logger.warning("Back-fill returned no bars — VWAP/move_open will be approximate.")
            return None

        # Prepend to today_bars (they arrive before the triggering WebSocket bar)
        backfilled = []
        for ts, row in missed.iterrows():
            backfilled.append({
                "open":   float(row["open"]),
                "high":   float(row["high"]),
                "low":    float(row["low"]),
                "close":  float(row["close"]),
                "volume": float(row["volume"]),
                "caldt":  ts,
            })

        # Insert before whatever bar triggered _prepare_session
        self._today_bars = backfilled + self._today_bars

        actual_open = float(missed["open"].iloc[0])
        logger.info(
            "Back-filled %d bars — actual 09:30 open=%.2f", len(backfilled), actual_open
        )
        return actual_open

    def _prepare_session(self, open_price: float) -> None:
        """
        Called on the very first bar of the day (09:30 ET).
        If we joined mid-session, back-fills missing bars first so that
        VWAP, move_open, and position sizing all use the correct 09:30 open.
        Computes all Layer-2 values and fixes position size for the session.
        """
        today = datetime.now(_ET).date()

        # Back-fill if we missed bars since 09:30
        actual_open = self._backfill_today()
        if actual_open is not None:
            open_price = actual_open

        # --- Vol-regime factor ---
        daily_rets = self._hist_daily["ret"].dropna()
        self._vol_regime = Indicators.vol_regime_factor(daily_rets)

        # --- Pre-market return (04:00 – 09:29 ET) ---
        pm_ret = 0.0
        raw_groups = self._raw_extended.groupby("day")
        if today in raw_groups.groups:
            pm_bars = raw_groups.get_group(today).between_time("04:00", "09:29")
            pm_ret  = Indicators.premarket_return(pm_bars)

        # --- Order imbalance (previous day's first 30 session bars) ---
        imbalance = 0.0
        all_days  = sorted(self._hist_df["day"].unique())
        if all_days:
            prev_day    = all_days[-1]
            prev_sess   = self._hist_df[self._hist_df["day"] == prev_day]
            first_30    = prev_sess.iloc[:30]
            imbalance   = Indicators.order_imbalance(first_30)

        # --- Flow composite ---
        self._flow_mult = Indicators.flow_composite_mult(pm_ret, imbalance)

        # --- Position size (fixed for the session) ---
        account   = self.trader.get_account()
        equity    = float(account["equity"])
        buying_power = float(account["buying_power"])
        daily_vol = self._dvol_today if not math.isnan(self._dvol_today) else None

        # Pass equity/2 as the sizing base so that max 4× leverage
        # maps to exactly 2× of total equity — within Alpaca's buying power limit.
        # The full dynamic range (0.25×–2.0× multiplier) is preserved.
        self._session_shares = self.sizer.shares(
            equity / 2, open_price, daily_vol,
            self._vol_regime, self._flow_mult,
        )

        # Safety net: if dynamic sizing still somehow exceeds buying power
        # (e.g. equity grew, or extreme flow_mult), hard-cap at 95% of buying power.
        max_shares_by_bp = math.floor(buying_power * 0.95 / open_price)
        if self._session_shares > max_shares_by_bp:
            logger.warning(
                "Capping shares %d → %d to fit buying_power=$%.0f",
                self._session_shares, max_shares_by_bp, buying_power,
            )
            self._session_shares = max_shares_by_bp

        # Leverage relative to total equity (not the halved base)
        leverage = (self._session_shares * open_price) / equity if equity > 0 else 0.0

        self._session_ready = True
        logger.info(
            "Session open — open=%.2f  vol_regime=%.2f  flow_mult=%.2f  "
            "pm_ret=%+.3f%%  imbalance=%+.2f  shares=%d  leverage=%.2f×",
            open_price, self._vol_regime, self._flow_mult,
            pm_ret * 100, imbalance,
            self._session_shares, leverage,
        )

    # ── Per-bar ───────────────────────────────────────────────────────────────

    def on_bar(self, bar: dict | pd.Series) -> None:
        """Callback — receives one 1-minute bar from the WebSocket or REST catchup.

        WebSocket path : bar is a pd.Series (from fetch.py _handler)
                         timestamp lives in bar.name / bar["caldt"]
        REST catchup   : bar is a plain dict with a "timestamp" key
        Normalise both into a dict before processing.
        """
        self._last_bar_time = datetime.now(_ET)   # heartbeat stamp — always ET-aware

        # ── Normalise pd.Series → dict ────────────────────────────────────
        if isinstance(bar, pd.Series):
            ts_raw = bar.name if isinstance(bar.name, (datetime, pd.Timestamp)) \
                     else bar["caldt"]
            bar = {
                "timestamp": ts_raw,
                "open":   float(bar["open"]),
                "high":   float(bar["high"]),
                "low":    float(bar["low"]),
                "close":  float(bar["close"]),
                "volume": float(bar["volume"]),
            }

        ts: datetime = bar["timestamp"]
        if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
            ts = ts.astimezone(_ET).replace(tzinfo=None)

        et_time = ts.time()

        # Ignore pre/post-market bars
        if not (_SESSION_OPEN <= et_time <= _EOD_CLOSE):
            return

        # Force-flat at EOD
        if et_time >= _EOD_CLOSE:
            if self._last_exposure != 0.0:
                logger.info("EOD — force-closing all positions")
                self.trader.close_all()
                self._last_exposure = 0.0
            return

        # Accumulate bar
        self._today_bars.append({
            "open":   float(bar["open"]),
            "high":   float(bar["high"]),
            "low":    float(bar["low"]),
            "close":  float(bar["close"]),
            "volume": float(bar["volume"]),
            "caldt":  ts,
        })

        # First bar: compute session sizing
        if len(self._today_bars) == 1:
            self._prepare_session(float(bar["open"]))
            return                      # need ≥ 2 bars for dual-bar confirmation

        if not self._session_ready or self._prev_close is None:
            return

        # Build today's feature DataFrame
        today_df = self._build_today_df()
        if today_df is None or today_df["sigma_open"].isna().all():
            logger.debug("sigma_open not ready at %s", et_time)
            return

        # Generate signal and take the latest exposure
        exposure_series  = self.signal_gen.generate(today_df, self._prev_close)
        desired_exposure = float(exposure_series.iloc[-1])

        if desired_exposure == self._last_exposure:
            return

        target_qty = int(desired_exposure * self._session_shares)
        logger.info(
            "Signal %+.0f → %+.0f  target=%d shares  bar=%s",
            self._last_exposure, desired_exposure, target_qty,
            et_time.strftime("%H:%M"),
        )
        self.trader.rebalance(self.symbol, target_qty)
        self._last_exposure = desired_exposure

    # ── Feature construction ──────────────────────────────────────────────────

    def _build_today_df(self) -> pd.DataFrame | None:
        """
        Build a feature-enriched DataFrame from today's bars so far.

        Columns produced (matching what EnhancedSignalGenerator expects):
            open, high, low, close, volume
            vwap         — cumulative intraday VWAP
            sigma_open   — from pre-computed sigma_lookup (lag-safe)
            spy_dvol     — carried from last historical value
            move_open    — abs % move from today's open
            min_from_open, minute_of_day

        sigma_open comes from _sigma_lookup which was computed at bootstrap from
        the 90-day rolling mean of move_open — identical to how the backtest
        computes it, with no look-ahead.
        """
        if not self._today_bars:
            return None

        df = pd.DataFrame(self._today_bars)
        df["caldt"] = pd.to_datetime(df["caldt"])
        df = df.set_index("caldt").sort_index()
        df["day"] = df.index.date

        # Session-position columns
        session_start = pd.Timedelta(hours=9, minutes=30)
        df["min_from_open"] = (
            (df.index - df.index.normalize()) - session_start
        ) / pd.Timedelta(minutes=1) + 1
        df["minute_of_day"] = df["min_from_open"].round().astype(int)

        # VWAP — compute directly (single-day, no groupby needed)
        hlc = (df["high"] + df["low"] + df["close"]) / 3
        df["vwap"] = (df["volume"] * hlc).cumsum() / df["volume"].cumsum()

        # move_open — absolute % from today's open
        day_open = df["open"].iloc[0]
        df["move_open"] = (df["close"] / day_open - 1).abs()

        # sigma_open — from the pre-session lookup (lag-safe 90-day mean)
        df["sigma_open"] = df["minute_of_day"].map(self._sigma_lookup)

        # spy_dvol — constant for the day (from last historical observation)
        df["spy_dvol"] = self._dvol_today

        return df

    # ── Daily refresh ─────────────────────────────────────────────────────────

    def _end_of_day_refresh(self) -> None:
        """
        Called after EOD close.  Re-fetches yesterday's full session bars and
        updates the historical DataFrame + sigma lookup so tomorrow's session
        starts with fresh context.
        """
        logger.info("End-of-day refresh — updating historical data ...")
        try:
            end   = datetime.now(_ET)
            start = end - timedelta(days=_HISTORY_DAYS + 15)
            raw   = self.fetcher.get_historical_bars(
                self.symbol, start, end, include_extended=True
            )
            if raw.empty:
                logger.warning("Refresh skipped — no data returned.")
                return

            self._raw_extended = raw.copy()
            if "day" not in self._raw_extended.columns:
                self._raw_extended["day"] = self._raw_extended.index.date
            self._raw_extended["day"] = pd.to_datetime(
                self._raw_extended["day"]
            ).dt.date

            raw_session = raw.between_time("09:30", "16:00")
            self._hist_df, self._hist_daily = self.preprocessor.transform(raw_session)

            all_days = sorted(self._hist_df["day"].unique())
            if all_days:
                last_day_df      = self._hist_df[self._hist_df["day"] == all_days[-1]]
                self._prev_close = float(last_day_df["close"].iloc[-1])
                self._dvol_today = float(last_day_df["spy_dvol"].iloc[0])

            self._sigma_lookup = self._compute_sigma_lookup()
            logger.info(
                "Refresh done — prev_close=%.2f  dvol=%.4f",
                self._prev_close or 0.0, self._dvol_today,
            )
        except Exception as exc:
            logger.error("End-of-day refresh failed: %s", exc)

        # Reset for next session
        self._today_bars    = []
        self._last_exposure = 0.0
        self._session_ready = False

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self) -> None:
        import time
        from datetime import datetime, timedelta

        logger.info(
            "Starting V6 live trader — symbol=%s  paper=%s",
            self.symbol, self.cfg.__dict__.get("paper", "?"),
        )
        self.fetcher.stream_bars(self.symbol, self.on_bar)
        self._last_bar_time: datetime = datetime.now(_ET)

        # ── Heartbeat loop ────────────────────────────────────────────────
        # Every 60 seconds check if the WebSocket has gone silent.
        # If no bar has arrived in 90+ seconds during market hours, fall
        # back to REST to fetch and process the missing bar(s).
        while True:
            time.sleep(60)

            now      = datetime.now(_ET)   # always Eastern Time
            et_time  = now.time()

            # Only check during regular session
            if not (_SESSION_OPEN <= et_time <= _EOD_CLOSE):
                continue

            gap = (now - self._last_bar_time).total_seconds()
            if gap > 90:
                logger.warning(
                    "WebSocket silent for %.0fs — fetching missed bars via REST",
                    gap,
                )
                try:
                    missed = self.fetcher.get_historical_bars(
                        self.symbol,
                        start=now - timedelta(minutes=5),
                        end=now,
                        include_extended=False,
                    )
                    for ts, row in missed.iterrows():
                        if ts > self._last_bar_time:
                            bar = {
                                "timestamp": ts,
                                "open":   row["open"],
                                "high":   row["high"],
                                "low":    row["low"],
                                "close":  row["close"],
                                "volume": row["volume"],
                            }
                            logger.info("REST catchup bar: %s", ts)
                            self.on_bar(bar)
                except Exception as exc:
                    logger.error("REST catchup failed: %s", exc)


# ── Baseline Live Trader (unchanged) ─────────────────────────────────────────

class LiveTrader:
    """Original baseline live trader (30-min clock, 14-day sigma)."""

    def __init__(self, app_cfg: AppConfig) -> None:
        self.cfg = app_cfg
        self.strategy_cfg = app_cfg.strategy
        self.symbol = self.strategy_cfg.symbol

        self.fetcher     = AlpacaFetcher(app_cfg.alpaca)
        self.trader      = AlpacaTrader(app_cfg.alpaca)
        self.preprocessor = Preprocessor(self.strategy_cfg)
        self.signal_gen  = SignalGenerator(self.strategy_cfg)
        self.sizer       = PositionSizer(self.strategy_cfg)

        self._buffer: list[dict] = []
        self._last_exposure: float = 0.0

        account = self.trader.get_account()
        logger.info(
            "Account ready: equity=$%.2f  buying_power=$%.2f  paper=%s",
            account["equity"], account["buying_power"], account["paper"],
        )

    def run(self) -> None:
        logger.info("Starting baseline live trader for %s ...", self.symbol)
        self.fetcher.stream_bars(self.symbol, self.on_bar)

    def on_bar(self, bar: dict | pd.Series) -> None:
        # Normalise WebSocket pd.Series → dict
        if isinstance(bar, pd.Series):
            ts_raw = bar.name if isinstance(bar.name, (datetime, pd.Timestamp)) \
                     else bar["caldt"]
            bar = {
                "timestamp": ts_raw,
                "open": float(bar["open"]), "high": float(bar["high"]),
                "low":  float(bar["low"]),  "close": float(bar["close"]),
                "volume": float(bar["volume"]),
            }

        ts: datetime = bar["timestamp"]
        if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
            ts = ts.astimezone(_ET).replace(tzinfo=None)

        et_time = ts.time()

        if not (time(9, 30) <= et_time <= time(15, 59)):
            return

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
            df_feat, _ = self.preprocessor.transform(df_buf)
        except Exception as e:
            logger.warning("Preprocessing failed: %s", e)
            return

        today     = df_feat["day"].iloc[-1]
        today_df  = df_feat[df_feat["day"] == today]

        if today_df["sigma_open"].isna().all() or len(today_df) < 2:
            return

        prev_days = df_feat[df_feat["day"] < today]["day"].unique()
        if len(prev_days) == 0:
            return

        prev_close       = df_feat[df_feat["day"] == prev_days[-1]]["close"].iloc[-1]
        exposure_series  = self.signal_gen.generate(today_df, prev_close)
        desired_exposure = float(exposure_series.iloc[-1])

        if desired_exposure == self._last_exposure:
            return

        equity        = self.trader.get_account()["equity"]
        open_px       = float(today_df["close"].iloc[0])
        daily_vol_val = today_df["spy_dvol"].iloc[0]
        daily_vol     = float(daily_vol_val) if (
            daily_vol_val is not None
            and not (isinstance(daily_vol_val, float) and math.isnan(daily_vol_val))
        ) else None

        shares     = self.sizer.shares(equity, open_px, daily_vol)
        target_qty = int(desired_exposure * shares)

        logger.info(
            "Signal: %.0f → %.0f  |  target=%d shares",
            self._last_exposure, desired_exposure, target_qty,
        )
        self.trader.rebalance(self.symbol, target_qty)
        self._last_exposure = desired_exposure


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live Intraday Momentum Trader")
    parser.add_argument("--config",   type=str, default="config/config.yaml")
    parser.add_argument("--baseline", action="store_true",
                        help="Run original baseline strategy instead of V6")
    return parser.parse_args()


def main() -> None:
    args     = parse_args()
    cfg_path = Path(args.config)

    if not cfg_path.exists():
        logger.error(
            "Config not found at %s — copy config/config.yaml and fill in your keys.",
            cfg_path,
        )
        sys.exit(1)

    app_cfg = AppConfig.load(cfg_path)

    if args.baseline:
        logger.info("Using baseline strategy (--baseline flag set)")
        live: LiveTrader | LiveTraderV6 = LiveTrader(app_cfg)
    else:
        logger.info("Using V6 Enhanced strategy")
        live = LiveTraderV6(app_cfg)

    def _shutdown(sig, frame):
        logger.info("Shutdown signal received — closing all positions ...")
        live.trader.close_all()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    live.run()


if __name__ == "__main__":
    main()
