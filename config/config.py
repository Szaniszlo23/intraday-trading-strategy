"""
Configuration for the intraday momentum strategy.

API credentials are loaded exclusively from environment variables so
they are never stored in files that could accidentally be committed.

Priority order for credentials:
  1. Actual environment variables (e.g. set in your shell, Docker, or Cloud Run)
  2. A .env file in the project root (for local development — gitignored)

All other strategy parameters (band_mult, trade_freq, etc.) can still
be tuned via config/config.yaml, which contains NO secrets.

Usage
-----
Local development:
    Copy .env.example → .env and fill in your keys.
    The app loads .env automatically via python-dotenv.

Docker / Cloud Run:
    Pass ALPACA_API_KEY and ALPACA_SECRET_KEY as environment variables.
    No .env file needed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load .env from project root if it exists (silently ignored if absent)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@dataclass
class AlpacaConfig:
    api_key: str = ""
    secret_key: str = ""
    paper: bool = True
    base_url: str = "https://paper-api.alpaca.markets"

    @classmethod
    def from_env(cls) -> "AlpacaConfig":
        """
        Build AlpacaConfig from environment variables.

        Required env vars:
            ALPACA_API_KEY      — your Alpaca API key
            ALPACA_SECRET_KEY   — your Alpaca secret key

        Optional env vars:
            ALPACA_PAPER        — "true" (default) or "false"
            ALPACA_BASE_URL     — override the API base URL
        """
        api_key = os.environ.get("ALPACA_API_KEY", "")
        secret_key = os.environ.get("ALPACA_SECRET_KEY", "")

        if not api_key or not secret_key:
            raise EnvironmentError(
                "Missing Alpaca credentials. "
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY as environment variables, "
                "or add them to your .env file (see .env.example)."
            )

        paper_raw = os.environ.get("ALPACA_PAPER", "true").lower()
        paper = paper_raw not in ("false", "0", "no")

        default_url = (
            "https://paper-api.alpaca.markets"
            if paper
            else "https://api.alpaca.markets"
        )
        base_url = os.environ.get("ALPACA_BASE_URL", default_url)

        return cls(api_key=api_key, secret_key=secret_key, paper=paper, base_url=base_url)


@dataclass
class StrategyConfig:
    # --- Signal construction ---
    band_mult: float = 1.0       # Noise-band multiplier (σ scaling)
    trade_freq: int = 30         # Minutes between signal evaluations
    rsi_period: int = 14         # RSI look-back window (bars)

    # --- RSI filter ---
    rsi_filter: bool = True      # Enable RSI confirmation filter
    rsi_long: int = 60           # Go long only when RSI > rsi_long
    rsi_short: int = 40          # Go short only when RSI < rsi_short

    # --- Position sizing ---
    sizing_type: str = "vol_target"  # "vol_target" | "full_notional"
    target_vol: float = 0.02     # Daily target volatility (vol-targeting mode)
    max_leverage: float = 4.0    # Maximum leverage cap

    # --- Execution & costs ---
    aum_0: float = 100_000.0     # Starting AUM ($)
    commission: float = 0.0035   # Per-share commission ($)
    min_comm: float = 0.35       # Minimum commission per order ($)

    # --- Data ---
    symbol: str = "SPY"
    vol_window: int = 14         # Days used for rolling daily volatility


@dataclass
class AppConfig:
    alpaca: AlpacaConfig = field(default_factory=AlpacaConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)

    @classmethod
    def load(cls, yaml_path: str | Path = "config/config.yaml") -> "AppConfig":
        """
        Load a full AppConfig.

        - Alpaca credentials come from environment variables / .env
        - Strategy parameters come from config/config.yaml
        """
        strategy = StrategyConfig()

        path = Path(yaml_path)
        if path.exists():
            with open(path) as f:
                raw = yaml.safe_load(f) or {}
            strategy_raw = raw.get("strategy", {})
            strategy = StrategyConfig(**strategy_raw)

        return cls(
            alpaca=AlpacaConfig.from_env(),
            strategy=strategy,
        )

    @classmethod
    def default(cls) -> "AppConfig":
        """
        Return a config with defaults — credentials from env, default strategy params.
        Useful for tests that mock the Alpaca client.
        """
        try:
            alpaca = AlpacaConfig.from_env()
        except EnvironmentError:
            alpaca = AlpacaConfig()  # empty keys — safe for unit tests
        return cls(alpaca=alpaca, strategy=StrategyConfig())
