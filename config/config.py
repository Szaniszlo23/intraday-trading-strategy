"""
Configuration for the intraday momentum strategy.

All strategy parameters, execution settings, and API credentials
are centralised here so they can be loaded from config.yaml without
scattering magic numbers across the codebase.
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AlpacaConfig:
    api_key: str = ""
    secret_key: str = ""
    paper: bool = True
    base_url: str = "https://paper-api.alpaca.markets"


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
    def from_yaml(cls, path: str | Path = "config/config.yaml") -> "AppConfig":
        """Load configuration from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)

        alpaca_raw = raw.get("alpaca", {})
        strategy_raw = raw.get("strategy", {})

        return cls(
            alpaca=AlpacaConfig(**alpaca_raw),
            strategy=StrategyConfig(**strategy_raw),
        )

    @classmethod
    def default(cls) -> "AppConfig":
        """Return a default configuration (useful for testing without config.yaml)."""
        return cls()
