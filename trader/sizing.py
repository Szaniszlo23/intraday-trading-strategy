"""
Position sizing for the intraday momentum strategy.

Two modes:
  vol_target    — scales shares so daily P&L vol ≈ target_vol * AUM,
                  capped at max_leverage.
  full_notional — trades one full AUM's worth of shares.
"""

from __future__ import annotations

import math

from config.config import StrategyConfig


class PositionSizer:
    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config or StrategyConfig()

    def shares(self, aum: float, open_price: float, daily_vol: float | None) -> int:
        """
        Compute the number of shares to trade.

        Parameters
        ----------
        aum        : current portfolio value ($)
        open_price : day's opening price
        daily_vol  : estimated 14-day rolling daily return volatility
                     (None / NaN → use max_leverage as fallback)

        Returns
        -------
        int — number of shares (always ≥ 0)
        """
        cfg = self.config

        if cfg.sizing_type == "full_notional":
            return round(aum / open_price)

        # vol_target mode
        if daily_vol is None or math.isnan(daily_vol) or daily_vol == 0:
            leverage = cfg.max_leverage
        else:
            leverage = min(cfg.target_vol / daily_vol, cfg.max_leverage)

        return round(aum / open_price * leverage)
