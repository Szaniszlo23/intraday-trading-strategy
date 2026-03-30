"""
Position sizing for the intraday momentum strategy.

PositionSizer         — Baseline: scales shares so daily P&L vol ≈ target_vol * AUM,
                        capped at max_leverage.  Also supports full_notional mode.

EnhancedPositionSizer — Enhanced: applies a Layer 2 multiplier M on top of the
                        same vol-target base:
                          M = clip(vol_regime_factor × flow_composite_mult, 0.25, 2.0)
                          shares = floor(AUM × min(4, 0.02/sigma_90d) × M / Open)
"""

from __future__ import annotations

import math

from config.config import StrategyConfig


class PositionSizer:
    """Baseline position sizer (unchanged)."""

    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config or StrategyConfig()

    def shares(self, aum: float, open_price: float, daily_vol: float | None) -> int:
        """
        Compute the number of shares to trade.

        Parameters
        ----------
        aum        : current portfolio value ($)
        open_price : day's opening price
        daily_vol  : rolling daily return volatility
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


class EnhancedPositionSizer:
    """
    Enhanced position sizer with Layer 2 sizing multiplier.

    Base formula (unchanged from paper):
        base_leverage = min(max_leverage, target_vol / daily_vol_90d)

    Layer 2 multiplier:
        M = clip(vol_regime_factor × flow_composite_mult, 0.25, 2.0)

    Final shares:
        shares = floor(AUM × base_leverage × M / Open)

    The multiplier never blocks a trade — it only scales size up or down.
    """

    _MAX_LEVERAGE: float = 4.0
    _TARGET_VOL:   float = 0.02
    _M_MIN:        float = 0.25
    _M_MAX:        float = 2.0

    def shares(
        self,
        aum: float,
        open_price: float,
        daily_vol: float | None,
        vol_regime: float,
        flow_mult: float,
    ) -> int:
        """
        Compute the number of shares for the enhanced strategy.

        Parameters
        ----------
        aum        : current portfolio value ($)
        open_price : day's opening price
        daily_vol  : 90-day rolling daily return volatility
                     (None / NaN → fallback to max_leverage)
        vol_regime : vol-regime factor from Indicators.vol_regime_factor()
        flow_mult  : flow composite multiplier from Indicators.flow_composite_mult()

        Returns
        -------
        int — number of shares (always ≥ 0)
        """
        if daily_vol is None or math.isnan(daily_vol) or daily_vol == 0:
            base_leverage = self._MAX_LEVERAGE
        else:
            base_leverage = min(self._TARGET_VOL / daily_vol, self._MAX_LEVERAGE)

        M = max(self._M_MIN, min(self._M_MAX, vol_regime * flow_mult))
        return math.floor(aum * base_leverage * M / open_price)
