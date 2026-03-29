"""
Order execution via the Alpaca Trading API.

AlpacaTrader provides a minimal interface for the live trading loop:
  submit_order()  — market order (buy or sell)
  get_position()  — current held shares
  rebalance()     — move position to a target quantity
  close_all()     — flatten all open positions (EOD)
  get_account()   — equity / buying power summary
"""

from __future__ import annotations

import logging

from config.config import AlpacaConfig

logger = logging.getLogger(__name__)


class AlpacaTrader:
    def __init__(self, config: AlpacaConfig | None = None) -> None:
        self.config = config or AlpacaConfig()
        self._client = self._build_client()

    def _build_client(self):
        from alpaca.trading.client import TradingClient

        return TradingClient(
            api_key=self.config.api_key,
            secret_key=self.config.secret_key,
            paper=self.config.paper,
            url_override=self.config.base_url if self.config.paper else None,
        )

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def submit_order(self, symbol: str, qty: int, side: str) -> object:
        """Submit a DAY market order. side = 'buy' | 'sell'."""
        from alpaca.trading.enums import OrderSide
        from alpaca.trading.requests import MarketOrderRequest, TimeInForce

        if qty <= 0:
            raise ValueError(f"qty must be positive, got {qty}")

        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )
        order = self._client.submit_order(request)
        logger.info("Order: %s %d %s  id=%s", side.upper(), qty, symbol, order.id)
        return order

    def close_all(self) -> list:
        """Cancel open orders and close all positions."""
        closed = self._client.close_all_positions(cancel_orders=True)
        logger.info("Closed all positions (%d)", len(closed))
        return closed

    # ------------------------------------------------------------------
    # Account / position queries
    # ------------------------------------------------------------------

    def get_position(self, symbol: str) -> int:
        """Return current net share quantity (0 if no position)."""
        try:
            pos = self._client.get_open_position(symbol)
            return int(float(pos.qty))
        except Exception:
            return 0

    def get_account(self) -> dict:
        acc = self._client.get_account()
        return {
            "equity": float(acc.equity),
            "buying_power": float(acc.buying_power),
            "cash": float(acc.cash),
            "portfolio_value": float(acc.portfolio_value),
            "paper": self.config.paper,
        }

    # ------------------------------------------------------------------
    # High-level rebalance
    # ------------------------------------------------------------------

    def rebalance(self, symbol: str, target_qty: int) -> None:
        """Bring position in symbol to exactly target_qty shares."""
        current = self.get_position(symbol)
        delta = target_qty - current
        if delta == 0:
            return
        self.submit_order(symbol, abs(delta), "buy" if delta > 0 else "sell")
