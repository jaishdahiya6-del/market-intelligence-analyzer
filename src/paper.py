"""Paper trading with virtual money. No broker connection exists; no real orders are ever placed."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class PaperPortfolio:
    cash: float = 100_000.0
    fee_bps: float = 10.0
    positions: dict = field(default_factory=dict)   # symbol -> qty
    log: list = field(default_factory=list)

    def _record(self, side, symbol, qty, price, fee, note):
        self.log.append({"time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "side": side,
                         "symbol": symbol, "qty": qty, "price": price, "fee": fee, "price_note": note})

    def buy(self, symbol: str, qty: float, price: float, note: str = "") -> None:
        if qty <= 0 or price <= 0:
            raise ValueError("Quantity and price must be positive.")
        fee = qty * price * self.fee_bps / 1e4
        if qty * price + fee > self.cash:
            raise ValueError("Insufficient virtual cash.")
        self.cash -= qty * price + fee
        self.positions[symbol] = self.positions.get(symbol, 0.0) + qty
        self._record("BUY", symbol, qty, price, fee, note)

    def sell(self, symbol: str, qty: float, price: float, note: str = "") -> None:
        if qty <= 0 or price <= 0:
            raise ValueError("Quantity and price must be positive.")
        if self.positions.get(symbol, 0.0) + 1e-12 < qty:
            raise ValueError("Cannot sell more than you hold (no shorting in paper mode).")
        fee = qty * price * self.fee_bps / 1e4
        self.cash += qty * price - fee
        self.positions[symbol] -= qty
        if self.positions[symbol] <= 1e-12:
            del self.positions[symbol]
        self._record("SELL", symbol, qty, price, fee, note)

    def equity(self, prices: dict) -> float:
        return self.cash + sum(q * prices.get(s, 0.0) for s, q in self.positions.items())
