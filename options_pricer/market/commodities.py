from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CommodityMarketData:
    """Commodity market data.

    Under Black-Scholes, a commodity option with convenience yield and storage cost
    is equivalent to an equity option where div_yield = convenience_yield - storage_cost.
    The div_yield property implements this mapping so all existing pricers work
    without modification.
    """

    spot: float                  # current commodity spot price
    rate: float                  # risk-free rate, annualised continuous compounding
    convenience_yield: float = 0.0
    storage_cost: float = 0.0

    @property
    def div_yield(self) -> float:
        """Net carry cost: convenience_yield minus storage_cost."""
        return self.convenience_yield - self.storage_cost

    def __post_init__(self) -> None:
        if self.spot <= 0:
            raise ValueError(f"spot must be positive, got {self.spot}")
