from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MarketData:
    spot: float       # current underlying price (S)
    rate: float       # risk-free rate, annualised continuous compounding (r)
    vol: float        # implied volatility, annualised (sigma)
    div_yield: float = 0.0  # continuous dividend yield (q)

    def __post_init__(self) -> None:
        if self.spot <= 0:
            raise ValueError(f"spot must be positive, got {self.spot}")
        if self.vol <= 0:
            raise ValueError(f"vol must be positive, got {self.vol}")
