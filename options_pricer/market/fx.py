from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FXMarketData:
    spot: float           # spot exchange rate (e.g. 1.08 for EURUSD)
    domestic_rate: float  # risk-free rate in pricing currency
    foreign_rate: float   # risk-free rate in foreign currency

    @property
    def rate(self) -> float:
        """Domestic rate — maps to the r parameter in Garman-Kohlhagen."""
        return self.domestic_rate

    @property
    def div_yield(self) -> float:
        """Foreign rate — acts as the continuous yield in Garman-Kohlhagen."""
        return self.foreign_rate

    def __post_init__(self) -> None:
        if self.spot <= 0:
            raise ValueError(f"spot must be positive, got {self.spot}")
