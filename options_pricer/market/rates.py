from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RatesMarketData:
    """Interest rate market data — stub for future implementation.

    Will hold a yield curve or short-rate model parameters when rates
    option pricing (swaptions, bond options) is implemented.
    """

    def __post_init__(self) -> None:
        raise NotImplementedError("RatesMarketData is not yet implemented")
