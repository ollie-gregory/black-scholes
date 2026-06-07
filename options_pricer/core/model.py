from __future__ import annotations

from typing import Protocol


class VolatilityModel(Protocol):
    vol: float


class MarketDataLike(Protocol):
    spot: float
    rate: float
    div_yield: float
