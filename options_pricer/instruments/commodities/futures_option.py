from __future__ import annotations

from dataclasses import dataclass

from ..equity.base import VanillaOption


@dataclass
class CommodityOption(VanillaOption):
    """Commodity futures option — stub for future implementation.

    Pricing uses CommodityMarketData where div_yield = convenience_yield - storage_cost,
    making it structurally identical to an equity option under Black-Scholes.
    """
