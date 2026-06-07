from __future__ import annotations

from dataclasses import dataclass

from ..equity.european import EuropeanOption
from ..equity.american import AmericanOption


@dataclass
class CommodityEuropeanOption(EuropeanOption):
    """European-style commodity option. Use with CommodityMarketData."""


@dataclass
class CommodityAmericanOption(AmericanOption):
    """American-style commodity option. Use with CommodityMarketData."""
