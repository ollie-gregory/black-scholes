from .core.types import Greeks
from .core.instrument import OptionType
from .instruments.equity.base import VanillaOption
from .instruments.equity.american import AmericanOption
from .instruments.equity.european import EuropeanOption
from .instruments.fx.european import FXEuropeanOption
from .instruments.fx.american import FXAmericanOption
from .market.equity import EquityMarketData
from .market.fx import FXMarketData
from .market.commodities import CommodityMarketData
from .models.black_scholes import BlackScholesModel
from .models.heston import HestonModel
from .models.merton import MertonModel
from .pricers.closed_form import ClosedFormPricer
from .pricers.finite_differences import FiniteDifferencesPricer
from .pricers.monte_carlo import MonteCarloPricer
from .pricers.tree import TreePricer

# Backward-compat aliases — never removed
Option = VanillaOption
MarketData = EquityMarketData

__all__ = [
    # core types
    "Greeks",
    "OptionType",
    # equity instruments
    "VanillaOption",
    "Option",          # alias for VanillaOption
    "EuropeanOption",
    "AmericanOption",
    # fx instruments
    "FXEuropeanOption",
    "FXAmericanOption",
    # market data
    "EquityMarketData",
    "MarketData",      # alias for EquityMarketData
    "FXMarketData",
    "CommodityMarketData",
    # models
    "BlackScholesModel",
    "HestonModel",
    "MertonModel",
    # pricers
    "ClosedFormPricer",
    "FiniteDifferencesPricer",
    "MonteCarloPricer",
    "TreePricer",
]
