from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..instruments.base import OptionType
from ..instruments.european import EuropeanOption
from ..market.market_data import MarketData
from ..models.black_scholes import BlackScholesModel


@dataclass
class MonteCarloPricer:
    """Monte Carlo pricer using terminal-value sampling under risk-neutral GBM."""

    n_paths: int = 100_000
    seed: int | None = None

    def price(
        self,
        inst: EuropeanOption,
        md: MarketData,
        model: BlackScholesModel,
    ) -> float:
        rng = np.random.default_rng(self.seed)
        z = rng.standard_normal(self.n_paths)

        # Risk-neutral terminal stock price: S_T = S * exp((r-q-σ²/2)T + σ√T·Z)
        s_t = md.spot * np.exp(
            (md.rate - md.div_yield - 0.5 * model.vol**2) * inst.expiry
            + model.vol * math.sqrt(inst.expiry) * z
        )

        if inst.option_type is OptionType.CALL:
            payoffs = np.maximum(s_t - inst.strike, 0.0)
        else:
            payoffs = np.maximum(inst.strike - s_t, 0.0)

        return float(math.exp(-md.rate * inst.expiry) * np.mean(payoffs))
