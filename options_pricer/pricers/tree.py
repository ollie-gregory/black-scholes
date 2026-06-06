from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..instruments.american import AmericanOption
from ..instruments.base import OptionType
from ..instruments.european import EuropeanOption
from ..market.market_data import MarketData
from ..models.black_scholes import BlackScholesModel


@dataclass
class TreePricer:
    """Cox-Ross-Rubinstein binomial tree pricer."""

    n_steps: int = 200

    def price(
        self,
        inst: EuropeanOption | AmericanOption,
        md: MarketData,
        model: BlackScholesModel,
    ) -> float:
        N = self.n_steps
        dt = inst.expiry / N

        # CRR up/down factors and risk-neutral probability
        u = math.exp(model.vol * math.sqrt(dt))
        d = 1.0 / u
        p = (math.exp((md.rate - md.div_yield) * dt) - d) / (u - d)
        disc = math.exp(-md.rate * dt)

        # Terminal asset prices: S_T[j] = S * u^j * d^(N-j), j = 0..N
        j = np.arange(N + 1)
        s_t = md.spot * (u**j) * (d ** (N - j))

        if inst.option_type is OptionType.CALL:
            v = np.maximum(s_t - inst.strike, 0.0)
        else:
            v = np.maximum(inst.strike - s_t, 0.0)

        # Backward induction: reduce N+1 values to 1 over N steps.
        # For American options, apply early exercise at each level.
        for k in range(1, N + 1):
            v = disc * (p * v[1:] + (1 - p) * v[:-1])
            if isinstance(inst, AmericanOption):
                step = N - k  # number of steps from the root
                j = np.arange(step + 1)
                s_nodes = md.spot * (u**j) * (d ** (step - j))
                if inst.option_type is OptionType.CALL:
                    v = np.maximum(v, s_nodes - inst.strike)
                else:
                    v = np.maximum(v, inst.strike - s_nodes)

        return float(v[0])
