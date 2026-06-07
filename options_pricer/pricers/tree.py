from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..instruments.american import AmericanOption
from ..instruments.base import OptionType
from ..instruments.european import EuropeanOption
from ..market.market_data import MarketData
from ..models.black_scholes import BlackScholesModel
from ..types import Greeks


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

    def greeks(
        self,
        inst: EuropeanOption | AmericanOption,
        md: MarketData,
        model: BlackScholesModel,
        d_vol: float = 0.001,
        d_rate: float = 1e-4,
    ) -> Greeks:
        N = self.n_steps
        dt = inst.expiry / N
        u = math.exp(model.vol * math.sqrt(dt))
        d = 1.0 / u
        p = (math.exp((md.rate - md.div_yield) * dt) - d) / (u - d)
        disc = math.exp(-md.rate * dt)

        j = np.arange(N + 1)
        s_t = md.spot * (u**j) * (d ** (N - j))

        if inst.option_type is OptionType.CALL:
            v = np.maximum(s_t - inst.strike, 0.0)
        else:
            v = np.maximum(inst.strike - s_t, 0.0)

        v1 = v2 = None
        for k in range(1, N + 1):
            v = disc * (p * v[1:] + (1 - p) * v[:-1])
            if isinstance(inst, AmericanOption):
                step = N - k
                j = np.arange(step + 1)
                s_nodes = md.spot * (u**j) * (d ** (step - j))
                if inst.option_type is OptionType.CALL:
                    v = np.maximum(v, s_nodes - inst.strike)
                else:
                    v = np.maximum(v, inst.strike - s_nodes)
            if k == N - 2:
                v2 = v.copy()
            elif k == N - 1:
                v1 = v.copy()

        v0 = float(v[0])

        # Delta: (V_u - V_d) / (S_u - S_d)
        delta = (float(v1[1]) - float(v1[0])) / (md.spot * (u - d))

        # Gamma: second derivative from step-2 nodes (u*d = 1, so S_ud = S)
        delta_u = (float(v2[2]) - float(v2[1])) / (md.spot * (u**2 - 1))
        delta_d = (float(v2[1]) - float(v2[0])) / (md.spot * (1 - d**2))
        gamma = (delta_u - delta_d) / (0.5 * md.spot * (u**2 - d**2))

        # Theta: same spot (S_ud = S since u*d=1), 2dt elapsed; per calendar day
        theta = (float(v2[1]) - v0) / (2 * dt) / 365

        # Vega: central difference, per 1 vol point
        vega = (
            self.price(inst, md, BlackScholesModel(vol=model.vol + d_vol))
            - self.price(inst, md, BlackScholesModel(vol=model.vol - d_vol))
        ) / (2 * d_vol) / 100

        # Rho: central difference, per 1 bp
        md_up = MarketData(spot=md.spot, rate=md.rate + d_rate, div_yield=md.div_yield)
        md_dn = MarketData(spot=md.spot, rate=md.rate - d_rate, div_yield=md.div_yield)
        rho = (self.price(inst, md_up, model) - self.price(inst, md_dn, model)) / (2 * d_rate) / 100

        return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
