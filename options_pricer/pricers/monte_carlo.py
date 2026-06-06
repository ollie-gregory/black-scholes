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
class MonteCarloPricer:
    """Monte Carlo pricer.

    European options use exact terminal-value sampling under risk-neutral GBM.
    American options use the Longstaff-Schwartz least-squares Monte Carlo
    algorithm with full path simulation and polynomial regression.
    """

    n_paths: int = 100_000
    n_steps: int = 50  # time steps; used only for American options
    seed: int | None = None

    def price(
        self,
        inst: EuropeanOption | AmericanOption,
        md: MarketData,
        model: BlackScholesModel,
    ) -> float:
        if isinstance(inst, AmericanOption):
            return self._price_american(inst, md, model)
        return self._price_european(inst, md, model)

    def _price_european(
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

    def _price_american(
        self,
        inst: AmericanOption,
        md: MarketData,
        model: BlackScholesModel,
    ) -> float:
        rng = np.random.default_rng(self.seed)
        N = self.n_steps
        dt = inst.expiry / N
        disc = math.exp(-md.rate * dt)

        # Simulate full paths: shape (n_paths, N+1)
        z = rng.standard_normal((self.n_paths, N))
        log_increments = (
            (md.rate - md.div_yield - 0.5 * model.vol**2) * dt
            + model.vol * math.sqrt(dt) * z
        )
        log_paths = np.concatenate(
            [np.zeros((self.n_paths, 1)), np.cumsum(log_increments, axis=1)],
            axis=1,
        )
        paths = md.spot * np.exp(log_paths)  # (n_paths, N+1)

        def payoff(s: np.ndarray) -> np.ndarray:
            if inst.option_type is OptionType.CALL:
                return np.maximum(s - inst.strike, 0.0)
            return np.maximum(inst.strike - s, 0.0)

        # cashflow[i] tracks the (undiscounted) optimal payoff for path i,
        # expressed in dollars at the current backward step.
        cashflow = payoff(paths[:, -1])

        # Backward induction from step N-1 down to 1
        for n in range(N - 1, 0, -1):
            cashflow = cashflow * disc  # discount from step n+1 to step n

            h = payoff(paths[:, n])
            itm = h > 0

            if itm.sum() >= 3:
                x = paths[itm, n]
                y = cashflow[itm]
                # Regress continuation value on [1, S, S²]
                A = np.column_stack([np.ones(len(x)), x, x**2])
                coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                continuation = A @ coeffs

                exercise = h[itm] > continuation
                itm_indices = np.where(itm)[0]
                cashflow[itm_indices[exercise]] = h[itm][exercise]

        cashflow = cashflow * disc  # final discount from step 1 to step 0

        price = float(np.mean(cashflow))

        # No-arbitrage floor: price >= intrinsic(spot)
        intrinsic_spot = float(payoff(np.array([md.spot]))[0])
        return max(price, intrinsic_spot)
