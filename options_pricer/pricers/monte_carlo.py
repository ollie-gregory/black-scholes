from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core.instrument import OptionType, VanillaInstrument
from ..core.model import MarketDataLike, VolatilityModel
from ..core.types import Greeks
from ..instruments.equity.american import AmericanOption
from ..market.equity import EquityMarketData
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
        inst: VanillaInstrument,
        md: MarketDataLike,
        model: VolatilityModel,
    ) -> float:
        if isinstance(inst, AmericanOption):
            return self._price_american(inst, md, model)
        return self._price_european(inst, md, model)

    def _price_european(
        self,
        inst: VanillaInstrument,
        md: MarketDataLike,
        model: VolatilityModel,
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
        inst: VanillaInstrument,
        md: MarketDataLike,
        model: VolatilityModel,
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

    def greeks(
        self,
        inst: VanillaInstrument,
        md: MarketDataLike,
        model: VolatilityModel,
    ) -> Greeks:
        """Estimate Greeks via central finite differences with common random numbers.

        A fixed seed ensures each bumped reprice uses identical random paths,
        so Monte Carlo noise largely cancels in the finite-difference ratios.
        """
        h_s = md.spot * 0.01  # 1 % spot bump
        h_v = 0.01            # 1 vol-point bump
        h_r = 0.001           # 10 bp rate bump

        def _price(
            spot: float = md.spot,
            rate: float = md.rate,
            vol: float = model.vol,
            div_yield: float = md.div_yield,
            expiry: float = inst.expiry,
        ) -> float:
            return self.price(
                inst.__class__(option_type=inst.option_type, strike=inst.strike, expiry=expiry),
                EquityMarketData(spot=spot, rate=rate, div_yield=div_yield),
                BlackScholesModel(vol=vol),
            )

        p0 = _price()
        p_s_up = _price(spot=md.spot + h_s)
        p_s_dn = _price(spot=md.spot - h_s)
        p_v_up = _price(vol=model.vol + h_v)
        p_v_dn = _price(vol=model.vol - h_v)
        p_r_up = _price(rate=md.rate + h_r)
        p_r_dn = _price(rate=md.rate - h_r)
        p_t_dn = _price(expiry=inst.expiry - 1.0 / 365)

        delta = (p_s_up - p_s_dn) / (2 * h_s)
        gamma = (p_s_up - 2 * p0 + p_s_dn) / h_s**2
        vega = (p_v_up - p_v_dn) / (2 * h_v) / 100   # per 1 vol point
        rho = (p_r_up - p_r_dn) / (2 * h_r) / 100    # per 1 bp
        theta = p_t_dn - p0                          # per calendar day

        return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
