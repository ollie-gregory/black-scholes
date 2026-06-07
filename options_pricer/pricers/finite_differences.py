from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

from ..core.instrument import OptionType, VanillaInstrument
from ..core.model import MarketDataLike, VolatilityModel
from ..core.types import Greeks
from ..instruments.equity.american import AmericanOption
from ..instruments.equity.european import EuropeanOption
from ..market.equity import EquityMarketData
from ..models.black_scholes import BlackScholesModel


@dataclass
class FiniteDifferencesPricer:
    """Crank-Nicolson finite differences pricer.

    Solves the Black-Scholes PDE on a uniform (S, τ) grid using the
    Crank-Nicolson scheme. For American options, the early exercise
    constraint is enforced after each time step via a max-projection
    against the intrinsic value (operator-splitting approach).
    """

    n_s: int = 200          # number of interior S grid points
    n_t: int = 200          # number of backward time steps
    s_max_factor: float = 3.0  # S_max = s_max_factor * max(spot, strike)

    def _solve(
        self,
        inst: VanillaInstrument,
        md: MarketDataLike,
        model: VolatilityModel,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run the Crank-Nicolson solver. Returns (s_grid, v_grid)."""
        M = self.n_s
        r, q = md.rate, md.div_yield
        sigma = model.vol
        K = inst.strike
        T = inst.expiry

        S_max = self.s_max_factor * max(md.spot, K)
        dS = S_max / (M + 1)
        s = dS * np.arange(1, M + 1)   # M interior S nodes
        dtau = T / self.n_t

        # BS PDE operator L coefficients at each interior node:
        #   L[V]_j = a_j * V_{j-1} + b_j * V_j + c_j * V_{j+1}
        alpha = 0.5 * sigma**2 * s**2 / dS**2
        beta = (r - q) * s / (2.0 * dS)
        a = alpha - beta    # sub-diagonal
        b = -2.0 * alpha - r  # main diagonal
        c = alpha + beta    # super-diagonal

        # Crank-Nicolson LHS matrix A = I - dtau/2 * L, stored in banded format
        # ab[0, 1:]  = upper diagonal of A
        # ab[1, :]   = main diagonal of A
        # ab[2, :-1] = lower diagonal of A
        ab = np.zeros((3, M))
        ab[0, 1:] = -0.5 * dtau * c[:-1]   # upper (c_j terms)
        ab[1, :] = 1.0 - 0.5 * dtau * b    # main
        ab[2, :-1] = -0.5 * dtau * a[1:]   # lower (a_j terms)

        # RHS multipliers for explicit side: B = I + dtau/2 * L
        rhs_main = 1.0 + 0.5 * dtau * b
        rhs_upper = 0.5 * dtau * c[:-1]
        rhs_lower = 0.5 * dtau * a[1:]

        # Terminal condition at τ=0 (option just expired)
        if inst.option_type is OptionType.CALL:
            v = np.maximum(s - K, 0.0)
        else:
            v = np.maximum(K - s, 0.0)

        def boundary(tau: float) -> tuple[float, float]:
            if inst.option_type is OptionType.CALL:
                v_left = 0.0
                v_right = S_max - K * math.exp(-r * tau)
            else:
                # American put: exercise immediately at S=0 gives K
                # European put: present value of K
                if isinstance(inst, AmericanOption):
                    v_left = K
                else:
                    v_left = K * math.exp(-r * tau)
                v_right = 0.0
            return v_left, v_right

        if inst.option_type is OptionType.CALL:
            intrinsic = np.maximum(s - K, 0.0)
        else:
            intrinsic = np.maximum(K - s, 0.0)

        v_left_prev, v_right_prev = boundary(0.0)

        for step in range(self.n_t):
            tau_new = (step + 1) * dtau
            v_left_new, v_right_new = boundary(tau_new)

            # Build RHS via tridiagonal matvec of B
            rhs = rhs_main * v
            rhs[1:] += rhs_lower * v[:-1]
            rhs[:-1] += rhs_upper * v[1:]

            # Boundary corrections from both explicit and implicit sides
            rhs[0] += 0.5 * dtau * a[0] * (v_left_prev + v_left_new)
            rhs[-1] += 0.5 * dtau * c[-1] * (v_right_prev + v_right_new)

            v = solve_banded((1, 1), ab, rhs)

            if isinstance(inst, AmericanOption):
                v = np.maximum(v, intrinsic)

            v_left_prev, v_right_prev = v_left_new, v_right_new

        return s, v

    def price(
        self,
        inst: VanillaInstrument,
        md: MarketDataLike,
        model: VolatilityModel,
    ) -> float:
        s, v = self._solve(inst, md, model)
        return float(np.interp(md.spot, s, v))

    def greeks(
        self,
        inst: VanillaInstrument,
        md: MarketDataLike,
        model: VolatilityModel,
    ) -> Greeks:
        S = md.spot
        s, v = self._solve(inst, md, model)
        dS = s[1] - s[0]

        # Delta and Gamma read directly from the solved grid via np.gradient
        dv = np.gradient(v, dS)
        d2v = np.gradient(dv, dS)
        delta = float(np.interp(S, s, dv))
        gamma = float(np.interp(S, s, d2v))

        v_mid = float(np.interp(S, s, v))

        # Vega: central difference in σ; result per 1 vol point (÷100)
        h_vol = 0.001
        v_vol_up = self.price(inst, md, BlackScholesModel(vol=model.vol + h_vol))
        v_vol_dn = self.price(inst, md, BlackScholesModel(vol=model.vol - h_vol))
        vega = (v_vol_up - v_vol_dn) / (2 * h_vol) / 100

        # Theta: 1-day backward difference; result per calendar day
        dt = 1.0 / 365
        if isinstance(inst, AmericanOption):
            inst_short = AmericanOption(
                option_type=inst.option_type, strike=inst.strike, expiry=inst.expiry - dt
            )
        else:
            inst_short = EuropeanOption(
                option_type=inst.option_type, strike=inst.strike, expiry=inst.expiry - dt
            )
        theta = (self.price(inst_short, md, model) - v_mid) / dt / 365

        # Rho: central difference in r; result per 1 pct point (÷100)
        h_r = 0.001
        v_r_up = self.price(inst, EquityMarketData(spot=S, rate=md.rate + h_r, div_yield=md.div_yield), model)
        v_r_dn = self.price(inst, EquityMarketData(spot=S, rate=md.rate - h_r, div_yield=md.div_yield), model)
        rho = (v_r_up - v_r_dn) / (2 * h_r) / 100

        return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
