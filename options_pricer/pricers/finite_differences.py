from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

from ..instruments.american import AmericanOption
from ..instruments.base import OptionType
from ..instruments.european import EuropeanOption
from ..market.market_data import MarketData
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

    def price(
        self,
        inst: EuropeanOption | AmericanOption,
        md: MarketData,
        model: BlackScholesModel,
    ) -> float:
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

        return float(np.interp(md.spot, s, v))
