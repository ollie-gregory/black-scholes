from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HestonModel:
    """Heston (1993) stochastic volatility model.

    The variance follows a mean-reverting CIR process:
        dv_t = kappa * (theta - v_t) dt + xi * sqrt(v_t) dW_t^v

    with correlation rho between the asset and variance Brownian motions.
    """

    v0: float     # initial variance (sigma_0^2)
    kappa: float  # mean-reversion speed
    theta: float  # long-run variance
    xi: float     # vol of vol
    rho: float    # correlation between asset and variance processes

    def __post_init__(self) -> None:
        raise NotImplementedError("HestonModel is not yet implemented")
