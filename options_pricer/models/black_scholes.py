from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BlackScholesModel:
    """Black-Scholes model: geometric Brownian motion with constant volatility."""

    vol: float  # annualised volatility (sigma), e.g. 0.20 for 20%

    def __post_init__(self) -> None:
        if self.vol <= 0:
            raise ValueError(f"vol must be positive, got {self.vol}")
