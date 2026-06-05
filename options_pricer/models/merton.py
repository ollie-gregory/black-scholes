from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MertonModel:
    """Merton (1976) jump-diffusion model.

    The asset follows GBM plus a compound Poisson jump process:
        dS_t/S_t = (mu - lambda * kbar) dt + sigma dW_t + (J - 1) dN_t

    where N_t is a Poisson process with intensity lambda and J is the jump size
    (log-normally distributed with mean jump_mean and vol jump_vol).
    """

    vol: float            # diffusion volatility (sigma)
    jump_intensity: float  # Poisson jump arrival rate (lambda), jumps per year
    jump_mean: float      # mean of log-jump size (mu_J)
    jump_vol: float       # volatility of log-jump size (sigma_J)

    def __post_init__(self) -> None:
        raise NotImplementedError("MertonModel is not yet implemented")
