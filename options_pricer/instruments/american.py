from __future__ import annotations

from dataclasses import dataclass

from .base import Option


@dataclass
class AmericanOption(Option):
    strike: float
    expiry: float  # time to expiry in years

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.strike <= 0:
            raise ValueError(f"strike must be positive, got {self.strike}")
        if self.expiry <= 0:
            raise ValueError(f"expiry must be positive, got {self.expiry}")
