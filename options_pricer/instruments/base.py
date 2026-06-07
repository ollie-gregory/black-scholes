from __future__ import annotations

from dataclasses import dataclass

from ..core.instrument import OptionType


@dataclass
class VanillaOption:
    option_type: OptionType
    strike: float
    expiry: float  # time to expiry in years

    def __post_init__(self) -> None:
        self.option_type = OptionType(self.option_type)
        if self.strike <= 0:
            raise ValueError(f"strike must be positive, got {self.strike}")
        if self.expiry <= 0:
            raise ValueError(f"expiry must be positive, got {self.expiry}")
