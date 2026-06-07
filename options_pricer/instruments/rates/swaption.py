from __future__ import annotations

from dataclasses import dataclass

from ...core.instrument import OptionType


@dataclass
class Swaption:
    """Interest rate swaption — stub for future implementation.

    The option_type (payer/receiver) is encoded as CALL (payer) and PUT (receiver)
    by convention. Strike is a rate, not a price, so Swaption does not inherit
    from VanillaOption.
    """

    option_type: OptionType   # CALL = payer swaption, PUT = receiver swaption
    strike_rate: float        # fixed rate strike (e.g. 0.05 for 5%)
    expiry: float             # option expiry in years
    tenor: float              # underlying swap tenor in years

    def __post_init__(self) -> None:
        raise NotImplementedError("Swaption pricing is not yet implemented")
