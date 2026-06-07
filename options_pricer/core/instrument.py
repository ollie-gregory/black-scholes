from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


@runtime_checkable
class VanillaInstrument(Protocol):
    option_type: OptionType
    strike: float
    expiry: float
