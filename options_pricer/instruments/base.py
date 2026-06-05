from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


@dataclass
class Option:
    option_type: OptionType

    def __post_init__(self) -> None:
        self.option_type = OptionType(self.option_type)
