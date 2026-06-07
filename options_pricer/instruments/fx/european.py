from __future__ import annotations

from dataclasses import dataclass

from ..equity.european import EuropeanOption


@dataclass
class FXEuropeanOption(EuropeanOption):
    currency_pair: str = ""  # e.g. "EURUSD" — for display/docs only
