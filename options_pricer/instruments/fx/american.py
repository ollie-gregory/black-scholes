from __future__ import annotations

from dataclasses import dataclass

from ..equity.american import AmericanOption


@dataclass
class FXAmericanOption(AmericanOption):
    currency_pair: str = ""  # e.g. "EURUSD" — for display/docs only
