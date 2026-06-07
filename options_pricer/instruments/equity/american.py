from __future__ import annotations

from dataclasses import dataclass

from .base import VanillaOption


@dataclass
class AmericanOption(VanillaOption):
    pass
