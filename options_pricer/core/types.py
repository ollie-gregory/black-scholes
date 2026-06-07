from __future__ import annotations

from typing import NamedTuple


class Greeks(NamedTuple):
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
