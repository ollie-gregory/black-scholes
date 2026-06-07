from __future__ import annotations

import math

from scipy.stats import norm

from ..core.instrument import OptionType, VanillaInstrument
from ..core.model import MarketDataLike, VolatilityModel
from ..core.types import Greeks
from ..models.black_scholes import BlackScholesModel


class ClosedFormPricer:
    """Closed-form Black-Scholes-Merton pricer for European options."""

    @staticmethod
    def _d1d2(
        inst: VanillaInstrument,
        md: MarketDataLike,
        model: VolatilityModel,
    ) -> tuple[float, float]:
        sqrt_t = math.sqrt(inst.expiry)
        d1 = (
            math.log(md.spot / inst.strike)
            + (md.rate - md.div_yield + 0.5 * model.vol**2) * inst.expiry
        ) / (model.vol * sqrt_t)
        return d1, d1 - model.vol * sqrt_t

    @classmethod
    def price(
        cls,
        inst: VanillaInstrument,
        md: MarketDataLike,
        model: VolatilityModel,
    ) -> float:
        d1, d2 = cls._d1d2(inst, md, model)
        df_r = math.exp(-md.rate * inst.expiry)
        df_q = math.exp(-md.div_yield * inst.expiry)

        if inst.option_type is OptionType.CALL:
            return md.spot * df_q * norm.cdf(d1) - inst.strike * df_r * norm.cdf(d2)
        else:
            return inst.strike * df_r * norm.cdf(-d2) - md.spot * df_q * norm.cdf(-d1)

    @classmethod
    def greeks(
        cls,
        inst: VanillaInstrument,
        md: MarketDataLike,
        model: VolatilityModel,
    ) -> Greeks:
        d1, d2 = cls._d1d2(inst, md, model)
        sqrt_t = math.sqrt(inst.expiry)
        df_r = math.exp(-md.rate * inst.expiry)
        df_q = math.exp(-md.div_yield * inst.expiry)

        gamma = df_q * norm.pdf(d1) / (md.spot * model.vol * sqrt_t)
        vega = md.spot * df_q * norm.pdf(d1) * sqrt_t / 100  # per 1 vol point

        if inst.option_type is OptionType.CALL:
            delta = df_q * norm.cdf(d1)
            theta = (
                -md.spot * df_q * norm.pdf(d1) * model.vol / (2 * sqrt_t)
                - md.rate * inst.strike * df_r * norm.cdf(d2)
                + md.div_yield * md.spot * df_q * norm.cdf(d1)
            ) / 365
            rho = inst.strike * inst.expiry * df_r * norm.cdf(d2) / 100  # per 1 bp
        else:
            delta = df_q * (norm.cdf(d1) - 1)
            theta = (
                -md.spot * df_q * norm.pdf(d1) * model.vol / (2 * sqrt_t)
                + md.rate * inst.strike * df_r * norm.cdf(-d2)
                - md.div_yield * md.spot * df_q * norm.cdf(-d1)
            ) / 365
            rho = -inst.strike * inst.expiry * df_r * norm.cdf(-d2) / 100  # per 1 bp

        return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)

    @classmethod
    def implied_vol(
        cls,
        inst: VanillaInstrument,
        md: MarketDataLike,
        market_price: float,
        tol: float = 1e-6,
        max_iter: int = 200,
    ) -> float:
        """Invert the BS formula via Newton-Raphson to recover implied volatility."""
        sigma = math.sqrt(2 * math.pi / inst.expiry) * market_price / md.spot

        if sigma <= 0:
            raise ValueError("market_price must be positive for implied vol calculation")

        for _ in range(max_iter):
            trial_model = BlackScholesModel(vol=sigma)
            trial_price = cls.price(inst, md, trial_model)
            trial_vega = cls.greeks(inst, md, trial_model).vega * 100  # undo /100

            diff = trial_price - market_price
            if abs(diff) < tol:
                return sigma

            if trial_vega < 1e-10:
                raise ValueError("Vega near zero — Newton-Raphson cannot converge")

            sigma -= diff / trial_vega

            if sigma <= 0:
                raise ValueError("Implied vol stepped below zero")

        raise ValueError(f"Implied vol did not converge within {max_iter} iterations")
