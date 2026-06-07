from __future__ import annotations

import math

import pytest

from options_pricer import (
    BlackScholesModel,
    ClosedFormPricer,
    EuropeanOption,
    MarketData,
    OptionType,
)

# Standard ATM reference: S=100, K=100, T=1yr, r=5%, σ=20%, q=0
ATM_MD = MarketData(spot=100.0, rate=0.05)
BS = BlackScholesModel(vol=0.20)
ATM_CALL = EuropeanOption(option_type=OptionType.CALL, strike=100.0, expiry=1.0)
ATM_PUT = EuropeanOption(option_type=OptionType.PUT, strike=100.0, expiry=1.0)


class TestPrice:
    def test_call_price_atm(self):
        assert ClosedFormPricer.price(ATM_CALL, ATM_MD, BS) == pytest.approx(10.4506, abs=1e-3)

    def test_put_price_atm(self):
        assert ClosedFormPricer.price(ATM_PUT, ATM_MD, BS) == pytest.approx(5.5735, abs=1e-3)

    def test_put_call_parity(self):
        c = ClosedFormPricer.price(ATM_CALL, ATM_MD, BS)
        p = ClosedFormPricer.price(ATM_PUT, ATM_MD, BS)
        forward = ATM_MD.spot - ATM_CALL.strike * math.exp(-ATM_MD.rate * ATM_CALL.expiry)
        assert c - p == pytest.approx(forward, abs=1e-6)

    def test_call_price_with_dividend(self):
        md_div = MarketData(spot=100.0, rate=0.05, div_yield=0.02)
        c_no_div = ClosedFormPricer.price(ATM_CALL, ATM_MD, BS)
        c_with_div = ClosedFormPricer.price(ATM_CALL, md_div, BS)
        assert c_with_div < c_no_div

    def test_deep_itm_call_price_positive(self):
        deep_itm = EuropeanOption(option_type=OptionType.CALL, strike=50.0, expiry=1.0)
        assert ClosedFormPricer.price(deep_itm, ATM_MD, BS) > 0

    def test_string_option_type_accepted(self):
        opt = EuropeanOption(option_type="call", strike=100.0, expiry=1.0)
        assert ClosedFormPricer.price(opt, ATM_MD, BS) == pytest.approx(
            ClosedFormPricer.price(ATM_CALL, ATM_MD, BS), rel=1e-9
        )


class TestGreeks:
    def test_call_delta_atm(self):
        g = ClosedFormPricer.greeks(ATM_CALL, ATM_MD, BS)
        assert g.delta == pytest.approx(0.6368, abs=1e-3)

    def test_put_delta_atm(self):
        g = ClosedFormPricer.greeks(ATM_PUT, ATM_MD, BS)
        assert g.delta == pytest.approx(-0.3632, abs=1e-3)

    def test_call_delta_bounds(self):
        g = ClosedFormPricer.greeks(ATM_CALL, ATM_MD, BS)
        assert 0.0 < g.delta < 1.0

    def test_put_delta_bounds(self):
        g = ClosedFormPricer.greeks(ATM_PUT, ATM_MD, BS)
        assert -1.0 < g.delta < 0.0

    def test_call_put_delta_sum(self):
        gc = ClosedFormPricer.greeks(ATM_CALL, ATM_MD, BS)
        gp = ClosedFormPricer.greeks(ATM_PUT, ATM_MD, BS)
        assert gc.delta - gp.delta == pytest.approx(1.0, abs=1e-9)

    def test_gamma_positive(self):
        assert ClosedFormPricer.greeks(ATM_CALL, ATM_MD, BS).gamma > 0
        assert ClosedFormPricer.greeks(ATM_PUT, ATM_MD, BS).gamma > 0

    def test_call_put_same_gamma(self):
        gc = ClosedFormPricer.greeks(ATM_CALL, ATM_MD, BS)
        gp = ClosedFormPricer.greeks(ATM_PUT, ATM_MD, BS)
        assert gc.gamma == pytest.approx(gp.gamma, rel=1e-9)

    def test_vega_positive(self):
        assert ClosedFormPricer.greeks(ATM_CALL, ATM_MD, BS).vega > 0
        assert ClosedFormPricer.greeks(ATM_PUT, ATM_MD, BS).vega > 0

    def test_call_put_same_vega(self):
        gc = ClosedFormPricer.greeks(ATM_CALL, ATM_MD, BS)
        gp = ClosedFormPricer.greeks(ATM_PUT, ATM_MD, BS)
        assert gc.vega == pytest.approx(gp.vega, rel=1e-9)

    def test_call_theta_negative(self):
        assert ClosedFormPricer.greeks(ATM_CALL, ATM_MD, BS).theta < 0

    def test_put_theta_negative(self):
        assert ClosedFormPricer.greeks(ATM_PUT, ATM_MD, BS).theta < 0

    def test_call_rho_positive(self):
        assert ClosedFormPricer.greeks(ATM_CALL, ATM_MD, BS).rho > 0

    def test_put_rho_negative(self):
        assert ClosedFormPricer.greeks(ATM_PUT, ATM_MD, BS).rho < 0

    def test_greeks_returns_named_tuple(self):
        g = ClosedFormPricer.greeks(ATM_CALL, ATM_MD, BS)
        for field in ("delta", "gamma", "vega", "theta", "rho"):
            assert hasattr(g, field)


class TestImpliedVol:
    def test_roundtrip_call(self):
        price = ClosedFormPricer.price(ATM_CALL, ATM_MD, BS)
        iv = ClosedFormPricer.implied_vol(ATM_CALL, ATM_MD, market_price=price)
        assert iv == pytest.approx(0.20, abs=1e-6)

    def test_roundtrip_put(self):
        price = ClosedFormPricer.price(ATM_PUT, ATM_MD, BS)
        iv = ClosedFormPricer.implied_vol(ATM_PUT, ATM_MD, market_price=price)
        assert iv == pytest.approx(0.20, abs=1e-6)

    def test_roundtrip_high_vol(self):
        model_50 = BlackScholesModel(vol=0.50)
        price = ClosedFormPricer.price(ATM_CALL, ATM_MD, model_50)
        iv = ClosedFormPricer.implied_vol(ATM_CALL, ATM_MD, market_price=price)
        assert iv == pytest.approx(0.50, abs=1e-6)

    def test_higher_price_implies_higher_vol(self):
        price = ClosedFormPricer.price(ATM_CALL, ATM_MD, BS)
        iv = ClosedFormPricer.implied_vol(ATM_CALL, ATM_MD, market_price=price * 1.1)
        assert iv > 0.20

    def test_roundtrip_with_dividend(self):
        md_div = MarketData(spot=100.0, rate=0.05, div_yield=0.02)
        model_25 = BlackScholesModel(vol=0.25)
        price = ClosedFormPricer.price(ATM_CALL, md_div, model_25)
        iv = ClosedFormPricer.implied_vol(ATM_CALL, md_div, market_price=price)
        assert iv == pytest.approx(0.25, abs=1e-6)


class TestFXClosedForm:
    """Verify Garman-Kohlhagen pricing via FXMarketData + existing ClosedFormPricer."""

    FX_MD = None  # set below after import

    def _md(self):
        from options_pricer import FXMarketData
        return FXMarketData(spot=1.08, domestic_rate=0.05, foreign_rate=0.03)

    def _opt(self, option_type="call"):
        from options_pricer import FXEuropeanOption
        return FXEuropeanOption(option_type=option_type, strike=1.08, expiry=1.0, currency_pair="EURUSD")

    def _bs(self):
        return BlackScholesModel(vol=0.10)

    def test_fx_call_price_positive(self):
        price = ClosedFormPricer.price(self._opt("call"), self._md(), self._bs())
        assert price > 0

    def test_fx_put_price_positive(self):
        price = ClosedFormPricer.price(self._opt("put"), self._md(), self._bs())
        assert price > 0

    def test_fx_put_call_parity(self):
        import math
        md = self._md()
        bs = self._bs()
        c = ClosedFormPricer.price(self._opt("call"), md, bs)
        p = ClosedFormPricer.price(self._opt("put"), md, bs)
        # GK put-call parity: C - P = S * e^(-r_f*T) - K * e^(-r_d*T)
        expected = md.spot * math.exp(-md.foreign_rate * 1.0) - self._opt().strike * math.exp(-md.domestic_rate * 1.0)
        assert c - p == pytest.approx(expected, abs=1e-9)

    def test_fx_zero_foreign_rate_equals_equity(self):
        """FX option with r_f=0 must equal equity option with div_yield=0."""
        from options_pricer import FXMarketData, FXEuropeanOption
        fx_opt = FXEuropeanOption(option_type="call", strike=100.0, expiry=1.0)
        fx_md = FXMarketData(spot=100.0, domestic_rate=0.05, foreign_rate=0.0)
        bs = BlackScholesModel(vol=0.20)

        fx_price = ClosedFormPricer.price(fx_opt, fx_md, bs)
        eq_price = ClosedFormPricer.price(ATM_CALL, ATM_MD, bs)
        assert fx_price == pytest.approx(eq_price, rel=1e-9)

    def test_fx_call_approx_gk_analytic(self):
        """S=1.08, K=1.08, T=1, r_d=5%, r_f=3%, σ=10% → GK call ≈ 0.0524."""
        price = ClosedFormPricer.price(self._opt("call"), self._md(), self._bs())
        assert price == pytest.approx(0.0524, abs=5e-4)

    def test_fx_greeks_delta_bounds(self):
        g = ClosedFormPricer.greeks(self._opt("call"), self._md(), self._bs())
        assert 0.0 < g.delta < 1.0

    def test_fx_implied_vol_roundtrip(self):
        opt = self._opt("call")
        md = self._md()
        bs = self._bs()
        price = ClosedFormPricer.price(opt, md, bs)
        iv = ClosedFormPricer.implied_vol(opt, md, market_price=price)
        assert iv == pytest.approx(0.10, abs=1e-6)


class TestValidation:
    def test_negative_strike_raises(self):
        with pytest.raises(ValueError, match="strike"):
            EuropeanOption(option_type=OptionType.CALL, strike=-1.0, expiry=1.0)

    def test_zero_strike_raises(self):
        with pytest.raises(ValueError, match="strike"):
            EuropeanOption(option_type=OptionType.CALL, strike=0.0, expiry=1.0)

    def test_zero_expiry_raises(self):
        with pytest.raises(ValueError, match="expiry"):
            EuropeanOption(option_type=OptionType.CALL, strike=100.0, expiry=0.0)

    def test_negative_expiry_raises(self):
        with pytest.raises(ValueError, match="expiry"):
            EuropeanOption(option_type=OptionType.CALL, strike=100.0, expiry=-1.0)

    def test_invalid_option_type_raises(self):
        with pytest.raises(ValueError):
            EuropeanOption(option_type="invalid", strike=100.0, expiry=1.0)
