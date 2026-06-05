from __future__ import annotations

import math

import pytest

from options_pricer import BlackScholes, EuropeanOption, MarketData, OptionType

# Standard ATM reference: S=100, K=100, T=1yr, r=5%, sigma=20%, q=0
# d1 = (ln(1) + (0.05 + 0.02) * 1) / 0.20 = 0.35
# d2 = 0.35 - 0.20 = 0.15
ATM_MD = MarketData(spot=100.0, rate=0.05, vol=0.20)
ATM_CALL = EuropeanOption(option_type=OptionType.CALL, strike=100.0, expiry=1.0)
ATM_PUT = EuropeanOption(option_type=OptionType.PUT, strike=100.0, expiry=1.0)


class TestPrice:
    def test_call_price_atm(self):
        assert BlackScholes.price(ATM_CALL, ATM_MD) == pytest.approx(10.4506, abs=1e-3)

    def test_put_price_atm(self):
        assert BlackScholes.price(ATM_PUT, ATM_MD) == pytest.approx(5.5735, abs=1e-3)

    def test_put_call_parity(self):
        c = BlackScholes.price(ATM_CALL, ATM_MD)
        p = BlackScholes.price(ATM_PUT, ATM_MD)
        forward = ATM_MD.spot - ATM_CALL.strike * math.exp(-ATM_MD.rate * ATM_CALL.expiry)
        assert c - p == pytest.approx(forward, abs=1e-6)

    def test_call_price_with_dividend(self):
        md = MarketData(spot=100.0, rate=0.05, vol=0.20, div_yield=0.02)
        c_no_div = BlackScholes.price(ATM_CALL, ATM_MD)
        c_with_div = BlackScholes.price(ATM_CALL, md)
        assert c_with_div < c_no_div

    def test_deep_itm_call_price_positive(self):
        deep_itm = EuropeanOption(option_type=OptionType.CALL, strike=50.0, expiry=1.0)
        assert BlackScholes.price(deep_itm, ATM_MD) > 0

    def test_string_option_type_accepted(self):
        opt = EuropeanOption(option_type="call", strike=100.0, expiry=1.0)
        assert BlackScholes.price(opt, ATM_MD) == pytest.approx(
            BlackScholes.price(ATM_CALL, ATM_MD), rel=1e-9
        )


class TestGreeks:
    def test_call_delta_atm(self):
        g = BlackScholes.greeks(ATM_CALL, ATM_MD)
        assert g.delta == pytest.approx(0.6368, abs=1e-3)

    def test_put_delta_atm(self):
        g = BlackScholes.greeks(ATM_PUT, ATM_MD)
        assert g.delta == pytest.approx(-0.3632, abs=1e-3)

    def test_call_delta_bounds(self):
        g = BlackScholes.greeks(ATM_CALL, ATM_MD)
        assert 0.0 < g.delta < 1.0

    def test_put_delta_bounds(self):
        g = BlackScholes.greeks(ATM_PUT, ATM_MD)
        assert -1.0 < g.delta < 0.0

    def test_call_put_delta_sum(self):
        # For q=0: call_delta - put_delta = 1 (put-call delta parity)
        gc = BlackScholes.greeks(ATM_CALL, ATM_MD)
        gp = BlackScholes.greeks(ATM_PUT, ATM_MD)
        assert gc.delta - gp.delta == pytest.approx(1.0, abs=1e-9)

    def test_gamma_positive(self):
        assert BlackScholes.greeks(ATM_CALL, ATM_MD).gamma > 0
        assert BlackScholes.greeks(ATM_PUT, ATM_MD).gamma > 0

    def test_call_put_same_gamma(self):
        gc = BlackScholes.greeks(ATM_CALL, ATM_MD)
        gp = BlackScholes.greeks(ATM_PUT, ATM_MD)
        assert gc.gamma == pytest.approx(gp.gamma, rel=1e-9)

    def test_vega_positive(self):
        assert BlackScholes.greeks(ATM_CALL, ATM_MD).vega > 0
        assert BlackScholes.greeks(ATM_PUT, ATM_MD).vega > 0

    def test_call_put_same_vega(self):
        gc = BlackScholes.greeks(ATM_CALL, ATM_MD)
        gp = BlackScholes.greeks(ATM_PUT, ATM_MD)
        assert gc.vega == pytest.approx(gp.vega, rel=1e-9)

    def test_call_theta_negative(self):
        assert BlackScholes.greeks(ATM_CALL, ATM_MD).theta < 0

    def test_put_theta_negative(self):
        assert BlackScholes.greeks(ATM_PUT, ATM_MD).theta < 0

    def test_call_rho_positive(self):
        assert BlackScholes.greeks(ATM_CALL, ATM_MD).rho > 0

    def test_put_rho_negative(self):
        assert BlackScholes.greeks(ATM_PUT, ATM_MD).rho < 0

    def test_greeks_returns_named_tuple(self):
        g = BlackScholes.greeks(ATM_CALL, ATM_MD)
        assert hasattr(g, "delta")
        assert hasattr(g, "gamma")
        assert hasattr(g, "vega")
        assert hasattr(g, "theta")
        assert hasattr(g, "rho")


class TestImpliedVol:
    def test_roundtrip_call(self):
        price = BlackScholes.price(ATM_CALL, ATM_MD)
        iv = BlackScholes.implied_vol(ATM_CALL, ATM_MD, market_price=price)
        assert iv == pytest.approx(0.20, abs=1e-6)

    def test_roundtrip_put(self):
        price = BlackScholes.price(ATM_PUT, ATM_MD)
        iv = BlackScholes.implied_vol(ATM_PUT, ATM_MD, market_price=price)
        assert iv == pytest.approx(0.20, abs=1e-6)

    def test_roundtrip_high_vol(self):
        md = MarketData(spot=100.0, rate=0.05, vol=0.50)
        price = BlackScholes.price(ATM_CALL, md)
        iv = BlackScholes.implied_vol(ATM_CALL, md, market_price=price)
        assert iv == pytest.approx(0.50, abs=1e-6)

    def test_higher_price_implies_higher_vol(self):
        price = BlackScholes.price(ATM_CALL, ATM_MD)
        iv = BlackScholes.implied_vol(ATM_CALL, ATM_MD, market_price=price * 1.1)
        assert iv > 0.20

    def test_roundtrip_with_dividend(self):
        md = MarketData(spot=100.0, rate=0.05, vol=0.25, div_yield=0.02)
        price = BlackScholes.price(ATM_CALL, md)
        iv = BlackScholes.implied_vol(ATM_CALL, md, market_price=price)
        assert iv == pytest.approx(0.25, abs=1e-6)


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

    def test_negative_spot_raises(self):
        with pytest.raises(ValueError, match="spot"):
            MarketData(spot=-1.0, rate=0.05, vol=0.20)

    def test_zero_spot_raises(self):
        with pytest.raises(ValueError, match="spot"):
            MarketData(spot=0.0, rate=0.05, vol=0.20)

    def test_zero_vol_raises(self):
        with pytest.raises(ValueError, match="vol"):
            MarketData(spot=100.0, rate=0.05, vol=0.0)

    def test_invalid_option_type_raises(self):
        with pytest.raises(ValueError):
            EuropeanOption(option_type="invalid", strike=100.0, expiry=1.0)
