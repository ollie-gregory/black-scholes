from __future__ import annotations

import pytest

from options_pricer import (
    AmericanOption,
    BlackScholesModel,
    ClosedFormPricer,
    EuropeanOption,
    MarketData,
    OptionType,
    TreePricer,
)

ATM_MD = MarketData(spot=100.0, rate=0.05)
BS = BlackScholesModel(vol=0.20)
ATM_CALL = EuropeanOption(option_type=OptionType.CALL, strike=100.0, expiry=1.0)
ATM_PUT = EuropeanOption(option_type=OptionType.PUT, strike=100.0, expiry=1.0)

TREE = TreePricer(n_steps=200)


class TestTreePricer:
    def test_call_price_close_to_closed_form(self):
        cf = ClosedFormPricer.price(ATM_CALL, ATM_MD, BS)
        tr = TREE.price(ATM_CALL, ATM_MD, BS)
        assert tr == pytest.approx(cf, abs=0.02)

    def test_put_price_close_to_closed_form(self):
        cf = ClosedFormPricer.price(ATM_PUT, ATM_MD, BS)
        tr = TREE.price(ATM_PUT, ATM_MD, BS)
        assert tr == pytest.approx(cf, abs=0.02)

    def test_put_call_parity(self):
        import math
        tr_call = TREE.price(ATM_CALL, ATM_MD, BS)
        tr_put = TREE.price(ATM_PUT, ATM_MD, BS)
        forward = ATM_MD.spot - ATM_CALL.strike * math.exp(-ATM_MD.rate * ATM_CALL.expiry)
        assert tr_call - tr_put == pytest.approx(forward, abs=0.02)

    def test_call_price_with_dividend(self):
        md_div = MarketData(spot=100.0, rate=0.05, div_yield=0.02)
        cf = ClosedFormPricer.price(ATM_CALL, md_div, BS)
        tr = TREE.price(ATM_CALL, md_div, BS)
        assert tr == pytest.approx(cf, abs=0.02)

    def test_deep_itm_call(self):
        deep_itm = EuropeanOption(option_type=OptionType.CALL, strike=50.0, expiry=1.0)
        cf = ClosedFormPricer.price(deep_itm, ATM_MD, BS)
        tr = TREE.price(deep_itm, ATM_MD, BS)
        assert tr == pytest.approx(cf, abs=0.05)

    def test_more_steps_more_accurate(self):
        cf = ClosedFormPricer.price(ATM_CALL, ATM_MD, BS)
        err_coarse = abs(TreePricer(n_steps=50).price(ATM_CALL, ATM_MD, BS) - cf)
        err_fine = abs(TreePricer(n_steps=500).price(ATM_CALL, ATM_MD, BS) - cf)
        assert err_fine < err_coarse

    def test_american_put_ge_european_put(self):
        am_put = AmericanOption(option_type=OptionType.PUT, strike=100.0, expiry=1.0)
        am = TREE.price(am_put, ATM_MD, BS)
        eur = TREE.price(ATM_PUT, ATM_MD, BS)
        assert am >= eur - 1e-6

    def test_american_call_no_dividend_equals_european(self):
        am_call = AmericanOption(option_type=OptionType.CALL, strike=100.0, expiry=1.0)
        tree = TreePricer(n_steps=500)
        am = tree.price(am_call, ATM_MD, BS)
        eur = tree.price(ATM_CALL, ATM_MD, BS)
        assert am == pytest.approx(eur, abs=0.01)

    def test_american_put_deep_itm_ge_intrinsic(self):
        deep = AmericanOption(option_type=OptionType.PUT, strike=120.0, expiry=1.0)
        tr = TREE.price(deep, ATM_MD, BS)
        assert tr >= 20.0 - 1e-6  # intrinsic = K - S = 120 - 100

    def test_american_call_with_dividend_ge_european_call(self):
        md_div = MarketData(spot=100.0, rate=0.05, div_yield=0.05)
        call_am = AmericanOption(option_type=OptionType.CALL, strike=100.0, expiry=1.0)
        am = TREE.price(call_am, md_div, BS)
        eur = TREE.price(ATM_CALL, md_div, BS)
        assert am >= eur - 1e-6


class TestTreeGreeks:
    def test_call_greeks_close_to_closed_form(self):
        tg = TREE.greeks(ATM_CALL, ATM_MD, BS)
        cg = ClosedFormPricer.greeks(ATM_CALL, ATM_MD, BS)
        assert tg.delta == pytest.approx(cg.delta, abs=0.01)
        assert tg.gamma == pytest.approx(cg.gamma, abs=0.001)
        assert tg.vega == pytest.approx(cg.vega, abs=0.02)
        assert tg.theta == pytest.approx(cg.theta, abs=0.01)
        assert tg.rho == pytest.approx(cg.rho, abs=0.05)

    def test_put_greeks_close_to_closed_form(self):
        tg = TREE.greeks(ATM_PUT, ATM_MD, BS)
        cg = ClosedFormPricer.greeks(ATM_PUT, ATM_MD, BS)
        assert tg.delta == pytest.approx(cg.delta, abs=0.01)
        assert tg.gamma == pytest.approx(cg.gamma, abs=0.001)
        assert tg.vega == pytest.approx(cg.vega, abs=0.02)
        assert tg.theta == pytest.approx(cg.theta, abs=0.01)
        assert tg.rho == pytest.approx(cg.rho, abs=0.05)

    def test_call_delta_bounds(self):
        assert 0.0 < TREE.greeks(ATM_CALL, ATM_MD, BS).delta < 1.0

    def test_put_delta_bounds(self):
        assert -1.0 < TREE.greeks(ATM_PUT, ATM_MD, BS).delta < 0.0

    def test_call_put_delta_sum(self):
        gc = TREE.greeks(ATM_CALL, ATM_MD, BS)
        gp = TREE.greeks(ATM_PUT, ATM_MD, BS)
        assert gc.delta - gp.delta == pytest.approx(1.0, abs=0.01)

    def test_gamma_positive(self):
        assert TREE.greeks(ATM_CALL, ATM_MD, BS).gamma > 0
        assert TREE.greeks(ATM_PUT, ATM_MD, BS).gamma > 0

    def test_call_put_same_gamma(self):
        gc = TREE.greeks(ATM_CALL, ATM_MD, BS)
        gp = TREE.greeks(ATM_PUT, ATM_MD, BS)
        assert gc.gamma == pytest.approx(gp.gamma, rel=0.01)

    def test_vega_positive(self):
        assert TREE.greeks(ATM_CALL, ATM_MD, BS).vega > 0
        assert TREE.greeks(ATM_PUT, ATM_MD, BS).vega > 0

    def test_call_put_same_vega(self):
        gc = TREE.greeks(ATM_CALL, ATM_MD, BS)
        gp = TREE.greeks(ATM_PUT, ATM_MD, BS)
        assert gc.vega == pytest.approx(gp.vega, rel=0.01)

    def test_call_theta_negative(self):
        assert TREE.greeks(ATM_CALL, ATM_MD, BS).theta < 0

    def test_put_theta_negative(self):
        assert TREE.greeks(ATM_PUT, ATM_MD, BS).theta < 0

    def test_call_rho_positive(self):
        assert TREE.greeks(ATM_CALL, ATM_MD, BS).rho > 0

    def test_put_rho_negative(self):
        assert TREE.greeks(ATM_PUT, ATM_MD, BS).rho < 0

    def test_greeks_returns_named_tuple(self):
        g = TREE.greeks(ATM_CALL, ATM_MD, BS)
        for field in ("delta", "gamma", "vega", "theta", "rho"):
            assert hasattr(g, field)

    def test_american_put_delta_bounds(self):
        am_put = AmericanOption(option_type=OptionType.PUT, strike=100.0, expiry=1.0)
        assert -1.0 < TREE.greeks(am_put, ATM_MD, BS).delta < 0.0

    def test_american_call_delta_bounds(self):
        am_call = AmericanOption(option_type=OptionType.CALL, strike=100.0, expiry=1.0)
        assert 0.0 < TREE.greeks(am_call, ATM_MD, BS).delta < 1.0

    def test_american_put_gamma_positive(self):
        am_put = AmericanOption(option_type=OptionType.PUT, strike=100.0, expiry=1.0)
        assert TREE.greeks(am_put, ATM_MD, BS).gamma > 0

    def test_american_put_vega_positive(self):
        am_put = AmericanOption(option_type=OptionType.PUT, strike=100.0, expiry=1.0)
        assert TREE.greeks(am_put, ATM_MD, BS).vega > 0

    def test_american_put_rho_negative(self):
        am_put = AmericanOption(option_type=OptionType.PUT, strike=100.0, expiry=1.0)
        assert TREE.greeks(am_put, ATM_MD, BS).rho < 0
