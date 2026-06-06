from __future__ import annotations

import pytest

from options_pricer import (
    AmericanOption,
    BlackScholesModel,
    ClosedFormPricer,
    EuropeanOption,
    FiniteDifferencesPricer,
    MarketData,
    OptionType,
    TreePricer,
)

ATM_MD = MarketData(spot=100.0, rate=0.05)
BS = BlackScholesModel(vol=0.20)
ATM_CALL_EUR = EuropeanOption(option_type=OptionType.CALL, strike=100.0, expiry=1.0)
ATM_PUT_EUR = EuropeanOption(option_type=OptionType.PUT, strike=100.0, expiry=1.0)
ATM_CALL_AM = AmericanOption(option_type=OptionType.CALL, strike=100.0, expiry=1.0)
ATM_PUT_AM = AmericanOption(option_type=OptionType.PUT, strike=100.0, expiry=1.0)

FD = FiniteDifferencesPricer(n_s=200, n_t=200)
TREE_REF = TreePricer(n_steps=1000)


class TestFiniteDifferencesPricer:
    def test_european_call_close_to_closed_form(self):
        cf = ClosedFormPricer.price(ATM_CALL_EUR, ATM_MD, BS)
        fd = FD.price(ATM_CALL_EUR, ATM_MD, BS)
        assert fd == pytest.approx(cf, abs=0.02)

    def test_european_put_close_to_closed_form(self):
        cf = ClosedFormPricer.price(ATM_PUT_EUR, ATM_MD, BS)
        fd = FD.price(ATM_PUT_EUR, ATM_MD, BS)
        assert fd == pytest.approx(cf, abs=0.02)

    def test_american_put_close_to_tree_reference(self):
        ref = TREE_REF.price(ATM_PUT_AM, ATM_MD, BS)
        fd = FD.price(ATM_PUT_AM, ATM_MD, BS)
        assert fd == pytest.approx(ref, abs=0.05)

    def test_american_call_no_dividend_equals_european(self):
        # American call on non-dividend stock: no early exercise benefit
        fd_am = FD.price(ATM_CALL_AM, ATM_MD, BS)
        fd_eur = FD.price(ATM_CALL_EUR, ATM_MD, BS)
        assert fd_am == pytest.approx(fd_eur, abs=0.01)

    def test_american_put_ge_european_put(self):
        fd_am = FD.price(ATM_PUT_AM, ATM_MD, BS)
        fd_eur = FD.price(ATM_PUT_EUR, ATM_MD, BS)
        assert fd_am >= fd_eur - 1e-6

    def test_american_put_ge_intrinsic(self):
        deep_itm = AmericanOption(option_type=OptionType.PUT, strike=120.0, expiry=1.0)
        fd = FD.price(deep_itm, ATM_MD, BS)
        assert fd >= 20.0 - 1e-6  # intrinsic = K - S = 120 - 100

    def test_american_call_with_dividend_ge_european_call(self):
        md_div = MarketData(spot=100.0, rate=0.05, div_yield=0.05)
        call_am = AmericanOption(option_type=OptionType.CALL, strike=100.0, expiry=1.0)
        call_eur = EuropeanOption(option_type=OptionType.CALL, strike=100.0, expiry=1.0)
        fd_am = FD.price(call_am, md_div, BS)
        fd_eur = FD.price(call_eur, md_div, BS)
        assert fd_am >= fd_eur - 1e-6

    def test_american_put_with_dividend_close_to_tree(self):
        md_div = MarketData(spot=100.0, rate=0.05, div_yield=0.03)
        put_am = AmericanOption(option_type=OptionType.PUT, strike=100.0, expiry=1.0)
        ref = TREE_REF.price(put_am, md_div, BS)
        fd = FD.price(put_am, md_div, BS)
        assert fd == pytest.approx(ref, abs=0.05)
