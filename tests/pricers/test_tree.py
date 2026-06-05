from __future__ import annotations

import pytest

from options_pricer import (
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
