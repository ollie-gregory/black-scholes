from __future__ import annotations

import pytest

from options_pricer import (
    BlackScholesModel,
    ClosedFormPricer,
    EuropeanOption,
    MarketData,
    MonteCarloPricer,
    OptionType,
)

ATM_MD = MarketData(spot=100.0, rate=0.05)
BS = BlackScholesModel(vol=0.20)
ATM_CALL = EuropeanOption(option_type=OptionType.CALL, strike=100.0, expiry=1.0)
ATM_PUT = EuropeanOption(option_type=OptionType.PUT, strike=100.0, expiry=1.0)

MC = MonteCarloPricer(n_paths=200_000, seed=42)


class TestMonteCarloPricer:
    def test_call_price_close_to_closed_form(self):
        cf = ClosedFormPricer.price(ATM_CALL, ATM_MD, BS)
        mc = MC.price(ATM_CALL, ATM_MD, BS)
        assert mc == pytest.approx(cf, abs=0.10)

    def test_put_price_close_to_closed_form(self):
        cf = ClosedFormPricer.price(ATM_PUT, ATM_MD, BS)
        mc = MC.price(ATM_PUT, ATM_MD, BS)
        assert mc == pytest.approx(cf, abs=0.10)

    def test_put_call_parity(self):
        import math
        mc_call = MC.price(ATM_CALL, ATM_MD, BS)
        mc_put = MC.price(ATM_PUT, ATM_MD, BS)
        forward = ATM_MD.spot - ATM_CALL.strike * math.exp(-ATM_MD.rate * ATM_CALL.expiry)
        assert mc_call - mc_put == pytest.approx(forward, abs=0.15)

    def test_seeded_results_reproducible(self):
        mc1 = MonteCarloPricer(n_paths=10_000, seed=99)
        mc2 = MonteCarloPricer(n_paths=10_000, seed=99)
        assert mc1.price(ATM_CALL, ATM_MD, BS) == mc2.price(ATM_CALL, ATM_MD, BS)

    def test_different_seeds_differ(self):
        mc1 = MonteCarloPricer(n_paths=10_000, seed=1)
        mc2 = MonteCarloPricer(n_paths=10_000, seed=2)
        assert mc1.price(ATM_CALL, ATM_MD, BS) != mc2.price(ATM_CALL, ATM_MD, BS)

    def test_call_price_with_dividend(self):
        md_div = MarketData(spot=100.0, rate=0.05, div_yield=0.02)
        cf = ClosedFormPricer.price(ATM_CALL, md_div, BS)
        mc = MC.price(ATM_CALL, md_div, BS)
        assert mc == pytest.approx(cf, abs=0.10)

    def test_more_paths_reduces_error(self):
        coarse = MonteCarloPricer(n_paths=1_000, seed=0).price(ATM_CALL, ATM_MD, BS)
        fine = MonteCarloPricer(n_paths=500_000, seed=0).price(ATM_CALL, ATM_MD, BS)
        cf = ClosedFormPricer.price(ATM_CALL, ATM_MD, BS)
        assert abs(fine - cf) < abs(coarse - cf)
