from __future__ import annotations

import pytest

from options_pricer import (
    AmericanOption,
    BlackScholesModel,
    ClosedFormPricer,
    EuropeanOption,
    MarketData,
    MonteCarloPricer,
    OptionType,
    TreePricer,
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

    def test_american_put_vs_tree_reference(self):
        am_put = AmericanOption(option_type=OptionType.PUT, strike=100.0, expiry=1.0)
        ref = TreePricer(n_steps=500).price(am_put, ATM_MD, BS)
        mc = MonteCarloPricer(n_paths=50_000, n_steps=50, seed=42).price(am_put, ATM_MD, BS)
        assert mc == pytest.approx(ref, abs=0.10)

    def test_american_put_ge_european_put(self):
        am_put = AmericanOption(option_type=OptionType.PUT, strike=100.0, expiry=1.0)
        am = MonteCarloPricer(n_paths=50_000, n_steps=50, seed=42).price(am_put, ATM_MD, BS)
        eur = MC.price(ATM_PUT, ATM_MD, BS)
        assert am >= eur - 0.10  # LSM has variance; generous tolerance

    def test_american_lsm_seeded_reproducible(self):
        am_put = AmericanOption(option_type=OptionType.PUT, strike=100.0, expiry=1.0)
        mc1 = MonteCarloPricer(n_paths=10_000, n_steps=50, seed=99)
        mc2 = MonteCarloPricer(n_paths=10_000, n_steps=50, seed=99)
        assert mc1.price(am_put, ATM_MD, BS) == mc2.price(am_put, ATM_MD, BS)

    def test_american_put_ge_intrinsic(self):
        deep = AmericanOption(option_type=OptionType.PUT, strike=120.0, expiry=1.0)
        mc = MonteCarloPricer(n_paths=50_000, n_steps=50, seed=42).price(deep, ATM_MD, BS)
        assert mc >= 20.0 - 1e-6  # intrinsic = K - S = 120 - 100


class TestMonteCarloGreeks:
    MC = MonteCarloPricer(n_paths=200_000, seed=42)

    def test_call_delta_bounds(self):
        g = self.MC.greeks(ATM_CALL, ATM_MD, BS)
        assert 0.0 < g.delta < 1.0

    def test_put_delta_bounds(self):
        g = self.MC.greeks(ATM_PUT, ATM_MD, BS)
        assert -1.0 < g.delta < 0.0

    def test_call_delta_close_to_closed_form(self):
        cf = ClosedFormPricer.greeks(ATM_CALL, ATM_MD, BS)
        mc = self.MC.greeks(ATM_CALL, ATM_MD, BS)
        assert mc.delta == pytest.approx(cf.delta, abs=0.05)

    def test_put_delta_close_to_closed_form(self):
        cf = ClosedFormPricer.greeks(ATM_PUT, ATM_MD, BS)
        mc = self.MC.greeks(ATM_PUT, ATM_MD, BS)
        assert mc.delta == pytest.approx(cf.delta, abs=0.05)

    def test_call_put_delta_sum(self):
        gc = self.MC.greeks(ATM_CALL, ATM_MD, BS)
        gp = self.MC.greeks(ATM_PUT, ATM_MD, BS)
        assert gc.delta - gp.delta == pytest.approx(1.0, abs=0.05)

    def test_gamma_positive(self):
        assert self.MC.greeks(ATM_CALL, ATM_MD, BS).gamma > 0
        assert self.MC.greeks(ATM_PUT, ATM_MD, BS).gamma > 0

    def test_gamma_close_to_closed_form(self):
        cf = ClosedFormPricer.greeks(ATM_CALL, ATM_MD, BS)
        mc = self.MC.greeks(ATM_CALL, ATM_MD, BS)
        assert mc.gamma == pytest.approx(cf.gamma, abs=0.005)

    def test_vega_positive(self):
        assert self.MC.greeks(ATM_CALL, ATM_MD, BS).vega > 0
        assert self.MC.greeks(ATM_PUT, ATM_MD, BS).vega > 0

    def test_vega_close_to_closed_form(self):
        cf = ClosedFormPricer.greeks(ATM_CALL, ATM_MD, BS)
        mc = self.MC.greeks(ATM_CALL, ATM_MD, BS)
        assert mc.vega == pytest.approx(cf.vega, abs=0.05)

    def test_theta_negative(self):
        assert self.MC.greeks(ATM_CALL, ATM_MD, BS).theta < 0
        assert self.MC.greeks(ATM_PUT, ATM_MD, BS).theta < 0

    def test_theta_close_to_closed_form(self):
        cf = ClosedFormPricer.greeks(ATM_CALL, ATM_MD, BS)
        mc = self.MC.greeks(ATM_CALL, ATM_MD, BS)
        assert mc.theta == pytest.approx(cf.theta, abs=0.01)

    def test_call_rho_positive(self):
        assert self.MC.greeks(ATM_CALL, ATM_MD, BS).rho > 0

    def test_put_rho_negative(self):
        assert self.MC.greeks(ATM_PUT, ATM_MD, BS).rho < 0

    def test_rho_close_to_closed_form(self):
        cf = ClosedFormPricer.greeks(ATM_CALL, ATM_MD, BS)
        mc = self.MC.greeks(ATM_CALL, ATM_MD, BS)
        assert mc.rho == pytest.approx(cf.rho, abs=0.10)

    def test_greeks_returns_named_fields(self):
        g = self.MC.greeks(ATM_CALL, ATM_MD, BS)
        for field in ("delta", "gamma", "vega", "theta", "rho"):
            assert hasattr(g, field)

    def test_american_put_delta_bounds(self):
        am_put = AmericanOption(option_type=OptionType.PUT, strike=100.0, expiry=1.0)
        mc = MonteCarloPricer(n_paths=50_000, n_steps=50, seed=42)
        g = mc.greeks(am_put, ATM_MD, BS)
        assert -1.0 < g.delta < 0.0

    def test_american_put_gamma_positive(self):
        am_put = AmericanOption(option_type=OptionType.PUT, strike=100.0, expiry=1.0)
        mc = MonteCarloPricer(n_paths=50_000, n_steps=50, seed=42)
        assert mc.greeks(am_put, ATM_MD, BS).gamma > 0
