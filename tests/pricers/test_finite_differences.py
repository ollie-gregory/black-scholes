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

CF_CALL = ClosedFormPricer.greeks(ATM_CALL_EUR, ATM_MD, BS)
CF_PUT = ClosedFormPricer.greeks(ATM_PUT_EUR, ATM_MD, BS)


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


class TestFiniteDifferencesGreeks:
    def test_returns_named_tuple(self):
        g = FD.greeks(ATM_CALL_EUR, ATM_MD, BS)
        for field in ("delta", "gamma", "vega", "theta", "rho"):
            assert hasattr(g, field)

    # --- European call vs closed form ---

    def test_european_call_delta(self):
        g = FD.greeks(ATM_CALL_EUR, ATM_MD, BS)
        assert g.delta == pytest.approx(CF_CALL.delta, abs=0.005)

    def test_european_call_gamma(self):
        g = FD.greeks(ATM_CALL_EUR, ATM_MD, BS)
        assert g.gamma == pytest.approx(CF_CALL.gamma, abs=0.001)

    def test_european_call_vega(self):
        g = FD.greeks(ATM_CALL_EUR, ATM_MD, BS)
        assert g.vega == pytest.approx(CF_CALL.vega, abs=0.005)

    def test_european_call_theta(self):
        g = FD.greeks(ATM_CALL_EUR, ATM_MD, BS)
        assert g.theta == pytest.approx(CF_CALL.theta, abs=0.001)

    def test_european_call_rho(self):
        g = FD.greeks(ATM_CALL_EUR, ATM_MD, BS)
        assert g.rho == pytest.approx(CF_CALL.rho, abs=0.005)

    # --- European put vs closed form ---

    def test_european_put_delta(self):
        g = FD.greeks(ATM_PUT_EUR, ATM_MD, BS)
        assert g.delta == pytest.approx(CF_PUT.delta, abs=0.005)

    def test_european_put_gamma(self):
        g = FD.greeks(ATM_PUT_EUR, ATM_MD, BS)
        assert g.gamma == pytest.approx(CF_PUT.gamma, abs=0.001)

    def test_european_put_vega(self):
        g = FD.greeks(ATM_PUT_EUR, ATM_MD, BS)
        assert g.vega == pytest.approx(CF_PUT.vega, abs=0.005)

    def test_european_put_theta(self):
        g = FD.greeks(ATM_PUT_EUR, ATM_MD, BS)
        assert g.theta == pytest.approx(CF_PUT.theta, abs=0.001)

    def test_european_put_rho(self):
        g = FD.greeks(ATM_PUT_EUR, ATM_MD, BS)
        assert g.rho == pytest.approx(CF_PUT.rho, abs=0.005)

    # --- Cross-instrument identities ---

    def test_call_put_delta_sum(self):
        gc = FD.greeks(ATM_CALL_EUR, ATM_MD, BS)
        gp = FD.greeks(ATM_PUT_EUR, ATM_MD, BS)
        assert gc.delta - gp.delta == pytest.approx(1.0, abs=0.005)

    def test_call_put_same_gamma(self):
        gc = FD.greeks(ATM_CALL_EUR, ATM_MD, BS)
        gp = FD.greeks(ATM_PUT_EUR, ATM_MD, BS)
        assert gc.gamma == pytest.approx(gp.gamma, abs=1e-6)

    def test_call_put_same_vega(self):
        gc = FD.greeks(ATM_CALL_EUR, ATM_MD, BS)
        gp = FD.greeks(ATM_PUT_EUR, ATM_MD, BS)
        assert gc.vega == pytest.approx(gp.vega, abs=1e-6)

    # --- Sign and bounds ---

    def test_call_delta_in_bounds(self):
        assert 0.0 < FD.greeks(ATM_CALL_EUR, ATM_MD, BS).delta < 1.0

    def test_put_delta_in_bounds(self):
        assert -1.0 < FD.greeks(ATM_PUT_EUR, ATM_MD, BS).delta < 0.0

    def test_gamma_positive(self):
        assert FD.greeks(ATM_CALL_EUR, ATM_MD, BS).gamma > 0
        assert FD.greeks(ATM_PUT_EUR, ATM_MD, BS).gamma > 0

    def test_vega_positive(self):
        assert FD.greeks(ATM_CALL_EUR, ATM_MD, BS).vega > 0
        assert FD.greeks(ATM_PUT_EUR, ATM_MD, BS).vega > 0

    def test_call_theta_negative(self):
        assert FD.greeks(ATM_CALL_EUR, ATM_MD, BS).theta < 0

    def test_put_theta_negative(self):
        assert FD.greeks(ATM_PUT_EUR, ATM_MD, BS).theta < 0

    def test_call_rho_positive(self):
        assert FD.greeks(ATM_CALL_EUR, ATM_MD, BS).rho > 0

    def test_put_rho_negative(self):
        assert FD.greeks(ATM_PUT_EUR, ATM_MD, BS).rho < 0

    # --- American options ---

    def test_american_put_delta_negative(self):
        assert FD.greeks(ATM_PUT_AM, ATM_MD, BS).delta < 0

    def test_american_put_delta_le_european(self):
        # Early exercise pulls American put delta more negative than European
        g_am = FD.greeks(ATM_PUT_AM, ATM_MD, BS)
        g_eur = FD.greeks(ATM_PUT_EUR, ATM_MD, BS)
        assert g_am.delta <= g_eur.delta + 0.005

    def test_american_call_delta_matches_european_no_dividend(self):
        # American call == European call when there is no dividend
        g_am = FD.greeks(ATM_CALL_AM, ATM_MD, BS)
        g_eur = FD.greeks(ATM_CALL_EUR, ATM_MD, BS)
        assert g_am.delta == pytest.approx(g_eur.delta, abs=0.005)

    def test_american_put_gamma_positive(self):
        assert FD.greeks(ATM_PUT_AM, ATM_MD, BS).gamma > 0

    def test_american_put_theta_negative(self):
        assert FD.greeks(ATM_PUT_AM, ATM_MD, BS).theta < 0
