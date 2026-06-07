from __future__ import annotations

import pytest

from options_pricer import FXEuropeanOption, FXAmericanOption, OptionType


class TestFXEuropeanOption:
    def test_valid_call_no_pair(self):
        opt = FXEuropeanOption(option_type=OptionType.CALL, strike=1.08, expiry=1.0)
        assert opt.option_type is OptionType.CALL
        assert opt.strike == 1.08
        assert opt.expiry == 1.0
        assert opt.currency_pair == ""

    def test_valid_call_with_pair(self):
        opt = FXEuropeanOption(option_type=OptionType.CALL, strike=1.08, expiry=1.0, currency_pair="EURUSD")
        assert opt.currency_pair == "EURUSD"

    def test_string_option_type_coercion(self):
        opt = FXEuropeanOption(option_type="put", strike=1.08, expiry=0.5)
        assert opt.option_type is OptionType.PUT

    def test_negative_strike_raises(self):
        with pytest.raises(ValueError, match="strike"):
            FXEuropeanOption(option_type=OptionType.CALL, strike=-1.08, expiry=1.0)

    def test_zero_expiry_raises(self):
        with pytest.raises(ValueError, match="expiry"):
            FXEuropeanOption(option_type=OptionType.CALL, strike=1.08, expiry=0.0)

    def test_is_european_option(self):
        from options_pricer import EuropeanOption
        opt = FXEuropeanOption(option_type=OptionType.CALL, strike=1.08, expiry=1.0)
        assert isinstance(opt, EuropeanOption)


class TestFXAmericanOption:
    def test_valid_put(self):
        opt = FXAmericanOption(option_type=OptionType.PUT, strike=1.08, expiry=1.0)
        assert opt.option_type is OptionType.PUT

    def test_is_american_option(self):
        from options_pricer import AmericanOption
        opt = FXAmericanOption(option_type=OptionType.CALL, strike=1.08, expiry=1.0)
        assert isinstance(opt, AmericanOption)
