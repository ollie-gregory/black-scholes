from __future__ import annotations

import pytest

from options_pricer import AmericanOption, OptionType


class TestAmericanOption:
    def test_valid_call(self):
        opt = AmericanOption(option_type=OptionType.CALL, strike=100.0, expiry=1.0)
        assert opt.option_type is OptionType.CALL
        assert opt.strike == 100.0
        assert opt.expiry == 1.0

    def test_valid_put(self):
        opt = AmericanOption(option_type=OptionType.PUT, strike=100.0, expiry=1.0)
        assert opt.option_type is OptionType.PUT

    def test_string_option_type_coercion_call(self):
        opt = AmericanOption(option_type="call", strike=100.0, expiry=1.0)
        assert opt.option_type is OptionType.CALL

    def test_string_option_type_coercion_put(self):
        opt = AmericanOption(option_type="put", strike=100.0, expiry=1.0)
        assert opt.option_type is OptionType.PUT

    def test_negative_strike_raises(self):
        with pytest.raises(ValueError, match="strike"):
            AmericanOption(option_type=OptionType.PUT, strike=-10.0, expiry=1.0)

    def test_zero_strike_raises(self):
        with pytest.raises(ValueError, match="strike"):
            AmericanOption(option_type=OptionType.PUT, strike=0.0, expiry=1.0)

    def test_negative_expiry_raises(self):
        with pytest.raises(ValueError, match="expiry"):
            AmericanOption(option_type=OptionType.CALL, strike=100.0, expiry=-1.0)

    def test_zero_expiry_raises(self):
        with pytest.raises(ValueError, match="expiry"):
            AmericanOption(option_type=OptionType.CALL, strike=100.0, expiry=0.0)
