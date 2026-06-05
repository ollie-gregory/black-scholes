from __future__ import annotations

import pytest

from options_pricer import MarketData


class TestMarketData:
    def test_valid_construction(self):
        md = MarketData(spot=100.0, rate=0.05)
        assert md.spot == 100.0
        assert md.rate == 0.05
        assert md.div_yield == 0.0

    def test_div_yield_defaults_to_zero(self):
        md = MarketData(spot=100.0, rate=0.05)
        assert md.div_yield == 0.0

    def test_div_yield_explicit(self):
        md = MarketData(spot=100.0, rate=0.05, div_yield=0.02)
        assert md.div_yield == 0.02

    def test_negative_rate_valid(self):
        md = MarketData(spot=100.0, rate=-0.005)
        assert md.rate == -0.005

    def test_zero_spot_raises(self):
        with pytest.raises(ValueError, match="spot"):
            MarketData(spot=0.0, rate=0.05)

    def test_negative_spot_raises(self):
        with pytest.raises(ValueError, match="spot"):
            MarketData(spot=-50.0, rate=0.05)

    def test_vol_field_does_not_exist(self):
        md = MarketData(spot=100.0, rate=0.05)
        assert not hasattr(md, "vol")
