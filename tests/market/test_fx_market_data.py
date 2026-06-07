from __future__ import annotations

import pytest

from options_pricer import FXMarketData


class TestFXMarketData:
    def test_valid_construction(self):
        md = FXMarketData(spot=1.08, domestic_rate=0.05, foreign_rate=0.03)
        assert md.spot == 1.08
        assert md.domestic_rate == 0.05
        assert md.foreign_rate == 0.03

    def test_rate_property_maps_to_domestic(self):
        md = FXMarketData(spot=1.08, domestic_rate=0.05, foreign_rate=0.03)
        assert md.rate == md.domestic_rate

    def test_div_yield_property_maps_to_foreign(self):
        md = FXMarketData(spot=1.08, domestic_rate=0.05, foreign_rate=0.03)
        assert md.div_yield == md.foreign_rate

    def test_zero_spot_raises(self):
        with pytest.raises(ValueError, match="spot"):
            FXMarketData(spot=0.0, domestic_rate=0.05, foreign_rate=0.03)

    def test_negative_spot_raises(self):
        with pytest.raises(ValueError, match="spot"):
            FXMarketData(spot=-1.0, domestic_rate=0.05, foreign_rate=0.03)

    def test_negative_rates_valid(self):
        md = FXMarketData(spot=1.08, domestic_rate=-0.005, foreign_rate=-0.01)
        assert md.rate == -0.005
        assert md.div_yield == -0.01
