from __future__ import annotations

import pytest

from options_pricer import BlackScholesModel


class TestBlackScholesModel:
    def test_valid_construction(self):
        model = BlackScholesModel(vol=0.20)
        assert model.vol == 0.20

    def test_zero_vol_raises(self):
        with pytest.raises(ValueError, match="vol"):
            BlackScholesModel(vol=0.0)

    def test_negative_vol_raises(self):
        with pytest.raises(ValueError, match="vol"):
            BlackScholesModel(vol=-0.10)

    def test_high_vol_valid(self):
        model = BlackScholesModel(vol=2.0)
        assert model.vol == 2.0
