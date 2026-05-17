"""Full coverage tests for modules.core.indicators."""

from __future__ import annotations

import numpy as np
import pytest

from modules.core.indicators import (
    calculate_beta,
    calculate_hurst,
    calculate_spread_statistics,
    calculate_z_score,
    generate_signal,
)


class TestGenerateSignal:
    def test_returns_zero_when_z_score_missing(self) -> None:
        assert generate_signal(None, 1.0, 1.0, None, False) == 0
        assert generate_signal(1.0, None, 1.0, None, False) == 0

    def test_delayed_entry_long(self) -> None:
        assert generate_signal(-0.5, -1.5, 1.0, None, True) == 1

    def test_delayed_entry_short(self) -> None:
        assert generate_signal(0.5, 1.5, 1.0, None, True) == -1

    def test_delayed_entry_neutral(self) -> None:
        assert generate_signal(0.0, 0.0, 1.0, None, True) == 0

    def test_standard_long_without_stop_loss(self) -> None:
        assert generate_signal(-1.2, -0.5, 1.0, None, False) == 1

    def test_standard_short_without_stop_loss(self) -> None:
        assert generate_signal(1.2, 0.5, 1.0, None, False) == -1

    def test_standard_neutral_without_stop_loss(self) -> None:
        assert generate_signal(0.0, 0.0, 1.0, None, False) == 0

    def test_standard_long_with_stop_loss(self) -> None:
        assert generate_signal(-2.0, -0.5, 1.0, 3.0, False) == 1

    def test_standard_short_with_stop_loss(self) -> None:
        assert generate_signal(2.0, 0.5, 1.0, 3.0, False) == -1

    def test_standard_neutral_with_stop_loss(self) -> None:
        assert generate_signal(4.0, 0.5, 1.0, 3.0, False) == 0


class TestCalculateBeta:
    def test_ols_beta(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        assert calculate_beta(x, y) == pytest.approx(0.5)

    def test_zero_variance_returns_zero(self) -> None:
        y = np.array([3.0, 3.0, 3.0])
        x = np.array([1.0, 2.0, 3.0])
        assert calculate_beta(x, y) == 0.0


class TestCalculateZScore:
    def test_none_when_std_zero(self) -> None:
        assert calculate_z_score(10.0, 5.0, 0.0) is None

    def test_standard_score(self) -> None:
        assert calculate_z_score(12.0, 10.0, 2.0) == pytest.approx(1.0)


class TestCalculateHurst:
    def test_short_series_returns_half(self) -> None:
        x = np.arange(10.0)
        y = np.arange(10.0)
        assert calculate_hurst(x, y, beta=1.0, max_lags=20) == 0.5

    def test_empty_lags_returns_half(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=10)
        y = rng.normal(size=10)
        assert calculate_hurst(x, y, beta=1.0, max_lags=2) == 0.5

    def test_polyfit_path(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        x = np.cumsum(rng.normal(size=n))
        y = np.cumsum(rng.normal(size=n))
        result = calculate_hurst(x, y, beta=0.5, max_lags=20)
        assert isinstance(result, float)
        assert np.isfinite(result)


class TestCalculateSpreadStatistics:
    def test_returns_last_mean_std(self) -> None:
        x = np.array([10.0, 11.0, 12.0])
        y = np.array([1.0, 2.0, 3.0])
        last, mean, std = calculate_spread_statistics(x, y, beta=2.0)
        spreads = x - 2.0 * y
        assert last == pytest.approx(spreads[-1])
        assert mean == pytest.approx(np.mean(spreads))
        assert std == pytest.approx(np.std(spreads, ddof=1))
