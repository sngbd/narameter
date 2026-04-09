"""Tests for Wasserstein distance computation."""

import numpy as np
import pytest

from narameter.distance.wasserstein import WassersteinCalculator
from narameter.config import DistanceConfig


@pytest.fixture
def calculator():
    """Create a calculator with default config (sliced)."""
    config = DistanceConfig(method="sliced", n_projections=100, seed=42)
    return WassersteinCalculator(config)


@pytest.fixture
def identical_distributions():
    """Two identical sets of embeddings."""
    np.random.seed(42)
    X = np.random.randn(50, 10)
    return X, X.copy()


@pytest.fixture
def different_distributions():
    """Two clearly different sets of embeddings."""
    np.random.seed(42)
    X_s = np.random.randn(50, 10)
    X_t = np.random.randn(50, 10) + 5.0  # shifted by 5 in all dimensions
    return X_s, X_t


@pytest.fixture
def slightly_different_distributions():
    """Two slightly different distributions."""
    np.random.seed(42)
    X_s = np.random.randn(50, 10)
    X_t = np.random.randn(50, 10) + 0.5  # small shift
    return X_s, X_t


class TestSlicedWasserstein:
    def test_identical_distributions_near_zero(self, calculator, identical_distributions):
        X_s, X_t = identical_distributions
        dist = calculator.compute_distance(X_s, X_t, method="sliced")
        assert dist < 0.1, f"Distance should be near 0 for identical data, got {dist}"

    def test_different_distributions_positive(self, calculator, different_distributions):
        X_s, X_t = different_distributions
        dist = calculator.compute_distance(X_s, X_t, method="sliced")
        assert dist > 0, "Distance should be positive for different distributions"

    def test_distance_ordering(self, calculator, identical_distributions, slightly_different_distributions, different_distributions):
        X_identical_s, X_identical_t = identical_distributions
        X_slight_s, X_slight_t = slightly_different_distributions
        X_diff_s, X_diff_t = different_distributions

        d_identical = calculator.compute_distance(X_identical_s, X_identical_t, method="sliced")
        d_slight = calculator.compute_distance(X_slight_s, X_slight_t, method="sliced")
        d_different = calculator.compute_distance(X_diff_s, X_diff_t, method="sliced")

        assert d_identical < d_slight < d_different, (
            f"Expected ordering: identical ({d_identical:.4f}) "
            f"< slight ({d_slight:.4f}) < different ({d_different:.4f})"
        )

    def test_non_negative(self, calculator):
        np.random.seed(123)
        X_s = np.random.randn(30, 10)
        X_t = np.random.randn(30, 10)
        dist = calculator.compute_distance(X_s, X_t, method="sliced")
        assert dist >= 0, f"Distance must be non-negative, got {dist}"


class TestEMD:
    def test_identical_distributions_near_zero(self, identical_distributions):
        config = DistanceConfig(method="emd")
        calculator = WassersteinCalculator(config)
        X_s, X_t = identical_distributions
        dist = calculator.compute_distance(X_s, X_t, method="emd")
        assert dist < 0.01, f"EMD should be near 0 for identical data, got {dist}"

    def test_different_distributions_positive(self, different_distributions):
        config = DistanceConfig(method="emd")
        calculator = WassersteinCalculator(config)
        X_s, X_t = different_distributions
        dist = calculator.compute_distance(X_s, X_t, method="emd")
        assert dist > 0, "EMD should be positive for different distributions"


class TestSinkhorn:
    def test_identical_distributions_near_zero(self, identical_distributions):
        config = DistanceConfig(method="sinkhorn", regularization=0.1)
        calculator = WassersteinCalculator(config)
        X_s, X_t = identical_distributions
        dist = calculator.compute_distance(X_s, X_t, method="sinkhorn")
        assert dist < 0.5, f"Sinkhorn should be relatively small for identical data, got {dist}"


class TestConsecutiveDistances:
    def test_returns_correct_shape(self, calculator):
        np.random.seed(42)
        period_embeddings = {
            "W1": np.random.randn(30, 10),
            "W2": np.random.randn(30, 10),
            "W3": np.random.randn(30, 10),
        }
        df = calculator.compute_consecutive(period_embeddings)
        assert len(df) == 2  # 3 periods → 2 transitions
        assert "distance" in df.columns
        assert "period_a" in df.columns
        assert "period_b" in df.columns


class TestDistanceMatrix:
    def test_matrix_is_symmetric(self, calculator):
        np.random.seed(42)
        period_embeddings = {
            "W1": np.random.randn(20, 10),
            "W2": np.random.randn(20, 10),
            "W3": np.random.randn(20, 10),
        }
        D, labels = calculator.compute_distance_matrix(period_embeddings)
        assert D.shape == (3, 3)
        np.testing.assert_allclose(D, D.T, atol=1e-6)

    def test_diagonal_is_zero(self, calculator):
        np.random.seed(42)
        period_embeddings = {
            "W1": np.random.randn(20, 10),
            "W2": np.random.randn(20, 10),
        }
        D, labels = calculator.compute_distance_matrix(period_embeddings)
        np.testing.assert_allclose(np.diag(D), 0.0)


class TestInputValidation:
    def test_rejects_1d_input(self, calculator):
        with pytest.raises(ValueError, match="2D"):
            calculator.compute_distance(np.array([1, 2, 3]), np.array([4, 5, 6]))

    def test_rejects_mismatched_dimensions(self, calculator):
        X_s = np.random.randn(10, 5)
        X_t = np.random.randn(10, 8)
        with pytest.raises(ValueError, match="dimensions must match"):
            calculator.compute_distance(X_s, X_t)

    def test_rejects_invalid_method(self):
        with pytest.raises(ValueError, match="Invalid method"):
            WassersteinCalculator(DistanceConfig(method="invalid"))
