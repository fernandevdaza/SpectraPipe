"""Unit tests for normalization module."""

import pytest
import numpy as np

from hsi_pipeline.spectra.normalize import (
    normalize_signature, validate_normalize_mode
)


class TestNormalizeSignature:
    """Tests for normalize_signature function."""

    def test_none_returns_copy(self):
        """Mode 'none' should return copy of input."""
        values = np.array([1.0, 2.0, 3.0])
        
        result = normalize_signature(values, "none")
        
        np.testing.assert_array_equal(result, values)
        assert result is not values  # Should be copy

    def test_minmax_scales_to_01(self):
        """Mode 'minmax' should scale to [0, 1]."""
        values = np.array([10.0, 20.0, 30.0])
        
        result = normalize_signature(values, "minmax")
        
        assert result.min() == 0.0
        assert result.max() == 1.0
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_minmax_constant_returns_zeros(self):
        """Mode 'minmax' with constant value should return zeros."""
        values = np.array([5.0, 5.0, 5.0])
        
        result = normalize_signature(values, "minmax")
        
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])

    def test_l2_unit_length(self):
        """Mode 'l2' should produce unit length vector."""
        values = np.array([3.0, 4.0, 0.0])  # Length 5
        
        result = normalize_signature(values, "l2")
        
        assert np.linalg.norm(result) == pytest.approx(1.0)
        np.testing.assert_allclose(result, [0.6, 0.8, 0.0])

    def test_l2_zero_returns_zeros(self):
        """Mode 'l2' with zero vector should return zeros."""
        values = np.array([0.0, 0.0, 0.0])
        
        result = normalize_signature(values, "l2")
        
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])

    def test_invalid_mode_raises(self):
        """Invalid mode should raise ValueError."""
        values = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError, match="Unknown"):
            normalize_signature(values, "invalid")


class TestValidateNormalizeMode:
    """Tests for validate_normalize_mode function."""

    def test_valid_modes(self):
        """Should accept valid modes."""
        assert validate_normalize_mode("none") == "none"
        assert validate_normalize_mode("minmax") == "minmax"
        assert validate_normalize_mode("l2") == "l2"

    def test_invalid_mode_raises(self):
        """Invalid mode should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid"):
            validate_normalize_mode("invalid")
