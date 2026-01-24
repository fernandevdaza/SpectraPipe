"""Unit tests for background suppression."""

import pytest
import numpy as np
from hsi_pipeline.postprocess.background_suppression import suppress_background


class TestSuppressBackground:
    """Tests for suppress_background function."""

    def test_subtract_mean_policy(self):
        """Should subtract background mean spectrum."""
        hsi = np.ones((32, 32, 31), dtype=np.float32)
        hsi[16:, :, :] = 2.0
        
        mask = np.zeros((32, 32), dtype=bool)
        mask[:16, :] = True
        
        result = suppress_background(hsi, mask, policy="subtract_mean")
        
        assert result is not None
        assert result.hsi_clean.shape == hsi.shape
        assert result.policy == "subtract_mean"

    def test_zero_background_policy(self):
        """Should zero out background pixels."""
        hsi = np.ones((32, 32, 31), dtype=np.float32)
        mask = np.zeros((32, 32), dtype=bool)
        mask[:16, :] = True
        
        result = suppress_background(hsi, mask, policy="zero_background")
        
        assert result is not None
        assert np.all(result.hsi_clean[~mask] == 0)
        assert np.all(result.hsi_clean[mask] == 1)

    def test_preserves_shape(self):
        """Output should have same shape as input."""
        hsi = np.random.rand(64, 64, 31).astype(np.float32)
        mask = np.random.random((64, 64)) > 0.5
        mask[0, 0] = True
        mask[-1, -1] = False
        
        result = suppress_background(hsi, mask)
        
        assert result.hsi_clean.shape == hsi.shape

    def test_empty_roi_returns_none(self):
        """Empty ROI should return None."""
        hsi = np.ones((32, 32, 31), dtype=np.float32)
        mask = np.zeros((32, 32), dtype=bool)
        
        result = suppress_background(hsi, mask)
        
        assert result is None

    def test_full_roi_returns_none(self):
        """Full ROI should return None."""
        hsi = np.ones((32, 32, 31), dtype=np.float32)
        mask = np.ones((32, 32), dtype=bool)
        
        result = suppress_background(hsi, mask)
        
        assert result is None

    def test_invalid_hsi_ndim_raises(self):
        """Should raise for non-3D HSI."""
        hsi = np.ones((32, 32), dtype=np.float32)
        mask = np.zeros((32, 32), dtype=bool)
        
        with pytest.raises(ValueError, match="3D"):
            suppress_background(hsi, mask)

    def test_mask_shape_mismatch_raises(self):
        """Should raise if mask shape doesn't match HSI."""
        hsi = np.ones((32, 32, 31), dtype=np.float32)
        mask = np.zeros((64, 64), dtype=bool)
        
        with pytest.raises(ValueError, match="doesn't match"):
            suppress_background(hsi, mask)

    def test_invalid_policy_raises(self):
        """Should raise for unknown policy."""
        hsi = np.ones((32, 32, 31), dtype=np.float32)
        mask = np.zeros((32, 32), dtype=bool)
        mask[:16, :] = True
        
        with pytest.raises(ValueError, match="Unknown policy"):
            suppress_background(hsi, mask, policy="invalid_policy")
