"""Unit tests for separability calculation."""

import pytest
import numpy as np
from hsi_pipeline.roi.separability import calculate_separability


class TestCalculateSeparability:
    """Tests for calculate_separability function."""

    def test_distinct_spectra_high_separability(self):
        """Distinct ROI/BG spectra should have high separability."""
        # Create HSI with distinct ROI/BG spectra
        hsi = np.zeros((32, 32, 31), dtype=np.float32)
        mask = np.zeros((32, 32), dtype=bool)
        
        # ROI (top half): spectrum peaks at band 0
        hsi[:16, :, 0] = 1.0
        mask[:16, :] = True
        
        # BG (bottom half): spectrum peaks at band 30
        hsi[16:, :, 30] = 1.0
        
        sep = calculate_separability(hsi, mask)
        
        assert sep is not None
        assert sep > 0.5  # Should be high

    def test_identical_spectra_zero_separability(self):
        """Identical ROI/BG spectra should have ~0 separability."""
        hsi = np.ones((32, 32, 31), dtype=np.float32)  # Uniform
        mask = np.zeros((32, 32), dtype=bool)
        mask[:16, :] = True
        
        sep = calculate_separability(hsi, mask)
        
        assert sep is not None
        assert sep < 0.1  # Should be low

    def test_empty_roi_returns_none(self):
        """Empty ROI should return None."""
        hsi = np.ones((32, 32, 31), dtype=np.float32)
        mask = np.zeros((32, 32), dtype=bool)  # All False
        
        sep = calculate_separability(hsi, mask)
        
        assert sep is None

    def test_full_roi_returns_none(self):
        """Full ROI should return None."""
        hsi = np.ones((32, 32, 31), dtype=np.float32)
        mask = np.ones((32, 32), dtype=bool)  # All True
        
        sep = calculate_separability(hsi, mask)
        
        assert sep is None

    def test_invalid_hsi_ndim_raises(self):
        """Should raise for non-3D HSI."""
        hsi = np.ones((32, 32), dtype=np.float32)
        mask = np.zeros((32, 32), dtype=bool)
        
        with pytest.raises(ValueError, match="3D"):
            calculate_separability(hsi, mask)

    def test_mask_shape_mismatch_raises(self):
        """Should raise if mask shape doesn't match HSI."""
        hsi = np.ones((32, 32, 31), dtype=np.float32)
        mask = np.zeros((64, 64), dtype=bool)
        
        with pytest.raises(ValueError, match="doesn't match"):
            calculate_separability(hsi, mask)

    def test_separability_in_range(self):
        """Separability should be in [0, 1] range."""
        hsi = np.random.rand(32, 32, 31).astype(np.float32)
        mask = np.random.random((32, 32)) > 0.5
        
        # Ensure not empty/full
        mask[0, 0] = True
        mask[-1, -1] = False
        
        sep = calculate_separability(hsi, mask)
        
        assert sep is not None
        assert 0.0 <= sep <= 1.0
