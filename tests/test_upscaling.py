"""Unit tests for upscaling module."""

import pytest
import numpy as np
from hsi_pipeline.upscaling.spatial import upscale_baseline, upscale_improved


class TestUpscaleBaseline:
    """Tests for upscale_baseline function."""

    def test_shape_2x(self):
        """Output shape should be (H*2, W*2, C) for factor=2."""
        hsi = np.random.rand(32, 32, 31).astype(np.float32)
        
        result = upscale_baseline(hsi, factor=2)
        
        assert result.shape == (64, 64, 31)

    def test_shape_4x(self):
        """Output shape should be (H*4, W*4, C) for factor=4."""
        hsi = np.random.rand(16, 16, 31).astype(np.float32)
        
        result = upscale_baseline(hsi, factor=4)
        
        assert result.shape == (64, 64, 31)

    def test_factor_1_no_change(self):
        """Factor=1 should return copy of input."""
        hsi = np.random.rand(32, 32, 31).astype(np.float32)
        
        result = upscale_baseline(hsi, factor=1)
        
        assert result.shape == hsi.shape
        np.testing.assert_array_equal(result, hsi)

    def test_preserves_dtype(self):
        """Output should preserve input dtype."""
        hsi = np.random.rand(16, 16, 31).astype(np.float32)
        
        result = upscale_baseline(hsi, factor=2)
        
        assert result.dtype == np.float32

    def test_invalid_ndim_raises(self):
        """Should raise for non-3D input."""
        hsi_2d = np.random.rand(32, 32)
        
        with pytest.raises(ValueError, match="3D"):
            upscale_baseline(hsi_2d, factor=2)

    def test_invalid_factor_raises(self):
        """Should raise for factor < 1."""
        hsi = np.random.rand(32, 32, 31).astype(np.float32)
        
        with pytest.raises(ValueError, match="Factor"):
            upscale_baseline(hsi, factor=0)

    def test_bands_preserved(self):
        """All spectral bands should be preserved."""
        hsi = np.random.rand(16, 16, 31).astype(np.float32)
        
        result = upscale_baseline(hsi, factor=2)
        
        assert result.shape[2] == 31


class TestUpscaleImproved:
    """Tests for upscale_improved function."""

    def test_shape_with_valid_guide(self):
        """Output shape should match expected upscaled dimensions."""
        hsi = np.random.rand(32, 32, 31).astype(np.float32)
        rgb_guide = np.random.rand(64, 64, 3).astype(np.float32)
        
        result = upscale_improved(hsi, rgb_guide, factor=2)
        
        assert result.shape == (64, 64, 31)

    def test_invalid_guide_shape_raises(self):
        """Should raise if RGB guide has wrong dimensions."""
        hsi = np.random.rand(32, 32, 31).astype(np.float32)
        rgb_wrong = np.random.rand(100, 100, 3).astype(np.float32)  # Wrong size
        
        with pytest.raises(ValueError, match="shape mismatch"):
            upscale_improved(hsi, rgb_wrong, factor=2)

    def test_invalid_hsi_ndim_raises(self):
        """Should raise for non-3D HSI input."""
        hsi_2d = np.random.rand(32, 32)
        rgb_guide = np.random.rand(64, 64, 3).astype(np.float32)
        
        with pytest.raises(ValueError, match="HSI must be 3D"):
            upscale_improved(hsi_2d, rgb_guide, factor=2)

    def test_invalid_guide_ndim_raises(self):
        """Should raise for non-3D RGB guide."""
        hsi = np.random.rand(32, 32, 31).astype(np.float32)
        rgb_2d = np.random.rand(64, 64)
        
        with pytest.raises(ValueError, match="RGB guide must be 3D"):
            upscale_improved(hsi, rgb_2d, factor=2)
