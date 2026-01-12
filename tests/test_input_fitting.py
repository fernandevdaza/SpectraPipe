"""Unit tests for input fitting module."""

import pytest
import numpy as np
from hsi_pipeline.preprocess.input_fitting import fit_input, unfit_output


class TestFitInput:
    """Tests for fit_input function."""

    def test_no_padding_needed(self):
        """Image already multiple of 32 should not be padded."""
        rgb = np.random.rand(64, 128, 3).astype(np.float32)
        result = fit_input(rgb, multiple=32)
        
        assert result.original_shape == (64, 128)
        assert result.fitted_shape == (64, 128)
        assert result.padding == (0, 0, 0, 0)
        assert result.fitted is rgb  # Same object, no copy

    def test_padding_needed(self):
        """Image not multiple of 32 should be padded."""
        rgb = np.random.rand(50, 70, 3).astype(np.float32)
        result = fit_input(rgb, multiple=32)
        
        assert result.original_shape == (50, 70)
        assert result.fitted_shape == (64, 96)  # Next multiples of 32
        assert result.fitted.shape == (64, 96, 3)
        assert sum(result.padding) > 0

    def test_padding_symmetric(self):
        """Padding should be approximately symmetric."""
        rgb = np.random.rand(31, 31, 3).astype(np.float32)
        result = fit_input(rgb, multiple=32)
        
        top, bottom, left, right = result.padding
        # For 31 -> 32, need 1 pixel total, so either (0,1) or (1,0)
        assert top + bottom == 1
        assert left + right == 1

    def test_determinism(self):
        """Same input should always produce same output shape."""
        rgb = np.random.rand(100, 150, 3).astype(np.float32)
        
        result1 = fit_input(rgb, multiple=32)
        result2 = fit_input(rgb, multiple=32)
        
        assert result1.fitted_shape == result2.fitted_shape
        assert result1.padding == result2.padding

    def test_preserves_dtype(self):
        """Output should preserve input dtype."""
        rgb_uint8 = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        result = fit_input(rgb_uint8, multiple=32)
        assert result.fitted.dtype == np.uint8

    def test_invalid_dimensions_raises(self):
        """Invalid input shape should raise ValueError."""
        rgb_2d = np.random.rand(100, 100)  # Missing channel dim
        with pytest.raises(ValueError):
            fit_input(rgb_2d)

    def test_zero_dimensions_raises(self):
        """Zero-sized image should raise ValueError."""
        rgb = np.zeros((0, 100, 3))
        with pytest.raises(ValueError):
            fit_input(rgb)


class TestUnfitOutput:
    """Tests for unfit_output function."""

    def test_no_crop_needed(self):
        """No padding means no cropping."""
        hsi = np.random.rand(64, 64, 31).astype(np.float32)
        result = unfit_output(hsi, original_shape=(64, 64), padding=(0, 0, 0, 0))
        assert result is hsi

    def test_crop_to_original(self):
        """Output should be cropped back to original dimensions."""
        hsi_padded = np.random.rand(64, 96, 31).astype(np.float32)
        padding = (7, 7, 13, 13)  # top, bottom, left, right
        
        result = unfit_output(hsi_padded, original_shape=(50, 70), padding=padding)
        
        assert result.shape == (50, 70, 31)

    def test_roundtrip(self):
        """fit_input -> inference -> unfit_output should preserve spatial dims."""
        original = np.random.rand(100, 150, 3).astype(np.float32)
        fit_result = fit_input(original, multiple=32)
        
        fake_hsi = np.random.rand(*fit_result.fitted_shape, 31).astype(np.float32)
        
        unfit_hsi = unfit_output(fake_hsi, fit_result.original_shape, fit_result.padding)
        
        assert unfit_hsi.shape[:2] == original.shape[:2]
        assert unfit_hsi.shape[2] == 31
