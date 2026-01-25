"""Unit tests for spectral signature extraction."""

import pytest
import numpy as np

from hsi_pipeline.spectra.extractor import (
    extract_pixel, extract_roi_aggregate,
    CoordinateError, ROIError, SpectralSignature
)


class TestExtractPixel:
    """Tests for extract_pixel function."""

    def test_valid_pixel_returns_vector(self):
        """Should return vector of length C for valid pixel."""
        hsi = np.random.rand(64, 64, 31).astype(np.float32)
        
        result = extract_pixel(hsi, x=10, y=20)
        
        assert isinstance(result, SpectralSignature)
        assert len(result.values) == 31
        assert result.source == "pixel"
        assert result.pixel_x == 10
        assert result.pixel_y == 20

    def test_values_match_hsi(self):
        """Extracted values should match HSI data."""
        hsi = np.arange(64 * 64 * 31).reshape(64, 64, 31).astype(np.float32)
        
        result = extract_pixel(hsi, x=5, y=10)
        
        expected = hsi[10, 5, :]  # y=row, x=col
        np.testing.assert_array_equal(result.values, expected)

    def test_x_negative_raises(self):
        """Should raise CoordinateError for negative x."""
        hsi = np.ones((64, 64, 31), dtype=np.float32)
        
        with pytest.raises(CoordinateError, match="X coordinate -1 out of range"):
            extract_pixel(hsi, x=-1, y=10)

    def test_x_out_of_range_raises(self):
        """Should raise CoordinateError for x >= width."""
        hsi = np.ones((64, 64, 31), dtype=np.float32)
        
        with pytest.raises(CoordinateError, match="X coordinate 64 out of range"):
            extract_pixel(hsi, x=64, y=10)

    def test_y_out_of_range_raises(self):
        """Should raise CoordinateError for y >= height."""
        hsi = np.ones((64, 64, 31), dtype=np.float32)
        
        with pytest.raises(CoordinateError, match="Y coordinate 64 out of range"):
            extract_pixel(hsi, x=10, y=64)

    def test_error_message_includes_valid_range(self):
        """Error message should include valid range info."""
        hsi = np.ones((100, 200, 31), dtype=np.float32)
        
        with pytest.raises(CoordinateError, match="0 to 199"):
            extract_pixel(hsi, x=200, y=0)


class TestExtractROIAggregate:
    """Tests for extract_roi_aggregate function."""

    def test_mean_aggregation(self):
        """Should compute mean over ROI pixels."""
        hsi = np.ones((10, 10, 31), dtype=np.float32) * 2.0
        mask = np.zeros((10, 10), dtype=bool)
        mask[:5, :] = True  # Top half
        
        result = extract_roi_aggregate(hsi, mask, "mean")
        
        assert result.roi_aggregation == "mean"
        assert result.roi_pixel_count == 50
        np.testing.assert_allclose(result.values, 2.0)

    def test_median_aggregation(self):
        """Should compute median over ROI pixels."""
        hsi = np.ones((10, 10, 31), dtype=np.float32)
        hsi[:5, :, :] = 1.0
        hsi[5:, :, :] = 3.0
        mask = np.ones((10, 10), dtype=bool)
        
        result = extract_roi_aggregate(hsi, mask, "median")
        
        assert result.roi_aggregation == "median"
        # Median of 50 ones and 50 threes = 2.0
        np.testing.assert_allclose(result.values, 2.0)

    def test_empty_mask_raises(self):
        """Should raise ROIError for empty mask."""
        hsi = np.ones((10, 10, 31), dtype=np.float32)
        mask = np.zeros((10, 10), dtype=bool)
        
        with pytest.raises(ROIError, match="empty"):
            extract_roi_aggregate(hsi, mask, "mean")

    def test_shape_mismatch_raises(self):
        """Should raise ROIError for mask shape mismatch."""
        hsi = np.ones((10, 10, 31), dtype=np.float32)
        mask = np.ones((20, 20), dtype=bool)
        
        with pytest.raises(ROIError, match="doesn't match"):
            extract_roi_aggregate(hsi, mask, "mean")

    def test_invalid_aggregation_raises(self):
        """Should raise ValueError for invalid aggregation."""
        hsi = np.ones((10, 10, 31), dtype=np.float32)
        mask = np.ones((10, 10), dtype=bool)
        
        with pytest.raises(ValueError, match="Unknown aggregation"):
            extract_roi_aggregate(hsi, mask, "sum")
