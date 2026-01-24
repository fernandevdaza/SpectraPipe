"""Unit tests for clean metrics."""

import pytest
import numpy as np
from hsi_pipeline.postprocess.clean_metrics import calculate_clean_metrics, spectral_angle


class TestSpectralAngle:
    """Tests for spectral_angle function."""

    def test_identical_vectors_zero_angle(self):
        """Identical vectors should have ~0 angle."""
        a = np.array([1.0, 2.0, 3.0])
        angle = spectral_angle(a, a)
        assert angle < 0.01

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have ~pi/2 angle."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        angle = spectral_angle(a, b)
        assert abs(angle - np.pi/2) < 0.01

    def test_zero_vector_returns_zero(self):
        """Zero vector should return 0 angle."""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 2.0, 3.0])
        angle = spectral_angle(a, b)
        assert angle == 0.0


class TestCalculateCleanMetrics:
    """Tests for calculate_clean_metrics function."""

    def test_returns_all_metrics(self):
        """Should return clean_separability, raw_clean_sam, raw_clean_rmse."""
        hsi_raw = np.random.rand(32, 32, 31).astype(np.float32)
        hsi_clean = hsi_raw * 0.9
        mask = np.zeros((32, 32), dtype=bool)
        mask[:16, :] = True
        
        metrics = calculate_clean_metrics(hsi_raw, hsi_clean, mask)
        
        assert metrics is not None
        assert "clean_separability" in metrics
        assert "raw_clean_sam" in metrics
        assert "raw_clean_rmse" in metrics

    def test_empty_roi_returns_none(self):
        """Empty ROI should return None."""
        hsi_raw = np.ones((32, 32, 31), dtype=np.float32)
        hsi_clean = hsi_raw.copy()
        mask = np.zeros((32, 32), dtype=bool)
        
        metrics = calculate_clean_metrics(hsi_raw, hsi_clean, mask)
        
        assert metrics is None

    def test_full_roi_returns_none(self):
        """Full ROI should return None."""
        hsi_raw = np.ones((32, 32, 31), dtype=np.float32)
        hsi_clean = hsi_raw.copy()
        mask = np.ones((32, 32), dtype=bool)
        
        metrics = calculate_clean_metrics(hsi_raw, hsi_clean, mask)
        
        assert metrics is None

    def test_shape_mismatch_raises(self):
        """Should raise if HSI shapes don't match."""
        hsi_raw = np.ones((32, 32, 31), dtype=np.float32)
        hsi_clean = np.ones((64, 64, 31), dtype=np.float32)
        mask = np.zeros((32, 32), dtype=bool)
        
        with pytest.raises(ValueError, match="shapes must match"):
            calculate_clean_metrics(hsi_raw, hsi_clean, mask)

    def test_mask_shape_mismatch_raises(self):
        """Should raise if mask shape doesn't match."""
        hsi_raw = np.ones((32, 32, 31), dtype=np.float32)
        hsi_clean = hsi_raw.copy()
        mask = np.zeros((64, 64), dtype=bool)
        
        with pytest.raises(ValueError, match="Mask shape"):
            calculate_clean_metrics(hsi_raw, hsi_clean, mask)
