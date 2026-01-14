"""Unit tests for ROI loader."""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from PIL import Image
from hsi_pipeline.roi.loader import (
    load_roi_mask,
    ROILoadError,
    ROIValidationError,
)


class TestLoadROIMask:
    """Tests for load_roi_mask function."""

    def setup_method(self):
        """Create temporary directory for each test."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up temporary directory after each test."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def _create_mask(self, shape, value=255):
        """Helper to create a mask image."""
        mask = np.full(shape, value, dtype=np.uint8)
        path = self.temp_dir / "mask.png"
        Image.fromarray(mask).save(path)
        return path

    def test_valid_binary_mask(self):
        """Should load valid binary mask successfully."""
        mask_path = self._create_mask((64, 64), value=255)
        
        result = load_roi_mask(mask_path, (64, 64))
        
        assert result.mask.shape == (64, 64)
        assert result.mask.dtype == bool
        assert result.coverage == 1.0
        assert len(result.warnings) == 1  # Full coverage warning

    def test_partial_coverage(self):
        """Should calculate correct coverage for partial mask."""
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[:32, :] = 255  # Top half is ROI
        path = self.temp_dir / "mask.png"
        Image.fromarray(mask).save(path)
        
        result = load_roi_mask(path, (64, 64))
        
        assert 0.49 < result.coverage < 0.51  # ~50%
        assert len(result.warnings) == 0

    def test_missing_file_raises(self):
        """Should raise ROILoadError for missing file."""
        with pytest.raises(ROILoadError, match="not found"):
            load_roi_mask(self.temp_dir / "nonexistent.png", (64, 64))

    def test_size_mismatch_raises(self):
        """Should raise ROIValidationError for size mismatch."""
        mask_path = self._create_mask((32, 32))
        
        with pytest.raises(ROIValidationError, match="size mismatch"):
            load_roi_mask(mask_path, (64, 64))

    def test_non_binary_binarizes_with_warning(self):
        """Should binarize non-binary mask and warn."""
        mask = np.full((64, 64), 128, dtype=np.uint8)  # Gray
        path = self.temp_dir / "mask.png"
        Image.fromarray(mask).save(path)
        
        result = load_roi_mask(path, (64, 64))
        
        assert any("binariz" in w.lower() for w in result.warnings)
        assert result.mask.dtype == bool

    def test_empty_mask_warns(self):
        """Should warn for empty mask (0% coverage)."""
        mask_path = self._create_mask((64, 64), value=0)
        
        result = load_roi_mask(mask_path, (64, 64))
        
        assert result.coverage == 0.0
        assert any("empty" in w.lower() for w in result.warnings)

    def test_full_mask_warns(self):
        """Should warn for full mask (100% coverage)."""
        mask_path = self._create_mask((64, 64), value=255)
        
        result = load_roi_mask(mask_path, (64, 64))
        
        assert result.coverage == 1.0
        assert any("full" in w.lower() for w in result.warnings)

    def test_corrupt_image_raises(self):
        """Should raise ROILoadError for corrupt image."""
        path = self.temp_dir / "corrupt.png"
        with open(path, "wb") as f:
            f.write(b"not an image")
        
        with pytest.raises(ROILoadError, match="decode"):
            load_roi_mask(path, (64, 64))
