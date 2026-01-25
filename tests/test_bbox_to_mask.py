"""Unit tests for bbox to mask conversion."""

import numpy as np

from hsi_pipeline.adapters.voc_parser import BBox
from hsi_pipeline.adapters.bbox_to_mask import (
    clamp_bbox, bboxes_to_mask, save_mask
)


class TestClampBbox:
    """Tests for clamp_bbox function."""

    def test_clamp_within_bounds(self):
        """Bbox within bounds should remain unchanged."""
        bbox = BBox(xmin=10, ymin=20, xmax=100, ymax=200)
        
        result = clamp_bbox(bbox, width=640, height=480)
        
        assert result.xmin == 10
        assert result.ymin == 20
        assert result.xmax == 100
        assert result.ymax == 200

    def test_clamp_negative_coords(self):
        """Negative coords should be clamped to 0."""
        bbox = BBox(xmin=-10, ymin=-20, xmax=100, ymax=200)
        
        result = clamp_bbox(bbox, width=640, height=480)
        
        assert result.xmin == 0
        assert result.ymin == 0

    def test_clamp_exceeds_bounds(self):
        """Coords exceeding bounds should be clamped."""
        bbox = BBox(xmin=10, ymin=20, xmax=700, ymax=500)
        
        result = clamp_bbox(bbox, width=640, height=480)
        
        assert result.xmax == 640
        assert result.ymax == 480


class TestBboxesToMask:
    """Tests for bboxes_to_mask function."""

    def test_single_bbox_mask(self):
        """Single bbox should create rectangular mask."""
        bboxes = [BBox(xmin=10, ymin=20, xmax=30, ymax=40)]
        
        mask = bboxes_to_mask(bboxes, width=100, height=100)
        
        assert mask.shape == (100, 100)
        assert mask[25, 20]  # Inside bbox
        assert not mask[0, 0]  # Outside bbox
        assert mask[20:40, 10:30].all()  # All inside bbox is True

    def test_multiple_bboxes_union(self):
        """Multiple bboxes should create union mask."""
        bboxes = [
            BBox(xmin=0, ymin=0, xmax=10, ymax=10),
            BBox(xmin=20, ymin=20, xmax=30, ymax=30)
        ]
        
        mask = bboxes_to_mask(bboxes, width=50, height=50)
        
        assert mask[5, 5]  # In first bbox
        assert mask[25, 25]  # In second bbox
        assert not mask[15, 15]  # Between bboxes

    def test_clamp_out_of_bounds_bbox(self):
        """Out-of-bounds bbox should be clamped."""
        bboxes = [BBox(xmin=-10, ymin=-10, xmax=150, ymax=150)]
        
        mask = bboxes_to_mask(bboxes, width=100, height=100, clamp=True)
        
        assert mask.shape == (100, 100)
        assert mask.all()  # Entire mask should be True after clamp

    def test_skip_invalid_bbox(self):
        """Invalid bbox (zero area) should be skipped."""
        bboxes = [
            BBox(xmin=10, ymin=10, xmax=10, ymax=20),  # Zero width
            BBox(xmin=20, ymin=20, xmax=30, ymax=30)   # Valid
        ]
        
        mask = bboxes_to_mask(bboxes, width=50, height=50)
        
        assert not mask[15, 10]  # Zero-width bbox not included
        assert mask[25, 25]  # Valid bbox included


class TestSaveMask:
    """Tests for save_mask function."""

    def test_save_png(self, tmp_path):
        """Should save mask as PNG."""
        mask = np.ones((64, 64), dtype=bool)
        
        result = save_mask(mask, tmp_path / "mask.png", format="png")
        
        assert result.exists()
        assert result.suffix == ".png"

    def test_save_npy(self, tmp_path):
        """Should save mask as NPY."""
        mask = np.ones((64, 64), dtype=bool)
        
        result = save_mask(mask, tmp_path / "mask.npy", format="npy")
        
        assert result.exists()
        loaded = np.load(result)
        assert loaded.shape == (64, 64)

    def test_save_npz(self, tmp_path):
        """Should save mask as NPZ."""
        mask = np.ones((64, 64), dtype=bool)
        
        result = save_mask(mask, tmp_path / "mask.npz", format="npz")
        
        assert result.exists()
        loaded = np.load(result)
        assert "mask" in loaded
