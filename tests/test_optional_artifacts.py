"""Unit tests for optional artifacts export functionality."""

import json
import numpy as np
from pathlib import Path
import tempfile
import shutil

from hsi_pipeline.export.manager import ExportManager


class TestSkippedTracking:
    """Tests for skipped artifact tracking."""

    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_mark_skipped_records_artifact(self):
        """Should record skipped artifact with reason."""
        exporter = ExportManager(self.temp_dir)
        
        exporter.mark_skipped("hsi_clean", "no ROI provided")
        
        skipped = exporter.list_skipped()
        assert len(skipped) == 1
        assert skipped[0]["artifact"] == "hsi_clean"
        assert skipped[0]["reason"] == "no ROI provided"

    def test_mark_skipped_multiple(self):
        """Should record multiple skipped artifacts."""
        exporter = ExportManager(self.temp_dir)
        
        exporter.mark_skipped("hsi_clean", "no ROI")
        exporter.mark_skipped("hsi_upscaled_baseline", "upscaling disabled")
        
        skipped = exporter.list_skipped()
        assert len(skipped) == 2


class TestExportROI:
    """Tests for ROI export functionality."""

    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_export_roi_png(self):
        """Should export ROI mask as PNG."""
        exporter = ExportManager(self.temp_dir)
        exporter.prepare_directory()
        mask = np.ones((64, 64), dtype=bool)
        
        path = exporter.export_roi(mask, export_as="png")
        
        assert path.exists()
        assert path.suffix == ".png"
        assert "roi_mask.png" in exporter.list_exported()

    def test_export_roi_npz(self):
        """Should export ROI mask as NPZ."""
        exporter = ExportManager(self.temp_dir)
        exporter.prepare_directory()
        mask = np.ones((64, 64), dtype=bool)
        
        path = exporter.export_roi(mask, export_as="npz")
        
        assert path.exists()
        loaded = np.load(path)
        assert "mask" in loaded

    def test_export_roi_ref(self):
        """Should export ROI as reference JSON."""
        exporter = ExportManager(self.temp_dir)
        exporter.prepare_directory()
        mask = np.zeros((64, 64), dtype=bool)
        mask[10:20, 10:20] = True  # 100 pixels
        
        path = exporter.export_roi(mask, source_path="/path/to/mask.png", export_as="ref")
        
        assert path.exists()
        with open(path) as f:
            ref = json.load(f)
        assert ref["shape"] == [64, 64]
        assert ref["pixel_count"] == 100
        assert ref["source_path"] == "/path/to/mask.png"


class TestExportSummary:
    """Tests for export summary functionality."""

    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_get_export_summary(self):
        """Should return summary with exported and skipped."""
        exporter = ExportManager(self.temp_dir)
        exporter.prepare_directory()
        
        exporter.export_array("hsi_raw", np.zeros((32, 32, 31)))
        exporter.mark_skipped("hsi_clean", "no ROI")
        
        summary = exporter.get_export_summary()
        
        assert summary["exported_count"] == 1
        assert summary["skipped_count"] == 1
        assert "hsi_raw_full.npz" in summary["exported"]
        assert summary["skipped"][0]["artifact"] == "hsi_clean"

    def test_summary_empty_when_nothing(self):
        """Summary should be empty if nothing exported/skipped."""
        exporter = ExportManager(self.temp_dir)
        
        summary = exporter.get_export_summary()
        
        assert summary["exported_count"] == 0
        assert summary["skipped_count"] == 0
