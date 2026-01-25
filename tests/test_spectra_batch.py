"""Unit tests for batch extraction module."""

import pytest
import numpy as np
from pathlib import Path

from hsi_pipeline.spectra.batch import (
    load_pixels_file, extract_batch, BatchError
)


class TestLoadPixelsFile:
    """Tests for load_pixels_file function."""

    def test_load_valid_csv(self, tmp_path):
        """Should load valid CSV with x,y columns."""
        filepath = tmp_path / "pixels.csv"
        filepath.write_text("x,y\n10,20\n30,40\n50,60\n")
        
        result = load_pixels_file(filepath)
        
        assert len(result) == 3
        assert result[0] == (10, 20)
        assert result[1] == (30, 40)

    def test_load_csv_no_header(self, tmp_path):
        """Should load CSV without header."""
        filepath = tmp_path / "pixels.csv"
        filepath.write_text("10,20\n30,40\n")
        
        result = load_pixels_file(filepath)
        
        assert len(result) == 2

    def test_file_not_found_raises(self):
        """Should raise BatchError for missing file."""
        with pytest.raises(BatchError, match="not found"):
            load_pixels_file(Path("/nonexistent/pixels.csv"))

    def test_empty_file_raises(self, tmp_path):
        """Should raise BatchError for empty file."""
        filepath = tmp_path / "pixels.csv"
        filepath.write_text("x,y\n")  # Only header
        
        with pytest.raises(BatchError, match="empty"):
            load_pixels_file(filepath)

    def test_invalid_coords_raises(self, tmp_path):
        """Should raise BatchError for invalid coordinates."""
        filepath = tmp_path / "pixels.csv"
        filepath.write_text("x,y\nabc,def\n")
        
        with pytest.raises(BatchError, match="Invalid coordinates"):
            load_pixels_file(filepath)


class TestExtractBatch:
    """Tests for extract_batch function."""

    def test_extracts_all_valid_pixels(self):
        """Should extract signatures for all valid pixels."""
        hsi = np.random.rand(100, 100, 31).astype(np.float32)
        pixels = [(10, 20), (30, 40), (50, 60)]
        
        result = extract_batch(hsi, pixels)
        
        assert result.total == 3
        assert result.success_count == 3
        assert result.fail_count == 0
        assert len(result.signatures) == 3

    def test_continues_on_error(self):
        """Should continue on error if fail_fast=False."""
        hsi = np.random.rand(50, 50, 31).astype(np.float32)
        pixels = [(10, 20), (100, 100), (30, 40)]  # (100,100) out of range
        
        result = extract_batch(hsi, pixels, fail_fast=False)
        
        assert result.success_count == 2
        assert result.fail_count == 1
        assert len(result.failed) == 1
        assert result.failed[0]["x"] == 100

    def test_stops_on_error_fail_fast(self):
        """Should stop on first error if fail_fast=True."""
        hsi = np.random.rand(50, 50, 31).astype(np.float32)
        pixels = [(10, 20), (100, 100), (30, 40)]
        
        result = extract_batch(hsi, pixels, fail_fast=True)
        
        assert result.success_count == 1
        assert result.fail_count == 1
        # Third pixel not processed
        assert len(result.signatures) == 1
