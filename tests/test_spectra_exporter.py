"""Unit tests for spectral signature exporter."""

import pytest
import json
import csv
import numpy as np

from hsi_pipeline.spectra.extractor import SpectralSignature
from hsi_pipeline.spectra.exporter import (
    export_csv, export_json, export_signature, get_export_filename
)


class TestGetExportFilename:
    """Tests for get_export_filename function."""

    def test_pixel_filename(self):
        """Should generate correct filename for pixel extraction."""
        sig = SpectralSignature(
            values=np.zeros(31),
            num_bands=31,
            source="pixel",
            artifact="raw",
            pixel_x=120,
            pixel_y=80
        )
        
        assert get_export_filename(sig, "csv") == "spectra_raw_pixel_120_80.csv"
        assert get_export_filename(sig, "json") == "spectra_raw_pixel_120_80.json"

    def test_roi_filename(self):
        """Should generate correct filename for ROI extraction."""
        sig = SpectralSignature(
            values=np.zeros(31),
            num_bands=31,
            source="roi",
            artifact="clean",
            roi_aggregation="mean",
            roi_pixel_count=100
        )
        
        assert get_export_filename(sig, "csv") == "spectra_clean_roi_mean.csv"
        assert get_export_filename(sig, "json") == "spectra_clean_roi_mean.json"


class TestExportCSV:
    """Tests for export_csv function."""

    def test_creates_csv_file(self, tmp_path):
        """Should create CSV file."""
        sig = SpectralSignature(
            values=np.array([0.1, 0.2, 0.3]),
            num_bands=3,
            source="pixel",
            artifact="raw",
            pixel_x=10,
            pixel_y=20
        )
        
        result = export_csv(sig, tmp_path)
        
        assert result.exists()
        assert result.suffix == ".csv"

    def test_csv_contains_all_bands(self, tmp_path):
        """CSV should contain one row per band."""
        values = np.arange(31, dtype=np.float32) / 10
        sig = SpectralSignature(
            values=values,
            num_bands=31,
            source="pixel",
            artifact="raw",
            pixel_x=0,
            pixel_y=0
        )
        
        result = export_csv(sig, tmp_path)
        
        with open(result) as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        assert len(rows) == 32  # header + 31 bands
        assert rows[0] == ["band", "value"]

    def test_csv_values_correct(self, tmp_path):
        """CSV values should match signature."""
        values = np.array([0.5, 1.0, 1.5])
        sig = SpectralSignature(
            values=values,
            num_bands=3,
            source="pixel",
            artifact="raw",
            pixel_x=0,
            pixel_y=0
        )
        
        result = export_csv(sig, tmp_path)
        
        with open(result) as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                assert float(row["value"]) == pytest.approx(values[i])


class TestExportJSON:
    """Tests for export_json function."""

    def test_creates_json_file(self, tmp_path):
        """Should create JSON file."""
        sig = SpectralSignature(
            values=np.array([0.1, 0.2, 0.3]),
            num_bands=3,
            source="pixel",
            artifact="raw",
            pixel_x=10,
            pixel_y=20
        )
        
        result = export_json(sig, tmp_path)
        
        assert result.exists()
        assert result.suffix == ".json"

    def test_json_contains_metadata(self, tmp_path):
        """JSON should contain metadata."""
        sig = SpectralSignature(
            values=np.array([0.1, 0.2, 0.3]),
            num_bands=3,
            source="pixel",
            artifact="raw",
            pixel_x=10,
            pixel_y=20
        )
        
        result = export_json(sig, tmp_path)
        
        with open(result) as f:
            data = json.load(f)
        
        assert data["artifact"] == "raw"
        assert data["source"] == "pixel"
        assert data["bands"] == 3
        assert data["pixel"]["x"] == 10
        assert data["pixel"]["y"] == 20
        assert len(data["values"]) == 3

    def test_json_roi_metadata(self, tmp_path):
        """JSON should contain ROI metadata for ROI extraction."""
        sig = SpectralSignature(
            values=np.array([0.5]),
            num_bands=1,
            source="roi",
            artifact="clean",
            roi_aggregation="median",
            roi_pixel_count=500
        )
        
        result = export_json(sig, tmp_path)
        
        with open(result) as f:
            data = json.load(f)
        
        assert data["roi"]["aggregation"] == "median"
        assert data["roi"]["pixel_count"] == 500


class TestExportSignature:
    """Tests for export_signature function."""

    def test_export_csv_only(self, tmp_path):
        """Should export only CSV when format='csv'."""
        sig = SpectralSignature(
            values=np.zeros(31),
            num_bands=31,
            source="pixel",
            artifact="raw",
            pixel_x=0,
            pixel_y=0
        )
        
        paths = export_signature(sig, tmp_path, "csv")
        
        assert len(paths) == 1
        assert paths[0].suffix == ".csv"

    def test_export_json_only(self, tmp_path):
        """Should export only JSON when format='json'."""
        sig = SpectralSignature(
            values=np.zeros(31),
            num_bands=31,
            source="pixel",
            artifact="raw",
            pixel_x=0,
            pixel_y=0
        )
        
        paths = export_signature(sig, tmp_path, "json")
        
        assert len(paths) == 1
        assert paths[0].suffix == ".json"

    def test_export_both(self, tmp_path):
        """Should export both CSV and JSON when format='both'."""
        sig = SpectralSignature(
            values=np.zeros(31),
            num_bands=31,
            source="pixel",
            artifact="raw",
            pixel_x=0,
            pixel_y=0
        )
        
        paths = export_signature(sig, tmp_path, "both")
        
        assert len(paths) == 2
        extensions = {p.suffix for p in paths}
        assert extensions == {".csv", ".json"}
