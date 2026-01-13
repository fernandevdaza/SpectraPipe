"""Unit tests for metrics reader."""

import pytest
import json
import tempfile
from pathlib import Path
import shutil
from hsi_pipeline.metrics.reader import (
    read_metrics,
    validate_metrics,
    MetricsNotFoundError,
    MetricsCorruptError,
)


class TestReadMetrics:
    """Tests for read_metrics function."""

    def setup_method(self):
        """Create temporary directory for each test."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up temporary directory after each test."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_valid_json(self):
        """Should read valid metrics.json successfully."""
        metrics_path = self.temp_dir / "metrics.json"
        data = {
            "hsi_shape": [512, 512, 31],
            "n_bands": 31,
            "execution_time_seconds": 12.5,
            "ensemble_enabled": True,
            "timestamp": "2026-01-13T12:00:00"
        }
        with open(metrics_path, "w") as f:
            json.dump(data, f)
        
        result = read_metrics(metrics_path)
        
        assert result.data == data
        # Optional section warnings are expected (separability, clean, upscaling)
        core_field_warnings = [w for w in result.warnings if "Missing field" in w]
        assert len(core_field_warnings) == 0

    def test_missing_file(self):
        """Should raise MetricsNotFoundError for missing file."""
        metrics_path = self.temp_dir / "missing.json"
        
        with pytest.raises(MetricsNotFoundError):
            read_metrics(metrics_path)

    def test_corrupt_json(self):
        """Should raise MetricsCorruptError for invalid JSON."""
        metrics_path = self.temp_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            f.write("{invalid json content")
        
        with pytest.raises(MetricsCorruptError):
            read_metrics(metrics_path)

    def test_non_object_json(self):
        """Should raise MetricsCorruptError for non-object JSON."""
        metrics_path = self.temp_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump([1, 2, 3], f)  # Array instead of object
        
        with pytest.raises(MetricsCorruptError):
            read_metrics(metrics_path)

    def test_partial_data_warns(self):
        """Should return warnings for missing fields."""
        metrics_path = self.temp_dir / "metrics.json"
        data = {"hsi_shape": [100, 100, 31]}  # Missing other fields
        with open(metrics_path, "w") as f:
            json.dump(data, f)
        
        result = read_metrics(metrics_path)
        
        assert len(result.warnings) > 0
        assert any("n_bands" in w for w in result.warnings)


class TestValidateMetrics:
    """Tests for validate_metrics function."""

    def test_complete_data_no_warnings(self):
        """Complete data should have no warnings."""
        data = {
            "hsi_shape": [512, 512, 31],
            "n_bands": 31,
            "execution_time_seconds": 12.5,
            "ensemble_enabled": True,
            "timestamp": "2026-01-13T12:00:00"
        }
        
        warnings = validate_metrics(data)
        
        # Only core field warnings should be absent; optional section hints are OK
        core_field_warnings = [w for w in warnings if "Missing field" in w or "Unexpected type" in w]
        assert len(core_field_warnings) == 0

    def test_wrong_type_warns(self):
        """Wrong field type should produce warning."""
        data = {
            "hsi_shape": "not a list",  # Should be list
            "n_bands": 31,
            "execution_time_seconds": 12.5,
            "ensemble_enabled": True,
            "timestamp": "2026-01-13T12:00:00"
        }
        
        warnings = validate_metrics(data)
        
        assert len(warnings) > 0
        assert any("hsi_shape" in w for w in warnings)

    def test_missing_field_warns(self):
        """Missing field should produce warning."""
        data = {
            "hsi_shape": [100, 100, 31],
            # Missing n_bands, execution_time_seconds, etc.
        }
        
        warnings = validate_metrics(data)
        
        assert len(warnings) >= 3
