"""Unit tests for export manager."""

import pytest
import json
import numpy as np
from pathlib import Path
import tempfile
import shutil
from hsi_pipeline.export.manager import ExportManager


class TestExportManager:
    """Tests for ExportManager class."""

    def setup_method(self):
        """Create temporary directory for each test."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up temporary directory after each test."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_prepare_directory_creates(self):
        """Should create output directory if it doesn't exist."""
        out_dir = self.temp_dir / "new_output"
        exporter = ExportManager(out_dir)
        
        assert not out_dir.exists()
        exporter.prepare_directory()
        assert out_dir.exists()

    def test_prepare_directory_not_a_dir(self):
        """Should raise if path exists but is not a directory."""
        file_path = self.temp_dir / "not_a_dir"
        file_path.touch()
        
        exporter = ExportManager(file_path)
        with pytest.raises(NotADirectoryError):
            exporter.prepare_directory()

    def test_get_path_standard_artifacts(self):
        """Should return correct paths for standard artifacts."""
        exporter = ExportManager(self.temp_dir, format="npz")
        
        assert exporter.get_path("hsi_raw") == self.temp_dir / "hsi_raw_full.npz"
        assert exporter.get_path("metrics") == self.temp_dir / "metrics.json"
        assert exporter.get_path("run_config") == self.temp_dir / "run_config.json"
        assert exporter.get_path("roi_mask") == self.temp_dir / "roi_mask.png"

    def test_export_array_npz(self):
        """Should export array as NPZ and be readable."""
        exporter = ExportManager(self.temp_dir, format="npz")
        exporter.prepare_directory()
        
        data = np.random.rand(64, 64, 31).astype(np.float32)
        path = exporter.export_array("hsi_raw", data)
        
        assert path.exists()
        assert path.suffix == ".npz"
        
        loaded = np.load(path, allow_pickle=True)
        # Schema v1 uses 'cube' key for HSI artifacts
        np.testing.assert_array_almost_equal(loaded["cube"], data)

    def test_export_json(self):
        """Should export valid JSON."""
        exporter = ExportManager(self.temp_dir)
        exporter.prepare_directory()
        
        data = {"key": "value", "number": 42}
        path = exporter.export_json("metrics", data)
        
        assert path.exists()
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_export_metrics_structure(self):
        """Should export metrics with required fields."""
        exporter = ExportManager(self.temp_dir)
        exporter.prepare_directory()
        
        path = exporter.export_metrics(
            hsi_shape=(100, 150, 31),
            execution_time=5.123,
            ensemble_enabled=True,
        )
        
        with open(path) as f:
            metrics = json.load(f)
        
        assert "hsi_shape" in metrics
        assert metrics["n_bands"] == 31
        assert "execution_time_seconds" in metrics
        assert "timestamp" in metrics

    def test_export_run_config_structure(self):
        """Should export run_config with required fields."""
        exporter = ExportManager(self.temp_dir)
        exporter.prepare_directory()
        
        path = exporter.export_run_config(
            config_dict={},
            input_path="/path/to/input.png",
            config_path="/path/to/config.yaml",
            fitting_info={"policy": "pad_to_multiple", "multiple": 32},
            pipeline_version="0.1.0",
        )
        
        with open(path) as f:
            config = json.load(f)
        
        assert config["meta"]["pipeline_version"] == "0.1.0"
        assert "timestamp" in config["meta"]
        assert "fitting" in config["meta"]
        assert config["meta"]["fitting"]["policy"] == "pad_to_multiple"
        assert config["meta"]["fitting"]["multiple"] == 32
        assert "config" in config

    def test_list_exported(self):
        """Should track exported artifacts."""
        exporter = ExportManager(self.temp_dir)
        exporter.prepare_directory()
        
        exporter.export_array("hsi_raw", np.zeros((32, 32, 31)))
        exporter.export_json("metrics", {"test": True})
        
        exported = exporter.list_exported()
        assert len(exported) == 2
        assert "hsi_raw_full.npz" in exported
        assert "metrics.json" in exported

    def test_overwrite_disabled_raises(self):
        """Should raise if overwrite is disabled and file exists."""
        exporter = ExportManager(self.temp_dir, overwrite=False)
        exporter.prepare_directory()
        
        data = np.zeros((32, 32, 31))
        exporter.export_array("hsi_raw", data)
        
        with pytest.raises(FileExistsError):
            exporter.export_array("hsi_raw", data)

    def test_overwrite_enabled_succeeds(self):
        """Should overwrite if enabled."""
        exporter = ExportManager(self.temp_dir, overwrite=True)
        exporter.prepare_directory()
        
        data1 = np.ones((32, 32, 31))
        data2 = np.zeros((32, 32, 31))
        
        exporter.export_array("hsi_raw", data1)
        exporter.export_array("hsi_raw", data2)
        
        path = exporter.get_path("hsi_raw")
        loaded = np.load(path, allow_pickle=True)
        # Schema v1 uses 'cube' key
        np.testing.assert_array_equal(loaded["cube"], data2)

    def test_cleanup_partial(self):
        """Should remove exported artifacts on cleanup."""
        exporter = ExportManager(self.temp_dir)
        exporter.prepare_directory()
        
        exporter.export_array("hsi_raw", np.zeros((32, 32, 31)))
        exporter.export_json("metrics", {"test": True})
        
        assert (self.temp_dir / "hsi_raw_full.npz").exists()
        assert (self.temp_dir / "metrics.json").exists()
        
        removed = exporter.cleanup_partial()
        
        assert len(removed) == 2
        assert not (self.temp_dir / "hsi_raw_full.npz").exists()
        assert not (self.temp_dir / "metrics.json").exists()
