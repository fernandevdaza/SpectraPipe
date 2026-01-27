"""Unit tests for HSI loader."""

import pytest
import numpy as np

from hsi_pipeline.spectra.loader import (
    load_hsi_artifact, HSINotFoundError, HSILoadError, LoadedHSI
)
from hsi_pipeline.export.npz_schema import save_npz_v1, NPZMetadata


class TestLoadHSIArtifact:
    """Tests for load_hsi_artifact function."""

    def test_load_npz_v1(self, tmp_path):
        """Should load NPZ file with schema v1 correctly."""
        hsi = np.random.rand(64, 64, 31).astype(np.float32)
        metadata = NPZMetadata(artifact="raw")
        save_npz_v1(tmp_path / "hsi_raw_full.npz", hsi, metadata)
        
        result = load_hsi_artifact(tmp_path, "raw")
        
        assert isinstance(result, LoadedHSI)
        assert result.shape == (64, 64, 31)
        assert result.artifact_type == "raw"
        np.testing.assert_array_equal(result.data, hsi)

    def test_legacy_schema_raises_error(self, tmp_path):
        """Should raise error for legacy NPZ with 'data' key."""
        hsi = np.random.rand(64, 64, 31).astype(np.float32)
        np.savez_compressed(tmp_path / "hsi_raw_full.npz", data=hsi)
        
        with pytest.raises(HSILoadError, match="Legacy NPZ schema not supported"):
            load_hsi_artifact(tmp_path, "raw")

    def test_load_clean_artifact(self, tmp_path):
        """Should load clean artifact when specified."""
        hsi = np.random.rand(64, 64, 31).astype(np.float32)
        metadata = NPZMetadata(artifact="clean")
        save_npz_v1(tmp_path / "hsi_clean_full.npz", hsi, metadata)
        
        result = load_hsi_artifact(tmp_path, "clean")
        
        assert result.artifact_type == "clean"

    def test_dir_not_found_raises(self):
        """Should raise HSINotFoundError for missing directory."""
        with pytest.raises(HSINotFoundError, match="not found"):
            load_hsi_artifact("/nonexistent/path", "raw")

    def test_artifact_not_found_raises(self, tmp_path):
        """Should raise HSINotFoundError for missing artifact."""
        with pytest.raises(HSINotFoundError, match="not found"):
            load_hsi_artifact(tmp_path, "raw")

    def test_error_message_suggests_run(self, tmp_path):
        """Error should suggest running pipeline first."""
        with pytest.raises(HSINotFoundError, match="Run 'spectrapipe run'"):
            load_hsi_artifact(tmp_path, "raw")

    def test_invalid_artifact_type_raises(self, tmp_path):
        """Should raise ValueError for unknown artifact type."""
        with pytest.raises(ValueError, match="Unknown artifact"):
            load_hsi_artifact(tmp_path, "invalid")
    
    def test_loads_wavelength_from_npz(self, tmp_path):
        """Should load wavelength_nm if present in NPZ."""
        hsi = np.random.rand(32, 32, 31).astype(np.float32)
        wavelength = np.linspace(400, 700, 31).astype(np.float32)
        metadata = NPZMetadata(artifact="raw")
        save_npz_v1(tmp_path / "hsi_raw_full.npz", hsi, metadata, wavelength)
        
        result = load_hsi_artifact(tmp_path, "raw")
        
        assert result.wavelength_nm is not None
        np.testing.assert_array_almost_equal(result.wavelength_nm, wavelength)
