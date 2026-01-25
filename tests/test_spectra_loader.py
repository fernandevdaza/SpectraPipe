"""Unit tests for HSI loader."""

import pytest
import numpy as np

from hsi_pipeline.spectra.loader import (
    load_hsi_artifact, HSINotFoundError, LoadedHSI
)


class TestLoadHSIArtifact:
    """Tests for load_hsi_artifact function."""

    def test_load_npz(self, tmp_path):
        """Should load NPZ file correctly."""
        hsi = np.random.rand(64, 64, 31).astype(np.float32)
        np.savez_compressed(tmp_path / "hsi_raw_full.npz", data=hsi)
        
        result = load_hsi_artifact(tmp_path, "raw")
        
        assert isinstance(result, LoadedHSI)
        assert result.shape == (64, 64, 31)
        assert result.artifact_type == "raw"
        np.testing.assert_array_equal(result.data, hsi)

    def test_load_npy_fallback(self, tmp_path):
        """Should fall back to NPY if NPZ doesn't exist."""
        hsi = np.random.rand(64, 64, 31).astype(np.float32)
        np.save(tmp_path / "hsi_raw_full.npy", hsi)
        
        result = load_hsi_artifact(tmp_path, "raw")
        
        assert result.path.suffix == ".npy"
        np.testing.assert_array_equal(result.data, hsi)

    def test_load_clean_artifact(self, tmp_path):
        """Should load clean artifact when specified."""
        hsi = np.random.rand(64, 64, 31).astype(np.float32)
        np.savez_compressed(tmp_path / "hsi_clean_full.npz", data=hsi)
        
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
