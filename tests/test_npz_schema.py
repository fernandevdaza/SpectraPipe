"""Unit tests for NPZ Schema v1."""

import pytest
import json
import numpy as np

from hsi_pipeline.export.npz_schema import (
    save_npz_v1, load_npz_v1, validate_npz_schema,
    NPZMetadata, NPZSchemaError,
    SCHEMA_VERSION, KEY_CUBE, KEY_METADATA, KEY_SCHEMA_VERSION
)


class TestNPZMetadata:
    """Tests for NPZMetadata dataclass."""

    def test_to_json_parseable(self):
        """Metadata should serialize to valid JSON."""
        metadata = NPZMetadata(artifact="raw", model_name="MST++")
        
        json_str = metadata.to_json()
        data = json.loads(json_str)
        
        assert data["artifact"] == "raw"
        assert data["model_name"] == "MST++"
        assert data["schema_version"] == SCHEMA_VERSION

    def test_from_json_roundtrip(self):
        """Metadata should roundtrip through JSON."""
        original = NPZMetadata(
            artifact="clean",
            input_path="/path/to/image.png",
            cube_shape=(64, 64, 31)
        )
        
        json_str = original.to_json()
        loaded = NPZMetadata.from_json(json_str)
        
        assert loaded.artifact == "clean"
        assert loaded.input_path == "/path/to/image.png"
        assert loaded.cube_shape == (64, 64, 31)


class TestSaveNPZv1:
    """Tests for save_npz_v1 function."""

    def test_saves_required_keys(self, tmp_path):
        """Should save cube, schema_version, and metadata."""
        cube = np.random.rand(64, 64, 31).astype(np.float32)
        metadata = NPZMetadata(artifact="raw")
        path = tmp_path / "test.npz"
        
        save_npz_v1(path, cube, metadata)
        
        loaded = np.load(path, allow_pickle=True)
        assert KEY_CUBE in loaded
        assert KEY_SCHEMA_VERSION in loaded
        assert KEY_METADATA in loaded
        assert loaded[KEY_CUBE].shape == (64, 64, 31)

    def test_saves_wavelength_if_provided(self, tmp_path):
        """Should save wavelength_nm when provided."""
        cube = np.random.rand(32, 32, 31).astype(np.float32)
        metadata = NPZMetadata(artifact="raw")
        wavelength = np.linspace(400, 700, 31)
        path = tmp_path / "test.npz"
        
        save_npz_v1(path, cube, metadata, wavelength)
        
        loaded = np.load(path, allow_pickle=True)
        assert "wavelength_nm" in loaded
        assert len(loaded["wavelength_nm"]) == 31

    def test_rejects_wrong_band_count(self, tmp_path):
        """Should reject cube with != 31 bands."""
        cube = np.random.rand(64, 64, 10).astype(np.float32)
        metadata = NPZMetadata(artifact="raw")
        path = tmp_path / "test.npz"
        
        with pytest.raises(NPZSchemaError, match="31 bands"):
            save_npz_v1(path, cube, metadata)

    def test_rejects_wrong_wavelength_count(self, tmp_path):
        """Should reject wavelength with != 31 values."""
        cube = np.random.rand(64, 64, 31).astype(np.float32)
        metadata = NPZMetadata(artifact="raw")
        wavelength = np.linspace(400, 700, 10)  # Wrong count
        path = tmp_path / "test.npz"
        
        with pytest.raises(NPZSchemaError, match="31 values"):
            save_npz_v1(path, cube, metadata, wavelength)


class TestLoadNPZv1:
    """Tests for load_npz_v1 function."""

    def test_loads_v1_npz(self, tmp_path):
        """Should load NPZ with schema v1."""
        cube = np.random.rand(64, 64, 31).astype(np.float32)
        metadata = NPZMetadata(artifact="clean")
        path = tmp_path / "test.npz"
        save_npz_v1(path, cube, metadata)
        
        result = load_npz_v1(path)
        
        assert result.cube.shape == (64, 64, 31)
        assert result.is_legacy is False
        assert result.schema_version == SCHEMA_VERSION
        assert result.metadata.artifact == "clean"

    def test_loads_legacy_npz_with_warning(self, tmp_path):
        """Should load legacy NPZ with 'data' key and warn."""
        cube = np.random.rand(32, 32, 31).astype(np.float32)
        path = tmp_path / "legacy.npz"
        np.savez_compressed(path, data=cube)
        
        with pytest.warns(DeprecationWarning, match="Legacy NPZ"):
            result = load_npz_v1(path)
        
        assert result.is_legacy is True
        assert result.cube.shape == (32, 32, 31)

    def test_fails_on_missing_keys(self, tmp_path):
        """Should fail if neither 'cube' nor 'data' present."""
        path = tmp_path / "invalid.npz"
        np.savez_compressed(path, other_key=np.zeros(10))
        
        with pytest.raises(NPZSchemaError, match="missing 'cube' or 'data'"):
            load_npz_v1(path)

    def test_fails_on_wrong_shape(self, tmp_path):
        """Should fail if cube has wrong number of bands."""
        path = tmp_path / "wrong.npz"
        np.savez_compressed(path, cube=np.zeros((64, 64, 10)))
        
        with pytest.raises(NPZSchemaError, match="31 bands"):
            load_npz_v1(path)

    def test_loads_wavelength(self, tmp_path):
        """Should load wavelength_nm if present."""
        cube = np.random.rand(32, 32, 31).astype(np.float32)
        wavelength = np.linspace(400, 700, 31).astype(np.float32)
        metadata = NPZMetadata(artifact="raw")
        path = tmp_path / "test.npz"
        save_npz_v1(path, cube, metadata, wavelength)
        
        result = load_npz_v1(path)
        
        assert result.wavelength_nm is not None
        np.testing.assert_array_almost_equal(result.wavelength_nm, wavelength)


class TestValidateNPZSchema:
    """Tests for validate_npz_schema function."""

    def test_valid_npz_passes(self, tmp_path):
        """Valid v1 NPZ should pass validation."""
        cube = np.random.rand(64, 64, 31).astype(np.float32)
        metadata = NPZMetadata(artifact="raw")
        path = tmp_path / "valid.npz"
        save_npz_v1(path, cube, metadata)
        
        assert validate_npz_schema(path) is True

    def test_legacy_npz_fails(self, tmp_path):
        """Legacy NPZ should fail strict validation."""
        cube = np.random.rand(32, 32, 31).astype(np.float32)
        path = tmp_path / "legacy.npz"
        np.savez_compressed(path, data=cube)
        
        with pytest.raises(NPZSchemaError, match="legacy"):
            validate_npz_schema(path)
