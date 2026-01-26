"""Unit tests for wavelength resolution."""

import pytest
import json
import numpy as np

from hsi_pipeline.spectra.wavelengths import (
    WavelengthError, resolve_wavelengths, load_wavelengths, generate_wavelengths
)


class TestResolveWavelengths:
    """Tests for resolve_wavelengths function."""

    def test_npz_only_uses_npz(self):
        """NPZ with wavelength_nm + no CLI → uses NPZ."""
        npz_wl = np.linspace(400, 700, 31).astype(np.float32)
        
        result = resolve_wavelengths(npz_wavelengths=npz_wl)
        
        assert result.source == "npz"
        assert result.is_override is False
        np.testing.assert_array_almost_equal(result.wavelength_nm, npz_wl)

    def test_no_npz_no_cli_fails(self):
        """NPZ without wavelength_nm + no CLI → error."""
        with pytest.raises(WavelengthError, match="No wavelength axis"):
            resolve_wavelengths()

    def test_cli_file_overrides_npz(self, tmp_path):
        """NPZ with wavelength_nm + CLI file → uses CLI + is_override."""
        npz_wl = np.linspace(400, 700, 31).astype(np.float32)
        cli_wl = np.linspace(450, 750, 31).tolist()
        wl_file = tmp_path / "wavelengths.json"
        with open(wl_file, "w") as f:
            json.dump(cli_wl, f)
        
        result = resolve_wavelengths(npz_wavelengths=npz_wl, cli_file=wl_file)
        
        assert result.source == "cli_file"
        assert result.is_override is True
        assert result.wavelength_nm[0] == 450

    def test_cli_params_overrides_npz(self):
        """NPZ with wavelength_nm + CLI params → uses CLI + is_override."""
        npz_wl = np.linspace(400, 700, 31).astype(np.float32)
        
        result = resolve_wavelengths(
            npz_wavelengths=npz_wl,
            cli_start=410,
            cli_step=10
        )
        
        assert result.source == "cli_params"
        assert result.is_override is True
        assert result.wavelength_nm[0] == 410

    def test_cli_params_without_npz(self):
        """No NPZ + CLI params → uses CLI."""
        result = resolve_wavelengths(cli_start=400, cli_step=10)
        
        assert result.source == "cli_params"
        assert result.is_override is False
        assert len(result.wavelength_nm) == 31

    def test_partial_cli_params_fails(self):
        """Only --wl-start without --wl-step → error."""
        with pytest.raises(WavelengthError, match="must be provided together"):
            resolve_wavelengths(cli_start=400)

    def test_invalid_step_fails(self):
        """--wl-step <= 0 → error."""
        with pytest.raises(WavelengthError, match="must be > 0"):
            resolve_wavelengths(cli_start=400, cli_step=-10)

    def test_npz_wrong_length_fails(self):
        """NPZ wavelength_nm with len != 31 → error."""
        bad_wl = np.linspace(400, 700, 20).astype(np.float32)
        
        with pytest.raises(WavelengthError, match="invalid length"):
            resolve_wavelengths(npz_wavelengths=bad_wl)


class TestLoadWavelengths:
    """Tests for load_wavelengths function."""

    def test_load_json_list(self, tmp_path):
        """Should load JSON list of wavelengths."""
        wl = list(range(400, 431))
        wl_file = tmp_path / "wl.json"
        with open(wl_file, "w") as f:
            json.dump(wl, f)
        
        result = load_wavelengths(wl_file)
        
        assert len(result) == 31
        assert result[0] == 400

    def test_load_json_dict(self, tmp_path):
        """Should load JSON dict with wavelengths key."""
        wl_file = tmp_path / "wl.json"
        with open(wl_file, "w") as f:
            json.dump({"wavelengths": list(range(400, 431))}, f)
        
        result = load_wavelengths(wl_file)
        
        assert len(result) == 31

    def test_wrong_length_fails(self, tmp_path):
        """Wavelengths file with != 31 values → error."""
        wl_file = tmp_path / "wl.json"
        with open(wl_file, "w") as f:
            json.dump(list(range(10)), f)
        
        with pytest.raises(WavelengthError, match="exactly 31"):
            load_wavelengths(wl_file)

    def test_file_not_found_fails(self, tmp_path):
        """Non-existent file → error."""
        with pytest.raises(WavelengthError, match="not found"):
            load_wavelengths(tmp_path / "nonexistent.json")


class TestGenerateWavelengths:
    """Tests for generate_wavelengths function."""

    def test_generates_31_values(self):
        """Should generate exactly 31 values."""
        wl = generate_wavelengths(400, 10)
        
        assert len(wl) == 31
        assert wl[0] == 400
        assert wl[-1] == 700  # 400 + 30*10
