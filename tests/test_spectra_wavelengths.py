"""Unit tests for wavelengths module."""

import pytest
import json
import numpy as np
from pathlib import Path

from hsi_pipeline.spectra.wavelengths import (
    load_wavelengths, generate_wavelengths, get_wavelengths,
    WavelengthError, NUM_BANDS
)


class TestLoadWavelengths:
    """Tests for load_wavelengths function."""

    def test_load_json_list(self, tmp_path):
        """Should load wavelengths from JSON list."""
        wl = list(range(400, 400 + NUM_BANDS))
        filepath = tmp_path / "wavelengths.json"
        with open(filepath, "w") as f:
            json.dump(wl, f)
        
        result = load_wavelengths(filepath)
        
        assert len(result) == NUM_BANDS
        np.testing.assert_array_equal(result, wl)

    def test_load_json_dict(self, tmp_path):
        """Should load wavelengths from JSON dict with 'wavelengths' key."""
        wl = list(range(400, 400 + NUM_BANDS))
        filepath = tmp_path / "wavelengths.json"
        with open(filepath, "w") as f:
            json.dump({"wavelengths": wl}, f)
        
        result = load_wavelengths(filepath)
        
        assert len(result) == NUM_BANDS

    def test_load_csv(self, tmp_path):
        """Should load wavelengths from CSV."""
        wl = list(range(400, 400 + NUM_BANDS))
        filepath = tmp_path / "wavelengths.csv"
        with open(filepath, "w") as f:
            for v in wl:
                f.write(f"{v}\n")
        
        result = load_wavelengths(filepath)
        
        assert len(result) == NUM_BANDS

    def test_file_not_found_raises(self):
        """Should raise WavelengthError for missing file."""
        with pytest.raises(WavelengthError, match="not found"):
            load_wavelengths(Path("/nonexistent/file.json"))

    def test_wrong_count_raises(self, tmp_path):
        """Should raise WavelengthError if not 31 values."""
        filepath = tmp_path / "wavelengths.json"
        with open(filepath, "w") as f:
            json.dump([400, 410, 420], f)  # Only 3 values
        
        with pytest.raises(WavelengthError, match="exactly 31"):
            load_wavelengths(filepath)

    def test_unsupported_format_raises(self, tmp_path):
        """Should raise WavelengthError for unsupported format."""
        filepath = tmp_path / "wavelengths.txt"
        filepath.write_text("400 410 420")
        
        with pytest.raises(WavelengthError, match="Unsupported"):
            load_wavelengths(filepath)


class TestGenerateWavelengths:
    """Tests for generate_wavelengths function."""

    def test_generates_31_values(self):
        """Should generate exactly 31 values."""
        result = generate_wavelengths(400, 10)
        
        assert len(result) == NUM_BANDS

    def test_correct_values(self):
        """Should generate correct sequence."""
        result = generate_wavelengths(400, 10)
        
        assert result[0] == 400
        assert result[1] == 410
        assert result[-1] == 400 + 30 * 10  # 700


class TestGetWavelengths:
    """Tests for get_wavelengths function."""

    def test_returns_none_if_no_config(self):
        """Should return None if no wavelength config provided."""
        result = get_wavelengths()
        
        assert result is None

    def test_loads_from_file(self, tmp_path):
        """Should load from file if provided."""
        wl = list(range(400, 400 + NUM_BANDS))
        filepath = tmp_path / "wavelengths.json"
        with open(filepath, "w") as f:
            json.dump(wl, f)
        
        result = get_wavelengths(file_path=filepath)
        
        assert len(result) == NUM_BANDS

    def test_generates_from_params(self):
        """Should generate from start/step if provided."""
        result = get_wavelengths(start_nm=400, step_nm=10)
        
        assert len(result) == NUM_BANDS
        assert result[0] == 400

    def test_partial_params_raises(self):
        """Should raise if only one of start/step provided."""
        with pytest.raises(WavelengthError, match="Both"):
            get_wavelengths(start_nm=400)
