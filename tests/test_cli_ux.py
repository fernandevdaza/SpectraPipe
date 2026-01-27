"""Tests for CLI UX improvements (US-17)."""

import pytest
from typer.testing import CliRunner
from hsi_pipeline.cli import app
from hsi_pipeline.utils.parsing import parse_pixels_inline

class TestParsingUtils:
    """Tests for parsing utilities."""
    
    def test_parse_pixels_inline_valid(self):
        """Should parse valid pixel string."""
        assert parse_pixels_inline("10,10;20,20") == [(10, 10), (20, 20)]
        assert parse_pixels_inline("  10 , 10 ; 20 , 20  ") == [(10, 10), (20, 20)]
        
    def test_parse_pixels_inline_single(self):
        """Should parse single pixel."""
        assert parse_pixels_inline("10,10") == [(10, 10)]
        
    def test_parse_pixels_inline_invalid_format(self):
        """Should raise ValueError on bad format."""
        with pytest.raises(ValueError, match="Invalid pixel format"):
            parse_pixels_inline("10,10,10")
        with pytest.raises(ValueError, match="Invalid pixel format"):
            parse_pixels_inline("10;20")
            
    def test_parse_pixels_inline_non_integer(self):
        """Should raise ValueError on non-integers."""
        with pytest.raises(ValueError, match="Invalid coordinates"):
            parse_pixels_inline("10,a")


runner = CliRunner()

class TestCLISpectraUX:
    """Tests for spectra command UX."""
    
    @pytest.fixture
    def mock_env(self, tmp_path):
        """Create mock environment."""
        runs = tmp_path / "runs"
        runs.mkdir()
        return runs
        
    def test_spectra_mutual_exclusion_pixels(self, mock_env):
        """Should error if --pixel and --pixels used together."""
        result = runner.invoke(app, [
            "spectra", 
            "--from", str(mock_env), 
            "--pixel", "10,10", 
            "--pixels", "20,20"
        ])
        assert result.exit_code != 0
        assert "Use only one extraction mode" in result.stdout
        
    def test_spectra_mutual_exclusion_pixels_file(self, mock_env):
        """Should error if --pixels and --pixels-file used together."""
        coords = mock_env / "coords.csv"
        coords.touch()
        
        result = runner.invoke(app, [
            "spectra",
            "--from", str(mock_env),
            "--pixels", "10,10",
            "--pixels-file", str(coords)
        ])
        assert result.exit_code != 0
        assert "Use only one extraction mode" in result.stdout

    def test_spectra_mutual_exclusion_wavelengths(self, mock_env):
        """Should error if --wavelengths and --wl-start used together."""
        wl = mock_env / "wl.csv"
        wl.touch()
        
        result = runner.invoke(app, [
            "spectra",
            "--from", str(mock_env),
            "--pixel", "10,10",
            "--wavelengths", str(wl),
            "--wl-start", "400"
        ])
        assert result.exit_code != 0
        assert "Cannot specify both --wavelengths and --wl-start" in result.stdout
        
    def test_spectra_missing_mode(self, mock_env):
        """Should error if no extraction mode specified."""
        result = runner.invoke(app, [
            "spectra",
            "--from", str(mock_env)
        ])
        assert result.exit_code != 0
        
        # Typer validation for missing required args might happen first if any
        # But here valid args provided, just logic check failure
        assert "Specify one of: --pixel, --pixels, --roi-agg, or --pixels-file" in result.stdout

    def test_no_args_shows_help(self):
        """Running without args should show help."""
        result = runner.invoke(app, [])
        # Typer/Click might return 0 or 2 depending on version/config
        assert result.exit_code in (0, 2)
        assert "Usage:" in result.stdout
