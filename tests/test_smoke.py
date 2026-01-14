"""Smoke test for CLI run command.

This test validates the full pipeline works end-to-end with minimal input.
"""

import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
from typer.testing import CliRunner
from hsi_pipeline.cli import app


runner = CliRunner()


@patch("hsi_pipeline.cli.rgb_to_hsi")
def test_cli_run_smoke(mock_rgb_to_hsi):
    """Smoke test: CLI run produces expected artifacts.
    
    Validates:
    - Exit code = 0
    - hsi_raw_full.npz exists
    - metrics.json exists
    - run_config.json exists
    """
    mock_rgb_to_hsi.return_value = np.zeros((64, 64, 31), dtype=np.float32)
    
    image_path = Path("tests/test_images/01.bmp").resolve()
    out_path = Path(tempfile.mkdtemp())
    
    try:
        result = runner.invoke(app, [
            "run",
            "--input", str(image_path),
            "--out", str(out_path)
        ])
        
        # Exit code 0
        assert result.exit_code == 0, f"Failed with: {result.stdout}"
        
        # Minimal artifacts exist
        assert (out_path / "hsi_raw_full.npz").exists(), "hsi_raw_full.npz missing"
        assert (out_path / "metrics.json").exists(), "metrics.json missing"
        assert (out_path / "run_config.json").exists(), "run_config.json missing"
        
        # Verify NPZ is readable
        loaded = np.load(out_path / "hsi_raw_full.npz")
        assert "data" in loaded
        
        # Verify JSON is valid
        import json
        with open(out_path / "metrics.json") as f:
            metrics = json.load(f)
        assert "hsi_shape" in metrics
        
    finally:
        shutil.rmtree(out_path)


def test_smoke_placeholder():
    """Basic smoke test placeholder for test discovery."""
    assert True