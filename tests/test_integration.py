"""Integration tests that run without mocking.

These tests are slow because they use the real MST++ model.
Run with: pytest tests/test_integration.py -v --slow
Skip in CI by default, enable with: pytest --slow
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from typer.testing import CliRunner
from hsi_pipeline.cli import app


runner = CliRunner()


@pytest.mark.slow
def test_integration_full_pipeline():
    """Integration test: Run full pipeline without mocks.
    
    This test validates the actual MST++ model works correctly.
    Skipped by default, run with --slow flag.
    """
    image_path = Path("tests/test_images/01.bmp").resolve()
    
    if not image_path.exists():
        pytest.skip("Test image not found")
    
    out_path = Path(tempfile.mkdtemp())
    
    try:
        result = runner.invoke(app, [
            "run",
            "--input", str(image_path),
            "--out", str(out_path),
            "--no-ensemble"  # Faster
        ])
        
        assert result.exit_code == 0, f"Failed: {result.stdout}"
        
        # Artifacts exist
        assert (out_path / "hsi_raw_full.npz").exists()
        assert (out_path / "metrics.json").exists()
        
        # Valid HSI output
        loaded = np.load(out_path / "hsi_raw_full.npz")
        hsi = loaded["data"]
        
        assert hsi.ndim == 3
        assert hsi.shape[2] == 31  # 31 bands
        assert hsi.dtype == np.float32
        assert not np.isnan(hsi).any(), "HSI contains NaN"
        assert not np.isinf(hsi).any(), "HSI contains Inf"
        
    finally:
        shutil.rmtree(out_path)


@pytest.mark.slow
def test_integration_dataset_command():
    """Integration test: Dataset command without mocks."""
    manifest_path = Path("tests/test_images/dataset_small/manifest.yaml").resolve()
    
    if not manifest_path.exists():
        pytest.skip("Test manifest not found")
    
    out_path = Path(tempfile.mkdtemp())
    
    try:
        result = runner.invoke(app, [
            "dataset",
            "--manifest", str(manifest_path),
            "--out", str(out_path),
            "--no-ensemble",
            "--on-error", "continue"
        ])
        
        assert result.exit_code == 0, f"Failed: {result.stdout}"
        
        # Report exists
        assert (out_path / "dataset_report.json").exists()
        
        import json
        with open(out_path / "dataset_report.json") as f:
            report = json.load(f)
        
        assert report["processed_ok"] > 0, "No samples processed"
        
    finally:
        shutil.rmtree(out_path)
