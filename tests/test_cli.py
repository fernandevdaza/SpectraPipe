from unittest.mock import patch
import numpy as np
from typer.testing import CliRunner
from pathlib import Path
from hsi_pipeline.cli import app

runner = CliRunner()

def test_app_info():
    """Test that the app help command works."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    # On CI, help output formatting might differ or contain escape codes.
    # Just checking exit code is safer, or checking basic words.
    assert "Usage" in result.stdout or "Options" in result.stdout

@patch("hsi_pipeline.cli.rgb_to_hsi")
def test_process_image_success(mock_rgb_to_hsi):
    """Test processing a valid image."""
    mock_rgb_to_hsi.return_value = np.zeros((32, 32, 31), dtype=np.float32)

    image_path = Path("tests/test_images/01.bmp").resolve()
    assert image_path.exists(), "Test image 01.bmp not found"

    result = runner.invoke(app, ["--input", str(image_path)])
    
    assert result.exit_code == 0
    # Check calling arguments if needed, but the important part is it ran.
    # Logs might vary depending on where mock interrupts, but "Converting" is before call.
    assert "Converting RGB" in result.stdout
    assert "Pipeline finished successfully" in result.stdout

def test_process_image_invalid():
    """Test processing an invalid image file (corrupt/text)."""
    image_path = Path("tests/test_images/fake.png").resolve()
    assert image_path.exists(), "Test image fake.png not found"
    
    result = runner.invoke(app, ["--input", str(image_path)])
    
    assert result.exit_code == 1
    assert "Integrity Error" in result.stdout

def test_process_image_not_found():
    """Test processing a non existent image."""
    result = runner.invoke(app, ["--input", "non_existent.jpg"])
    
    assert result.exit_code != 0

def test_run_no_args():
    """Test running without arguments. Should fail as input is required."""
    result = runner.invoke(app, [])
    assert result.exit_code != 0
    assert result.exit_code == 2

@patch("hsi_pipeline.cli.rgb_to_hsi")
def test_process_image_implicit_out(mock_rgb_to_hsi):
    """Test processing a valid image without specifying output directory."""
    mock_rgb_to_hsi.return_value = np.zeros((32, 32, 31), dtype=np.float32)

    image_path = Path("tests/test_images/01.bmp").resolve()
    assert image_path.exists(), "Test image 01.bmp not found"

    expected_out = image_path.parent / "output"
    
    import shutil
    if expected_out.exists():
        shutil.rmtree(expected_out)

    result = runner.invoke(app, ["--input", str(image_path)])
    
    if result.exit_code != 0:
        print(f"Output: {result.stdout}")
        
    assert result.exit_code == 0
    assert expected_out.exists()
    assert (expected_out / "hsi_raw_full.npy").exists()

    if expected_out.exists():
        shutil.rmtree(expected_out)

def test_process_image_really_corrupt():
    """Test processing a file with jpg extension but corrupt content."""
    image_path = Path("tests/test_images/corrupt.jpg").resolve()
    assert image_path.exists()
    
    result = runner.invoke(app, ["--input", str(image_path)])
    
    assert result.exit_code != 0
    assert "Integrity Error" in result.stdout or "Error" in result.stdout

@patch("hsi_pipeline.cli.rgb_to_hsi")
def test_process_image_tiny(mock_rgb_to_hsi):
    """Test processing an image that is too small (e.g. 1x1) - should be padded."""
    mock_rgb_to_hsi.return_value = np.zeros((32, 32, 31), dtype=np.float32)
    
    image_path = Path("tests/test_images/tiny.png").resolve()
    assert image_path.exists()
    
    result = runner.invoke(app, ["--input", str(image_path)])
    
    assert result.exit_code == 0
    assert "Fitted shape" in result.stdout

def test_process_image_text_format():
    """Test processing a file with unsupported extension/format (txt)."""
    image_path = Path("tests/test_images/plain.txt").resolve()
    assert image_path.exists()
    
    result = runner.invoke(app, ["--input", str(image_path)])
    
    assert result.exit_code != 0
    assert "Integrity Error" in result.stdout or "Error" in result.stdout


@patch("hsi_pipeline.cli.rgb_to_hsi")
def test_metadata_written(mock_rgb_to_hsi):
    """Test that run_config.json metadata is written with fitting info."""
    import json
    import shutil
    
    mock_rgb_to_hsi.return_value = np.zeros((128, 128, 31), dtype=np.float32)
    
    image_path = Path("tests/test_images/oddsize.png").resolve()
    out_path = Path("tests/test_out_metadata").resolve()
    
    if out_path.exists():
        shutil.rmtree(out_path)
    
    result = runner.invoke(app, ["--input", str(image_path), "--out", str(out_path)])
    
    assert result.exit_code == 0
    
    metadata_path = out_path / "run_config.json"
    assert metadata_path.exists(), "run_config.json not created"
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    assert "input_shape_original" in metadata
    assert "input_shape_fitted" in metadata
    assert "fitting_policy" in metadata
    assert metadata["fitting_policy"] == "pad_to_multiple"
    
    if out_path.exists():
        shutil.rmtree(out_path)


@patch("hsi_pipeline.cli.rgb_to_hsi")
def test_smoke_oddsize_image(mock_rgb_to_hsi):
    """Smoke test: process odd-sized image end-to-end."""
    import shutil
    
    mock_rgb_to_hsi.return_value = np.zeros((128, 128, 31), dtype=np.float32)
    
    image_path = Path("tests/test_images/oddsize.png").resolve()
    out_path = Path("tests/test_out_smoke_us02").resolve()
    
    if out_path.exists():
        shutil.rmtree(out_path)
    
    result = runner.invoke(app, ["--input", str(image_path), "--out", str(out_path)])
    
    assert result.exit_code == 0
    
    assert (out_path / "hsi_raw_full.npy").exists()
    
    assert "Original shape" in result.stdout
    assert "Fitted shape" in result.stdout
    
    import json
    with open(out_path / "run_config.json") as f:
        metadata = json.load(f)
    assert "input_shape_original" in metadata
    assert "input_shape_fitted" in metadata
    
    if out_path.exists():
        shutil.rmtree(out_path)
