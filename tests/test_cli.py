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

def test_process_image_tiny():
    """Test processing an image that is too small (e.g. 1x1)."""
    image_path = Path("tests/test_images/tiny.png").resolve()
    assert image_path.exists()
    
    result = runner.invoke(app, ["--input", str(image_path)])
    
    assert result.exit_code != 0

def test_process_image_text_format():
    """Test processing a file with unsupported extension/format (txt)."""
    image_path = Path("tests/test_images/plain.txt").resolve()
    assert image_path.exists()
    
    result = runner.invoke(app, ["--input", str(image_path)])
    
    assert result.exit_code != 0
    assert "Integrity Error" in result.stdout or "Error" in result.stdout
