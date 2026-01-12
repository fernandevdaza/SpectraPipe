import pytest
from typer.testing import CliRunner
from pathlib import Path
from hsi_pipeline.cli import app

runner = CliRunner()

def test_app_info():
    """Test that the app help command works."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--input" in result.stdout
    assert "--out" in result.stdout

def test_process_image_success():
    """Test processing a valid image."""
    image_path = Path("tests/test_images/01.bmp").resolve()
    assert image_path.exists(), "Test image 01.bmp not found"

    result = runner.invoke(app, ["--input", str(image_path)])
    
    assert result.exit_code == 0
    assert f"Loading image: {image_path}" in result.stdout
    assert "Converting RGB" in result.stdout

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

def test_process_image_implicit_out():
    """Test processing a valid image without specifying output directory."""
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
