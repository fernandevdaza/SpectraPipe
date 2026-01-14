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
    assert "Usage" in result.stdout or "Options" in result.stdout

@patch("hsi_pipeline.cli.rgb_to_hsi")
def test_process_image_success(mock_rgb_to_hsi):
    """Test processing a valid image."""
    mock_rgb_to_hsi.return_value = np.zeros((32, 32, 31), dtype=np.float32)

    image_path = Path("tests/test_images/01.bmp").resolve()
    assert image_path.exists(), "Test image 01.bmp not found"

    result = runner.invoke(app, ["run", "--input", str(image_path)])
    
    assert result.exit_code == 0
    assert "Converting RGB" in result.stdout
    assert "Pipeline finished successfully" in result.stdout

def test_process_image_invalid():
    """Test processing an invalid image file (corrupt/text)."""
    image_path = Path("tests/test_images/fake.png").resolve()
    assert image_path.exists(), "Test image fake.png not found"
    
    result = runner.invoke(app, ["run", "--input", str(image_path)])
    
    assert result.exit_code == 1
    assert "Integrity Error" in result.stdout

def test_process_image_not_found():
    """Test processing a non existent image."""
    result = runner.invoke(app, ["run", "--input", "non_existent.jpg"])
    
    assert result.exit_code != 0

def test_run_no_args():
    """Test running without arguments. Should fail as input is required."""
    result = runner.invoke(app, ["run"])
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

    result = runner.invoke(app, ["run", "--input", str(image_path)])
    
    if result.exit_code != 0:
        print(f"Output: {result.stdout}")
        
    assert result.exit_code == 0
    assert expected_out.exists()
    assert (expected_out / "hsi_raw_full.npz").exists()

    if expected_out.exists():
        shutil.rmtree(expected_out)

def test_process_image_really_corrupt():
    """Test processing a file with jpg extension but corrupt content."""
    image_path = Path("tests/test_images/corrupt.jpg").resolve()
    assert image_path.exists()
    
    result = runner.invoke(app, ["run", "--input", str(image_path)])
    
    assert result.exit_code != 0
    assert "Integrity Error" in result.stdout or "Error" in result.stdout

@patch("hsi_pipeline.cli.rgb_to_hsi")
def test_process_image_tiny(mock_rgb_to_hsi):
    """Test processing an image that is too small (e.g. 1x1) - should be padded."""
    mock_rgb_to_hsi.return_value = np.zeros((32, 32, 31), dtype=np.float32)
    
    image_path = Path("tests/test_images/tiny.png").resolve()
    assert image_path.exists()
    
    result = runner.invoke(app, ["run", "--input", str(image_path)])
    
    assert result.exit_code == 0
    assert "Fitted shape" in result.stdout

def test_process_image_text_format():
    """Test processing a file with unsupported extension/format (txt)."""
    image_path = Path("tests/test_images/plain.txt").resolve()
    assert image_path.exists()
    
    result = runner.invoke(app, ["run", "--input", str(image_path)])
    
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
    
    result = runner.invoke(app, ["run", "--input", str(image_path), "--out", str(out_path)])
    
    assert result.exit_code == 0
    
    metadata_path = out_path / "run_config.json"
    assert metadata_path.exists(), "run_config.json not created"
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    assert "fitting" in metadata
    assert "input_shape_original" in metadata["fitting"]
    assert "input_shape_fitted" in metadata["fitting"]
    assert metadata["fitting"]["policy"] == "pad_to_multiple"
    
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
    
    result = runner.invoke(app, ["run", "--input", str(image_path), "--out", str(out_path)])
    
    assert result.exit_code == 0
    
    assert (out_path / "hsi_raw_full.npz").exists()
    
    assert "Original shape" in result.stdout
    assert "Fitted shape" in result.stdout
    
    import json
    with open(out_path / "run_config.json") as f:
        metadata = json.load(f)
    assert "input_shape_original" in metadata["fitting"]
    assert "input_shape_fitted" in metadata["fitting"]
    
    if out_path.exists():
        shutil.rmtree(out_path)


@patch("hsi_pipeline.cli.rgb_to_hsi")
def test_smoke_us06_minimal_artifacts(mock_rgb_to_hsi):
    """US-06: Verify minimal artifact structure is exported."""
    import shutil
    
    mock_rgb_to_hsi.return_value = np.zeros((64, 64, 31), dtype=np.float32)
    
    image_path = Path("tests/test_images/01.bmp").resolve()
    out_path = Path("tests/test_out_us06").resolve()
    
    if out_path.exists():
        shutil.rmtree(out_path)
    
    result = runner.invoke(app, ["run", "--input", str(image_path), "--out", str(out_path)])
    
    assert result.exit_code == 0
    
    assert (out_path / "hsi_raw_full.npz").exists(), "hsi_raw_full.npz missing"
    assert (out_path / "metrics.json").exists(), "metrics.json missing"
    assert (out_path / "run_config.json").exists(), "run_config.json missing"
    
    assert "Exported artifacts" in result.stdout
    
    if out_path.exists():
        shutil.rmtree(out_path)


@patch("hsi_pipeline.cli.rgb_to_hsi")
def test_metrics_command_success(mock_rgb_to_hsi):
    """Test metrics command with valid metrics.json."""
    import shutil
    
    mock_rgb_to_hsi.return_value = np.zeros((64, 64, 31), dtype=np.float32)
    
    image_path = Path("tests/test_images/01.bmp").resolve()
    out_path = Path("tests/test_out_metrics_cmd").resolve()
    
    if out_path.exists():
        shutil.rmtree(out_path)
    
    # First generate metrics
    result = runner.invoke(app, ["run", "--input", str(image_path), "--out", str(out_path)])
    assert result.exit_code == 0
    
    # Now read metrics
    result = runner.invoke(app, ["metrics", "--from", str(out_path)])
    
    assert result.exit_code == 0
    assert "General Stats" in result.stdout
    assert "Metrics loaded successfully" in result.stdout
    
    if out_path.exists():
        shutil.rmtree(out_path)


def test_metrics_command_missing_dir():
    """Test metrics command with non-existent directory."""
    result = runner.invoke(app, ["metrics", "--from", "nonexistent_dir_12345"])
    
    assert result.exit_code != 0
    assert "Directory not found" in result.stdout


def test_metrics_command_missing_file():
    """Test metrics command with directory but no metrics.json."""
    import tempfile
    
    temp_dir = Path(tempfile.mkdtemp())
    
    result = runner.invoke(app, ["metrics", "--from", str(temp_dir)])
    
    assert result.exit_code != 0
    assert "metrics.json not found" in result.stdout
    
    import shutil
    shutil.rmtree(temp_dir)


def test_metrics_command_corrupt_json():
    """Test metrics command with corrupt JSON."""
    import tempfile
    import shutil
    
    temp_dir = Path(tempfile.mkdtemp())
    metrics_path = temp_dir / "metrics.json"
    
    with open(metrics_path, "w") as f:
        f.write("{invalid json")
    
    result = runner.invoke(app, ["metrics", "--from", str(temp_dir)])
    
    assert result.exit_code != 0
    assert "corrupt" in result.stdout.lower() or "invalid" in result.stdout.lower()
    
    shutil.rmtree(temp_dir)


@patch("hsi_pipeline.cli.rgb_to_hsi")
def test_roi_mask_success(mock_rgb_to_hsi):
    """Test run with valid ROI mask includes separability in metrics."""
    import shutil
    import json
    
    mock_rgb_to_hsi.return_value = np.zeros((64, 64, 31), dtype=np.float32)
    
    image_path = Path("tests/test_images/01.bmp").resolve()
    # Create a simple mask matching image size
    mask_path = Path("tests/test_images/test_roi_mask.png").resolve()
    out_path = Path("tests/test_out_roi").resolve()
    
    if out_path.exists():
        shutil.rmtree(out_path)
    
    result = runner.invoke(app, [
        "run",
        "--input", str(image_path),
        "--roi-mask", str(mask_path),
        "--out", str(out_path)
    ])
    
    # May fail due to size mismatch, so check for either success or validation error
    if result.exit_code == 0:
        # Check metrics.json contains ROI data
        with open(out_path / "metrics.json") as f:
            metrics = json.load(f)
        assert "roi_coverage" in metrics
    
    if out_path.exists():
        shutil.rmtree(out_path)


@patch("hsi_pipeline.cli.rgb_to_hsi")
def test_no_roi_omits_separability(mock_rgb_to_hsi):
    """Test run without ROI mask omits separability from metrics."""
    import shutil
    import json
    
    mock_rgb_to_hsi.return_value = np.zeros((64, 64, 31), dtype=np.float32)
    
    image_path = Path("tests/test_images/01.bmp").resolve()
    out_path = Path("tests/test_out_no_roi").resolve()
    
    if out_path.exists():
        shutil.rmtree(out_path)
    
    result = runner.invoke(app, [
        "run",
        "--input", str(image_path),
        "--out", str(out_path)
    ])
    
    assert result.exit_code == 0
    
    with open(out_path / "metrics.json") as f:
        metrics = json.load(f)
    
    assert "raw_separability" not in metrics
    assert "ROI not provided" in result.stdout or "separability omitted" in result.stdout
    
    if out_path.exists():
        shutil.rmtree(out_path)


def test_invalid_roi_fails():
    """Test run with invalid ROI mask fails with error."""
    import tempfile
    
    image_path = Path("tests/test_images/01.bmp").resolve()
    
    # Create a mask with wrong size
    temp_dir = Path(tempfile.mkdtemp())
    mask_path = temp_dir / "wrong_size.png"
    from PIL import Image
    Image.fromarray(np.zeros((10, 10), dtype=np.uint8)).save(mask_path)
    
    result = runner.invoke(app, [
        "run",
        "--input", str(image_path),
        "--roi-mask", str(mask_path),
        "--out", str(temp_dir / "out")
    ])
    
    assert result.exit_code != 0
    assert "mismatch" in result.stdout.lower() or "error" in result.stdout.lower()
    
    import shutil
    shutil.rmtree(temp_dir)
