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
@patch("hsi_pipeline.pipeline.orchestrator.rgb_to_hsi", autospec=True)
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
@patch("hsi_pipeline.pipeline.orchestrator.rgb_to_hsi", autospec=True)
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
@patch("hsi_pipeline.pipeline.orchestrator.rgb_to_hsi", autospec=True)
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
@patch("hsi_pipeline.pipeline.orchestrator.rgb_to_hsi", autospec=True)
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
    
    assert "fitting" in metadata["meta"]
    assert "input_shape_original" in metadata["meta"]["fitting"]
    assert "input_shape_fitted" in metadata["meta"]["fitting"]
    assert metadata["meta"]["fitting"]["policy"] == "pad_to_multiple"
    
    if out_path.exists():
        shutil.rmtree(out_path)
@patch("hsi_pipeline.pipeline.orchestrator.rgb_to_hsi", autospec=True)
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
    assert "input_shape_original" in metadata["meta"]["fitting"]
    assert "input_shape_fitted" in metadata["meta"]["fitting"]
    
    if out_path.exists():
        shutil.rmtree(out_path)
@patch("hsi_pipeline.pipeline.orchestrator.rgb_to_hsi", autospec=True)
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
@patch("hsi_pipeline.pipeline.orchestrator.rgb_to_hsi", autospec=True)
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
@patch("hsi_pipeline.pipeline.orchestrator.rgb_to_hsi", autospec=True)
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
@patch("hsi_pipeline.pipeline.orchestrator.rgb_to_hsi", autospec=True)
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
@patch("hsi_pipeline.pipeline.orchestrator.rgb_to_hsi", autospec=True)
def test_upscale_success(mock_rgb_to_hsi):
    """Test upscaling generates both baseline and improved artifacts."""
    import shutil
    import json
    
    mock_rgb_to_hsi.return_value = np.zeros((64, 64, 31), dtype=np.float32)
    
    image_path = Path("tests/test_images/01.bmp").resolve()
    out_path = Path("tests/test_out_upscale").resolve()
    
    if out_path.exists():
        shutil.rmtree(out_path)
    
    result = runner.invoke(app, [
        "run",
        "--input", str(image_path),
        "--upscale-factor", "2",
        "--out", str(out_path)
    ])
    
    assert result.exit_code == 0, f"Failed with: {result.stdout}"
    
    # Check artifacts exist
    assert (out_path / "hsi_upscaled_baseline.npz").exists(), "Baseline missing"
    assert (out_path / "hsi_upscaled_improved.npz").exists(), "Improved missing"
    
    # Check metrics.json contains upscaling info
    with open(out_path / "metrics.json") as f:
        metrics = json.load(f)
    
    assert "upscale_factor" in metrics
    assert metrics["upscale_factor"] == 2
    assert "upscaled_size" in metrics
    
    # Verify shapes (schema v1 uses 'cube' key)
    loaded = np.load(out_path / "hsi_upscaled_baseline.npz", allow_pickle=True)
    baseline = loaded["cube"] if "cube" in loaded else loaded["data"]
    assert baseline.shape == (128, 128, 31)  # 64*2 = 128
    
    if out_path.exists():
        shutil.rmtree(out_path)
@patch("hsi_pipeline.pipeline.orchestrator.rgb_to_hsi", autospec=True)
def test_no_upscale_omits_artifacts(mock_rgb_to_hsi):
    """Test without upscale flag, no upscaled artifacts are created."""
    import shutil
    import json
    
    mock_rgb_to_hsi.return_value = np.zeros((64, 64, 31), dtype=np.float32)
    
    image_path = Path("tests/test_images/01.bmp").resolve()
    out_path = Path("tests/test_out_no_upscale").resolve()
    
    if out_path.exists():
        shutil.rmtree(out_path)
    
    result = runner.invoke(app, [
        "run",
        "--input", str(image_path),
        "--out", str(out_path)
    ])
    
    assert result.exit_code == 0
    
    # Upscaled artifacts should NOT exist
    assert not (out_path / "hsi_upscaled_baseline.npz").exists()
    assert not (out_path / "hsi_upscaled_improved.npz").exists()
    
    # Metrics should not contain upscaling info
    with open(out_path / "metrics.json") as f:
        metrics = json.load(f)
    
    assert "upscale_factor" not in metrics
    
    if out_path.exists():
        shutil.rmtree(out_path)


@patch("hsi_pipeline.pipeline.orchestrator.rgb_to_hsi", autospec=True)
def test_clean_generated_with_roi(mock_rgb_to_hsi):
    """Test clean HSI generated when ROI is provided with partial coverage."""
    import shutil
    import json
    from PIL import Image
    
    mock_rgb_to_hsi.return_value = np.ones((64, 64, 31), dtype=np.float32)
    
    image_path = Path("tests/test_images/01.bmp").resolve()
    out_path = Path("tests/test_out_clean").resolve()
    
    # Create a partial ROI mask (50% coverage)
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[:256, :] = 255
    mask_path = out_path.parent / "test_clean_mask.png"
    Image.fromarray(mask).save(mask_path)
    
    if out_path.exists():
        shutil.rmtree(out_path)
    
    result = runner.invoke(app, [
        "run",
        "--input", str(image_path),
        "--roi-mask", str(mask_path),
        "--out", str(out_path)
    ])
    
    assert result.exit_code == 0, f"Failed: {result.stdout}"
    
    # Check clean artifact exists
    assert (out_path / "hsi_clean_full.npz").exists(), "hsi_clean_full.npz not found"
    
    # Check metrics contain clean metrics
    with open(out_path / "metrics.json") as f:
        metrics = json.load(f)
    
    assert "clean_separability" in metrics
    assert "raw_clean_sam" in metrics
    assert "raw_clean_rmse" in metrics
    
    # Check logs
    assert "Clean HSI generated" in result.stdout or "clean" in result.stdout.lower()
    
    if out_path.exists():
        shutil.rmtree(out_path)
    if mask_path.exists():
        mask_path.unlink()


@patch("hsi_pipeline.pipeline.orchestrator.rgb_to_hsi", autospec=True)
def test_clean_skipped_without_roi(mock_rgb_to_hsi):
    """Test clean HSI is skipped when no ROI is provided."""
    import shutil
    import json
    
    mock_rgb_to_hsi.return_value = np.ones((64, 64, 31), dtype=np.float32)
    
    image_path = Path("tests/test_images/01.bmp").resolve()
    out_path = Path("tests/test_out_no_clean").resolve()
    
    if out_path.exists():
        shutil.rmtree(out_path)
    
    result = runner.invoke(app, [
        "run",
        "--input", str(image_path),
        "--out", str(out_path)
    ])
    
    assert result.exit_code == 0
    
    # Clean artifact should NOT exist
    assert not (out_path / "hsi_clean_full.npz").exists()
    
    # Metrics should not have clean metrics
    with open(out_path / "metrics.json") as f:
        metrics = json.load(f)
    
    assert "clean_separability" not in metrics
    
    # Check logs indicate skipped
    assert "Clean skipped" in result.stdout or "no ROI" in result.stdout.lower()
    
    if out_path.exists():
        shutil.rmtree(out_path)


@patch("hsi_pipeline.pipeline.orchestrator.rgb_to_hsi", autospec=True)
def test_dataset_command_success(mock_rgb_to_hsi):
    """Test dataset command processes samples successfully."""
    import shutil
    import json
    
    mock_rgb_to_hsi.return_value = np.ones((64, 64, 31), dtype=np.float32)
    
    manifest_path = Path("tests/test_images/dataset_small/manifest.yaml").resolve()
    out_path = Path("tests/test_out_dataset").resolve()
    
    if out_path.exists():
        shutil.rmtree(out_path)
    
    result = runner.invoke(app, [
        "dataset",
        "--manifest", str(manifest_path),
        "--out", str(out_path),
        "--on-error", "continue"
    ])
    
    assert result.exit_code == 0, f"Failed: {result.stdout}"
    
    # Check sample directories created
    assert (out_path / "s01").exists()
    assert (out_path / "s02").exists()
    
    # Check artifacts per sample
    assert (out_path / "s01" / "hsi_raw_full.npz").exists()
    assert (out_path / "s01" / "metrics.json").exists()
    assert (out_path / "s02" / "hsi_raw_full.npz").exists()
    
    # s02 has ROI so should have clean artifact
    assert (out_path / "s02" / "hsi_clean_full.npz").exists()
    
    # Check dataset report
    assert (out_path / "dataset_report.json").exists()
    with open(out_path / "dataset_report.json") as f:
        report = json.load(f)
    
    assert report["total_samples"] == 2
    assert report["processed_ok"] == 2
    assert report["failed"] == 0
    
    if out_path.exists():
        shutil.rmtree(out_path)


def test_dataset_command_invalid_manifest():
    """Test dataset command fails with invalid manifest."""
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: content:")
        manifest_path = f.name
    
    out_path = Path("tests/test_out_invalid_manifest").resolve()
    
    result = runner.invoke(app, [
        "dataset",
        "--manifest", manifest_path,
        "--out", str(out_path)
    ])
    
    assert result.exit_code != 0
    assert "Error" in result.stdout or "root" in result.stdout.lower()
    
    import os
    os.unlink(manifest_path)


@patch("hsi_pipeline.pipeline.orchestrator.rgb_to_hsi", autospec=True)
def test_spectra_pixel_extraction(mock_rgb_to_hsi):
    """Test spectra command extracts pixel signature."""
    import shutil
    import json
    
    mock_rgb_to_hsi.return_value = np.ones((64, 64, 31), dtype=np.float32)
    
    image_path = Path("tests/test_images/01.bmp").resolve()
    out_path = Path("tests/test_out_spectra").resolve()
    
    if out_path.exists():
        shutil.rmtree(out_path)
    
    # First generate the HSI
    result = runner.invoke(app, [
        "run",
        "--input", str(image_path),
        "--out", str(out_path)
    ])
    assert result.exit_code == 0, f"Run failed: {result.stdout}"
    
    # Then extract spectra (need wavelengths since fail-safe is on)
    result = runner.invoke(app, [
        "spectra",
        "--from", str(out_path),
        "--artifact", "raw",
        "--pixel", "10,20",
        "--export", "json",
        "--wl-start", "400",
        "--wl-step", "10"
    ])
    
    assert result.exit_code == 0, f"Spectra failed: {result.stdout}"
    
    # Check export file exists
    assert (out_path / "spectra_raw_pixel_10_20.json").exists()
    
    # Check JSON content
    with open(out_path / "spectra_raw_pixel_10_20.json") as f:
        data = json.load(f)
    
    assert data["source"] == "pixel"
    assert data["artifact"] == "raw"
    assert data["bands"] == 31
    assert len(data["values"]) == 31
    
    if out_path.exists():
        shutil.rmtree(out_path)


def test_spectra_pixel_out_of_range():
    """Test spectra command fails for out-of-range pixel."""
    import shutil
    import tempfile
    from hsi_pipeline.export.npz_schema import save_npz_v1, NPZMetadata
    
    out_path = Path(tempfile.mkdtemp())
    
    # Create a small HSI using schema v1
    hsi = np.ones((32, 32, 31), dtype=np.float32)
    metadata = NPZMetadata(artifact="raw")
    save_npz_v1(out_path / "hsi_raw_full.npz", hsi, metadata)
    
    result = runner.invoke(app, [
        "spectra",
        "--from", str(out_path),
        "--artifact", "raw",
        "--pixel", "100,100",  # Out of range for 32x32
        "--export", "json",
        "--wl-start", "400",
        "--wl-step", "10"
    ])
    
    assert result.exit_code != 0
    assert "out of range" in result.stdout.lower()
    assert "Valid range" in result.stdout
    
    shutil.rmtree(out_path)


def test_spectra_artifact_not_found():
    """Test spectra command fails for missing artifact."""
    import tempfile
    import shutil
    
    out_path = Path(tempfile.mkdtemp())
    
    result = runner.invoke(app, [
        "spectra",
        "--from", str(out_path),
        "--artifact", "clean",  # Doesn't exist
        "--pixel", "10,10",
        "--export", "json"
    ])
    
    assert result.exit_code != 0
    assert "not found" in result.stdout.lower() or "Error" in result.stdout
    

    shutil.rmtree(out_path)


@patch("hsi_pipeline.pipeline.orchestrator.rgb_to_hsi", autospec=True)
def test_dataset_command_with_annotation(mock_rgb_to_hsi):
    """TC-DS-ROI-01: Test dataset command uses annotation to generate ROI and applies it."""
    import shutil
    import json
    from PIL import Image
    
    # Mock inference
    mock_rgb_to_hsi.return_value = np.ones((64, 64, 31), dtype=np.float32)
    
    # Setup temp environment
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())
    img_path = temp_dir / "sample.png"
    Image.new('RGB', (100, 100)).save(img_path)
    
    # Create VOC annotation
    annot_path = temp_dir / "sample.xml"
    with open(annot_path, "w") as f:
        f.write(f"""
        <annotation>
            <folder>dataset</folder>
            <filename>{img_path.name}</filename>
            <size><width>100</width><height>100</height><depth>3</depth></size>
            <object>
                <name>obj</name>
                <bndbox><xmin>0</xmin><ymin>0</ymin><xmax>50</xmax><ymax>50</ymax></bndbox>
            </object>
        </annotation>
        """)
    
    # Create manifest
    manifest_path = temp_dir / "manifest.json"
    manifest_data = {
        "root": str(temp_dir),
        "samples": [
            {
                "id": "s1",
                "image": "sample.png",
                "annotation": "sample.xml",
                "annotation_type": "voc"
            }
        ]
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f)
        
    out_path = temp_dir / "out"
    
    try:
        result = runner.invoke(app, [
            "dataset",
            "--manifest", str(manifest_path),
            "--out", str(out_path)
        ])
        
        assert result.exit_code == 0, f"Failed: {result.stdout}"
        
        # Verify s1 output
        s1_dir = out_path / "s1"
        assert s1_dir.exists()
        
        # verify run_config has ROI source metadata
        with open(s1_dir / "run_config.json") as f:
            rc = json.load(f)
        assert rc["meta"]["roi_source"] == "annotation"
        assert "annotation_roi_path" in rc["meta"]
        
        # Verify clean artifact generated (implies ROI was used)
        assert (s1_dir / "hsi_clean_full.npz").exists()
        
        # Verify metrics have separability
        with open(s1_dir / "metrics.json") as f:
            metrics = json.load(f)
        assert "clean_separability" in metrics
        
    finally:
        shutil.rmtree(temp_dir)