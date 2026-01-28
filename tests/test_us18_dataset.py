"""Tests for US-18: Dataset Annotation Mask Usage."""

import pytest
import json
import yaml
from PIL import Image
from typer.testing import CliRunner

from hsi_pipeline.cli import app

runner = CliRunner()

class TestUS18DatasetAnnotation:
    """Tests for using generated annotation masks in dataset command."""
    
    @pytest.fixture
    def mock_dataset(self, tmp_path):
        """Create a mock dataset with images and annotations."""
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        
        # Sample 1: Annotation ONLY
        img1 = dataset_dir / "sample1.jpg"
        Image.new('RGB', (100, 100), color='white').save(img1)
        
        annot1 = dataset_dir / "sample1.json"
        with open(annot1, "w") as f:
            json.dump({
                "sample1.jpg123": {
                    "filename": "sample1.jpg",
                    "regions": [
                        {"shape_attributes": {"name": "rect", "x": 0, "y": 0, "width": 50, "height": 50}, "region_attributes": {}}
                    ]
                }
            }, f)
            
        # Sample 2: Annotation AND ROI Mask (Conflict)
        img2 = dataset_dir / "sample2.jpg"
        Image.new('RGB', (100, 100), color='white').save(img2)
        
        annot2 = dataset_dir / "sample2.json"
        with open(annot2, "w") as f:
            json.dump({
                "sample2.jpg123": {
                    "filename": "sample2.jpg",
                    "regions": [{"shape_attributes": {"name": "rect", "x": 0, "y": 0, "width": 50, "height": 50}, "region_attributes": {}}]
                }
            }, f)
            
        mask2 = dataset_dir / "sample2_mask.png"
        Image.new('L', (100, 100), color=255).save(mask2) # Full white mask
        
        # Sample 3: No annotation, no mask (Baseline)
        img3 = dataset_dir / "sample3.jpg"
        Image.new('RGB', (100, 100), color='white').save(img3)
        
        # Manifest
        manifest = {
            "root": str(dataset_dir),
            "samples": [
                {
                    "id": "sample1",
                    "image": "sample1.jpg",
                    "annotation": "sample1.json",
                    "annotation_type": "via"
                },
                {
                    "id": "sample2",
                    "image": "sample2.jpg",
                    "annotation": "sample2.json",
                    "annotation_type": "via",
                    "roi_mask": "sample2_mask.png"
                },
                {
                    "id": "sample3",
                    "image": "sample3.jpg"
                }
            ]
        }
        
        manifest_path = tmp_path / "manifest.yaml"
        with open(manifest_path, "w") as f:
            yaml.dump(manifest, f)
            
        return manifest_path, dataset_dir, tmp_path / "output"

    def test_dataset_uses_annotation_mask(self, mock_dataset):
        """Sample 1 (Annotation only) should use generated mask."""
        manifest_path, _, output_dir = mock_dataset
        
        result = runner.invoke(app, [
            "dataset",
            "--manifest", str(manifest_path),
            "--out", str(output_dir)
        ])
        
        assert result.exit_code == 0
        
        # Check Sample 1
        s1_out = output_dir / "sample1"
        assert (s1_out / "roi_mask_annot.png").exists()
        
        s1_config = json.loads((s1_out / "run_config.json").read_text())
        assert s1_config["meta"].get("roi_source") == "annotation"
        assert "annotation_roi_path" in s1_config["meta"]

    def test_dataset_prioritizes_explicit_mask_and_warns(self, mock_dataset):
        """Sample 2 (Both) should use explicit mask and warn."""
        manifest_path, _, output_dir = mock_dataset
        
        # Patch the global console to capture output, as it might bypass CliRunner capture
        from unittest.mock import patch
        from rich.console import Console
        from io import StringIO
        
        capture_io = StringIO()
        mock_console = Console(file=capture_io, force_terminal=True)
        
        with patch("hsi_pipeline.cli.console", mock_console):
            result = runner.invoke(app, [
                "dataset",
                "--manifest", str(manifest_path),
                "--out", str(output_dir)
            ])
        
        assert result.exit_code == 0
        
        # Check warning in captured output
        output_str = capture_io.getvalue()
        assert "Warning" in output_str
        assert "Both annotation and roi_mask specified" in output_str
        
        # Check Sample 2
        s2_out = output_dir / "sample2"
        # Since logic calls process_annotation anyway, the annot mask IS generated
        assert (s2_out / "roi_mask_annot.png").exists()
        
        # BUT config should say mask_file used
        s2_config = json.loads((s2_out / "run_config.json").read_text())
        assert s2_config["meta"].get("roi_source") == "mask_file"
        
    def test_dataset_no_mask(self, mock_dataset):
        """Sample 3 should have roi_source: none."""
        manifest_path, _, output_dir = mock_dataset
        
        runner.invoke(app, [
            "dataset",
            "--manifest", str(manifest_path),
            "--out", str(output_dir)
        ])
        
        s3_out = output_dir / "sample3"
        s3_config = json.loads((s3_out / "run_config.json").read_text())
        assert s3_config["meta"].get("roi_source") == "none"
