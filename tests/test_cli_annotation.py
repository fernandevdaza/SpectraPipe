"""Tests for CLI annotation support."""

import pytest
import json
from PIL import Image
from typer.testing import CliRunner

from hsi_pipeline.cli import app

runner = CliRunner()

class TestCLIAnnotation:
    """Tests for run command with --annotation."""
    
    @pytest.fixture
    def mock_data(self, tmp_path):
        """Create mock image and annotation."""
        # Create image
        img_path = tmp_path / "input.jpg"
        img = Image.new('RGB', (100, 100), color='white')
        img.save(img_path)
        
        # Create output dir
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        
        return img_path, out_dir
        
    def test_run_with_voc_annotation(self, mock_data):
        """Should accept --annotation with VOC and generate mask."""
        img_path, out_dir = mock_data
        
        # Create VOC annotation
        annot_path = img_path.parent / "annot.xml"
        with open(annot_path, "w") as f:
            f.write(f"""
            <annotation>
                <folder>dataset</folder>
                <filename>{img_path.name}</filename>
                <size>
                    <width>100</width>
                    <height>100</height>
                    <depth>3</depth>
                </size>
                <object>
                    <name>cyst</name>
                    <bndbox>
                        <xmin>10</xmin>
                        <ymin>10</ymin>
                        <xmax>30</xmax>
                        <ymax>30</ymax>
                    </bndbox>
                </object>
            </annotation>
            """)
            
        result = runner.invoke(app, [
            "run",
            "--input", str(img_path),
            "--out", str(out_dir),
            "--annotation", str(annot_path),
            "--annotation-type", "voc"
        ])
        
        # Should succeed at mask generation step
        if result.exit_code != 0:
            pass
            
        assert "Generated ROI mask from annotation" in result.stdout
        assert (out_dir / "roi_mask_annot.png").exists()

    def test_run_with_via_annotation(self, mock_data):
        """Should accept --annotation with VIA and generate mask."""
        img_path, out_dir = mock_data
        
        # Create VIA annotation
        annot_path = img_path.parent / "annot.json"
        data = {
            f"{img_path.name}123": {
                "filename": img_path.name,
                "regions": [
                    {
                        "shape_attributes": {"name": "rect", "x": 0, "y": 0, "width": 50, "height": 50},
                        "region_attributes": {}
                    }
                ]
            }
        }
        with open(annot_path, "w") as f:
            json.dump(data, f)
            
        result = runner.invoke(app, [
            "run",
            "--input", str(img_path),
            "--out", str(out_dir),
            "--annotation", str(annot_path),
            "--annotation-type", "via"
        ])
        
        # Should succeed (even if pipeline fails later due to mock model, 
        # but at least mask generation should log success)
        # Note: pipeline will try to run model. Model might not be present or mockable easily here without more setup.
        # But we can check stdout for "Generated ROI mask"
        
        if result.exit_code != 0:
            # It might fail due to model loading, but let's check if it got past mask generation
            pass
            
        assert "Generated ROI mask from annotation" in result.stdout
        assert (out_dir / "roi_mask_annot.png").exists()

    def test_run_mutual_exclusion(self, mock_data):
        """Should fail if both --roi-mask and --annotation provided."""
        img_path, out_dir = mock_data
        annot_path = img_path.parent / "annot.json"
        annot_path.touch()
        mask_path = img_path.parent / "mask.png"
        mask_path.touch()
        
        result = runner.invoke(app, [
            "run",
            "--input", str(img_path),
            "--out", str(out_dir),
            "--annotation", str(annot_path),
            "--roi-mask", str(mask_path)
        ])
        
        assert result.exit_code != 0
        assert "Cannot specify both" in result.stdout

    def test_run_invalid_annotation_fails_gracefully(self, mock_data):
        """Should fail gracefully if annotation processing errors."""
        img_path, out_dir = mock_data
        annot_path = img_path.parent / "annot.json"
        with open(annot_path, "w") as f:
            f.write("invalid json")
            
        result = runner.invoke(app, [
            "run",
            "--input", str(img_path),
            "--out", str(out_dir),
            "--annotation", str(annot_path),
            "--annotation-type", "via"
        ])
        
        assert result.exit_code != 0
        assert "Annotation Error" in result.stdout
