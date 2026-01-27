"""Tests for annotation processor."""

import pytest
import json
import numpy as np
from PIL import Image
from unittest.mock import Mock

from hsi_pipeline.dataset.annotation_processor import process_sample_annotation, AnnotationError
from hsi_pipeline.manifest.parser import Sample

class TestProcessSampleAnnotation:
    """Tests for process_sample_annotation function."""
    
    @pytest.fixture
    def mock_sample(self, tmp_path):
        """Create a mock sample with necessary files."""
        # Create dummy image
        img_path = tmp_path / "test_image.jpg"
        img = Image.new('RGB', (100, 100), color='white')
        img.save(img_path)
        
        sample = Mock(spec=Sample)
        sample.id = "sample1"
        sample.image = "test_image.jpg"
        sample.image_resolved = img_path
        sample.annotation = "annotations.json"
        
        return sample
    
    def test_process_via_annotation_success(self, mock_sample, tmp_path):
        """Should process VIA annotation correctly."""
        # Create VIA annotation file
        annot_path = tmp_path / "annotations.json"
        mock_sample.annotation_resolved = annot_path
        mock_sample.annotation_type = "via"
        
        data = {
            "test_image.jpg123": {
                "filename": "test_image.jpg",
                "regions": [
                    {
                        "shape_attributes": {
                            "name": "rect",
                            "x": 10, "y": 10, "width": 20, "height": 20
                        },
                        "region_attributes": {}
                    }
                ]
            }
        }
        with open(annot_path, "w") as f:
            json.dump(data, f)
            
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        mask, saved_path = process_sample_annotation(mock_sample, output_dir)
        
        assert mask is not None
        assert mask.shape == (100, 100)
        assert np.sum(mask) > 0
        assert saved_path.exists()
        assert saved_path.name == "roi_mask_annot.png"

    def test_via_image_not_found_raises(self, mock_sample, tmp_path):
        """Should raise error if image not found in VIA project."""
        annot_path = tmp_path / "annotations.json"
        mock_sample.annotation_resolved = annot_path
        mock_sample.annotation_type = "via"
        
        # Annotation for DIFFERENT image
        data = {
            "other.jpg123": {
                "filename": "other.jpg",
                "regions": []
            }
        }
        with open(annot_path, "w") as f:
            json.dump(data, f)
            
        output_dir = tmp_path
        
        with pytest.raises(AnnotationError, match="not found in VIA project"):
            process_sample_annotation(mock_sample, output_dir)

    def test_via_parse_error_raises(self, mock_sample, tmp_path):
        """Should raise AnnotationError if VIA parsing fails."""
        annot_path = tmp_path / "annotations.json"
        mock_sample.annotation_resolved = annot_path
        mock_sample.annotation_type = "via"
        
        with open(annot_path, "w") as f:
            f.write("invalid json")
            
        output_dir = tmp_path
        
        with pytest.raises(AnnotationError, match="VIA parse error"):
            process_sample_annotation(mock_sample, output_dir)

    def test_voc_annotation(self, mock_sample, tmp_path):
        """Should process VOC annotation (sanity check)."""
        annot_path = tmp_path / "annot.xml"
        mock_sample.annotation_resolved = annot_path
        mock_sample.annotation_type = "voc"
        
        with open(annot_path, "w") as f:
            f.write("""
            <annotation>
                <folder>dataset</folder>
                <filename>test_image.jpg</filename>
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
            
        output_dir = tmp_path / "output_voc"
        output_dir.mkdir()
        
        mask, saved_path = process_sample_annotation(mock_sample, output_dir)
        
        assert mask is not None
        assert saved_path.name == "roi_mask_annot.png" 
