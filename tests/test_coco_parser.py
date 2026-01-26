"""Unit tests for COCO JSON parser."""

import pytest
import json
from pathlib import Path

from hsi_pipeline.adapters.coco_parser import (
    parse_coco_json, get_coco_annotation_for_image,
    COCOParseError, COCODataset
)


class TestParseCOCOJSON:
    """Tests for parse_coco_json function."""

    def test_parse_valid_json(self, tmp_path):
        """Should parse valid COCO JSON."""
        coco_data = {
            "images": [
                {"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480}
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "bbox": [100, 50, 200, 150], "category_id": 1}
            ],
            "categories": [{"id": 1, "name": "dog"}]
        }
        json_path = tmp_path / "coco.json"
        with open(json_path, "w") as f:
            json.dump(coco_data, f)
        
        result = parse_coco_json(json_path)
        
        assert isinstance(result, COCODataset)
        assert 1 in result.images
        assert result.images[1].file_name == "image1.jpg"
        assert len(result.images[1].bboxes) == 1
        assert result.images[1].bboxes[0].xmin == 100

    def test_multiple_images(self, tmp_path):
        """Should parse multiple images."""
        coco_data = {
            "images": [
                {"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480},
                {"id": 2, "file_name": "img2.jpg", "width": 800, "height": 600}
            ],
            "annotations": []
        }
        json_path = tmp_path / "coco.json"
        with open(json_path, "w") as f:
            json.dump(coco_data, f)
        
        result = parse_coco_json(json_path)
        
        assert len(result.images) == 2
        assert "img1.jpg" in result.by_filename

    def test_bbox_conversion(self, tmp_path):
        """COCO bbox [x,y,w,h] should convert to xmin,ymin,xmax,ymax."""
        coco_data = {
            "images": [{"id": 1, "file_name": "img.jpg", "width": 640, "height": 480}],
            "annotations": [
                {"id": 1, "image_id": 1, "bbox": [10, 20, 100, 50]}  # x,y,w,h
            ]
        }
        json_path = tmp_path / "coco.json"
        with open(json_path, "w") as f:
            json.dump(coco_data, f)
        
        result = parse_coco_json(json_path)
        bbox = result.images[1].bboxes[0]
        
        assert bbox.xmin == 10
        assert bbox.ymin == 20
        assert bbox.xmax == 110  # x + w
        assert bbox.ymax == 70   # y + h

    def test_file_not_found_raises(self):
        """Should raise COCOParseError for missing file."""
        with pytest.raises(COCOParseError, match="not found"):
            parse_coco_json(Path("/nonexistent/coco.json"))

    def test_invalid_json_raises(self, tmp_path):
        """Should raise COCOParseError for invalid JSON."""
        json_path = tmp_path / "invalid.json"
        json_path.write_text("{not valid json}")
        
        with pytest.raises(COCOParseError, match="Invalid JSON"):
            parse_coco_json(json_path)

    def test_no_images_raises(self, tmp_path):
        """Should raise COCOParseError if no images array."""
        coco_data = {"annotations": []}
        json_path = tmp_path / "coco.json"
        with open(json_path, "w") as f:
            json.dump(coco_data, f)
        
        with pytest.raises(COCOParseError, match="No 'images'"):
            parse_coco_json(json_path)


class TestGetCOCOAnnotation:
    """Tests for get_coco_annotation_for_image function."""

    @pytest.fixture
    def sample_dataset(self, tmp_path):
        coco_data = {
            "images": [
                {"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480},
                {"id": 2, "file_name": "subdir/image2.jpg", "width": 800, "height": 600}
            ],
            "annotations": []
        }
        json_path = tmp_path / "coco.json"
        with open(json_path, "w") as f:
            json.dump(coco_data, f)
        return parse_coco_json(json_path)

    def test_get_by_image_id(self, sample_dataset):
        """Should find image by ID."""
        result = get_coco_annotation_for_image(sample_dataset, image_id=1)
        
        assert result.file_name == "image1.jpg"

    def test_get_by_filename(self, sample_dataset):
        """Should find image by filename."""
        result = get_coco_annotation_for_image(sample_dataset, file_name="image1.jpg")
        
        assert result.image_id == 1

    def test_get_by_basename_match(self, sample_dataset):
        """Should match by basename if full path doesn't match."""
        result = get_coco_annotation_for_image(sample_dataset, file_name="image2.jpg")
        
        assert result.image_id == 2

    def test_image_not_found_raises(self, sample_dataset):
        """Should raise COCOParseError for unknown image."""
        with pytest.raises(COCOParseError, match="not found"):
            get_coco_annotation_for_image(sample_dataset, file_name="unknown.jpg")
