"""Unit tests for VIA parser."""

import json
import pytest
import numpy as np
from pathlib import Path
from PIL import Image

from hsi_pipeline.adapters.via_parser import (
    parse_via_json, via_regions_to_mask, ellipse_to_mask, 
    polygon_to_mask, rect_to_mask, VIAParseError
)

class TestVIAParser:
    """Tests for VIA JSON parsing."""
    
    def test_parse_valid_via_json(self, tmp_path):
        """Should parse valid VIA JSON correctly."""
        data = {
            "image1.jpg123": {
                "filename": "image1.jpg",
                "size": 123,
                "regions": [
                    {
                        "shape_attributes": {
                            "name": "ellipse",
                            "cx": 100, "cy": 100, "rx": 50, "ry": 30, "theta": 0
                        },
                        "region_attributes": {"label": "cyst"}
                    }
                ]
            }
        }
        
        path = tmp_path / "via.json"
        with open(path, "w") as f:
            json.dump(data, f)
            
        project = parse_via_json(path)
        
        assert "image1.jpg" in project.images
        annot = project.images["image1.jpg"]
        assert annot.filename == "image1.jpg"
        assert len(annot.regions) == 1
        assert annot.regions[0].shape_attributes["name"] == "ellipse"

    def test_parse_via_with_metadata_key(self, tmp_path):
        """Should parse VIA JSON with _via_img_metadata key."""
        data = {
            "_via_settings": {},
            "_via_img_metadata": {
                "image1.jpg": {
                    "filename": "image1.jpg",
                    "regions": []
                }
            }
        }
        
        path = tmp_path / "via_meta.json"
        with open(path, "w") as f:
            json.dump(data, f)
            
        project = parse_via_json(path)
        assert "image1.jpg" in project.images

    def test_file_not_found_raises(self):
        """Should raise VIAParseError if file missing."""
        with pytest.raises(VIAParseError):
            parse_via_json(Path("nonexistent.json"))

    def test_invalid_json_raises(self, tmp_path):
        """Should raise VIAParseError for invalid JSON."""
        path = tmp_path / "invalid.json"
        path.write_text("{invalid json")
        
        with pytest.raises(VIAParseError):
            parse_via_json(path)


class TestMaskGeneration:
    """Tests for mask generation functions."""
    
    def test_ellipse_to_mask(self):
        """Should generate mask for ellipse."""
        mask = ellipse_to_mask(50, 50, 20, 10, 0, 100, 100)
        assert mask.shape == (100, 100)
        assert mask.dtype == bool
        assert np.sum(mask) > 0
        # Center should be True
        assert mask[50, 50]

    def test_polygon_to_mask(self):
        """Should generate mask for polygon."""
        xs = [10, 20, 20, 10]
        ys = [10, 10, 20, 20]
        mask = polygon_to_mask(xs, ys, 30, 30)
        
        assert mask.shape == (30, 30)
        assert np.sum(mask) > 0
        assert mask[15, 15]

    def test_rect_to_mask(self):
        """Should generate mask for rectangle."""
        mask = rect_to_mask(10, 10, 20, 20, 50, 50)
        
        assert mask.shape == (50, 50)
        # Check area approx 20*20 = 400
        # Simple check: center is filled
        assert mask[20, 20]

    def test_via_regions_to_mask_multiple(self):
        """Should combine multiple regions into one mask."""
        from hsi_pipeline.adapters.via_parser import VIARegion
        
        regions = [
            VIARegion(
                shape_attributes={"name": "rect", "x": 0, "y": 0, "width": 10, "height": 10},
                region_attributes={}
            ),
            VIARegion(
                shape_attributes={"name": "rect", "x": 20, "y": 20, "width": 10, "height": 10},
                region_attributes={}
            )
        ]
        
        mask = via_regions_to_mask(regions, 40, 40)
        
        assert mask[5, 5]
        assert mask[25, 25]
        assert not mask[15, 15]  # Space between

    def test_via_regions_unknown_shape_ignored(self):
        """Should ignore unknown shapes."""
        from hsi_pipeline.adapters.via_parser import VIARegion
        regions = [
            VIARegion(
                shape_attributes={"name": "unknown"},
                region_attributes={}
            )
        ]
        mask = via_regions_to_mask(regions, 10, 10)
        assert np.sum(mask) == 0
