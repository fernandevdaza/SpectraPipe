"""VIA (VGG Image Annotator) JSON parser and mask generator.

Supports shapes: ellipse, polygon, rect.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from PIL import Image, ImageDraw


class VIAParseError(Exception):
    """Error parsing VIA JSON."""
    pass


@dataclass
class VIARegion:
    """A single annotated region in VIA."""
    shape_attributes: Dict[str, Any]
    region_attributes: Dict[str, Any]


@dataclass
class VIAAnnotation:
    """VIA annotation for a single image."""
    filename: str
    size: int
    regions: List[VIARegion]


@dataclass
class VIAProject:
    """Parsed VIA project containing multiple image annotations."""
    images: Dict[str, VIAAnnotation] = field(default_factory=dict)


def parse_via_json(path: Path) -> VIAProject:
    """Parse a VIA JSON file.
    
    Args:
        path: Path to VIA JSON file.
        
    Returns:
        VIAProject object containing parsed annotations.
        
    Raises:
        VIAParseError: If JSON is invalid or missing required fields.
    """
    path = Path(path)
    if not path.exists():
        raise VIAParseError(f"VIA file not found: {path}")
        
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise VIAParseError(f"Invalid JSON: {e}")

    project = VIAProject()
    
    # VIA Key format: "filename + size" e.g. "image.jpg12345"
    # But usually we just iterate over values
    if not isinstance(data, dict):
         raise VIAParseError("Root element must be a dictionary (VIA project structure)")

    # Check if it's a VIA project (has '_via_settings') or just the regions dict
    # Usually VIA exports have the image map directly or inside '_via_img_metadata'
    
    img_metadata = data
    if "_via_img_metadata" in data:
        img_metadata = data["_via_img_metadata"]

    for key, item in img_metadata.items():
        if not isinstance(item, dict):
            continue
            
        filename = item.get("filename")
        if not filename:
            continue
            
        size = item.get("size", 0)
        regions_data = item.get("regions", [])
        
        parsed_regions = []
        for region in regions_data:
            shape_attr = region.get("shape_attributes", {})
            region_attr = region.get("region_attributes", {})
            if shape_attr:
                parsed_regions.append(VIARegion(
                    shape_attributes=shape_attr,
                    region_attributes=region_attr
                ))
        
        # Use filename as key for easier lookup later, or keep original key
        # We will map by filename essentially
        project.images[filename] = VIAAnnotation(
            filename=filename,
            size=size,
            regions=parsed_regions
        )
        
    return project


def ellipse_to_mask(cx: float, cy: float, rx: float, ry: float, theta: float, width: int, height: int) -> np.ndarray:
    """Convert rotated ellipse to binary mask."""
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    # Generate ellipse points with rotation
    # Theta in VIA is usually radians. 
    # Use enough points for smoothness.
    points = []
    steps = 72 # Every 5 degrees
    
    # Validations for radii
    if rx <= 0 or ry <= 0:
        return np.zeros((height, width), dtype=bool)

    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    
    for i in range(steps):
        angle = 2 * math.pi * i / steps
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        # Parametric equation for rotated ellipse
        # x = cx + rx*cos(t)*cos(theta) - ry*sin(t)*sin(theta)
        # y = cy + rx*cos(t)*sin(theta) + ry*sin(t)*cos(theta)
        
        x = cx + rx * cos_a * cos_theta - ry * sin_a * sin_theta
        y = cy + rx * cos_a * sin_theta + ry * sin_a * cos_theta
        points.append((x, y))
    
    if len(points) > 2:
        draw.polygon(points, fill=1, outline=1)
    
    return np.array(mask, dtype=bool)


def polygon_to_mask(all_points_x: List[float], all_points_y: List[float], width: int, height: int) -> np.ndarray:
    """Convert polygon to binary mask."""
    if len(all_points_x) != len(all_points_y) or len(all_points_x) < 3:
        return np.zeros((height, width), dtype=bool)
        
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    points = list(zip(all_points_x, all_points_y))
    draw.polygon(points, fill=1, outline=1)
    
    return np.array(mask, dtype=bool)


def rect_to_mask(x: float, y: float, w: float, h: float, width: int, height: int) -> np.ndarray:
    """Convert rectangle to binary mask."""
    if w <= 0 or h <= 0:
        return np.zeros((height, width), dtype=bool)

    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    draw.rectangle([x, y, x + w, y + h], fill=1, outline=1)
    
    return np.array(mask, dtype=bool)


def via_regions_to_mask(regions: List[VIARegion], width: int, height: int) -> np.ndarray:
    """Convert a list of VIA regions to a unified binary mask.
    
    Args:
        regions: List of VIARegion objects.
        width: Image width.
        height: Image height.
        
    Returns:
        Binary mask (boolean numpy array) where True indicates ROI.
    """
    combined_mask = np.zeros((height, width), dtype=bool)
    
    for region in regions:
        shape = region.shape_attributes
        name = shape.get("name")
        
        if name == "ellipse":
            mask = ellipse_to_mask(
                cx=float(shape.get("cx", 0)),
                cy=float(shape.get("cy", 0)),
                rx=float(shape.get("rx", 0)),
                ry=float(shape.get("ry", 0)),
                theta=float(shape.get("theta", 0)),
                width=width,
                height=height
            )
            combined_mask |= mask
            
        elif name == "polygon":
            xs = shape.get("all_points_x", [])
            ys = shape.get("all_points_y", [])
            mask = polygon_to_mask(xs, ys, width, height)
            combined_mask |= mask
            
        elif name == "rect":
            mask = rect_to_mask(
                x=float(shape.get("x", 0)),
                y=float(shape.get("y", 0)),
                w=float(shape.get("width", 0)),
                h=float(shape.get("height", 0)),
                width=width,
                height=height
            )
            combined_mask |= mask
            
    return combined_mask
