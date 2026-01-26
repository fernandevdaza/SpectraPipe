"""VOC XML annotation parser."""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List
from dataclasses import dataclass


class VOCParseError(Exception):
    """Error parsing VOC XML file."""
    pass


@dataclass
class BBox:
    """Bounding box representation."""
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    label: str = ""
    
    @property
    def width(self) -> int:
        return self.xmax - self.xmin
    
    @property
    def height(self) -> int:
        return self.ymax - self.ymin


@dataclass
class VOCAnnotation:
    """Parsed VOC annotation."""
    filename: str
    width: int
    height: int
    bboxes: List[BBox]


def parse_voc_xml(path: Path) -> VOCAnnotation:
    """Parse VOC XML annotation file.
    
    Args:
        path: Path to VOC XML file.
    
    Returns:
        VOCAnnotation with filename, dimensions, and bboxes.
    
    Raises:
        VOCParseError: If XML is invalid or required tags are missing.
    """
    path = Path(path)
    
    if not path.exists():
        raise VOCParseError(f"VOC XML file not found: {path}")
    
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except ET.ParseError as e:
        raise VOCParseError(f"Invalid XML: {e}")
    
    # Get filename
    filename_elem = root.find("filename")
    if filename_elem is None or not filename_elem.text:
        raise VOCParseError("Missing <filename> tag in VOC XML")
    filename = filename_elem.text
    
    # Get size
    size = root.find("size")
    if size is None:
        raise VOCParseError("Missing <size> tag in VOC XML")
    
    width_elem = size.find("width")
    height_elem = size.find("height")
    
    if width_elem is None or height_elem is None:
        raise VOCParseError("Missing <width> or <height> in <size>")
    
    try:
        width = int(width_elem.text)
        height = int(height_elem.text)
    except (ValueError, TypeError):
        raise VOCParseError("Invalid width/height values in <size>")
    
    # Parse objects/bboxes
    bboxes = []
    for obj in root.findall("object"):
        bbox_elem = obj.find("bndbox")
        if bbox_elem is None:
            continue  # Skip objects without bbox
        
        try:
            xmin = int(float(bbox_elem.find("xmin").text))
            ymin = int(float(bbox_elem.find("ymin").text))
            xmax = int(float(bbox_elem.find("xmax").text))
            ymax = int(float(bbox_elem.find("ymax").text))
        except (AttributeError, ValueError, TypeError):
            raise VOCParseError("Invalid bbox coordinates in <bndbox>")
        
        name_elem = obj.find("name")
        label = name_elem.text if name_elem is not None else ""
        
        bboxes.append(BBox(
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            label=label
        ))
    
    if not bboxes:
        raise VOCParseError("No valid bboxes found in VOC XML")
    
    return VOCAnnotation(
        filename=filename,
        width=width,
        height=height,
        bboxes=bboxes
    )
