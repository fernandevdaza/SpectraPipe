"""COCO JSON annotation parser."""

import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from .voc_parser import BBox


class COCOParseError(Exception):
    """Error parsing COCO JSON file."""
    pass


@dataclass
class COCOImageAnnotation:
    """Annotations for a single image in COCO format."""
    image_id: int
    file_name: str
    width: int
    height: int
    bboxes: List[BBox] = field(default_factory=list)


@dataclass
class COCODataset:
    """Parsed COCO dataset."""
    images: Dict[int, COCOImageAnnotation]  # image_id -> annotation
    by_filename: Dict[str, COCOImageAnnotation]  # file_name -> annotation


def parse_coco_json(path: Path) -> COCODataset:
    """Parse COCO JSON annotation file.
    
    Args:
        path: Path to COCO JSON file.
    
    Returns:
        COCODataset with images indexed by id and filename.
    
    Raises:
        COCOParseError: If JSON is invalid or schema is unexpected.
    """
    path = Path(path)
    
    if not path.exists():
        raise COCOParseError(f"COCO JSON file not found: {path}")
    
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise COCOParseError(f"Invalid JSON: {e}")
    
    if not isinstance(data, dict):
        raise COCOParseError("COCO JSON must be an object")
    
    # Parse images
    images_list = data.get("images", [])
    if not images_list:
        raise COCOParseError("No 'images' array in COCO JSON")
    
    images: Dict[int, COCOImageAnnotation] = {}
    by_filename: Dict[str, COCOImageAnnotation] = {}
    
    for img in images_list:
        try:
            image_id = int(img["id"])
            file_name = str(img["file_name"])
            width = int(img["width"])
            height = int(img["height"])
        except (KeyError, ValueError, TypeError) as e:
            raise COCOParseError(f"Invalid image entry: {e}")
        
        annot = COCOImageAnnotation(
            image_id=image_id,
            file_name=file_name,
            width=width,
            height=height
        )
        images[image_id] = annot
        by_filename[file_name] = annot
    
    # Parse annotations (bboxes)
    annotations = data.get("annotations", [])
    
    for ann in annotations:
        try:
            image_id = int(ann["image_id"])
            bbox = ann.get("bbox")  # [x, y, width, height] in COCO format
        except (KeyError, ValueError, TypeError):
            continue  # Skip invalid annotations
        
        if image_id not in images:
            continue  # Skip annotations for unknown images
        
        if bbox and len(bbox) >= 4:
            x, y, w, h = bbox[:4]
            images[image_id].bboxes.append(BBox(
                xmin=int(x),
                ymin=int(y),
                xmax=int(x + w),
                ymax=int(y + h),
                label=str(ann.get("category_id", ""))
            ))
    
    return COCODataset(images=images, by_filename=by_filename)


def get_coco_annotation_for_image(
    dataset: COCODataset,
    file_name: Optional[str] = None,
    image_id: Optional[int] = None
) -> COCOImageAnnotation:
    """Get annotation for a specific image.
    
    Args:
        dataset: Parsed COCO dataset.
        file_name: Image filename to look up.
        image_id: Image ID to look up.
    
    Returns:
        COCOImageAnnotation for the image.
    
    Raises:
        COCOParseError: If image not found.
    """
    if image_id is not None and image_id in dataset.images:
        return dataset.images[image_id]
    
    if file_name is not None and file_name in dataset.by_filename:
        return dataset.by_filename[file_name]
    
    # Try matching by basename
    if file_name is not None:
        basename = Path(file_name).name
        for fn, annot in dataset.by_filename.items():
            if Path(fn).name == basename:
                return annot
    
    raise COCOParseError(
        f"Image not found in COCO annotations: "
        f"file_name={file_name}, image_id={image_id}"
    )
