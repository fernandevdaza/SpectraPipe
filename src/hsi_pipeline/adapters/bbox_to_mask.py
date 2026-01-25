"""Convert bounding boxes to ROI masks."""

import numpy as np
from typing import List, Tuple
from pathlib import Path

from .voc_parser import BBox


def clamp_bbox(
    bbox: BBox,
    width: int,
    height: int
) -> BBox:
    """Clamp bounding box to image dimensions.
    
    Args:
        bbox: Input bounding box.
        width: Image width.
        height: Image height.
    
    Returns:
        Clamped bounding box.
    """
    return BBox(
        xmin=max(0, min(bbox.xmin, width - 1)),
        ymin=max(0, min(bbox.ymin, height - 1)),
        xmax=max(0, min(bbox.xmax, width)),
        ymax=max(0, min(bbox.ymax, height)),
        label=bbox.label
    )


def bboxes_to_mask(
    bboxes: List[BBox],
    width: int,
    height: int,
    clamp: bool = True
) -> np.ndarray:
    """Convert list of bounding boxes to binary mask.
    
    Multiple bboxes are combined with OR (union).
    
    Args:
        bboxes: List of bounding boxes.
        width: Output mask width.
        height: Output mask height.
        clamp: Whether to clamp bboxes to image bounds.
    
    Returns:
        Binary mask (H, W) with True inside bboxes.
    """
    mask = np.zeros((height, width), dtype=bool)
    
    for bbox in bboxes:
        if clamp:
            bbox = clamp_bbox(bbox, width, height)
        
        # Skip invalid bboxes
        if bbox.xmax <= bbox.xmin or bbox.ymax <= bbox.ymin:
            continue
        
        mask[bbox.ymin:bbox.ymax, bbox.xmin:bbox.xmax] = True
    
    return mask


def save_mask(
    mask: np.ndarray,
    output_path: Path,
    format: str = "png"
) -> Path:
    """Save mask to file.
    
    Args:
        mask: Binary mask (H, W).
        output_path: Output file path.
        format: Output format ('png', 'npy', 'npz').
    
    Returns:
        Path to saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "png":
        from PIL import Image
        # Convert bool to uint8 for PNG
        img = Image.fromarray((mask * 255).astype(np.uint8))
        img.save(output_path)
    elif format == "npy":
        np.save(output_path, mask)
    elif format == "npz":
        np.savez_compressed(output_path, mask=mask)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return output_path


def generate_roi_from_annotation(
    annotation_path: Path,
    annotation_type: str,
    image_size: Tuple[int, int],
    image_filename: str = None,
    output_path: Path = None,
    output_format: str = "png"
) -> Tuple[np.ndarray, Path]:
    """Generate ROI mask from annotation file.
    
    Args:
        annotation_path: Path to VOC XML or COCO JSON.
        annotation_type: 'voc' or 'coco'.
        image_size: (height, width) of the image.
        image_filename: For COCO, the filename to look up.
        output_path: Optional path to save mask.
        output_format: Format for saving ('png', 'npy', 'npz').
    
    Returns:
        Tuple of (mask array, output path or None).
    """
    height, width = image_size
    
    if annotation_type == "voc":
        from .voc_parser import parse_voc_xml
        annot = parse_voc_xml(annotation_path)
        bboxes = annot.bboxes
    
    elif annotation_type == "coco":
        from .coco_parser import parse_coco_json, get_coco_annotation_for_image
        dataset = parse_coco_json(annotation_path)
        img_annot = get_coco_annotation_for_image(dataset, file_name=image_filename)
        bboxes = img_annot.bboxes
    
    else:
        raise ValueError(f"Unknown annotation type: {annotation_type}. Use 'voc' or 'coco'.")
    
    mask = bboxes_to_mask(bboxes, width, height)
    
    saved_path = None
    if output_path:
        saved_path = save_mask(mask, output_path, output_format)
    
    return mask, saved_path
