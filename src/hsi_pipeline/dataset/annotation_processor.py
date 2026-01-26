"""Annotation processing utilities for dataset runner."""

from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from PIL import Image

from ..manifest.parser import Sample
from ..adapters.voc_parser import parse_voc_xml, VOCParseError
from ..adapters.coco_parser import (
    parse_coco_json, get_coco_annotation_for_image, COCOParseError
)
from ..adapters.bbox_to_mask import bboxes_to_mask, save_mask


class AnnotationError(Exception):
    """Error processing annotation."""
    pass


def process_sample_annotation(
    sample: Sample,
    output_dir: Path,
    coco_dataset_cache: Optional[dict] = None
) -> Tuple[Optional[np.ndarray], Optional[Path]]:
    """Process annotation for a sample and generate ROI mask.
    
    Args:
        sample: Sample with annotation info.
        output_dir: Directory to save generated mask.
        coco_dataset_cache: Cache for parsed COCO datasets (path -> COCODataset).
    
    Returns:
        Tuple of (mask array, saved path) or (None, None) if no annotation.
    
    Raises:
        AnnotationError: If annotation cannot be processed.
    """
    if not sample.annotation or not sample.annotation_type:
        return None, None
    
    if sample.annotation_resolved is None:
        raise AnnotationError(
            f"Annotation file not resolved for sample {sample.id}"
        )
    
    if not sample.annotation_resolved.exists():
        raise AnnotationError(
            f"Annotation file not found: {sample.annotation_resolved}"
        )
    
    # Get image dimensions
    if sample.image_resolved is None:
        raise AnnotationError(f"Image not resolved for sample {sample.id}")
    
    try:
        with Image.open(sample.image_resolved) as img:
            width, height = img.size
    except Exception as e:
        raise AnnotationError(f"Failed to read image dimensions: {e}")
    
    annotation_type = sample.annotation_type.lower()
    
    try:
        if annotation_type == "voc":
            annot = parse_voc_xml(sample.annotation_resolved)
            bboxes = annot.bboxes
        
        elif annotation_type == "coco":
            # Use cache if available
            cache_key = str(sample.annotation_resolved)
            
            if coco_dataset_cache is not None and cache_key in coco_dataset_cache:
                dataset = coco_dataset_cache[cache_key]
            else:
                dataset = parse_coco_json(sample.annotation_resolved)
                if coco_dataset_cache is not None:
                    coco_dataset_cache[cache_key] = dataset
            
            img_annot = get_coco_annotation_for_image(
                dataset,
                file_name=sample.image
            )
            bboxes = img_annot.bboxes
        
        else:
            raise AnnotationError(
                f"Unknown annotation type: {annotation_type}. Use 'voc' or 'coco'."
            )
        
    except (VOCParseError, COCOParseError) as e:
        raise AnnotationError(str(e))
    
    if not bboxes:
        raise AnnotationError(f"No bboxes found in annotation for {sample.id}")
    
    # Generate mask
    mask = bboxes_to_mask(bboxes, width, height, clamp=True)
    
    # Save mask
    output_path = output_dir / "roi_mask_bbox.png"
    saved_path = save_mask(mask, output_path, format="png")
    
    return mask, saved_path
