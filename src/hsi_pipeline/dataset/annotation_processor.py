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
from ..adapters.via_parser import (
    parse_via_json, via_regions_to_mask, VIAParseError
)


class AnnotationError(Exception):
    """Error processing annotation."""
    pass


def process_single_annotation(
    image_path: Path,
    annotation_path: Path,
    annotation_type: str,
    output_dir: Path,
    coco_dataset_cache: Optional[dict] = None,
    image_filename: Optional[str] = None
) -> Tuple[np.ndarray, Path]:
    """Process a single annotation and generate ROI mask.
    
    Args:
        image_path: Path to the image file.
        annotation_path: Path to the annotation file.
        annotation_type: Type of annotation ('voc', 'coco', 'via').
        output_dir: Directory to save generated mask.
        coco_dataset_cache: Optional cache for COCO datasets.
        image_filename: Filename to look up in annotation (defaults to image filename).
        
    Returns:
        Tuple of (mask array, saved path).
        
    Raises:
        AnnotationError: If processing fails.
    """
    if not image_path.exists():
        raise AnnotationError(f"Image not found: {image_path}")
        
    if not annotation_path.exists():
        raise AnnotationError(f"Annotation file not found: {annotation_path}")
        
    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except Exception as e:
        raise AnnotationError(f"Failed to read image dimensions: {e}")
        
    annotation_type = annotation_type.lower()
    if image_filename is None:
        image_filename = image_path.name
        
    try:
        mask = None
        
        if annotation_type == "via":
            try:
                project = parse_via_json(annotation_path)
            except VIAParseError as e:
                raise AnnotationError(f"VIA parse error: {e}")
            
            if image_filename not in project.images:
                raise AnnotationError(
                    f"Image '{image_filename}' not found in VIA project {annotation_path.name}"
                )
            
            via_annot = project.images[image_filename]
            if not via_annot.regions:
                raise AnnotationError(f"No regions found for {image_filename} in VIA annotation")
            
            mask = via_regions_to_mask(via_annot.regions, width, height)
            
        elif annotation_type == "voc":
            annot = parse_voc_xml(annotation_path)
            if not annot.bboxes:
                raise AnnotationError("No bboxes found in VOC annotation")
            mask = bboxes_to_mask(annot.bboxes, width, height, clamp=True)
            
        elif annotation_type == "coco":
            cache_key = str(annotation_path)
            if coco_dataset_cache is not None and cache_key in coco_dataset_cache:
                dataset = coco_dataset_cache[cache_key]
            else:
                dataset = parse_coco_json(annotation_path)
                if coco_dataset_cache is not None:
                    coco_dataset_cache[cache_key] = dataset
            
            img_annot = get_coco_annotation_for_image(
                dataset,
                file_name=image_filename
            )
            if not img_annot.bboxes:
                raise AnnotationError(f"No bboxes found in COCO annotation for {image_filename}")
            mask = bboxes_to_mask(img_annot.bboxes, width, height, clamp=True)
            
        else:
            raise AnnotationError(
                f"Unknown annotation type: {annotation_type}. Use 'voc', 'coco', or 'via'."
            )
            
    except (VOCParseError, COCOParseError, VIAParseError) as e:
        raise AnnotationError(str(e))
        
    # Save mask
    output_path = output_dir / "roi_mask_annot.png"
    saved_path = save_mask(mask, output_path, format="png")
    
    return mask, saved_path


def process_sample_annotation(
    sample: Sample,
    output_dir: Path,
    coco_dataset_cache: Optional[dict] = None
) -> Tuple[Optional[np.ndarray], Optional[Path]]:
    """Process annotation for a sample and generate ROI mask.
    
    Args:
        sample: Sample with annotation info.
        output_dir: Directory to save generated mask.
        coco_dataset_cache: Cache for parsed COCO datasets.
    
    Returns:
        Tuple of (mask array, saved path) or (None, None).
    """
    if not sample.annotation or not sample.annotation_type:
        return None, None
        
    if sample.annotation_resolved is None:
        raise AnnotationError(f"Annotation file not resolved for sample {sample.id}")
        
    if sample.image_resolved is None:
        raise AnnotationError(f"Image not resolved for sample {sample.id}")
        
    return process_single_annotation(
        image_path=sample.image_resolved,
        annotation_path=sample.annotation_resolved,
        annotation_type=sample.annotation_type,
        output_dir=output_dir,
        coco_dataset_cache=coco_dataset_cache,
        image_filename=sample.image
    )
