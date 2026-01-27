"""Pipeline orchestrator for HSI processing.

This module centralizes the core pipeline logic, separating it from CLI concerns.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..preprocess.input_fitting import FittingResult, fit_input, unfit_output
from ..transforms.rgb_to_hsi import rgb_to_hsi
from ..types import RunConfig


@dataclass
class ROIData:
    """ROI processing results."""
    mask: np.ndarray
    mask_resized: np.ndarray  # Resized to match HSI dimensions
    coverage: float
    path: Optional[str] = None
    warnings: list[str] = field(default_factory=list)


@dataclass 
class CleanData:
    """Clean HSI processing results."""
    hsi_clean: np.ndarray
    policy: str
    bg_mean_spectrum: np.ndarray


@dataclass
class UpscaleData:
    """Upscaling results."""
    hsi_baseline: np.ndarray
    hsi_improved: np.ndarray
    factor: int


@dataclass
class PipelineInput:
    """Input specification for pipeline execution."""
    rgb: np.ndarray                    # RGB image (H, W, 3)
    config: RunConfig                  # Pipeline configuration
    roi_mask_path: Optional[Path] = None  # Optional ROI mask path
    upscale_factor: Optional[int] = None  # Optional upscale factor
    use_ensemble: bool = True          # Whether to use ensemble inference


@dataclass
class PipelineOutput:
    """Complete result of pipeline execution."""
    # Core outputs
    hsi_raw: np.ndarray                # Raw HSI cube (H, W, 31)
    fit_result: FittingResult          # Input fitting metadata
    
    # Optional outputs
    hsi_clean: Optional[np.ndarray] = None
    roi_data: Optional[ROIData] = None
    clean_data: Optional[CleanData] = None
    upscale_data: Optional[UpscaleData] = None
    
    # Metrics
    raw_separability: Optional[float] = None
    clean_metrics: Optional[dict] = None
    execution_time: float = 0.0
    
    # Original RGB (for upscaling reference)
    rgb_original: Optional[np.ndarray] = None


class PipelineOrchestrator:
    """Orchestrates the HSI processing pipeline.
    
    This class encapsulates the core pipeline flow:
    1. Input fitting (pad to multiple of 32)
    2. RGB → HSI inference via MST++
    3. ROI processing (if mask provided)
    4. Background suppression (if ROI with partial coverage)
    5. Upscaling (if factor provided)
    6. Metrics computation
    """
    
    def run(self, input: PipelineInput) -> PipelineOutput:
        """Execute the full pipeline.
        
        Args:
            input: Pipeline input specification.
            
        Returns:
            PipelineOutput with all results and metrics.
            
        Raises:
            ValueError: If input validation fails.
            FileNotFoundError: If ROI mask path doesn't exist.
        """
        start_time = time.perf_counter()
        
        # Step 1: Input fitting
        fit_result = self._fit_input(input.rgb, input.config)
        
        # Step 2: RGB → HSI inference
        hsi_fitted = rgb_to_hsi(
            fit_result.fitted, 
            input.config, 
            ensemble_override=input.use_ensemble
        )
        hsi_raw = unfit_output(
            hsi_fitted, 
            fit_result.original_shape, 
            fit_result.padding
        )
        
        # Initialize output
        output = PipelineOutput(
            hsi_raw=hsi_raw,
            fit_result=fit_result,
            rgb_original=input.rgb,
        )
        
        # Step 3: ROI processing (if provided)
        if input.roi_mask_path is not None:
            output.roi_data = self._process_roi(
                input.roi_mask_path, 
                fit_result.original_shape[:2],
                hsi_raw.shape[:2]
            )
            
            # Calculate raw separability
            output.raw_separability = self._calculate_separability(
                hsi_raw, 
                output.roi_data.mask_resized
            )
            
            # Step 4: Background suppression (if partial coverage)
            if 0 < output.roi_data.coverage < 1:
                clean_result = self._generate_clean(
                    hsi_raw, 
                    output.roi_data.mask_resized
                )
                if clean_result is not None:
                    output.hsi_clean = clean_result.hsi_clean
                    output.clean_data = clean_result
                    output.clean_metrics = self._calculate_clean_metrics(
                        hsi_raw, 
                        clean_result.hsi_clean, 
                        output.roi_data.mask_resized
                    )
        
        # Step 5: Upscaling (if factor provided)
        if input.upscale_factor is not None:
            output.upscale_data = self._upscale(
                hsi_raw, 
                input.rgb, 
                input.upscale_factor
            )
        
        # Record execution time
        output.execution_time = time.perf_counter() - start_time
        
        return output
    
    def _fit_input(self, rgb: np.ndarray, config: RunConfig) -> FittingResult:
        """Fit input image to model requirements."""
        return fit_input(
            rgb, 
            multiple=config.fitting.multiple, 
            policy=config.fitting.policy
        )
    
    def _process_roi(
        self, 
        roi_path: Path, 
        original_shape: tuple[int, int],
        hsi_shape: tuple[int, int]
    ) -> ROIData:
        """Load and process ROI mask."""
        from ..roi.loader import load_roi_mask
        
        roi_result = load_roi_mask(roi_path, original_shape)
        
        # Resize mask to match HSI dimensions if needed
        mask_resized = roi_result.mask
        if roi_result.mask.shape != hsi_shape:
            mask_resized = cv2.resize(
                roi_result.mask.astype(np.uint8),
                (hsi_shape[1], hsi_shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        
        return ROIData(
            mask=roi_result.mask,
            mask_resized=mask_resized,
            coverage=roi_result.coverage,
            path=roi_result.path,
            warnings=roi_result.warnings,
        )
    
    def _calculate_separability(
        self, 
        hsi: np.ndarray, 
        mask: np.ndarray
    ) -> Optional[float]:
        """Calculate ROI vs background separability."""
        from ..roi.separability import calculate_separability
        return calculate_separability(hsi, mask)
    
    def _generate_clean(
        self, 
        hsi: np.ndarray, 
        mask: np.ndarray
    ) -> Optional[CleanData]:
        """Apply background suppression."""
        from ..postprocess.background_suppression import suppress_background
        
        result = suppress_background(hsi, mask, policy="subtract_mean")
        if result is None:
            return None
            
        return CleanData(
            hsi_clean=result.hsi_clean,
            policy=result.policy,
            bg_mean_spectrum=result.bg_mean_spectrum,
        )
    
    def _calculate_clean_metrics(
        self, 
        hsi_raw: np.ndarray, 
        hsi_clean: np.ndarray, 
        mask: np.ndarray
    ) -> dict:
        """Calculate clean vs raw metrics."""
        from ..postprocess.clean_metrics import calculate_clean_metrics
        return calculate_clean_metrics(hsi_raw, hsi_clean, mask)
    
    def _upscale(
        self, 
        hsi: np.ndarray, 
        rgb: np.ndarray, 
        factor: int
    ) -> UpscaleData:
        """Apply upscaling (baseline and improved)."""
        from scipy.ndimage import zoom
        from ..upscaling.spatial import upscale_baseline, upscale_improved
        
        # Baseline: bicubic per-band
        hsi_baseline = upscale_baseline(hsi, factor=factor)
        
        # Improved: edge-guided with RGB reference
        rgb_matched = cv2.resize(rgb, (hsi.shape[1], hsi.shape[0]))
        rgb_upscaled = zoom(rgb_matched, (factor, factor, 1), order=3)
        hsi_improved = upscale_improved(hsi, rgb_upscaled, factor=factor)
        
        return UpscaleData(
            hsi_baseline=hsi_baseline,
            hsi_improved=hsi_improved,
            factor=factor,
        )
