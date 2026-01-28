"""Helper functions for CLI commands that use the orchestrator."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from rich.console import Console

from .pipeline.orchestrator import PipelineOutput
from .export.manager import ExportManager


def load_image(path: Path, console: Console) -> np.ndarray:
    """Load and validate RGB image.
    
    Args:
        path: Path to image file.
        console: Console for error output.
        
    Returns:
        RGB image as numpy array (H, W, 3).
        
    Raises:
        SystemExit: If image cannot be loaded.
    """
    import typer
    
    try:
        with Image.open(path) as img:
            img.load()
    except (IOError, SyntaxError, UnidentifiedImageError) as e:
        console.print("[red]Integrity Error:[/red] Failed to load image data (corrupt or unsupported)")
        console.print(f"[yellow]Detail:[/yellow] {e}")
        raise typer.Exit(1)
    
    rgb = cv2.imread(str(path))
    if rgb is None:
        console.print("[red]Error:[/red] OpenCV failed to load the image (format might not be supported by cv2).")
        raise typer.Exit(1)
    
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)


def export_pipeline_output(
    output: PipelineOutput,
    exporter: ExportManager,
    input_path: Path,
    config_path: Path,
    config_dict: dict,
    pipeline_version: str,
    console: Console,
) -> None:
    """Export all pipeline outputs using ExportManager.
    
    Args:
        output: Pipeline output to export.
        exporter: Export manager instance.
        input_path: Original input image path.
        config_path: Configuration file path.
        config_dict: Pipeline configuration dictionary.
        pipeline_version: Pipeline version string.
        console: Console for logging.
    """
    # Export raw HSI
    exporter.export_array("hsi_raw", output.hsi_raw)
    
    # Export clean HSI if available
    if output.hsi_clean is not None:
        exporter.export_array("hsi_clean", output.hsi_clean)
    
    # Export upscaled HSI if available
    if output.upscale_data is not None:
        exporter.export_array("hsi_upscaled_baseline", output.upscale_data.hsi_baseline)
        exporter.export_array("hsi_upscaled_improved", output.upscale_data.hsi_improved)
    
    # Build metrics extra data
    metrics_extra = {}
    
    if output.roi_data is not None:
        metrics_extra["raw_separability"] = output.raw_separability
        metrics_extra["roi_coverage"] = output.roi_data.coverage
        metrics_extra["roi_mask_path"] = output.roi_data.path
    
    if output.clean_metrics is not None:
        metrics_extra.update(output.clean_metrics)
    
    if output.upscale_data is not None:
        metrics_extra["upscale_factor"] = output.upscale_data.factor
        metrics_extra["upscaled_size"] = list(output.upscale_data.hsi_baseline.shape[:2])
        metrics_extra["upscaling_methods"] = ["baseline_bicubic", "improved_edge_guided"]
    
    exporter.export_metrics(
        hsi_shape=output.hsi_raw.shape,
        execution_time=output.execution_time,
        ensemble_enabled=True,  # Will be overridden by caller if needed
        extra=metrics_extra if metrics_extra else None,
    )
    
    fitting_info = {
        "policy": output.fit_result.policy,
        "padding": list(output.fit_result.padding),
        "input_shape_original": list(output.fit_result.original_shape),
        "input_shape_fitted": list(output.fit_result.fitted_shape),
    }
    
    run_config_extra = {
        "hsi_shape": list(output.hsi_raw.shape),
    }
    
    if output.roi_data is not None:
        run_config_extra["roi"] = {
            "mask_path": output.roi_data.path,
            "coverage": output.roi_data.coverage,
        }
    
    if output.clean_data is not None:
        run_config_extra["clean"] = {
            "enabled": True,
            "policy": output.clean_data.policy,
        }
    
    if output.upscale_data is not None:
        run_config_extra["upscaling"] = {
            "enabled": True,
            "factor": output.upscale_data.factor,
            "methods": ["baseline_bicubic", "improved_edge_guided"],
            "upscaled_size": list(output.upscale_data.hsi_baseline.shape[:2]),
        }
    
    exporter.export_run_config(
        config_dict=config_dict,
        input_path=str(input_path),
        config_path=str(config_path),
        fitting_info=fitting_info,
        pipeline_version=pipeline_version,
        extra=run_config_extra,
    )
    
    if output.roi_data is None:
        exporter.mark_skipped("hsi_clean", "no ROI provided")
        exporter.mark_skipped("roi_mask", "no ROI provided")
    elif output.hsi_clean is None:
        exporter.mark_skipped("hsi_clean", "ROI empty or full")
    
    if output.upscale_data is None:
        exporter.mark_skipped("hsi_upscaled_baseline", "upscaling not requested")
        exporter.mark_skipped("hsi_upscaled_improved", "upscaling not requested")


def log_pipeline_progress(output: PipelineOutput, console: Console) -> None:
    """Log pipeline progress and results to console.
    
    Args:
        output: Pipeline output.
        console: Console for logging.
    """
    # Log ROI info
    if output.roi_data is not None:
        for warning in output.roi_data.warnings:
            console.print(f"[yellow]ROI Warning:[/yellow] {warning}")
        console.print(f"  ROI coverage: {output.roi_data.coverage:.1%}")
        
        if output.raw_separability is not None:
            console.print(f"  Raw separability: {output.raw_separability:.4f}")
        else:
            console.print("  Raw separability: [yellow]NA[/yellow] (empty or full ROI)")
    else:
        console.print("[dim]ROI not provided → separability omitted[/dim]")
    
    if output.hsi_clean is not None:
        console.print(f"  Policy: {output.clean_data.policy}")
        console.print(f"  ✓ Clean HSI generated: {output.hsi_clean.shape}")
        
        if output.clean_metrics:
            console.print(f"  Clean separability: {output.clean_metrics['clean_separability']:.4f}")
            console.print(f"  Raw-Clean SAM: {output.clean_metrics['raw_clean_sam']:.4f}")
            console.print(f"  Raw-Clean RMSE: {output.clean_metrics['raw_clean_rmse']:.4f}")
    elif output.roi_data is not None:
        console.print("[yellow]Clean skipped:[/yellow] ROI is empty or full (100%)")
    else:
        console.print("[dim]Clean skipped (no ROI mask)[/dim]")
    
    if output.upscale_data is not None:
        console.print(f"  ✓ Upscaling complete: {output.hsi_raw.shape[:2]} → {output.upscale_data.hsi_baseline.shape[:2]}")
    else:
        console.print("[dim]Upscaling not requested[/dim]")
