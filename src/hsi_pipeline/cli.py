"""Command-line interface for SpectraPipe."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
import numpy as np

from .pipeline.run import load_config
from .pipeline.orchestrator import PipelineInput, PipelineOrchestrator
from .export.manager import ExportManager
from .cli_helpers import load_image, export_pipeline_output, log_pipeline_progress

PIPELINE_VERSION = "0.1.0"

app = typer.Typer(
    name="SpectraPipe",
    help="A Reproducible HSI-to-RGB Reconstruction Pipeline",
    add_completion=False,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
console = Console()

@app.command(
    no_args_is_help=True,
)
def run(
    input: Path = typer.Option(
        ...,
        "--input", "-i",
        help="Path to RGB image",
        exists=True,
        file_okay=True,
        readable=True,
        resolve_path=True
    ),
    out: Optional[Path] = typer.Option(
        None,
        "--out", "-o",
        help="Path to output directory",
        file_okay=False,
        dir_okay=True,
        resolve_path=True
    ),
    config: Path = typer.Option(
        "configs/defaults.yaml",
        "--config", "-c",
        help="Path to configuration YAML file",
        exists=True,
        file_okay=True,
        readable=True,
        resolve_path=True
    ),
    no_ensemble: bool = typer.Option(
        False,
        "--no-ensemble",
        help="Disable model ensembling (faster but potentially less accurate)."
    ),
    roi_mask: Optional[Path] = typer.Option(
        None,
        "--roi-mask", "-r",
        help="Path to ROI mask image for separability calculation",
        exists=True,
        file_okay=True,
        readable=True,
        resolve_path=True
    ),
    upscale_factor: Optional[int] = typer.Option(
        None,
        "--upscale-factor", "-u",
        help="Upscaling factor (e.g., 2 for 2x upscaling). Generates baseline and improved upscaled outputs.",
        min=2,
        max=8
    ),
    annotation: Optional[Path] = typer.Option(
        None,
        "--annotation", "-a",
        help="Path to annotation file (VOC, COCO, VIA) to generate ROI mask",
        exists=True,
        file_okay=True,
        readable=True,
        resolve_path=True
    ),
    annotation_type: str = typer.Option(
        "via",
        "--annotation-type",
        help="Type of annotation file: 'voc', 'coco', 'via'",
        show_default=True
    ),
):
    """Run the full HSI processing pipeline."""
    from .roi.loader import ROILoadError, ROIValidationError
    from .pipeline.run import merge_cli_overrides
    from .dataset.annotation_processor import process_single_annotation, AnnotationError
    
    cfg = load_config(config)
    
    # Apply CLI overrides (CLI > config.yaml > defaults)
    cfg = merge_cli_overrides(
        cfg,
        no_ensemble=no_ensemble,
        upscale_factor=upscale_factor,
    )
    
    if out is None:
        out = input.parent / cfg.export.default_dir
    
    exporter = ExportManager(out_dir=out, format="npz", overwrite=cfg.export.overwrite)
    
    console.print(f"Loading image: {input}")
    
    rgb = load_image(input, console)
    
    console.print(f"Fitting input: policy={cfg.fitting.policy}, multiple={cfg.fitting.multiple}")
    
    try:
        exporter.prepare_directory()
        
        # Handle annotation if provided
        effective_roi_mask = roi_mask
        
        if annotation:
            if roi_mask:
                console.print("[red]Error:[/red] Cannot specify both --roi-mask and --annotation.")
                raise typer.Exit(1)
            
            console.print(f"Processing annotation: {annotation} (type={annotation_type})")
            try:
                # Generate mask from annotation
                _, mask_path = process_single_annotation(
                    image_path=input,
                    annotation_path=annotation,
                    annotation_type=annotation_type,
                    output_dir=out,
                    image_filename=input.name
                )
                effective_roi_mask = mask_path
                console.print(f"Generated ROI mask from annotation: {mask_path}")
            except AnnotationError as e:
                console.print(f"[red]Annotation Error:[/red] {e}")
                raise typer.Exit(1)
        
        orchestrator = PipelineOrchestrator()
        
        console.print("Converting RGB → HSI using MST++...")
        
        # Determine upscale factor (from CLI or config)
        effective_upscale = upscale_factor if upscale_factor else (
            cfg.upscaling.factor if cfg.upscaling.enabled else None
        )
        
        pipeline_input = PipelineInput(
            rgb=rgb,
            config=cfg,
            roi_mask_path=effective_roi_mask,
            upscale_factor=effective_upscale,
            use_ensemble=cfg.model.ensemble,
        )
        
        result = orchestrator.run(pipeline_input)
        
        console.print(f"  Original shape: {result.fit_result.original_shape[0]}x{result.fit_result.original_shape[1]}")
        console.print(f"  Fitted shape:   {result.fit_result.fitted_shape[0]}x{result.fit_result.fitted_shape[1]}")
        
        log_pipeline_progress(result, console)
        
        export_pipeline_output(
            output=result,
            exporter=exporter,
            input_path=input,
            config_path=config,
            config_dict=cfg.to_dict(),
            pipeline_version=PIPELINE_VERSION,
            console=console,
        )
        
        console.print("")
        exporter.log_export_summary(console)
        
    except ROILoadError as e:
        console.print(f"[red]ROI Load Error:[/red] {e}")
        console.print("[yellow]Suggestion:[/yellow] Check that the mask file exists and is a valid image.")
        raise typer.Exit(1)
    except ROIValidationError as e:
        console.print(f"[red]ROI Validation Error:[/red] {e}")
        console.print("[yellow]Suggestion:[/yellow] Ensure mask dimensions match input image.")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]Fitting Error:[/red] {e}")
        console.print("[yellow]Suggestion:[/yellow] Ensure image has valid dimensions and is RGB (H, W, 3).")
        raise typer.Exit(1)
    except NotADirectoryError as e:
        console.print(f"[red]Output Error:[/red] {e}")
        console.print("[yellow]Suggestion:[/yellow] Ensure --out points to a directory, not a file.")
        raise typer.Exit(1)
    except PermissionError as e:
        console.print(f"[red]Permission Error:[/red] {e}")
        console.print("[yellow]Suggestion:[/yellow] Check folder permissions or choose a different output path.")
        raise typer.Exit(1)
    except FileExistsError as e:
        console.print(f"[red]Collision Error:[/red] {e}")
        console.print("[yellow]Suggestion:[/yellow] Use a different output directory or enable overwrite.")
        raise typer.Exit(1)
    except OSError as e:
        console.print("[red]System Error:[/red] Failed to create/access output directory.")
        console.print(f"Detail: {e}")
        raise typer.Exit(1)
    except Exception as e:
        removed = exporter.cleanup_partial()
        console.print("[red]Pipeline Error:[/red] Execution failed unexpectedly.")
        console.print(f"[yellow]Detail:[/yellow] {e}")
        if removed:
            console.print(f"[yellow]Cleanup:[/yellow] Removed {len(removed)} partial artifact(s): {', '.join(removed)}")
        console.print("[yellow]Suggestion:[/yellow] Check input image and model configuration.")
        raise typer.Exit(1)
    
    
    # Exported artifacts summary is already logged by exporter.log_export_summary(console)
    
    table = Table(title="SpectraPipe Execution Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Input Image", str(input))
    table.add_row("Original Shape", f"{result.fit_result.original_shape[0]}x{result.fit_result.original_shape[1]}")
    table.add_row("Fitted Shape", f"{result.fit_result.fitted_shape[0]}x{result.fit_result.fitted_shape[1]}")
    table.add_row("Fitting Policy", result.fit_result.policy)
    table.add_row("Output Directory", str(out))
    table.add_row("Config File", str(config))
    table.add_row("Ensemble Enabled", "No" if no_ensemble else "Yes")
    table.add_row("HSI Shape", str(result.hsi_raw.shape))
    table.add_row("Total Time", f"{result.execution_time:.2f}s")
    
    console.print(table)
    console.print("[green]✓[/green] Pipeline finished successfully.")


@app.command(
    no_args_is_help=True,
)
def metrics(
    from_dir: Path = typer.Option(
        ...,
        "--from", "-f",
        help="Output directory containing metrics.json",
        resolve_path=True
    ),
):
    """Display a human-readable summary of metrics from a previous run."""
    from .metrics.reader import read_metrics, MetricsNotFoundError, MetricsCorruptError
    from .metrics.formatter import format_metrics, print_warnings
    
    if not from_dir.exists():
        console.print(f"[red]Error:[/red] Directory not found: {from_dir}")
        console.print("[yellow]Suggestion:[/yellow] Check the path and try again.")
        raise typer.Exit(1)
    
    if not from_dir.is_dir():
        console.print(f"[red]Error:[/red] Path is not a directory: {from_dir}")
        console.print("[yellow]Suggestion:[/yellow] Provide a directory path, not a file.")
        raise typer.Exit(1)
    
    metrics_path = from_dir / "metrics.json"
    
    try:
        result = read_metrics(metrics_path)
    except MetricsNotFoundError:
        console.print(f"[red]Error:[/red] metrics.json not found in: {from_dir}")
        console.print("[yellow]Suggestion:[/yellow] Run `spectrapipe run --input <image> --out <dir>` first to generate metrics.")
        raise typer.Exit(1)
    except MetricsCorruptError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("[yellow]Suggestion:[/yellow] Regenerate metrics with `spectrapipe run --input <image> --out <dir>`.")
        raise typer.Exit(1)
    
    if result.warnings:
        print_warnings(result.warnings, console)
    
    format_metrics(result.data, console)
    
    console.print("[green]✓[/green] Metrics loaded successfully.")


@app.command("dataset", no_args_is_help=True)
def dataset_run(
    manifest: Path = typer.Option(
        ...,
        "--manifest", "-m",
        help="Path to dataset manifest file (YAML or JSON)",
        exists=True,
        file_okay=True,
        readable=True,
        resolve_path=True
    ),
    out: Path = typer.Option(
        ...,
        "--out", "-o",
        help="Path to output directory for all samples",
        file_okay=False,
        dir_okay=True,
        resolve_path=True
    ),
    config: Path = typer.Option(
        "configs/defaults.yaml",
        "--config", "-c",
        help="Path to configuration YAML file",
        exists=True,
        file_okay=True,
        readable=True,
        resolve_path=True
    ),
    on_error: str = typer.Option(
        "continue",
        "--on-error",
        help="Policy on sample failure: 'continue' or 'abort'",
    ),
    no_ensemble: bool = typer.Option(
        False,
        "--no-ensemble",
        help="Disable model ensembling",
    ),
    upscale_factor: Optional[int] = typer.Option(
        None,
        "--upscale-factor", "-u",
        help="Upscaling factor for all samples",
        min=2,
        max=8
    ),
    on_annot_error: str = typer.Option(
        "continue",
        "--on-annot-error",
        help="Policy on annotation parse failure: 'continue' or 'abort'",
    ),
):
    """Run the pipeline on a dataset defined by a manifest."""
    from .manifest.parser import (
        parse_manifest, ManifestNotFoundError, ManifestParseError, ManifestValidationError
    )
    from .dataset.runner import run_dataset, write_dataset_report, OnErrorPolicy
    
    console.print(f"[bold]SpectraPipe Dataset Processing[/bold] v{PIPELINE_VERSION}")
    console.print(f"Manifest: {manifest}")
    
    if on_error not in ("continue", "abort"):
        console.print(f"[red]Error:[/red] --on-error must be 'continue' or 'abort', got '{on_error}'")
        raise typer.Exit(1)
    
    try:
        parsed_manifest = parse_manifest(manifest)
    except ManifestNotFoundError as e:
        console.print(f"[red]Manifest Error:[/red] {e}")
        console.print("[yellow]Suggestion:[/yellow] Check the manifest file path.")
        raise typer.Exit(1)
    except ManifestParseError as e:
        console.print(f"[red]Parse Error:[/red] {e}")
        console.print("[yellow]Suggestion:[/yellow] Fix the YAML/JSON syntax error.")
        raise typer.Exit(1)
    except ManifestValidationError as e:
        console.print(f"[red]Validation Error:[/red] {e}")
        console.print("[yellow]Suggestion:[/yellow] Check manifest schema requirements.")
        raise typer.Exit(1)
    
    console.print(f"Root: {parsed_manifest.root}")
    console.print(f"Samples: {len(parsed_manifest.samples)}")
    console.print(f"On-error policy: {on_error}")
    console.print("")
    
    from .pipeline.run import merge_cli_overrides
    
    cfg = load_config(config)
    
    # Apply CLI overrides (CLI > config.yaml > defaults)
    # Note: on_error/on_annot_error CLI flags override config if provided
    cfg = merge_cli_overrides(
        cfg,
        no_ensemble=no_ensemble,
        upscale_factor=upscale_factor,
        on_error=on_error if on_error != cfg.dataset.on_error else None,
        on_annot_error=on_annot_error if on_annot_error != cfg.dataset.on_annot_error else None,
    )
    
    # Use effective values from merged config
    effective_on_error = cfg.dataset.on_error
    effective_on_annot_error = cfg.dataset.on_annot_error
    effective_upscale = upscale_factor if upscale_factor else (
        cfg.upscaling.factor if cfg.upscaling.enabled else None
    )
    
    # Create policy from effective value
    policy = OnErrorPolicy.CONTINUE if effective_on_error == "continue" else OnErrorPolicy.ABORT
    
    coco_cache = {}  # Cache for parsed COCO datasets
    orchestrator = PipelineOrchestrator()
    
    def process_sample(sample, sample_out):
        """Process a single sample using the orchestrator."""
        from PIL import Image, UnidentifiedImageError
        
        # Variables for finally block
        roi_source = "none"
        annotation_roi_path = None
        
        try:
            image_path = sample.image_resolved
            if not image_path or not image_path.exists():
                raise FileNotFoundError(f"Image not found: {sample.image}")
            
            try:
                rgb = np.array(Image.open(image_path).convert('RGB'))
            except UnidentifiedImageError:
                raise ValueError(f"Invalid image format: {image_path}")
            
            # Determine ROI mask path (from sample or annotation)
            roi_mask_path = sample.roi_mask_resolved
            if roi_mask_path:
                roi_source = "mask_file"
            
            # Process annotation if present (generates ROI from bboxes)
            if sample.annotation and sample.annotation_type:
                from .dataset.annotation_processor import process_sample_annotation, AnnotationError
                
                # Check for conflict with explicit ROI mask
                if roi_mask_path:
                    console.print(f"  [yellow]Warning:[/yellow] Sample {sample.id}: Both annotation and roi_mask specified. Using roi_mask ({roi_mask_path}).")
                
                try:
                    mask, annotation_roi_path = process_sample_annotation(
                        sample, sample_out, coco_dataset_cache=coco_cache
                    )
                    
                    # Use generated mask if no explicit mask provided
                    if not roi_mask_path and annotation_roi_path:
                        roi_mask_path = annotation_roi_path
                        roi_source = "annotation"
                        
                except AnnotationError as e:
                    if effective_on_annot_error == "abort":
                        raise
                    else:
                        console.print(f"  [yellow]Annotation warning:[/yellow] {e}")
            
            # Run pipeline via orchestrator
            pipeline_input = PipelineInput(
                rgb=rgb,
                config=cfg,
                roi_mask_path=roi_mask_path,
                upscale_factor=effective_upscale,
                use_ensemble=cfg.model.ensemble,
            )
            
            result = orchestrator.run(pipeline_input)
            
            # Export results
            exporter = ExportManager(sample_out)
            export_pipeline_output(
                output=result,
                exporter=exporter,
                input_path=image_path,
                config_path=config,
                config_dict=cfg.to_dict(),
                pipeline_version=PIPELINE_VERSION,
                console=console,
            )

        finally:
            # Add sample-specific metadata to run_config
            # Always update run_config with annotation/ROI info
            import json
            import os
            run_config_path = sample_out / "run_config.json"
            
            # Ensure directory exists (defensive, as requested)
            os.makedirs(os.path.dirname(run_config_path), exist_ok=True)
            
            run_config_data = {}
            if run_config_path.exists():
                try:
                    with open(run_config_path) as f:
                        run_config_data = json.load(f)
                except json.JSONDecodeError:
                    # If corrupt, overwrite
                    pass
            
            # Ensure meta section exists
            if "meta" not in run_config_data:
                run_config_data["meta"] = {}
                
            # Ensure config section exists
            if "config" not in run_config_data:
                run_config_data["config"] = cfg.to_dict()
                
            run_config_data["meta"]["sample_id"] = sample.id
            if annotation_roi_path:
                run_config_data["meta"]["annotation_roi_path"] = str(annotation_roi_path)
                run_config_data["meta"]["annotation_type"] = sample.annotation_type
            
            run_config_data["meta"]["roi_source"] = roi_source
            
            with open(run_config_path, "w") as f:
                json.dump(run_config_data, f, indent=2)
    
    report = run_dataset(
        manifest=parsed_manifest,
        output_dir=out,
        process_fn=process_sample,
        on_error=policy,
        console=console
    )
    
    report_path = write_dataset_report(report, out)
    
    console.print("")
    console.print("[bold]Dataset Processing Complete[/bold]")
    console.print(f"  Total samples: {report.total_samples}")
    console.print(f"  Processed OK:  [green]{report.processed_ok}[/green]")
    console.print(f"  Failed:        [red]{report.failed}[/red]")
    console.print(f"  Total time:    {report.total_time:.2f}s")
    console.print(f"  Report:        {report_path}")
    
    if report.failed > 0:
        console.print("")
        console.print("[yellow]Failures:[/yellow]")
        for failure in report.failures:
            console.print(f"  - {failure['sample_id']}: {failure['reason']}")
    
    if report.failed > 0 and policy == OnErrorPolicy.ABORT:
        raise typer.Exit(1)


@app.command("spectra", no_args_is_help=True)
def spectra_extract(
    from_dir: Path = typer.Option(
        ...,
        "--from",
        help="Directory containing pipeline outputs",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True
    ),
    artifact: str = typer.Option(
        "raw",
        "--artifact", "-a",
        help="Which HSI artifact to extract from: 'raw' or 'clean'"
    ),
    pixel: Optional[str] = typer.Option(
        None,
        "--pixel", "-p",
        help="Pixel coordinates as 'x,y' (e.g., '120,80')"
    ),
    pixels: Optional[str] = typer.Option(
        None,
        "--pixels",
        help="Multiple pixels as 'x1,y1;x2,y2' (e.g., '10,10;20,20')"
    ),
    roi_agg: Optional[str] = typer.Option(
        None,
        "--roi-agg",
        help="ROI aggregation method: 'mean' or 'median'"
    ),
    roi_mask: Optional[Path] = typer.Option(
        None,
        "--roi-mask",
        help="Path to ROI mask image",
        exists=True,
        file_okay=True,
        readable=True,
        resolve_path=True
    ),
    pixels_file: Optional[Path] = typer.Option(
        None,
        "--pixels-file",
        help="CSV file with x,y columns for batch extraction",
        exists=True,
        file_okay=True,
        readable=True,
        resolve_path=True
    ),
    wavelengths_file: Optional[Path] = typer.Option(
        None,
        "--wavelengths",
        help="CSV/JSON file with 31 wavelength values",
        exists=True,
        file_okay=True,
        readable=True,
        resolve_path=True
    ),
    wl_start: Optional[float] = typer.Option(
        None,
        "--wl-start",
        help="Starting wavelength in nm (use with --wl-step)"
    ),
    wl_step: Optional[float] = typer.Option(
        None,
        "--wl-step",
        help="Wavelength step in nm (use with --wl-start)"
    ),
    normalize: str = typer.Option(
        "none",
        "--normalize", "-n",
        help="Normalization mode: 'none', 'minmax', or 'l2'"
    ),
    export: str = typer.Option(
        "both",
        "--export", "-e",
        help="Export format: 'csv', 'json', or 'both'"
    ),
    export_out: Optional[Path] = typer.Option(
        None,
        "--export-out",
        help="Output directory for exports (default: same as --from)"
    ),
    plot: bool = typer.Option(
        False,
        "--plot",
        help="Generate plot PNG"
    ),
    plot_first_n: int = typer.Option(
        10,
        "--plot-first-n",
        help="Max number of plots in batch mode"
    ),
):
    """Extract spectral signature from HSI artifact."""
    from .spectra.loader import load_hsi_artifact, HSINotFoundError, HSILoadError
    from .spectra.extractor import (
        extract_pixel, extract_roi_aggregate,
        CoordinateError, ROIError, SpectralSignature
    )
    from .spectra.exporter import export_signature, export_batch_csv, export_batch_json
    from .spectra.wavelengths import WavelengthError
    from .spectra.normalize import normalize_signature, validate_normalize_mode
    from .spectra.batch import load_pixels_file, extract_batch, BatchError
    from .utils.parsing import parse_pixels_inline
    from rich.table import Table
    
    console.print("[bold]Spectral Signature Extraction[/bold]")
    console.print(f"Source: {from_dir}")
    console.print(f"Artifact: {artifact}")
    
    if artifact not in ("raw", "clean"):
        console.print(f"[red]Error:[/red] Invalid artifact '{artifact}'. Use 'raw' or 'clean'.")
        raise typer.Exit(1)
    
    if export not in ("csv", "json", "both"):
        console.print(f"[red]Error:[/red] Invalid export format '{export}'. Use 'csv', 'json', or 'both'.")
        raise typer.Exit(1)
        
    # Validation: Wavelengths
    if wavelengths_file and (wl_start is not None or wl_step is not None):
        console.print("[red]Error:[/red] Cannot specify both --wavelengths and --wl-start/--wl-step.")
        raise typer.Exit(1)
    
    try:
        norm_mode = validate_normalize_mode(normalize)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    
    # Validation: Extraction Mode
    modes = sum([
        pixel is not None, 
        pixels is not None,
        roi_agg is not None, 
        pixels_file is not None
    ])
    
    if modes == 0:
        console.print("[red]Error:[/red] Specify one of: --pixel, --pixels, --roi-agg, or --pixels-file")
        console.print("[yellow]Suggestion:[/yellow] Use --pixel 'x,y' or --pixels 'x1,y1;x2,y2' or --roi-agg mean/median or --pixels-file file.csv")
        raise typer.Exit(1)
    if modes > 1:
        console.print("[red]Error:[/red] Use only one extraction mode: --pixel, --pixels, --roi-agg, or --pixels-file")
        raise typer.Exit(1)
    
    try:
        loaded = load_hsi_artifact(from_dir, artifact)
        console.print(f"Loaded: {loaded.path.name} (shape: {loaded.shape})")
    except HSINotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("[yellow]Suggestion:[/yellow] Run 'spectrapipe run' first to generate outputs.")
        raise typer.Exit(1)
    except HSILoadError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    
    # Resolve wavelengths with priority: CLI override > NPZ default
    from .spectra.wavelengths import resolve_wavelengths
    try:
        wl_result = resolve_wavelengths(
            npz_wavelengths=loaded.wavelength_nm,
            cli_file=wavelengths_file,
            cli_start=wl_start,
            cli_step=wl_step
        )
        wavelengths = wl_result.wavelength_nm
        
        # Log wavelength source
        if wl_result.is_override:
            console.print("[yellow]Using wavelengths from CLI (override).[/yellow] NPZ contains wavelength_nm too.")
        else:
            source_desc = {
                "npz": "NPZ file",
                "cli_file": f"file ({wavelengths_file})",
                "cli_params": f"CLI params (start={wl_start}, step={wl_step})"
            }
            console.print(f"Wavelengths: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm (from {source_desc[wl_result.source]})")
    except WavelengthError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    

    
    export_dir = export_out if export_out else from_dir
    source_path = str(loaded.path)
    
    # ---- BATCH MODE ----
    if pixels_file is not None or pixels is not None:
        if pixels_file:
            console.print(f"Mode: Batch (from {pixels_file.name})")
            try:
                pixels_list = load_pixels_file(pixels_file)
            except BatchError as e:
                console.print(f"[red]Error:[/red] {e}")
                raise typer.Exit(1)
        else:
            console.print("Mode: Batch (inline)")
            try:
                pixels_list = parse_pixels_inline(pixels)
            except ValueError as e:
                console.print(f"[red]Error:[/red] {e}")
                raise typer.Exit(1)
                
        console.print(f"Pixels to extract: {len(pixels_list)}")
        
        batch_result = extract_batch(loaded.data, pixels_list, artifact, fail_fast=False)
        
        for sig in batch_result.signatures:
            sig.values = normalize_signature(sig.values, norm_mode)
        
        console.print(f"Extracted: {batch_result.success_count}/{batch_result.total}")
        if batch_result.fail_count > 0:
            console.print(f"[yellow]Failed: {batch_result.fail_count}[/yellow]")
        
        paths = []
        if export in ("csv", "both"):
            paths.append(export_batch_csv(batch_result.signatures, export_dir, artifact, wavelengths, norm_mode))
        if export in ("json", "both"):
            paths.append(export_batch_json(batch_result.signatures, export_dir, artifact, wavelengths, norm_mode, batch_result.failed))
        
        console.print("\n[green]Exported:[/green]")
        for p in paths:
            console.print(f"  {p}")
        
        if plot and batch_result.signatures:
            from .spectra.plot import plot_batch_signatures
            plot_paths = plot_batch_signatures(batch_result.signatures, wavelengths, export_dir, artifact, plot_first_n)
            console.print(f"\n[green]Plots generated: {len(plot_paths)}[/green]")
        
        return
    
    # ---- SINGLE EXTRACTION (pixel or ROI) ----
    signature: SpectralSignature
    
    if pixel is not None:
        try:
            parts = pixel.split(",")
            if len(parts) != 2:
                raise ValueError("Expected 'x,y' format")
            x, y = int(parts[0].strip()), int(parts[1].strip())
        except ValueError:
            console.print(f"[red]Error:[/red] Invalid pixel format: {pixel}")
            console.print("[yellow]Suggestion:[/yellow] Use format 'x,y' (e.g., '120,80')")
            raise typer.Exit(1)
        
        try:
            signature = extract_pixel(loaded.data, x, y, artifact)
            console.print(f"Extracted: pixel ({x}, {y})")
        except CoordinateError as e:
            console.print(f"[red]Error:[/red] {e}")
            h, w = loaded.data.shape[:2]
            console.print(f"[yellow]Valid range:[/yellow] x: 0-{w-1}, y: 0-{h-1}")
            raise typer.Exit(1)
    
    else:  # roi_agg
        if roi_agg not in ("mean", "median"):
            console.print(f"[red]Error:[/red] Invalid aggregation '{roi_agg}'. Use 'mean' or 'median'.")
            raise typer.Exit(1)
        
        if roi_mask is None:
            console.print("[red]Error:[/red] --roi-mask required when using --roi-agg")
            raise typer.Exit(1)
        
        from .roi.loader import load_roi_mask, ROILoadError, ROIValidationError
        
        try:
            roi_result = load_roi_mask(roi_mask, loaded.data.shape[:2])
            if roi_result.coverage == 0:
                console.print("[red]Error:[/red] ROI mask is empty (no True pixels)")
                raise typer.Exit(1)
            console.print(f"ROI: {roi_mask.name} (coverage: {roi_result.coverage:.1%})")
        except (ROILoadError, ROIValidationError) as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)
        
        try:
            signature = extract_roi_aggregate(loaded.data, roi_result.mask, roi_agg, artifact)
            console.print(f"Extracted: ROI {roi_agg} ({signature.roi_pixel_count} pixels)")
        except ROIError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)
    
    # Apply normalization
    signature.values = normalize_signature(signature.values, norm_mode)
    if norm_mode != "none":
        console.print(f"Normalized: {norm_mode}")
    
    # Display signature
    console.print("")
    table = Table(title=f"Spectral Signature ({signature.num_bands} bands)")
    table.add_column("Band", justify="right", style="cyan")
    if wavelengths is not None:
        table.add_column("λ (nm)", justify="right")
    table.add_column("Value", justify="right")
    
    values = signature.values
    for i in range(min(10, len(values))):
        if wavelengths is not None:
            table.add_row(str(i + 1), f"{wavelengths[i]:.1f}", f"{values[i]:.6f}")
        else:
            table.add_row(str(i + 1), f"{values[i]:.6f}")
    
    if len(values) > 15:
        if wavelengths is not None:
            table.add_row("...", "...", "...")
        else:
            table.add_row("...", "...")
        for i in range(len(values) - 5, len(values)):
            if wavelengths is not None:
                table.add_row(str(i + 1), f"{wavelengths[i]:.1f}", f"{values[i]:.6f}")
            else:
                table.add_row(str(i + 1), f"{values[i]:.6f}")
    elif len(values) > 10:
        for i in range(10, len(values)):
            if wavelengths is not None:
                table.add_row(str(i + 1), f"{wavelengths[i]:.1f}", f"{values[i]:.6f}")
            else:
                table.add_row(str(i + 1), f"{values[i]:.6f}")
    
    console.print(table)
    console.print(f"Min: {values.min():.6f}  Max: {values.max():.6f}  Mean: {values.mean():.6f}")
    
    # Export
    try:
        paths = export_signature(
            signature, export_dir, export,
            wavelengths=wavelengths,
            normalize_mode=norm_mode,
            source_path=source_path
        )
        console.print("\n[green]Exported:[/green]")
        for p in paths:
            console.print(f"  {p}")
    except Exception as e:
        console.print(f"[red]Export Error:[/red] {e}")
        raise typer.Exit(1)
    
    # Plot
    if plot:
        from .spectra.plot import plot_signature as do_plot
        if signature.source == "pixel":
            plot_name = f"spectra_{artifact}_pixel_{signature.pixel_x}_{signature.pixel_y}.png"
            title = f"Spectral Signature - Pixel ({signature.pixel_x}, {signature.pixel_y})"
        else:
            plot_name = f"spectra_{artifact}_roi_{signature.roi_aggregation}.png"
            title = f"Spectral Signature - ROI {signature.roi_aggregation}"
        
        plot_path = do_plot(values, export_dir / plot_name, wavelengths, title=title)
        console.print(f"\n[green]Plot:[/green] {plot_path}")


def main():
    app()

if __name__ == "__main":
    main()