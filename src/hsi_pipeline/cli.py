"""Command-line interface for SpectraPipe - HSI Pipeline."""

from pathlib import Path
from typing import Optional
from .pipeline.run import load_config
from .transforms.rgb_to_hsi import rgb_to_hsi
from .preprocess.input_fitting import fit_input, unfit_output
from .export.manager import ExportManager
import typer
from rich.console import Console
import numpy as np

PIPELINE_VERSION = "0.1.0"

app = typer.Typer(
    name="SpectraPipe - HSI Pipeline",
    help="A Hyperspectral Image Processing CLI utility!",
    add_completion=False,
)
console = Console()

@app.command()
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
):
    """Run the full HSI processing pipeline."""
    import cv2
    from PIL import Image, UnidentifiedImageError 

    import time
    from rich.table import Table

    start_time = time.perf_counter()

    cfg = load_config(config)

    if out is None:
        out = input.parent / "output"

    exporter = ExportManager(out_dir=out, format="npz", overwrite=True)

    console.print(f"Loading image: {input}")

    try:
        with Image.open(input) as img:
            img.load()
    except (IOError, SyntaxError, UnidentifiedImageError) as e:
        console.print("[red]Integrity Error:[/red] Failed to load image data (corrupt or unsupported)")
        console.print(f"[yellow]Detail:[/yellow] {e}")
        raise typer.Exit(1)
    
    rgb = cv2.imread(str(input))
    if rgb is None:
        console.print("[red]Error:[/red] OpenCV failed to load the image (format might not be supported by cv2).")
        raise typer.Exit(1)

    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    console.print(f"Fitting input: policy={cfg.fitting.policy}, multiple={cfg.fitting.multiple}")
    try:
        fit_result = fit_input(rgb, multiple=cfg.fitting.multiple, policy=cfg.fitting.policy)
    except ValueError as e:
        console.print(f"[red]Fitting Error:[/red] {e}")
        console.print("[yellow]Suggestion:[/yellow] Ensure image has valid dimensions and is RGB (H, W, 3).")
        raise typer.Exit(1)

    console.print(f"  Original shape: {fit_result.original_shape[0]}x{fit_result.original_shape[1]}")
    console.print(f"  Fitted shape:   {fit_result.fitted_shape[0]}x{fit_result.fitted_shape[1]}")

    console.print("Converting RGB → HSI using MST++...")

    use_ensemble = not no_ensemble
    
    try:
        exporter.prepare_directory()

        hsi_fitted = rgb_to_hsi(fit_result.fitted, cfg, ensemble_override=use_ensemble)
        
        hsi = unfit_output(hsi_fitted, fit_result.original_shape, fit_result.padding)

        exporter.export_array("hsi_raw", hsi)

        roi_result = None
        raw_separability = None
        
        if roi_mask is not None:
            from .roi.loader import load_roi_mask, ROILoadError, ROIValidationError
            from .roi.separability import calculate_separability
            
            console.print(f"Loading ROI mask: {roi_mask}")
            try:
                roi_result = load_roi_mask(roi_mask, fit_result.original_shape[:2])
                
                for warning in roi_result.warnings:
                    console.print(f"[yellow]ROI Warning:[/yellow] {warning}")
                
                console.print(f"  ROI coverage: {roi_result.coverage:.1%}")
                
                # Resize mask to match HSI dimensions if needed
                roi_mask_hsi = roi_result.mask
                if roi_result.mask.shape != hsi.shape[:2]:
                    import cv2 as cv2_sep
                    roi_mask_hsi = cv2_sep.resize(
                        roi_result.mask.astype(np.uint8),
                        (hsi.shape[1], hsi.shape[0]),
                        interpolation=cv2_sep.INTER_NEAREST
                    ).astype(bool)
                
                console.print("Calculating separability...")
                raw_separability = calculate_separability(hsi, roi_mask_hsi)
                
                if raw_separability is not None:
                    console.print(f"  Raw separability: {raw_separability:.4f}")
                else:
                    console.print("  Raw separability: [yellow]NA[/yellow] (empty or full ROI)")
                    
            except ROILoadError as e:
                console.print(f"[red]ROI Load Error:[/red] {e}")
                console.print("[yellow]Suggestion:[/yellow] Check that the mask file exists and is a valid image.")
                raise typer.Exit(1)
            except ROIValidationError as e:
                console.print(f"[red]ROI Validation Error:[/red] {e}")
                console.print("[yellow]Suggestion:[/yellow] Ensure mask dimensions match input image.")
                raise typer.Exit(1)
        else:
            console.print("[dim]ROI not provided → separability omitted[/dim]")

        clean_result = None
        clean_metrics_data = None
        
        if roi_result is not None and roi_result.coverage > 0 and roi_result.coverage < 1:
            from .postprocess.background_suppression import suppress_background
            from .postprocess.clean_metrics import calculate_clean_metrics
            import cv2 as cv2_clean
            
            console.print("Generating clean HSI (background suppression)...")
            
            # Resize mask to match HSI dimensions if needed
            roi_mask_for_clean = roi_result.mask
            if roi_result.mask.shape != hsi.shape[:2]:
                roi_mask_for_clean = cv2_clean.resize(
                    roi_result.mask.astype(np.uint8),
                    (hsi.shape[1], hsi.shape[0]),
                    interpolation=cv2_clean.INTER_NEAREST
                ).astype(bool)
            
            clean_result = suppress_background(hsi, roi_mask_for_clean, policy="subtract_mean")
            
            if clean_result is not None:
                exporter.export_array("hsi_clean", clean_result.hsi_clean)
                console.print(f"  Policy: {clean_result.policy}")
                console.print(f"  ✓ Clean HSI generated: {clean_result.hsi_clean.shape}")
                
                console.print("Calculating clean metrics...")
                clean_metrics_data = calculate_clean_metrics(hsi, clean_result.hsi_clean, roi_mask_for_clean)
                
                if clean_metrics_data:
                    console.print(f"  Clean separability: {clean_metrics_data['clean_separability']:.4f}")
                    console.print(f"  Raw-Clean SAM: {clean_metrics_data['raw_clean_sam']:.4f}")
                    console.print(f"  Raw-Clean RMSE: {clean_metrics_data['raw_clean_rmse']:.4f}")
            else:
                console.print("[yellow]Clean skipped:[/yellow] Could not compute (empty/full ROI)")
        elif roi_result is not None:
            console.print("[yellow]Clean skipped:[/yellow] ROI is empty or full (100%)")
        else:
            console.print("[dim]Clean skipped (no ROI mask)[/dim]")

        end_time = time.perf_counter()
        duration = end_time - start_time

        roi_extra = {}
        if roi_result is not None:
            roi_extra["raw_separability"] = raw_separability
            roi_extra["roi_coverage"] = roi_result.coverage
            roi_extra["roi_mask_path"] = roi_result.path
        
        upscaling_extra = {}
        if upscale_factor is not None:
            from .upscaling.spatial import upscale_baseline, upscale_improved
            
            console.print(f"Upscaling HSI by factor {upscale_factor}x...")
            
            console.print("  Running baseline (bicubic per-band)...")
            hsi_baseline = upscale_baseline(hsi, factor=upscale_factor)
            exporter.export_array("hsi_upscaled_baseline", hsi_baseline)
            console.print(f"    Output shape: {hsi_baseline.shape}")
            
            console.print("  Running improved (edge-guided)...")
            from scipy.ndimage import zoom
            import cv2 as cv2_resize
            
            rgb_matched = cv2_resize.resize(rgb, (hsi.shape[1], hsi.shape[0]))
            rgb_upscaled = zoom(rgb_matched, (upscale_factor, upscale_factor, 1), order=3)
            
            try:
                hsi_improved = upscale_improved(hsi, rgb_upscaled, factor=upscale_factor)
                exporter.export_array("hsi_upscaled_improved", hsi_improved)
                console.print(f"    Output shape: {hsi_improved.shape}")
            except ValueError as e:
                console.print(f"[red]Upscaling Error:[/red] {e}")
                console.print("[yellow]Suggestion:[/yellow] Ensure RGB guide dimensions match expected upscaled size.")
                raise typer.Exit(1)
            
            upscaling_extra["upscale_factor"] = upscale_factor
            upscaling_extra["upscaled_size"] = list(hsi_baseline.shape[:2])
            upscaling_extra["upscaling_methods"] = ["baseline_bicubic", "improved_edge_guided"]
            
            console.print(f"  ✓ Upscaling complete: {hsi.shape[:2]} → {hsi_baseline.shape[:2]}")
        else:
            console.print("[dim]Upscaling not requested[/dim]")
        
        metrics_extra = {**roi_extra, **upscaling_extra}
        
        if clean_metrics_data:
            metrics_extra.update(clean_metrics_data)
        
        exporter.export_metrics(
            hsi_shape=hsi.shape,
            execution_time=duration,
            ensemble_enabled=use_ensemble,
            extra=metrics_extra if metrics_extra else None,
        )

        fitting_info = {
            "policy": fit_result.policy,
            "multiple": cfg.fitting.multiple,
            "padding": list(fit_result.padding),
            "input_shape_original": list(fit_result.original_shape),
            "input_shape_fitted": list(fit_result.fitted_shape),
        }
        
        run_config_extra = {
            "ensemble_enabled": use_ensemble,
            "hsi_shape": list(hsi.shape)
        }
        
        if roi_result is not None:
            run_config_extra["roi"] = {
                "mask_path": roi_result.path,
                "policy": "fail_on_invalid",
                "binarize_threshold": 127,
                "coverage": roi_result.coverage,
            }
        
        if clean_result is not None:
            run_config_extra["clean"] = {
                "enabled": True,
                "policy": clean_result.policy,
            }
        
        if upscale_factor is not None:
            run_config_extra["upscaling"] = {
                "enabled": True,
                "factor": upscale_factor,
                "methods": ["baseline_bicubic", "improved_edge_guided"],
                "upscaled_size": list(hsi_baseline.shape[:2]),
            }
        
        exporter.export_run_config(
            input_path=str(input),
            config_path=str(config),
            fitting_info=fitting_info,
            pipeline_version=PIPELINE_VERSION,
            extra=run_config_extra,
        )

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

    exported = exporter.list_exported()
    console.print(f"\n[bold]Exported artifacts ({len(exported)}):[/bold]")
    for artifact in exported:
        console.print(f"  • {artifact}")

    table = Table(title="SpectraPipe Execution Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Input Image", str(input))
    table.add_row("Original Shape", f"{fit_result.original_shape[0]}x{fit_result.original_shape[1]}")
    table.add_row("Fitted Shape", f"{fit_result.fitted_shape[0]}x{fit_result.fitted_shape[1]}")
    table.add_row("Fitting Policy", fit_result.policy)
    table.add_row("Output Directory", str(out))
    table.add_row("Config File", str(config))
    table.add_row("Ensemble Enabled", "No" if no_ensemble else "Yes")
    table.add_row("HSI Shape", str(hsi.shape))
    table.add_row("Total Time", f"{duration:.2f}s")

    console.print(table)
    console.print("[green]✓[/green] Pipeline finished successfully.")


@app.command()
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


@app.command("dataset")
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
    
    policy = OnErrorPolicy.CONTINUE if on_error == "continue" else OnErrorPolicy.ABORT
    
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
    
    cfg = load_config(config)
    use_ensemble = not no_ensemble
    
    def process_sample(sample, sample_out):
        """Process a single sample."""
        import cv2
        from PIL import Image, UnidentifiedImageError
        
        image_path = sample.image_resolved
        if not image_path or not image_path.exists():
            raise FileNotFoundError(f"Image not found: {sample.image}")
        
        try:
            rgb = np.array(Image.open(image_path).convert('RGB'))
        except UnidentifiedImageError:
            raise ValueError(f"Invalid image format: {image_path}")
        
        fit_result = fit_input(
            rgb,
            multiple=cfg.fitting.multiple,
            policy=cfg.fitting.policy
        )
        rgb_fitted = fit_result.fitted
        
        hsi = rgb_to_hsi(rgb_fitted, cfg, ensemble_override=use_ensemble)
        
        exporter = ExportManager(sample_out)
        exporter.export_array("hsi_raw", hsi)
        
        roi_mask_for_metrics = None
        raw_separability = None
        clean_metrics_data = None
        clean_result_obj = None
        
        if sample.roi_mask_resolved:
            roi_path = sample.roi_mask_resolved
            if not roi_path.exists():
                raise FileNotFoundError(f"ROI mask not found: {sample.roi_mask}")
            
            from .roi.loader import load_roi_mask
            from .roi.separability import calculate_separability
            
            roi_result = load_roi_mask(roi_path, fit_result.original_shape[:2])
            
            roi_mask_for_metrics = roi_result.mask
            if roi_result.mask.shape != hsi.shape[:2]:
                roi_mask_for_metrics = cv2.resize(
                    roi_result.mask.astype(np.uint8),
                    (hsi.shape[1], hsi.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            
            raw_separability = calculate_separability(hsi, roi_mask_for_metrics)
            
            if roi_result.coverage > 0 and roi_result.coverage < 1:
                from .postprocess.background_suppression import suppress_background
                from .postprocess.clean_metrics import calculate_clean_metrics
                
                clean_result_obj = suppress_background(hsi, roi_mask_for_metrics, policy="subtract_mean")
                if clean_result_obj:
                    exporter.export_array("hsi_clean", clean_result_obj.hsi_clean)
                    clean_metrics_data = calculate_clean_metrics(hsi, clean_result_obj.hsi_clean, roi_mask_for_metrics)
        
        if upscale_factor:
            from .upscaling.spatial import upscale_baseline, upscale_improved
            from scipy.ndimage import zoom
            
            hsi_baseline = upscale_baseline(hsi, factor=upscale_factor)
            exporter.export_array("hsi_upscaled_baseline", hsi_baseline)
            
            rgb_matched = cv2.resize(rgb, (hsi.shape[1], hsi.shape[0]))
            rgb_upscaled = zoom(rgb_matched, (upscale_factor, upscale_factor, 1), order=3)
            hsi_improved = upscale_improved(hsi, rgb_upscaled, factor=upscale_factor)
            exporter.export_array("hsi_upscaled_improved", hsi_improved)
        
        metrics_extra = {}
        if raw_separability is not None:
            metrics_extra["raw_separability"] = raw_separability
        if sample.roi_mask_resolved:
            metrics_extra["roi_mask_path"] = str(sample.roi_mask_resolved)
        if clean_metrics_data:
            metrics_extra.update(clean_metrics_data)
        if upscale_factor:
            metrics_extra["upscale_factor"] = upscale_factor
            metrics_extra["upscaled_size"] = [hsi.shape[0] * upscale_factor, hsi.shape[1] * upscale_factor]
        
        exporter.export_metrics(
            hsi_shape=hsi.shape,
            execution_time=0,
            ensemble_enabled=use_ensemble,
            extra=metrics_extra if metrics_extra else None
        )
        
        run_config_data = {
            "sample_id": sample.id,
            "image_path": str(sample.image_resolved),
            "fitting": {
                "policy": fit_result.policy,
                "input_shape_original": list(fit_result.original_shape),
                "input_shape_fitted": list(fit_result.fitted_shape),
            },
            "ensemble_enabled": use_ensemble,
        }
        if sample.roi_mask_resolved:
            run_config_data["roi_mask_path"] = str(sample.roi_mask_resolved)
        if clean_result_obj:
            run_config_data["clean"] = {"enabled": True, "policy": clean_result_obj.policy}
        if upscale_factor:
            run_config_data["upscaling"] = {"enabled": True, "factor": upscale_factor}
        
        exporter.export_run_config(
            input_path=image_path,
            config_path=config,
            fitting_info=run_config_data.get("fitting", {}),
            extra=run_config_data
        )
    
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


def main():
    app()

if __name__ == "__main":
    main()