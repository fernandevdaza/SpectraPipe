# SpectraPipe - HSI Pipeline

SpectraPipe is a CLI utility for Hyperspectral Image (HSI) processing. It converts RGB images to 31-band hyperspectral data using MST++ deep learning model.

## Installation

```bash
git clone https://github.com/fernandev/SpectraPipe.git
cd SpectraPipe
poetry install
```

## Usage

### Run Pipeline

```bash
# Basic usage (output to <input_dir>/output/)
poetry run spectrapipe run --input data/sample_rgb.png

# Full options
poetry run spectrapipe run \
  --input data/sample_rgb.png \
  --out runs/experiment_01 \
  --config configs/custom.yaml \
  --no-ensemble
```

### View Metrics

```bash
# Display metrics from a previous run
poetry run spectrapipe metrics --from runs/experiment_01
```

Example output:
```
╭─────────────────────────────────────────╮
│   SpectraPipe Metrics Summary           │
╰─────────────────────────────────────────╯

📊 General Stats
  HSI Shape     (512, 512, 31)
  Bands         31
  Execution Time 12.50s
  Ensemble      Yes

✓ Metrics loaded successfully.
```

## Output Contract

Every successful run produces the following artifacts in `--out`:

| File | Description | Always |
|------|-------------|--------|
| `hsi_raw_full.npz` | HSI cube (H, W, 31) | ✓ |
| `metrics.json` | Execution metrics | ✓ |
| `run_config.json` | Full run configuration | ✓ |
| `hsi_clean_full.npz` | Cleaned HSI (if ROI provided) | ○ |
| `hsi_upscaled_*.npz` | Upscaled HSI variants | ○ |
| `roi_mask.png` | ROI mask (if provided) | ○ |

## Features

- **RGB to HSI Conversion**: MST++ model reconstructs 31 spectral bands (400-700nm)
- **Automatic Input Fitting**: Handles any image size via padding
- **Ensemble Support**: Test-time augmentation for improved accuracy
- **Consistent Export**: Standardized artifact naming and metadata
- **Metrics Viewer**: Human-readable summary of execution results