# SpectraPipe: HSI to RGB Reconstruction CLI

![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-beta-orange)

**SpectraPipe** is a CLI tool designed for scientists and researchers seeking to automate, standardize, and guarantee reproducibility in the spectral reconstruction process (RGB to HSI) using the **MST++** model.

Unlike ad-hoc scripts, SpectraPipe offers a structured workflow for handling hyperspectral datasets, ensuring consistency in preprocessing, inference, and metric calculation.

## 🚀 Key Features

*   **SOTA Reconstruction:** MST++ model implementation for RGB -> HSI conversion.
*   **Annotation Handling:** Native support for ROI masks and COCO, Pascal VOC, and VGG Image Annotator (VIA) formats.
*   **Robust Preprocessing:**
    *   Input fitting with *reflection padding* (multiples of 32).
    *   Bicubic and RGB edge-guided upscaling.
    *   Automatic background suppression using ROI masks.
*   **Standardized Output:** Cubes exported to `.npz` format (Schema v1) with complete, reproducible metadata.
*   **Integrated Metrics:** Automatic calculation of separability metrics, SAM (Spectral Angle Mapper), and RMSE.

## 📦 Installation

```bash
# Install from source
git clone https://github.com/fernandevdaza/SpectraPipe.git
cd SpectraPipe
poetry install
# Or pip install -e .
```

### 🧱 Model Application Setup

SpectraPipe currently relies on the external `MST-plus-plus` repository and pretrained weights.

#### 1. Clone MST++
Clone the official repository into `external/`:

```bash
mkdir -p external
git clone https://github.com/caiyuanhao1998/MST-plus-plus.git external/MST-plus-plus
```

#### 2. Download Weights
Download the pre-trained weights (`mst_plus_plus.pth`) and place them in the weights directory.

*   **Download Link:** [Google Drive (mst_plus_plus.pth)](https://drive.google.com/file/d/18X6RkcQaIuiV5gRbswo7GLv7WJG9M_WM/view?usp=sharing)
*   **Target Path:** `src/hsi_pipeline/weights/model_zoo/mst_plus_plus.pth`

```bash
mkdir -p src/hsi_pipeline/weights/model_zoo/
# Move downloaded file:
mv ~/Downloads/mst_plus_plus.pth src/hsi_pipeline/weights/model_zoo/
```

## 🛠️ Usage (CLI)

SpectraPipe operates via 4 essential commands:

### 1. Process Single Image (`run`)

Ideal for quick tests or single-sample inference.

```bash
spectrapipe run --input input.png --roi-mask mask.png --out results/sample_01
# Or using annotations directly
spectrapipe run --input input.png --annotation labels.xml --annotation-type voc
```

### 2. Batch Processing (`dataset`)

Ensures reproducibility by loading a configuration manifest.

```bash
spectrapipe dataset --manifest manifest.yaml --out results/batch_run
```

### 3. Signature Extraction (`spectra`)

Extracts and processes spectral signatures from generated HSI cubes.

```bash
# Extract specific pixel
spectrapipe spectra --from results/sample_01 --artifact raw --pixel 120,80

# Extract mean ROI signature
spectrapipe spectra --from results/sample_01 --artifact clean --roi-agg mean --roi-mask mask.png
```

### 4. Metrics Report (`metrics`)

Reads processed artifacts and displays a readable metrics summary.

```bash
spectrapipe metrics --from results/sample_01
```

## 📄 Configuration & Data Contracts

### Reproducibility Standards
SpectraPipe allows you to define strict experimental setups.

*   **Manifest (`manifest.yaml`)**: Defines the dataset source and samples.
*   **Global Config (`config.yaml`)**: Controls model parameters and export behavior.

> [!NOTE] 
> For detailed specifications of the **Input formats**, **Output Schemas (NPZ v1)**, and **Metadata structures**, please refer to the [Data Contracts specification](DATA_CONTRACTS.md).

Example `config.yaml`:
```yaml
model:
  name: mst_plus_plus
  ensemble: true
fitting:
  policy: pad_to_multiple
export:
  format: npz
  include_wavelengths: false
```

## 🗺️ Roadmap

*   [x] MST++ Integration.
*   [x] COCO/VOC/VIA Support.
*   [x] Schema v1 for NPZ (Embedded Metadata).
*   [ ] **Modular Architecture:** Make the Deep Learning backbone interchangeable.
*   [ ] Docker Support.

## 📚 Citation

If you use SpectraPipe in your research, please cite:

```bibtex
@software{spectrapipe2026,
  author = {Said Fernando Daza Peña},
  title = {SpectraPipe: A Reproducible HSI-to-RGB Reconstruction Pipeline},
  year = {2026},
  url = {https://github.com/fernandevdaza/SpectraPipe}
}
```
