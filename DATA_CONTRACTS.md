# SpectraPipe Data Contracts

This document specifies the data formats, schemas, and contracts for the SpectraPipe system inputs and artifacts.

---

## 1. Input: Dataset Manifest
**File:** `manifest.json` or `manifest.yaml`  
**Description:** Defines a batch of samples to be processed.

### Fields
| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `root` | string | **Yes** | Root directory path. All relative paths in samples are resolved against this. |
| `samples` | list | Cond.* | List of sample definitions. Required if `pattern` is missing. |
| `pattern` | string | Cond.* | Glob pattern to discover images (e.g., `"images/*.jpg"`). Required if `samples` is missing. |

\* *Exactly one of `samples` or `pattern` must be provided.*

### Sample Object
| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `id` | string | **Yes** | Unique identifier for the sample. Used for output folder naming. |
| `image` | string | **Yes** | Path to the RGB image (relative to `root` or absolute). |
| `roi_mask` | string | No | Path to binary mask for ROI (optional). |
| `annotation`| string | No | Path to annotation file (VOC/COCO/VIA). |
| `annotation_type`| string | No | Type of annotation: `"voc"`, `"coco"`, or `"via"`. |

### JSON Schema
```json
{
  "type": "object",
  "required": ["root"],
  "properties": {
    "root": { "type": "string" },
    "pattern": { "type": "string" },
    "samples": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["id", "image"],
        "properties": {
          "id": { "type": "string" },
          "image": { "type": "string" },
          "roi_mask": { "type": "string" },
          "annotation": { "type": "string" },
          "annotation_type": { "enum": ["voc", "coco", "via"] }
        }
      }
    }
  }
}
```

---

## 2. Configuration: Pipeline Config
**File:** `config.yaml`  
**Description:** Controls pipeline behavior globally.

### Fields
| Section | Field | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| **model** | `name` | string | `"mst_plus_plus"` | Model architecture name. |
| | `weights_path` | string | `".../mst_plus_plus.pth"` | Path to .pth weights file. |
| | `device` | string | `"auto"` | Execution device: `"auto"`, `"cuda"`, `"cpu"`. |
| | `ensemble` | bool | `true` | Enable test-time augmentation (8x). |
| | `ensemble_mode` | string | `"mean"` | Aggregation strategy: `"mean"` or `"median"`. |
| **fitting** | `multiple` | int | `32` | Factor to pad images to (CNN constraint). |
| | `policy` | string | `"pad_to_multiple"` | Fitting policy. |
| **upscaling**| `enabled` | bool | `false` | Enable spatial upscaling by default. |
| | `factor` | int | `2` | Upscaling factor (2-8). |
| **export** | `format` | string | `"npz"` | Output format. |
| | `overwrite` | bool | `true` | Overwrite existing files. |

### JSON Schema (YAML Map)
```json
{
  "type": "object",
  "properties": {
    "model": {
      "type": "object",
      "properties": {
        "ensemble": { "type": "boolean" },
        "weights_path": { "type": "string" }
      }
    },
    "upscaling": {
      "type": "object",
      "properties": {
        "enabled": { "type": "boolean" },
        "factor": { "type": "integer", "minimum": 2 }
      }
    }
  }
}
```

---

## 3. Artifact: Run Configuration
**File:** `run_config.json`  
**Description:** Persisted snapshot of the configuration and metadata used for a specific execution. Guaranteed to exist for every successful run.

### Fields
| Section | Field | Type | Description |
| :--- | :--- | :--- | :--- |
| **meta** | `pipeline_version` | string | Version of SpectraPipe used. |
| | `timestamp` | string | ISO 8601 execution time. |
| | `input_path` | string | Absolute path to input image. |
| | `output_dir` | string | Absolute path to output directory. |
| | `fitting` | object | Details about applied padding (`original_shape`, `padding`). |
| | `sample_id` | string | (Batch only) ID of the sample. |
| **config** | *(object)* | object | Complete copy of the `RunConfig` used (see Section 2). |

### JSON Schema
```json
{
  "type": "object",
  "required": ["meta", "config"],
  "properties": {
    "meta": {
      "type": "object",
      "required": ["timestamp", "input_path", "pipeline_version"],
      "properties": {
        "pipeline_version": { "type": "string" },
        "timestamp": { "type": "string", "format": "date-time" },
        "fitting": { 
            "type": "object",
            "properties": {
                "original_shape": { "type": "array", "items": { "type": "integer" } },
                "padding": { "type": "array" }
            }
        }
      }
    },
    "config": { "type": "object" }
  }
}
```

---

## 4. Artifact: Metrics
**File:** `metrics.json`  
**Description:** Computed quality and performance metrics for a single run.

### Fields
| Field | Type | Description |
| :--- | :--- | :--- |
| `hsi_shape` | `[H, W, C]` | Dimensions of the generated HSI cube. |
| `n_bands` | int | Number of spectral bands (typically 31). |
| `execution_time_seconds` | float | Total pipeline runtime in seconds. |
| `ensemble_enabled` | bool | Whether ensembling was active. |
| `clean_metrics` | object | (Optional) if ROI provided. Contains `sam_mean`, `sid_sum`, etc. |
| `raw_separability` | float | (Optional) Separability score if ROI provided. |

### JSON Schema
```json
{
  "type": "object",
  "required": ["hsi_shape", "execution_time_seconds", "timestamp"],
  "properties": {
    "hsi_shape": { 
        "type": "array", 
        "items": { "type": "integer" },
        "minItems": 3,
        "maxItems": 3
    },
    "execution_time_seconds": { "type": "number" },
    "ensemble_enabled": { "type": "boolean" },
    "clean_metrics": { "type": "object", "nullable": true }
  }
}
```

---

## 5. Artifact: Dataset Report
**File:** `dataset_report.json`  
**Description:** Aggregate summary generated after a `dataset` batch command.

### Fields
| Field | Type | Description |
| :--- | :--- | :--- |
| `total_samples` | int | Total number of samples in manifest. |
| `processed_ok` | int | Count of successfully processed samples. |
| `failed` | int | Count of failures. |
| `total_time_seconds` | float | Total batch execution time. |
| `failures` | list | List of failure objects `{sample_id, reason}`. |
| `samples` | list | List of result objects for every sample. |

### JSON Schema
```json
{
  "type": "object",
  "required": ["total_samples", "processed_ok", "failed", "samples"],
  "properties": {
    "total_samples": { "type": "integer" },
    "processed_ok": { "type": "integer" },
    "failed": { "type": "integer" },
    "failures": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "sample_id": { "type": "string" },
          "reason": { "type": "string" }
        }
      }
    },
    "samples": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "sample_id": { "type": "string" },
          "success": { "type": "boolean" },
          "output_dir": { "type": ["string", "null"] },
          "error": { "type": ["string", "null"] }
        }
      }
    }
  }
}
```
---

## 6. Artifact: HSI Cube (NPZ Schema v1)
**File:** `hsi_*.npz` (raw, clean, upscaled)  
**Description:** Standardized storage for hyperspectral cubes with metadata.

### Keys
| Key | Type | Description |
| :--- | :--- | :--- |
| `cube` | `float32` array | 3D Hyperspectral Cube `[H, W, Bands]`. |
| `metadata` | `object` (JSON) | Metadata containing `pipeline_version`, `artifact_type`, and shapes. |
| `wavelength_nm` | `float32` array | (Optional) Wavelength values for the spectral bands. |
| `schema_version` | `int` | Version identifier (currently `1`). |

### Metadata Object
```json
{
    "artifact": "raw|clean|upscaled_baseline|...",
    "pipeline_version": "0.1.0",
    "original_shape": [H, W],
    "fitted_shape": [H', W'],
    "timestamp": "ISO8601"
}
```
