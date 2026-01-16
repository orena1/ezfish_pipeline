# Segmentation Guide

## Overview

The pipeline uses Cellpose for cell segmentation on both HCR and 2P images. This document covers segmentation workflows, mask operations, and feature extraction.

## Cellpose Segmentation

### HCR Segmentation

**Function:** `run_cellpose()` in `src/segmentation.py`

**When it runs:** During per-plane HCR processing.

**Configuration:**
```hjson
"HCR_cellpose": {
  "model_path": "/path/to/cellpose_model",
  "diameter": 15,
  "gpu": true,
  "cellpose_channel": "DAPI"
}
```

**Output:**
```
OUTPUT/HCR/cellpose/
├── HCR1_to_HCR1_masks.tiff
├── HCR2_to_HCR1_masks.tiff
└── ...
```

### 2P Segmentation

**Function:** `run_cellpose_2p()` in `src/segmentation.py`

**When it runs:** During per-plane 2P processing.

**Configuration:**
```hjson
"2p_cellpose": {
  "model_path": "/path/to/cellpose_model",
  "diameter": 7,
  "gpu": true
}
```

**Output:**
```
OUTPUT/2P/cellpose/
├── lowres_meanImg_C0_plane0_seg_rotated.tiff
└── ...
```

## Parameter Tuning

### diameter

Most critical parameter. Set to expected cell diameter in pixels.

| Image Type | Typical Range |
|------------|---------------|
| HCR (lower mag) | 10-20 |
| 2P (higher mag) | 5-10 |

**Tuning:**
- Too small → oversegmentation (cells split)
- Too large → undersegmentation (cells merged)
- Measure a few cells manually to calibrate

### Model Selection

Different Cellpose models for different image types:
- `cyto2` - General cytoplasmic signal
- `nuclei` - Nuclear staining (DAPI)
- Custom trained models for specific tissue types

## Mask Operations

### Mask Alignment

**Function:** `align_masks()` in `src/segmentation.py`

Aligns 2P masks to HCR space using computed transformations.

**Process:**
1. Load 2P segmentation masks
2. Apply 2P-to-HCR transformation
3. Use nearest-neighbor interpolation to preserve mask IDs
4. Save aligned masks

### Mask Matching

**Function:** `match_masks()` in `src/segmentation.py`

Matches cells between different segmentations using IoU (Intersection over Union).

**Algorithm:**
1. For each mask in source, find overlapping masks in target
2. Compute IoU for each overlap
3. Accept matches above IoU threshold
4. Apply bidirectional filtering for 1-to-1 mapping

**IoU Threshold:**
- Default: 0.3
- Higher → fewer matches, higher confidence
- Lower → more matches, lower confidence

### Mask Merging

**Function:** `merge_masks()` in `src/segmentation.py`

Combines matched masks from all rounds/planes into unified tables.

**Output columns:**
- `cell_id`: Unique identifier
- `HCR{X}_mask_id`: Mask ID in each HCR round
- `twop_plane{X}_mask_id`: Mask ID in each 2P plane
- `iou_{source}_to_{target}`: IoU scores for QC

## Feature Extraction

### Probe Intensity Extraction

**Function:** `extract_probs_intensities()` in `src/segmentation.py`

Extracts fluorescence intensity for each cell and probe.

**Features extracted:**
- Mean intensity
- Median intensity
- Percentile values (e.g., 75th, 90th)
- Neuropil-corrected values

**Neuropil estimation:**
- Ring around cell (configurable radius)
- Excludes other cell masks
- Used for background subtraction

### Electrophysiology Intensity Extraction

**Function:** `extract_electrophysiology_intensities()` in `src/segmentation.py`

Extracts Suite2p traces for 2P cells.

**Process:**
1. Map aligned 2P masks to Suite2p ROIs
2. Extract corresponding traces
3. Include in feature table

## Output Tables

### Per-Plane Tables

```
OUTPUT/MERGED/aligned_extracted_features/
├── full_table_mean_twop_plane0.pkl
├── full_table_mean_twop_plane1.pkl
└── ...
```

### Table Columns

| Column | Description |
|--------|-------------|
| cell_id | Unique cell identifier |
| HCR{X}_{probe}_mean | Mean probe intensity |
| HCR{X}_{probe}_neuropil | Neuropil background |
| twop_plane{X}_mask_id | 2P mask ID |
| iou_* | Match quality scores |
| plane | Plane number |

## Quality Control

### Segmentation QC

1. **Visual inspection**: Overlay masks on raw images
2. **Cell count check**: Compare to expected cell density
3. **Size distribution**: Check for bimodal (split cells) or heavy tail (merged)

### Matching QC

1. **IoU distribution**: Check histogram of IoU scores
2. **Match rate**: Fraction of cells with matches
3. **Bidirectional consistency**: Same matches in both directions

### Feature QC

1. **Intensity distributions**: Check for outliers
2. **Neuropil correlation**: Should be lower than soma
3. **Cross-round consistency**: Same cells should have similar probe profiles

## Troubleshooting

| Issue | Likely Cause | Solution |
|-------|--------------|----------|
| Oversegmentation | diameter too small | Increase diameter |
| Undersegmentation | diameter too large | Decrease diameter |
| Few matches | IoU threshold too high | Lower threshold or check alignment |
| Many false matches | IoU threshold too low | Raise threshold |
| Missing cells | Low signal or wrong model | Check raw images, try different model |
| High neuropil | Ring too large | Reduce neuropil radius |

## Code Reference

| Function | Purpose |
|----------|---------|
| `run_cellpose()` | HCR 3D segmentation |
| `run_cellpose_2p()` | 2P 2D segmentation |
| `align_masks()` | Transform masks to common space |
| `match_masks()` | IoU-based cell matching |
| `merge_masks()` | Combine into feature tables |
| `extract_probs_intensities()` | Fluorescence quantification |
| `convex_mask()` | ROI masking |
