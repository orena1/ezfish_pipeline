# Segmentation Guide

## Cellpose Segmentation

### HCR
- **Function**: `run_cellpose()` in `src/segmentation.py`
- **Output**: `OUTPUT/HCR/cellpose/HCR{N}_to_HCR1_masks.tiff`
- **Config**: `params.HCR_cellpose` — `model_path`, `diameter` (typ. 10-20), `gpu`, `cellpose_channel`

### 2P
- **Function**: `run_cellpose_2p()` in `src/segmentation.py`
- **Output**: `OUTPUT/2P/cellpose/lowres_meanImg_C{ch}_plane{N}_seg_rotated.tiff`
- **Config**: `params.2p_cellpose` — `model_path`, `diameter` (typ. 5-10), `gpu`

### Diameter Tuning
Most critical parameter. Measure a few cells manually to calibrate.
- Too small → oversegmentation (cells split)
- Too large → undersegmentation (cells merged)

---

## Mask Operations

### Alignment (`align_masks()`)
Applies 2P-to-HCR transform to masks using nearest-neighbor interpolation (preserves mask IDs).

### Matching (`match_masks()`)
IoU-based cell matching between segmentations:
1. Find overlapping masks between source and target
2. Accept matches above IoU threshold (default: 0.3)
3. Bidirectional filtering for 1-to-1 mapping

### Merging (`merge_masks()`)
Combines matched masks from all rounds/planes into feature tables.

---

## Feature Extraction

- **Probe intensity** (`extract_probs_intensities()`): mean, median, percentile values with neuropil subtraction
- **Suite2p traces** (`extract_electrophysiology_intensities()`): maps aligned 2P masks to Suite2p ROIs

## Output Tables

`OUTPUT/MERGED/aligned_extracted_features/full_table_{feature}_twop_plane{N}.pkl`

Key columns: `cell_id`, `HCR{X}_{probe}_mean`, `HCR{X}_{probe}_neuropil`, `twop_plane{X}_mask_id`, `iou_*`

---

## QA

- **Segmentation**: overlay masks on raw images; check cell count and size distribution
- **Matching**: check IoU histogram and match rate
- **Features**: check intensity distributions and cross-round consistency
