# File Naming Conventions

## Overview

Understanding file naming conventions is critical for:
- Debugging missing file errors
- Understanding pipeline state
- Manual intervention when needed

## Output Directory Structure

```
{base_path}/{mouse_name}/OUTPUT/
├── 2P/
│   ├── cellpose/           # 2P segmentation outputs
│   ├── tile/               # Tile processing outputs
│   │   └── stitched/       # Stitched images
│   └── registered/         # Registration outputs
│       └── QA/             # QA overlays
├── HCR/
│   ├── full_registered_stacks/  # Registered HCR rounds
│   └── cellpose/           # HCR segmentation outputs
└── MERGED/
    └── aligned_extracted_features/  # Final feature tables
```

## 2P Files

### Low-Res Mean Images

**Pattern:** `lowres_meanImg_C{channel}_plane{plane}.tiff`

**Example:** `lowres_meanImg_C0_plane0.tiff`

**Created by:** `extract_suite2p_registered_planes()` in `functional.py`

### Tile Files (Hi-Res Mode)

**Pattern:** `hires_{color}_run{run}_tile{tile}.tiff`

**Example:** `hires_green_run001_tile0.tiff`

**Created by:** External acquisition or tile processing

### Stitched Files

**Pattern:** `hires_stitched_plane{plane}.tiff`
**Rotated:** `hires_stitched_plane{plane}_rotated.tiff`

**Example:** `hires_stitched_plane0_rotated.tiff`

**Created by:** `auto_stitch_tiles()` in `auto_stitching.py`

### 2P Segmentation

**Pattern:** `lowres_meanImg_C{channel}_plane{plane}_seg_rotated.tiff`

**Example:** `lowres_meanImg_C0_plane0_seg_rotated.tiff`

**Created by:** `run_cellpose_2p()` in `segmentation.py`

### Registered 2P Images

**Pattern:** `twop_plane{plane}_aligned_3d.tiff`

**Example:** `twop_plane0_aligned_3d.tiff`

**Created by:** `twop_to_hcr_registration()` in `registrations.py`

### Landmark Files

**Manual:** `plane{plane}_TO_HCR{round}_landmarks.csv`
**Auto-refined:** `plane{plane}_TO_HCR{round}_landmarks_auto.csv`

**Example:** `plane0_TO_HCR1_landmarks.csv`

**Created by:** User (manual) or `twop_to_hcr_registration()` (auto)

### QA Overlays

**Pattern:** `plane{plane}_{BEFORE|AFTER}_registration_overlay.tiff`

**Example:** `plane0_AFTER_registration_overlay.tiff`

**Created by:** `twop_to_hcr_registration()` in `registrations.py`

## HCR Files

### Registered Stacks

**Pattern:** `HCR{source}_to_HCR{reference}.tiff`

**Example:** `HCR2_to_HCR1.tiff`

**Created by:** `register_rounds()` in `registrations.py`

### HCR Segmentation

**Pattern:** `HCR{round}_to_HCR{reference}_masks.tiff`

**Example:** `HCR1_to_HCR1_masks.tiff`

**Created by:** `run_cellpose()` in `segmentation.py`

## Merged Files

### Feature Tables

**Pattern:** `full_table_{feature}_twop_plane{plane}.pkl`

**Example:** `full_table_mean_twop_plane0.pkl`

**Created by:** `merge_masks()` in `segmentation.py`

## Files by Pipeline Mode

### Full Pipeline + Hi-Res Stitched

| File | Present |
|------|---------|
| Tile files | ✓ |
| Stitched files | ✓ |
| Low-res mean | ✓ |
| 2P segmentation | ✓ |
| Registered 2P | ✓ |
| Landmarks | ✓ |
| QA overlays | ✓ |
| HCR registered | ✓ |
| HCR segmentation | ✓ |
| Feature tables | ✓ |

### Full Pipeline + Standard

| File | Present |
|------|---------|
| Tile files | ✗ |
| Stitched files | ✗ |
| Low-res mean | ✓ |
| 2P segmentation | ✓ |
| Registered 2P | ✓ |
| Landmarks | ✓ |
| QA overlays | ✓ |
| HCR registered | ✓ |
| HCR segmentation | ✓ |
| Feature tables | ✓ |

### HCR-Only Mode

| File | Present |
|------|---------|
| Tile files | ✗ |
| Stitched files | ✗ |
| Low-res mean | ✗ |
| 2P segmentation | ✗ |
| Registered 2P | ✗ |
| Landmarks | ✗ |
| QA overlays | ✗ |
| HCR registered | ✓ |
| HCR segmentation | ✓ |
| Feature tables | Partial |

## Common File Errors

### "File not found"

1. Check mode matches expected files
2. Verify naming convention matches pattern
3. Check previous pipeline stage completed

### "File already exists" (skip)

Pipeline skips steps with existing outputs. To re-run:
1. Delete specific output file
2. Or delete OUTPUT folder for fresh run

### Partial/Corrupted Files

May occur from interrupted runs. Delete file and re-run.

## Version Suffixes

Some files may have version suffixes:
- `_v2` - Updated algorithm version
- `_auto` - Automatically generated (vs manual)
- `_rotated` - Rotation applied for alignment

## Checking Pipeline State

To determine which steps are complete, check for existence of key files:

| File Exists | Step Complete |
|-------------|---------------|
| `lowres_meanImg_*.tiff` | Suite2p extraction |
| `hires_stitched_*.tiff` | Tile stitching |
| `*_seg_rotated.tiff` | 2P segmentation |
| `HCR*_to_HCR*.tiff` | HCR registration |
| `*_masks.tiff` | HCR segmentation |
| `twop_*_aligned_3d.tiff` | 2P-HCR registration |
| `full_table_*.pkl` | Feature extraction |
