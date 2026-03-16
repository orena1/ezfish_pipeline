# Resume & Re-Entry Guide

## How Completion Is Detected

The pipeline checks output file existence: file exists → skip, no file → run.

**Warning**: Partial/corrupted files from interrupted runs will be treated as complete.

## Common Scenarios

### Clean Re-Run
Delete `OUTPUT/` folder entirely, then re-run.

### Re-Do Specific Step
Delete only that step's output files, then re-run. Pipeline skips completed steps.

```bash
# Example: Re-do 2P-HCR registration for plane 0
rm OUTPUT/2P/registered/twop_plane0_aligned_3d.tiff
rm OUTPUT/2P/registered/QA/plane0_*.tiff
python master_pipeline.py manifest.hjson
```

### Resume After Crash
Check last completed step (file timestamps), delete any partial files, re-run.

**Signs of partial files**: unusually small size, recent timestamp during crash, read errors downstream.

### Change Parameters
Update manifest, delete affected outputs, re-run.

| Parameter Changed | Delete |
|-------------------|--------|
| `twop_to_hcr_registration.*` | `twop_*_aligned_3d.tiff`, QA overlays |
| `HCR_cellpose.*` | `*_masks.tiff` in `HCR/cellpose` |
| `2p_cellpose.*` | `*_seg_rotated.tiff` |
| `stitching.*` | `hires_stitched_*.tiff` |

### Add New Plane
Update manifest `additional_functional_planes`, re-run. Existing planes preserved.

### Switch Modes
- Full → HCR-only: add `--only_hcr` (2P outputs preserved, not used)
- HCR-only → Full: remove `--only_hcr` (skipped 2P steps will process)

## Safe Deletion Patterns

```bash
# Registration only
rm OUTPUT/2P/registered/twop_plane*_aligned_3d.tiff

# 2P segmentation only
rm OUTPUT/2P/cellpose/*_seg_rotated.tiff

# HCR segmentation only
rm OUTPUT/HCR/cellpose/*_masks.tiff

# Feature tables only
rm OUTPUT/MERGED/aligned_extracted_features/*.pkl

# Everything for one plane
rm OUTPUT/2P/cellpose/*plane0*
rm OUTPUT/2P/registered/*plane0*
rm OUTPUT/MERGED/aligned_extracted_features/*plane0*
```
