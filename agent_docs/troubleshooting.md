# Troubleshooting Guide

## File Not Found Errors

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| Reference round not found | Wrong `reference_round` or path issue | Check manifest HCR path and round number |
| Landmark file not found | Not yet created, or naming mismatch | Create in BigWarp; check `plane{X}_TO_HCR{Y}_landmarks.csv` |
| Tile file not found | Wrong `use_automated_stitching` setting | Verify manifest setting and tile naming: `hires_{color}_run{X}_tile{Y}.tiff` |

## Registration Errors

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| Global alignment failed | Rotation range too narrow or low signal | Widen `rotation_range`; check input images |
| TPS warping failed (singular matrix) | Too few/collinear landmarks | Add 10-20 landmarks distributed across FOV |
| Tile shift exceeds maximum | Local distortion > `tile_xy_max` | Increase `tile_xy_max`; check global alignment first |

## Segmentation Errors

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| Cellpose model not found | Wrong `model_path` | Use absolute path in manifest |
| No cells detected (0 masks) | Wrong diameter or low signal | Measure cells manually; adjust `diameter` |
| Oversegmentation | Diameter too small | Increase `diameter` |
| Undersegmentation | Diameter too large | Decrease `diameter` |

## Stitching Errors

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| No valid shifts found | Wrong `overlap_fraction` or low signal | Verify overlap matches acquisition; try manual stitching |
| Visible seams | Poor shift detection or feathering | Check `shift_report.json`; increase overlap |
| Memory error | Too many/large tiles | Process planes separately |

## Pipeline Flow Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| Step skipped unexpectedly | Output file exists from previous run | Delete specific output file |
| Steps running out of order | Partial file from failed run | Delete partial/corrupt files |
| Different results on re-run | Mixed parameter outputs | Clean re-run for consistency |
| Aligned 2P masks have 0 cells | Masks eroded during registration | Use QA overlays for visualization; re-transform without erosion for figures |

## Memory Errors

Large images or many tiles can exhaust RAM. Solutions:
1. Process planes separately
2. Reduce tile count
3. Close other applications

## Quick Diagnostics

1. Check QA overlays — visual inspection reveals most issues
2. Check file sizes — unusually small = likely corrupt
3. Check file timestamps — recent during crash = partial
4. Do a clean re-run if state is confused
