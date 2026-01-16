# Troubleshooting Guide

## Overview

This guide covers common errors, their causes, and solutions. Organized by error type and pipeline stage.

## File Not Found Errors

### "Reference round not found"

**Message:** `Reference round HCR{X} not found at {path}`

**Causes:**
- HCR data not in expected location
- Wrong `reference_round` in manifest
- Path formatting issue (forward vs backslash)

**Solutions:**
1. Verify HCR files exist at expected path
2. Check `HCR_confocal_imaging.reference_round` in manifest
3. On Windows, ensure paths use proper formatting

---

### "Landmark file not found"

**Message:** `Landmark file not found: {path}`

**Causes:**
- Landmarks not yet created
- Wrong naming convention
- Running full pipeline before creating landmarks

**Solutions:**
1. Create landmarks in BigWarp (see [registration_guide.md](registration_guide.md))
2. Check file name matches pattern: `plane{X}_TO_HCR{Y}_landmarks.csv`
3. Verify file is in `OUTPUT/2P/registered/`

---

### "Tile file not found"

**Message:** `Tile {X} not found: {path}`

**Causes:**
- Wrong `use_automated_stitching` setting
- Tile files not generated/moved
- Naming mismatch

**Solutions:**
1. Verify `use_automated_stitching: true` in manifest
2. Check tile files follow pattern: `hires_{color}_run{X}_tile{Y}.tiff`
3. Ensure tiles are in `OUTPUT/2P/tile/`

---

## Registration Errors

### "Global alignment failed"

**Message:** `Global alignment failed: no valid rotation found`

**Causes:**
- Images too different to align
- Rotation range too narrow
- Very low signal in one image

**Solutions:**
1. Widen `rotation_range` (e.g., [-10, 10])
2. Check input images have sufficient signal
3. Verify images are from same region

---

### "TPS warping failed"

**Message:** `TPS transformation failed: singular matrix`

**Causes:**
- Too few landmarks (need >3)
- Landmarks are collinear
- Duplicate landmarks

**Solutions:**
1. Add more landmarks (10-20 recommended)
2. Distribute landmarks across FOV
3. Remove any duplicate points

---

### "Tile shift exceeds maximum"

**Message:** `Tile shift ({X}, {Y}) exceeds tile_xy_max`

**Causes:**
- Local distortion larger than allowed
- Wrong tile_xy_max setting
- Registration fundamentally wrong

**Solutions:**
1. Increase `tile_xy_max` parameter
2. Check global alignment was successful
3. Review QA overlays for issues

---

## Segmentation Errors

### "Cellpose model not found"

**Message:** `Model not found: {path}`

**Causes:**
- Wrong model path in manifest
- Model file moved/deleted

**Solutions:**
1. Verify `model_path` in `HCR_cellpose` or `2p_cellpose`
2. Use absolute path to model file

---

### "No cells detected"

**Message:** `Cellpose returned 0 masks`

**Causes:**
- diameter parameter wrong
- Very low signal
- Wrong channel selected

**Solutions:**
1. Adjust `diameter` parameter (measure cells manually)
2. Check input image has signal
3. Verify `cellpose_channel` matches data

---

### "Too many cells detected" (oversegmentation)

**Causes:**
- diameter too small
- Noise being segmented

**Solutions:**
1. Increase `diameter` parameter
2. Check for background noise issues

---

## Memory Errors

### "MemoryError" or "Out of memory"

**Causes:**
- Images too large
- Too many tiles
- Multiple large arrays in memory

**Solutions:**
1. Process planes separately
2. Reduce tile count if possible
3. Close other applications
4. Increase system RAM/swap

---

## Stitching Errors

### "No valid shifts found"

**Message:** `Auto-stitch failed: no valid pairwise shifts`

**Causes:**
- Wrong `overlap_fraction`
- Low signal in overlap regions
- Tiles don't actually overlap

**Solutions:**
1. Verify `overlap_fraction` matches acquisition
2. Check tile signal quality
3. Fall back to manual stitching

---

### "Visible seams in stitched image"

**Causes:**
- Incorrect shift detection
- Insufficient feathering
- Brightness differences between tiles

**Solutions:**
1. Check shift quality in `shift_report.json`
2. Increase overlap fraction
3. Consider manual stitching

---

## QA Issues

### "Registration looks wrong in QA overlay"

**Symptoms:**
- Cells don't align between channels
- Visible rotation error
- Large translation offset

**Causes:**
- Parameters not optimal
- Landmarks incorrect
- Input images problematic

**Solutions:**
1. Adjust parameters (see [parameter_tuning.md](parameter_tuning.md))
2. Re-check/re-create landmarks
3. Verify input images are correct

---

### "Edge artifacts in registered image"

**Causes:**
- erosion too low
- TPS warping creating edge distortion

**Solutions:**
1. Increase `erosion` parameter
2. Add landmarks near edges

---

## Pipeline Flow Issues

### "Step skipped unexpectedly"

**Cause:** Output file already exists (from previous run).

**Solutions:**
1. Delete specific output file to re-run step
2. Check [resume_and_reentry.md](resume_and_reentry.md) for patterns

---

### "Steps running out of order"

**Cause:** Dependency not met but output file exists from failed run.

**Solutions:**
1. Delete partial/corrupt output files
2. Run from clean state if needed

---

### "Different results on re-run"

**Causes:**
- Parameters changed between runs
- Partial outputs mixed with new
- Random initialization in algorithms

**Solutions:**
1. Do clean re-run for consistent results
2. Track parameter changes in manifest commits
3. Set random seeds if reproducibility needed

---

## Mode-Specific Issues

### HCR-Only Mode Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Missing 2P outputs | Expected - mode skips 2P | Remove `--only_hcr` for 2P |
| Incomplete tables | No 2P data in HCR-only | Expected behavior |

### Hi-Res Stitched Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Wrong tile count | Naming mismatch | Check tile file names |
| Memory errors | Large stitched image | Process per-plane |

---

## Getting Help

If issue not resolved:

1. **Check existing docs**: May be covered elsewhere
2. **Collect diagnostics**:
   - Error message (full traceback)
   - Manifest parameters
   - File paths involved
   - Input image properties (size, signal)
3. **Check QA overlays**: Visual inspection often reveals issue
4. **Try clean re-run**: Eliminates state corruption

---

## Quick Reference: Common Fixes

| Symptom | Quick Fix |
|---------|-----------|
| Registration wrong | Adjust erosion, tile_sizes |
| Cells merged | Decrease Cellpose diameter |
| Cells split | Increase Cellpose diameter |
| Step skipped | Delete output file |
| Memory error | Process smaller regions |
| File not found | Check paths and naming |
| Stitching failed | Check overlap_fraction |
