# Auto-Stitching Guide

## Overview

The auto-stitching system combines multiple image tiles into a single stitched image using phase correlation and edge detection. This document covers the algorithm, failure modes, and safeguards.

## Algorithm

### Phase 1: Shift Detection

For each pair of adjacent tiles:
1. Extract overlap regions based on `overlap_fraction`
2. Apply phase correlation (FFT-based) to find optimal XY shift
3. Score shift quality using correlation peak and edge detection

### Phase 2: Global Position Optimization

1. Build graph of pairwise shifts
2. Compute global tile positions via least-squares optimization
3. Detect and handle outlier shifts

### Phase 3: Blending

1. Place tiles at computed positions
2. Apply linear feathering in overlap regions
3. Generate final stitched image

## Key Parameters

```hjson
"auto_stitch_params": {
  "overlap_fraction": 0.50,
  "noise_floor": 15.0
}
```

| Parameter | Effect | Tuning |
|-----------|--------|--------|
| overlap_fraction | Expected tile overlap | Must match actual acquisition |
| noise_floor | Signal threshold | Increase for noisy images |

## Failure Modes & Safeguards

### 1. Incorrect Overlap Fraction

**Symptom:** Tiles misaligned or stitching fails completely.

**Safeguard:** Validate overlap region contains sufficient signal.

**Solution:** Verify `overlap_fraction` matches acquisition settings.

---

### 2. Low Signal in Overlap

**Symptom:** Poor shift detection, visible seams.

**Safeguard:** Check signal-to-noise in overlap region; warn if below threshold.

**Solution:** Adjust `noise_floor` or use manual stitching.

---

### 3. Repetitive Patterns

**Symptom:** Phase correlation finds wrong shift due to pattern ambiguity.

**Safeguard:** Multi-scale verification of shift consistency.

**Solution:** Use landmarks or manual stitching.

---

### 4. Empty Tiles

**Symptom:** Stitching fails or produces artifacts.

**Safeguard:** Detect and skip empty tiles.

**Solution:** Verify tile files contain valid data.

---

### 5. Memory Exhaustion

**Symptom:** Process crashes during blending.

**Safeguard:** Process tiles in chunks if total size exceeds threshold.

**Solution:** Reduce tile count or process planes separately.

---

### 6. Inconsistent Z-Planes

**Symptom:** Different planes have different optimal shifts.

**Safeguard:** Compute shifts per Z-plane, detect outliers.

**Solution:** Use median shift across planes if inconsistent.

---

### 7. Tile Naming Mismatch

**Symptom:** "File not found" errors.

**Safeguard:** Validate tile files exist before processing.

**Solution:** Check naming convention: `hires_{color}_run{X}_tile{Y}.tiff`

---

### 8. Edge Artifacts

**Symptom:** Visible edges at tile boundaries.

**Safeguard:** Apply feathering in overlap regions.

**Solution:** Increase overlap or use larger feathering region.

## Fallback Strategy

If auto-stitching fails, the pipeline supports manual fallback:

1. **Warning issued**: Pipeline warns about auto-stitch failure
2. **Manual option**: User can provide pre-stitched images
3. **Continue processing**: Pipeline proceeds with available data

To use manual stitching:
1. Stitch tiles externally (e.g., Fiji/ImageJ)
2. Save as `hires_stitched_plane{X}.tiff`
3. Set `use_automated_stitching: false` in manifest
4. Re-run pipeline

## Output Files

**Intermediate:**
```
OUTPUT/2P/tile/
├── shift_report.json  # Computed shifts and quality scores
└── ...
```

**Final:**
```
OUTPUT/2P/tile/stitched/
├── hires_stitched_plane0.tiff
├── hires_stitched_plane0_rotated.tiff
└── ...
```

## Quality Checks

After stitching, verify:

1. **No visible seams** at tile boundaries
2. **Consistent brightness** across tiles
3. **Correct alignment** of features crossing boundaries
4. **No missing regions** where tiles should overlap

## Troubleshooting

| Issue | Likely Cause | Solution |
|-------|--------------|----------|
| "No valid shifts found" | Low signal or wrong overlap | Check `overlap_fraction` |
| Visible seams | Poor feathering | Verify overlap regions |
| Tiles misaligned | Wrong shift detected | Try manual stitching |
| Memory error | Too many/large tiles | Process planes separately |
| File not found | Naming mismatch | Check tile file names |

## Code Reference

| Function | File | Purpose |
|----------|------|---------|
| `auto_stitch_tiles()` | `src/auto_stitching.py` | Main entry point |
| `find_best_shift()` | `src/auto_stitching.py` | Phase correlation |
| `compute_pairwise_shifts()` | `src/auto_stitching.py` | All tile pairs |
| `compute_global_positions()` | `src/auto_stitching.py` | Optimize positions |
| `stitch_single_plane_channel()` | `src/auto_stitching.py` | Blend tiles |
