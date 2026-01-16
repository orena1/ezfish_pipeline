# Parameter Tuning Guide

## Overview

This document provides a complete reference for all tunable parameters, their effects, and recommended values for different scenarios.

## Parameter Naming (Updated)

The following parameters have been renamed for clarity. Both old and new names are supported for backward compatibility:

| Old Name | New Name | Location |
|----------|----------|----------|
| `rotation_2p_to_HCRspec` | `rotation_2p_to_HCR` | `params` |
| `HCR_to_HCR_params` | `HCR_to_HCR_registration` | `params` |
| `auto_stitch_params` | `stitching` | `params` |
| `HCR_probe_intensity_extraction` | `intensity_extraction` | `params` |

Additionally, `HCR_to_HCR_registration.downsampling` now accepts an array `[x, y, z]` instead of separate `red_mut_x`, `red_mut_y`, `red_mut_z` fields.

## Low-Res to High-Res Registration Parameters

Location: `params.lowres_to_hires_registration`

### Global SIFT Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| n_features | Integer | 5000 | Max SIFT keypoints to detect |
| ratio_threshold | Float | 0.75 | Lowe's ratio test threshold |
| ransac_reproj_threshold | Float | 5.0 | RANSAC reprojection threshold (pixels) |
| min_matches | Integer | 10 | Minimum matches required |

### Tile Refinement Parameters

After global SIFT alignment, tile-based local SIFT refines registration to handle quadrant-level misalignments.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| tile_refinement | Boolean | true | Enable/disable tile-based refinement |
| tile_size | Integer | 200 | Tile size in pixels |
| tile_overlap | Float | 0.3 | Overlap fraction between tiles |
| tile_n_features | Integer | 1000 | SIFT features per tile |
| tile_min_matches | Integer | 5 | Minimum matches for valid tile |

**Tuning guidance:**
- **tile_size**: Larger = faster, captures broader distortions. Smaller = more local correction but needs more features
- **tile_overlap**: Higher = smoother blending between tiles (uses RBF interpolation)
- Set `tile_refinement: false` to disable and use global SIFT only

**Example:**
```hjson
"lowres_to_hires_registration": {
    // Global SIFT
    "n_features": 5000,
    "ratio_threshold": 0.75,

    // Tile refinement
    "tile_refinement": true,
    "tile_size": 200,
    "tile_overlap": 0.3
}
```

---

## 2P-to-HCR Registration Parameters

Location: `params.twop_to_hcr_registration`

### erosion

**Type:** Integer
**Default:** 2
**Range:** 0-10

**Effect:** Number of pixels to remove from image edges after TPS warping.

**Tuning guidance:**
- Increase if seeing edge artifacts in QA overlays
- Decrease if losing too much valid data at edges
- Higher values for images with more distortion at edges

| Scenario | Recommended |
|----------|-------------|
| Normal | 2 |
| Heavy edge artifacts | 4-6 |
| Minimal distortion | 0-1 |

---

### rotation_range

**Type:** List [min, max]
**Default:** [-1, 1]
**Unit:** Degrees

**Effect:** Search range for rotation during global alignment.

**Tuning guidance:**
- Wider range if images are significantly rotated
- Narrow range speeds up processing if rotation is known to be small

| Scenario | Recommended |
|----------|-------------|
| Well-aligned data | [-1, 1] |
| Unknown orientation | [-5, 5] |
| Significant rotation | [-10, 10] |

---

### rotation_step

**Type:** Float
**Default:** 0.5
**Unit:** Degrees

**Effect:** Step size for rotation search grid.

**Tuning guidance:**
- Smaller = more precise but slower
- Larger = faster but may miss optimal rotation

| Scenario | Recommended |
|----------|-------------|
| Fast processing | 1.0 |
| High precision | 0.25 |
| Normal | 0.5 |

---

### z_range_global

**Type:** List [min, max]
**Default:** [-20, 20]
**Unit:** Z-slices

**Effect:** Search range for Z-offset during global alignment.

**Tuning guidance:**
- Increase if Z-alignment consistently off
- Narrow range if Z-alignment is known to be close

| Scenario | Recommended |
|----------|-------------|
| Normal | [-20, 20] |
| Large Z-offset | [-50, 50] |
| Known Z-alignment | [-5, 5] |

---

### tile_sizes

**Type:** List of integers
**Default:** [150, 75]
**Unit:** Pixels

**Effect:** Tile dimensions for local refinement pyramid. Multiple sizes = coarse-to-fine refinement.

**Tuning guidance:**
- Larger tiles = faster, less local correction
- Smaller tiles = slower, more precise local correction
- Use multiple sizes for progressive refinement

| Scenario | Recommended |
|----------|-------------|
| Fast processing | [200] |
| Normal | [150, 75] |
| High precision | [150, 100, 50] |
| Small cells | [100, 50, 25] |

---

### tile_overlap

**Type:** Float
**Default:** 0.30
**Range:** 0.1-0.5

**Effect:** Fraction of overlap between adjacent tiles.

**Tuning guidance:**
- Higher overlap = smoother transitions, more computation
- Lower overlap = faster, may have visible tile boundaries

| Scenario | Recommended |
|----------|-------------|
| Normal | 0.30 |
| Smooth result | 0.40-0.50 |
| Fast | 0.20 |

---

### tile_xy_max

**Type:** Integer
**Default:** 20
**Unit:** Pixels

**Effect:** Maximum allowed shift per tile during local refinement.

**Tuning guidance:**
- Higher = allows more local correction
- Lower = prevents overcorrection artifacts
- Too high can cause unrealistic warping

| Scenario | Recommended |
|----------|-------------|
| Normal | 20 |
| Conservative | 10 |
| Large local distortion | 30-40 |

---

## Auto-Stitching Parameters

Location: `params.stitching` (was `params.auto_stitch_params`)

### overlap_fraction

**Type:** Float
**Default:** 0.50
**Range:** 0.2-0.7

**Effect:** Expected overlap between adjacent tiles.

**Tuning guidance:**
- Must match actual acquisition overlap
- Incorrect values cause stitching failures

---

### noise_floor

**Type:** Float
**Default:** 15.0

**Effect:** Threshold for signal detection in stitching.

**Tuning guidance:**
- Increase for noisy images
- Decrease for low-signal images

---

## Cellpose Parameters

### HCR_cellpose

Location: `params.HCR_cellpose`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model_path | String | Required | Path to Cellpose model |
| diameter | Integer | 15 | Expected cell diameter in pixels |
| gpu | Boolean | true | Use GPU acceleration |
| cellpose_channel | String | Required | Channel for segmentation (e.g., "DAPI") |

### 2p_cellpose

Location: `params.2p_cellpose`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model_path | String | Required | Path to Cellpose model |
| diameter | Integer | 7 | Expected cell diameter (usually smaller for 2P) |
| gpu | Boolean | true | Use GPU acceleration |

---

## Scenario-Based Configurations

### Conservative (High Quality)

```hjson
"twop_to_hcr_registration": {
  "erosion": 3,
  "rotation_range": [-2, 2],
  "rotation_step": 0.25,
  "z_range_global": [-30, 30],
  "tile_sizes": [150, 100, 50],
  "tile_overlap": 0.40,
  "tile_xy_max": 15
}
```

### Fast Processing

```hjson
"twop_to_hcr_registration": {
  "erosion": 2,
  "rotation_range": [-1, 1],
  "rotation_step": 1.0,
  "z_range_global": [-10, 10],
  "tile_sizes": [200],
  "tile_overlap": 0.20,
  "tile_xy_max": 20
}
```

### Small Cells

```hjson
"twop_to_hcr_registration": {
  "erosion": 2,
  "rotation_range": [-1, 1],
  "rotation_step": 0.5,
  "z_range_global": [-20, 20],
  "tile_sizes": [100, 50, 25],
  "tile_overlap": 0.35,
  "tile_xy_max": 10
}
```

### Large Local Distortion

```hjson
"twop_to_hcr_registration": {
  "erosion": 4,
  "rotation_range": [-3, 3],
  "rotation_step": 0.5,
  "z_range_global": [-40, 40],
  "tile_sizes": [150, 75],
  "tile_overlap": 0.40,
  "tile_xy_max": 35
}
```

## Parameter Interaction Effects

| Parameter A | Parameter B | Interaction |
|-------------|-------------|-------------|
| tile_sizes | tile_xy_max | Smaller tiles need smaller max shift |
| erosion | tile_sizes | Higher erosion may reduce effective tile area |
| rotation_range | rotation_step | Wide range + large step = coarse search |

## When to Adjust Parameters

1. **After viewing QA overlays**: Most adjustment decisions come from visual inspection
2. **After registration failures**: Error messages often indicate which parameter to adjust
3. **For new experimental setups**: New microscopes/protocols may need parameter re-tuning
4. **Performance optimization**: After quality is acceptable, tune for speed
