# Registration Guide

## Overview

The pipeline performs multiple types of image registration. Each type serves a **different purpose** and operates on **different image pairs**:

| Registration Type | Source | Target | Modality | Config |
|-------------------|--------|--------|----------|--------|
| **HCR-to-HCR** | HCR round N | HCR round 1 | Same (confocal) | Always automatic |
| **Low-Res to High-Res** | 2P low-res | 2P high-res stitched | Same (2P only) | `automation.lowres_to_hires` |
| **2P-to-HCR** | 2P masks | HCR reference | Cross-modality | `automation.twop_to_hcr` |

**Important:** These are distinct registration steps that happen at different stages of the pipeline.

**Quick config reference:**
```hjson
"automation": {
    "lowres_to_hires": "manual",  // "manual" (landmarks) or "auto"
    "twop_to_hcr": "manual"       // "manual" or "auto"
}
```

**Key distinction:**
| Registration | Landmarks Required? | Automation Behavior |
|--------------|---------------------|---------------------|
| **Low-Res to High-Res** | Only if `"manual"` | `"auto"`: No landmarks needed, uses SIFT feature matching |
| **2P-to-HCR** | **Always** (for TPS warp) | `"auto"`: Refines landmarks automatically, generates per-plane `_auto.csv` files |

---

## Registration Types

### 1. HCR Round-to-Round Registration

**Purpose:** Align multiple HCR confocal imaging rounds to a reference round.

**Scope:** HCR images only (confocal-to-confocal alignment).

**Method:** BigStream feature point matching + deformable B-spline registration.

**Key function:** `register_rounds()` in `src/registrations.py`

**When it runs:** Automatically during HCR processing stage.

**Parameters:**
```hjson
"HCR_to_HCR_params": {
  "red_mut_x": 3,
  "red_mut_y": 3,
  "red_mut_z": 2
}
```

**Output files:**
```
OUTPUT/HCR/full_registered_stacks/
├── HCR2_to_HCR1.tiff
├── HCR3_to_HCR1.tiff
└── ...
```

---

### 2. Low-Res to High-Res Registration (2P Internal)

**Purpose:** Align low-resolution 2P mean images to high-resolution 2P stitched images.

**Scope:** 2P images only (2P-to-2P alignment within the same modality).

**Why needed:** Low-res functional recordings need to be mapped to high-res structural images before cross-modality registration to HCR.

**Pipeline configuration:**

```hjson
"automation": {
    "lowres_to_hires": "manual"  // or "auto"
}
```

| Config Value | Method | Landmarks Required? | Description |
|--------------|--------|---------------------|-------------|
| `"manual"` (default) | TPS + manual landmarks | **Yes** - user creates in BigWarp | Most control, requires manual work. Pipeline prompts if missing. |
| `"auto"` | SIFT + RANSAC + tile refinement | **No** - fully automated | Uses feature matching. No user input needed. Best for most cases. |

**Key functions:**

| Function | File | Used When |
|----------|------|-----------|
| `register_lowres_to_hires_landmarks()` | `src/registrations_landmarks.py` | `lowres_to_hires: "landmarks"` |
| `register_lowres_to_hires()` | `src/registrations.py` | `lowres_to_hires: "auto"` |
| `register_lowres_to_hires_single_plane()` | `src/registrations_utils.py` | Core algorithm (called by auto) |

**Direct usage (for testing/custom scripts):**

```python
from src.registrations_utils import register_lowres_to_hires_single_plane

# Both images are 2P (same modality, different resolutions)
params, aligned = register_lowres_to_hires_single_plane(
    lowres_2d,   # 2P low-res mean image
    hires_2d,    # 2P high-res stitched image
    method='sift'  # or 'orb', 'phase_correlation'
)
```

**Method options for `register_lowres_to_hires_single_plane()`:**

| Method | Description | When to Use |
|--------|-------------|-------------|
| `'sift'` | SIFT feature matching + RANSAC affine | **Recommended** for non-trivial warping |
| `'orb'` | ORB feature matching + RANSAC affine | Faster alternative, may be less robust |
| `'phase_correlation'` | Phase correlation + affine optimization | Legacy method, simpler distortions only |

**Output format:**
```python
{
    'scale_x': float,      # X scaling factor
    'scale_y': float,      # Y scaling factor
    'rotation': float,     # Rotation in degrees
    'shear': float,        # Shear component
    'shift_x': float,      # X translation (pixels)
    'shift_y': float,      # Y translation (pixels)
    'similarity': float,   # NCC score (0-1)
    'method': str,         # 'sift', 'orb', or 'phase_correlation'
    'n_inliers': int       # (SIFT/ORB only) RANSAC inlier count
}
```

#### Tile-Based Refinement (Auto Mode)

After global SIFT alignment, the pipeline applies **per-tile local corrections** to handle non-uniform distortions. This was extensively tested in `tests/lowres_hires_registration_test.ipynb`.

**Key findings from testing (2026-01):**

| Warp Method | NCC vs Baseline | Notes |
|-------------|-----------------|-------|
| **Direct Per-Tile** | **+2.1%** | Best method - apply each tile's correction directly |
| Local RBF | +1.2% | Moderate improvement |
| Feathered Blending | -9% | Worse than baseline |
| Griddata Interpolation | -6% to -15% | Significantly worse |

**Critical change: CLAHE preprocessing**

The single most important factor for successful tile-based registration is **CLAHE (Contrast Limited Adaptive Histogram Equalization)**. Without CLAHE, SIFT struggles to find features in tiles with uneven illumination or low contrast cells.

| With CLAHE | Without CLAHE |
|------------|---------------|
| **23.9 avg inliers** | 17.3 avg inliers |
| +38% more feature matches | Baseline |
| Reliable tile corrections | Many tiles fail with "no_features" |

**Other tuned parameters (secondary importance):**

| Parameter | Best Value | Notes |
|-----------|------------|-------|
| `ratio_threshold` | **0.8** | More permissive matching (vs 0.7 default) |
| `ransac_reproj_threshold` | **5.0** | More permissive RANSAC (vs 3.0 default) |
| `min_matches` | **3** | Allow fewer matches per tile |
| `min_feature_size` | **8** | Filter noise, keep cell-like features |
| `max_spatial_distance` | **0** | No spatial limit performs best |
| `blend_width` | **10** | Blending region at tile boundaries |
| `tile_size_lowres` | **300** | Optimal tile size (in low-res pixels) |

**Implementation details:**

1. **CLAHE enhancement** (critical): `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))` - enables feature detection in low-contrast regions
2. **Percentile normalization**: Uses 1-99th percentile (robust to hot pixels)
3. **Direct tile application**: Each tile's affine is applied to its region only (no interpolation)
4. **NCC quality gating**: Reject corrections that worsen alignment
5. **Feathered blending**: Soft blend at tile boundaries

**QA Output files:**
```
OUTPUT/2P/lowres_to_hires/
├── plane0_BEFORE_overlay.tiff      # Hires vs upsampled lowres (no alignment)
├── plane0_AFTER_global_overlay.tiff # After global SIFT alignment
├── plane0_AFTER_tiling_overlay.tiff # After tile-based refinement
└── lowres_plane0_masks_in_hires_space.tiff
```

---

### 3. 2P-to-HCR Registration (Cross-Modality)

**Purpose:** Align 2P masks to HCR reference space for cell matching.

**Scope:** Cross-modality alignment (2P to HCR confocal). This is the final registration step that enables matching cells between modalities.

**Landmarks:** **Always required** for the initial TPS warp. Unlike low-res to high-res registration, this cross-modality alignment cannot be fully automated.

**Pipeline configuration:**

```hjson
"automation": {
    "twop_to_hcr": "manual"  // or "auto"
}
```

| Config Value | Behavior | Landmarks |
|--------------|----------|-----------|
| `"manual"` (default) | Manual mode - prompts user if landmarks missing | User creates initial landmarks in BigWarp |
| `"auto"` | Auto mode - skips if no landmarks, refines if landmarks exist | Uses existing landmarks, saves refined `_auto.csv` for each plane |

**What automation does (when `"auto"`):**
1. Runs mask-mask alignment for each plane
2. Generates **per-plane refined landmarks** (`plane0_TO_HCR1_auto.csv`, `plane1_TO_HCR1_auto.csv`, etc.)
3. User can edit `_auto.csv` files in BigWarp to fine-tune specific planes
4. Subsequent runs will use `_auto.csv` files automatically

**Method:** Multi-stage approach:
1. **TPS Warp**: Apply landmark-based distortion correction
2. **Erosion**: Remove edge artifacts
3. **Global Alignment**: Coarse rotation + XY + Z search via FFT
4. **Local Refinement**: Tile-based shift optimization

**Key function:** `twop_to_hcr_registration()` in `src/registrations.py`

**Parameters:**
```hjson
"twop_to_hcr_registration": {
  "erosion": 2,
  "rotation_range": [-1, 1],
  "rotation_step": 0.5,
  "z_range_global": [-20, 20],
  "tile_sizes": [150, 75],
  "tile_overlap": 0.30,
  "tile_xy_max": 20
}
```

**Output files:**
```
OUTPUT/2P/registered/
├── twop_plane0_aligned_3d.tiff
├── plane0_TO_HCR1_auto.csv          # Per-plane refined landmarks
├── plane1_TO_HCR1_auto.csv
├── QA/
│   ├── plane0_BEFORE_registration_overlay.tiff
│   └── plane0_AFTER_registration_overlay.tiff
└── ...
```

## Landmark Workflow

### Creating Landmarks (BigWarp)

1. Open BigWarp with 2P image as "moving" and HCR as "fixed"
2. Select corresponding features in both images
3. Export landmarks as CSV

**Best practices:**
- Use 10-20 landmarks (not just 3 minimum)
- Cover entire FOV, not just center
- Choose distinctive features: blood vessels, cell clusters, boundaries
- Avoid areas with poor signal

### Landmark CSV Format

```csv
,x,y
0,123.45,678.90
1,234.56,789.01
...
```

### Auto-Refined Landmarks

After running 2P-to-HCR registration with automation enabled, refined landmarks are saved:
```
plane0_TO_HCR1_landmarks_auto.csv
```

These can be used for subsequent runs instead of manual landmarks.

## Registration Stage Details

### Stage 1: TPS Warp

Applies thin-plate spline transformation using landmarks to correct non-linear distortions.

**Why needed:** 2P and HCR have different optical distortions.

### Stage 2: Erosion

Removes `erosion` pixels from image edges.

**Why needed:** Edge artifacts from TPS warping can degrade alignment quality.

**Tuning:**
- Increase if seeing edge artifacts in QA overlays
- Decrease if losing too much valid data

### Stage 3: Global Alignment

FFT-based search for:
- Rotation within `rotation_range`
- XY translation
- Z-offset within `z_range_global`

**Why needed:** Coarse alignment before fine-tuning.

### Stage 4: Local Tile-Based Refinement

Divides image into tiles and optimizes shift per tile, then interpolates smooth shift field.

**Why needed:** Local non-linear distortions not captured by global alignment.

**Parameters:**
- `tile_sizes`: Tile dimensions for refinement pyramid (larger → coarser)
- `tile_overlap`: Fraction of overlap between tiles
- `tile_xy_max`: Maximum allowed shift per tile

## QA Overlays

The pipeline generates before/after overlay images for visual QA:
- **Red channel**: HCR reference
- **Green channel**: 2P aligned

**Location:** `OUTPUT/2P/registered/QA/`

**What to check:**
- Cell alignment (should overlap in "after")
- Edge artifacts (should be minimal)
- Global orientation (no rotation errors)

## Automation Checkpoints

When `automation.twop_to_hcr: true`, the pipeline:
1. Runs automated registration
2. Generates QA overlays
3. Prompts user to accept/refine/skip
4. Saves refined landmarks if accepted

This enables interactive refinement during batch processing.

## Troubleshooting Registration

| Issue | Likely Cause | Solution |
|-------|--------------|----------|
| Poor global alignment | Rotation range too narrow | Increase `rotation_range` |
| Edge artifacts | Erosion too low | Increase `erosion` parameter |
| Local misalignment | Tile size too large | Use smaller `tile_sizes` |
| Registration fails | Missing landmarks | Create/check landmark CSV |
| Memory errors | Images too large | Process smaller regions |

See [troubleshooting.md](troubleshooting.md) for more detailed solutions.

---

## Post-TPS Local Refinement: History of Failed Attempts

> **Status:** ABANDONED (2026-01-12)
>
> Multiple approaches were systematically tested to improve 2P-to-HCR mask alignment beyond TPS landmark-based registration. **None improved IoU.** The TPS warp (~21% IoU) appears to be the ceiling for this modality pairing.

### Problem Statement

After TPS landmark-based registration, the 2P masks and HCR masks have ~21% IoU. The goal was to apply local refinement to improve sub-cell alignment, under the hypothesis that there were micro-scale (1-10px) warps that varied locally.

### What Was Tried

#### Test Notebooks

| Notebook | Approach | Result |
|----------|----------|--------|
| `tests/twop_to_hcr_registration_test_lowres_v4.ipynb` | Global transform from aggregated feature matches | Zero IoU improvement |
| `tests/twop_to_hcr_registration_test_lowres_v5.ipynb` | Local tile-based feature matching (75-150px tiles) | Zero IoU improvement |
| `tests/twop_to_hcr_registration_test_lowres_v6.ipynb` | Comprehensive multi-strategy sweep | All methods failed or hurt IoU |

#### v6 Comprehensive Sweep (Final Attempt)

**Intensity-Based Methods:**
- HCR channels tested: DAPI, GCAMP
- Tile sizes: 20, 40, 60, 80, 100 px
- Preprocessing: raw, CLAHE, local_norm, edges_sobel, log, adaptive_thresh, tophat
- Feature detection: blob_log, blob_doh, blob_dog, intensity_peaks, harris, shi_tomasi, SIFT, ORB, AKAZE, BRISK
- Matching: nearest neighbor, mutual nearest, ratio test
- **Result:** No configuration improved IoU

**Mask-Based Methods (Binary IoU Optimization):**
- Tile sizes: 20, 30, 40, 50 px
- Max shifts: 3, 5, 7, 10 px (highly constrained)
- Min coverage: 0.02, 0.03, 0.05
- **Result:** ALL configurations DECREASED IoU (best was -3.5%, worst was -5%)

**Optical Flow:**
- Farneback dense flow
- **Result:** No improvement

**Phase Correlation:**
- Per-tile sub-pixel refinement
- **Result:** No improvement

### Why It Doesn't Work

The consistent failure across all methods suggests:

1. **TPS is already optimal:** The landmark-based registration is as good as possible given the data
2. **Fundamental mismatch:** 2P and HCR see different cell populations or morphologies
3. **Z-ambiguity:** The z-surface sampling introduces enough uncertainty that pixel-level alignment isn't meaningful
4. **Not a warp problem:** The residual error isn't due to geometric distortion but to actual differences in what each modality captures

### Implications for Pipeline

1. **Accept current alignment:** ~21% IoU is the practical limit
2. **Use centroid-based matching:** Instead of mask overlap, match cells by centroid proximity
3. **Manual landmarks:** If better alignment is needed for specific regions, add more TPS landmarks manually
4. **Don't waste time on local refinement:** Future attempts at automated sub-cell alignment are unlikely to succeed

### Key Files for Reference

- Plan document: `C:\Users\jsinghal\.claude\plans\modular-watching-swing.md`
- v6 notebook: `tests/twop_to_hcr_registration_test_lowres_v6.ipynb`
- Helper functions cell added for function ordering fixes
