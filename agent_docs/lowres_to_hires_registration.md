# Low-Res to Hi-Res 2P Registration Pipeline

## Overview

This document describes the optimal configuration for registering low-resolution 2P images to high-resolution 2P images of the same field of view. The pipeline was developed and validated using the notebook `tests/lowres_hires_registration_test.ipynb`.

**Key insight**: Distortions between low-res and hi-res 2P images are **local and independent** - they do not follow a smooth spatial pattern. This means interpolation-based methods (RBF, triangulation) hurt alignment quality. Direct per-tile correction is optimal.

---

## Pipeline Stages

### Stage 1: Global Alignment (SIFT)

Corrects gross translation, rotation, and scale differences.

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Method** | SIFT + FLANN + RANSAC | Scale-invariant feature matching |
| **n_features** | 500 | Maximum keypoints to detect |
| **ratio_threshold** | 0.75 | Lowe's ratio test (lower = stricter) |
| **ransac_reproj_threshold** | 5.0 | Pixels tolerance for inlier detection |
| **Transform** | Affine (2x3) | Applied via `cv2.warpAffine()` |

### Stage 2: Tile-Based Local Refinement

Corrects local distortions that vary across the image.

#### Tile Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Tile size** | 300px in low-res space | ~600px in hi-res after scaling |
| **Overlap** | 30% | Balance between coverage and speed |
| **Min overlap fraction** | 50% | Skip edge tiles with insufficient content |

#### Feature Detection (Per-Tile SIFT)

| Parameter | Value | Notes |
|-----------|-------|-------|
| **n_features** | 100 | Per tile (smaller region needs fewer) |
| **ratio_threshold** | 0.7 | Slightly stricter for tiles |
| **ransac_reproj_threshold** | 3.0 | Tighter for local refinement |
| **min_matches** | 4 | Minimum inliers to accept transform |
| **min_feature_size** | 8 | Filter out noise, keep cell-sized features |
| **max_spatial_distance** | 40px | Reject matches too far apart spatially |
| **local_normalize** | CLAHE | Enhances local contrast for feature detection |

#### Quality Gates

| Gate | Value | Purpose |
|------|-------|---------|
| **NCC improvement required** | YES | Reject shifts that worsen alignment |
| **Max shift** | 40px | Reject unreasonably large corrections |
| **Min overlap** | 50% | Skip tiles at edges with different FOV content |

### Stage 3: Warp Field Generation

Combines tile corrections into final warped image.

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Method** | Direct Per-Tile | No interpolation between tiles |
| **blend_width** | 40px | Soft feathering at tile edges |

---

## Why Direct Per-Tile Wins

We tested 5 warp combination methods:

| Rank | Method | Best NCC | vs Baseline |
|------|--------|----------|-------------|
| 1 | **Direct Per-Tile** | 0.7778 | **+7.34%** |
| 2 | Local RBF (Gaussian) | 0.6766 | -6.63% |
| 3 | Feathered Blending | 0.6596 | -8.98% |
| 4 | Piecewise Affine | 0.6188 | -14.61% |
| 5 | Global RBF (TPS) | 0.6166 | -14.91% |

**Key finding**: Local methods outperformed global methods by **26%**.

Interpolation methods fail because:
- Tile A might need +3px shift in X
- Adjacent Tile B might need -2px shift in X
- These aren't points on a smooth curve - they're independent corrections
- Interpolation averages conflicting shifts and gets both tiles wrong

---

## Edge Handling

Near image borders, low-res and hi-res may have different field-of-view content.

| Issue | Solution |
|-------|----------|
| Different FOV at borders | Compute valid overlap mask between images |
| Partial tiles | Skip tiles with <50% valid overlap |
| NCC on partial tiles | Use masked NCC - only compare overlapping regions |

---

## Expected Results

| Stage | NCC | Change |
|-------|-----|--------|
| Raw upsampled | ~0.65 | baseline |
| After global SIFT | 0.7247 | +11.5% |
| After tile refinement | 0.7778 | +7.34% additional |
| **Total improvement** | | **~20%** |

Typical tile success rate: **93.8%** with proper parameter tuning.

---

## Code Configuration

```python
# Global SIFT config
global_config = {
    'n_features': 500,
    'ratio_threshold': 0.75,
    'ransac_reproj_threshold': 5.0,
}

# Tile SIFT config
tile_config = {
    'n_features': 100,
    'ratio_threshold': 0.7,
    'ransac_reproj_threshold': 3.0,
    'min_matches': 4,
    'min_feature_size': 8,
    'max_spatial_distance': 40,
    'local_normalize': 'clahe',
}

# Tile grid config
TILE_SIZE_LOWRES = 300  # pixels in low-res space
OVERLAP_FRACTION = 0.3
MIN_OVERLAP_FRACTION = 0.5

# Quality gates
MAX_SHIFT = 40  # pixels
REQUIRE_NCC_IMPROVEMENT = True

# Warp combination
WARP_METHOD = 'direct_per_tile'
BLEND_WIDTH = 40  # pixels
```

---

## Key Learnings

1. **CLAHE helps feature detection** - enhances local contrast, but must be applied consistently to both images during matching

2. **Cell-size filtering works** - `min_feature_size=8` removes noise while keeping biologically relevant features (cell diameter 5-12px in low-res)

3. **NCC quality gate is essential** - without it, bad tile corrections pollute the result

4. **Don't interpolate** - the distortions aren't smooth; direct per-tile application preserves local corrections without averaging them away

5. **Spatial distance constraint helps** - `max_spatial_distance=40` rejects false matches between distant features

---

## Related Files

- **Development notebook**: `tests/lowres_hires_registration_test.ipynb`
- **Production implementation**: TBD (to be integrated into `src/registrations.py`)

---

*Last updated: 2025-01*
