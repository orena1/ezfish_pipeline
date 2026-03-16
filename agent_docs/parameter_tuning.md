# Parameter Tuning Guide

## 2P-to-HCR Registration (`params.twop_to_hcr_registration`)

Uses a 5-tier progressive refinement: Global search → Double affine (2 RANSAC passes) → Double local tiles (300px + 100px with RBF smoothing). See [flexible_alignment_strategy.md](flexible_alignment_strategy.md) for algorithm details.

### Commonly Tuned Parameters

| Parameter | Default | Description | Tuning |
|-----------|---------|-------------|--------|
| `erosion` | 2 | Per-cell 2P mask shrink before alignment (pixels). Includes cell-cell boundary separation. | Increase (3-4) for edge artifacts; decrease (1) if losing small cells |
| `erosion_hcr` | 0 | Per-cell HCR mask shrink (pixels). 0 = no erosion. | Try 1 if HCR cells bleed into neighbors |
| `z_range_global` | [-15, 15] | Z-offset search range (slices) | Widen ([-25, 25]) for large Z offset |
| `xy_max_global` | 500 | XY search range (pixels) | Should cover max expected offset after TPS |
| `hull_margin` | -100 | Inward hull erosion for global search (px, negative = inward) | Less negative (-50) if few landmarks / poor TPS coverage |

### Occasionally Tuned Parameters

| Parameter | Default | Description | Tuning |
|-----------|---------|-------------|--------|
| `fov_crop_margin` | 150 | Margin around 2P FOV for tight crop (px) | Increase if cells near edges are lost |
| `quad_blend_dist` | 300 | Z-map: quad-to-plane blend distance (px) | Rarely needs changing |

### Tier 1-4 Matching Parameters (Rarely Tuned)

Per-cell FFT-IoU matching uses four progressively tighter tiers. **Patch:search ratio must be >= 1.5:1** (below this, matches become unreliable).

| Parameter | Default | Used After | Patch Size | Search Range |
|-----------|---------|-----------|------------|-------------|
| `patch_radius_coarse` | 120 | Global search | 240px | +/-150px |
| `search_xy_coarse` | 150 | | | |
| `search_z_coarse` | 9 | | | +/-9 slices |
| `patch_radius_tight` | 60 | Affine pass 1 | 120px | +/-60px |
| `search_xy_tight` | 60 | | | |
| `search_z_tight` | 4 | | | +/-4 slices |
| `patch_radius_fine` | 30 | Affine pass 2 | 60px | +/-30px |
| `search_xy_fine` | 30 | | | |
| `search_z_fine` | 2 | | | +/-2 slices |
| `patch_radius_ultrafine` | 25 | Local 300px tiles | 50px | +/-8px |
| `search_xy_ultrafine` | 8 | | | |
| `search_z_ultrafine` | 1 | | | +/-1 slice |

### Quality Thresholds (Rarely Tuned)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_iou_global` | 0.06 | Min per-cell IoU for affine RANSAC |
| `min_gain_global` | 0.005 | Min IoU gain for affine RANSAC |
| `min_iou_local` | 0.10 | Min per-cell IoU for local tile RANSAC |
| `min_gain_local` | 0.01 | Min IoU gain for local tile RANSAC |
| `ransac_residual` | 10.0 | RANSAC inlier threshold (px) |
| `min_peak_ratio` | 1.02 | Min FFT peak confidence |
| `fov_dilation` | 10 | Per-cell FOV dilation for FFT-IoU |

### Tile Parameters (Rarely Tuned)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tile_size_coarse` | 300 | First local tile pass (px) |
| `tile_size_fine` | 100 | Second local tile pass (px) |
| `tile_overlap` | 0.5 | Tile overlap fraction |
| `min_cells_per_tile` | 8 | Min cells for 300px tiles |
| `min_cells_per_tile_fine` | 4 | Min cells for 100px tiles |
| `rbf_smoothing_coarse` | 50 | RBF regularization for 300px tiles |
| `rbf_smoothing_fine` | 100 | RBF regularization for 100px tiles |

### Scenario Configs

**Large Z offset**: `z_range_global: [-25, 25]`

**Large XY offset after TPS**: `xy_max_global: 600`

**Few landmarks / poor coverage**: `hull_margin: -50`

**Very sparse cells (<200)**: `min_cells_per_tile: 4, min_cells_per_tile_fine: 2`

**Over-erosion removing small cells**: `erosion: 1`

---

## Auto-Stitching (`params.auto_stitch_params`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `overlap_fraction` | 0.50 | Expected tile overlap — must match acquisition |
| `noise_floor` | 15.0 | Signal threshold — increase for noisy images |
| `min_signal_frac` | 0.01 | Min fraction of pixels with signal |
| `upsample_factor` | 10 | Sub-pixel registration precision |

---

## Cellpose

**HCR** (`params.HCR_cellpose`): `model_path`, `diameter` (typ. 10-20), `gpu`, `cellpose_channel`

**2P** (`params.2p_cellpose`): `model_path`, `diameter` (typ. 5-10), `gpu`

---

## When to Adjust

1. After viewing QA overlays (most common trigger)
2. After registration failures (error messages indicate which parameter)
3. For new experimental setups (different microscopes/protocols)
