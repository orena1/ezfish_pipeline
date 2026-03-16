# Registration Guide

## Registration Types

| Type | Source → Target | Method | Config |
|------|----------------|--------|--------|
| **HCR-to-HCR** | HCR round N → HCR round 1 | BigStream (automatic) | Always runs |
| **Low-Res to High-Res** | 2P low-res → 2P hi-res | SIFT or landmarks | `automation.lowres_to_hires` |
| **2P-to-HCR** | 2P masks → HCR space | TPS + global + local | `automation.twop_to_hcr` |

```hjson
"automation": {
    "lowres_to_hires": "auto",   // "manual" (landmarks) or "auto" (SIFT)
    "twop_to_hcr": "manual"      // "manual" or "auto"
}
```

---

## 1. HCR Round-to-Round

BigStream feature matching + B-spline registration. Fully automatic.

**Function**: `register_rounds()` in `src/registrations.py`
**Output**: `OUTPUT/HCR/full_registered_stacks/HCR{N}_to_HCR1.tiff`

---

## 2. Low-Res to High-Res (2P Internal)

Aligns low-res functional recordings to hi-res structural images.

| Config | Method | Landmarks? |
|--------|--------|-----------|
| `"auto"` | SIFT + RANSAC + tile refinement | No |
| `"manual"` | TPS with user landmarks | Yes (BigWarp) |

**Key functions**: `register_lowres_to_hires()`, `register_lowres_to_hires_single_plane()`

**Auto mode** uses CLAHE preprocessing + direct per-tile correction (no interpolation between tiles). Per-tile application outperforms all interpolation methods by ~7-26%.

**QA outputs**:
```
OUTPUT/2P/lowres_to_hires/
├── plane0_BEFORE_overlay.tiff
├── plane0_AFTER_global_overlay.tiff
└── plane0_AFTER_tiling_overlay.tiff
```

---

## 3. 2P-to-HCR (Cross-Modality)

Multi-stage alignment of 2P masks into HCR reference space using a 5-tier progressive refinement strategy (v8 algorithm).

**Landmarks always required** for the initial TPS warp (cross-modality can't be fully automated).

| Config | Behavior |
|--------|----------|
| `"manual"` | Prompts if landmarks missing |
| `"auto"` | Skips if no landmarks; refines existing → saves `_auto.csv` per plane |

**5-Tier Strategy**:
1. **TPS Warp** — landmark-based non-rigid correction
2. **Erosion** — per-cell shrink to remove edge artifacts
3. **Global Search** — moving-mask IoU with inward convex hull (XY + Z, no rotation)
4. **Double Affine** — two RANSAC passes with shrinking feature matching (captures residual tilt/shear)
5. **Double Local Tiles** — 300px then 100px tile-based RANSAC with RBF smoothing + border anchoring

**Key insight**: Global search handles bulk translation but cannot correct tilt. The double affine captures spatially-varying tilt residuals through iteratively improving match quality (coarse → tight patches). Local tiles handle per-region deformation the affine can't capture.

**Key function**: `twop_to_hcr_registration()` in `src/registrations.py`

**Output**:
```
OUTPUT/2P/registered/
├── twop_plane0_aligned_3d.tiff
├── twop_plane0_registration_params.npz
├── plane0_TO_HCR1_auto.csv
└── QA/plane0_{BEFORE|AFTER}_registration_overlay.tiff
```

**Registration params file** (`_registration_params.npz`) contains:
- `global_theta` (always 0.0), `global_dy/dx/dz` — global transform
- `cumulative_dy/dx/dz` — composed local shift fields (affine + tiles)
- `z_map_local` — final Z-coordinate map
- `iou_baseline/global/local` — IoU at baseline/global/final stages
- `algorithm_version` — version identifier (8)
- `iou_affine_pass1`, `iou_affine_composed`, `iou_local_300`, `iou_fine_100` — per-stage IoU
- `crop_offsets` — bounding box used during alignment

---

## Landmark Workflow

### Creating Landmarks (BigWarp)
1. Open BigWarp: 2P as "moving", HCR as "fixed"
2. Select 10-20 corresponding features covering entire FOV
3. Choose distinctive features: blood vessels, cell clusters, boundaries
4. Export as CSV

### Auto-Refined Landmarks
After running with `twop_to_hcr: true`, refined landmarks are saved as `plane{N}_TO_HCR1_auto.csv`. These encode the global translation in the landmark coordinates — using them with TPS gives you TPS+global in one step. (No rotation is applied; v8 does not use rotation search.)

---

## QA Overlays

Generated at `OUTPUT/2P/registered/QA/`:
- **Red channel**: HCR reference
- **Green channel**: 2P aligned

**Check for**: cell overlap, edge artifacts, systematic shifts.

---

## Troubleshooting

| Issue | Likely Cause | Fix |
|-------|-------------|-----|
| Poor global alignment | XY/Z search range too narrow | Widen `xy_max_global` or `z_range_global` |
| Edge artifacts | Erosion too low | Increase `erosion` |
| Affine worsens IoU | Patch:search ratio below 1.5:1 | Increase patch radii or decrease search radii |
| Local tiles no improvement | Cells too sparse | Reduce `min_cells_per_tile` |
| Registration fails | Missing landmarks | Create/check landmark CSV |
| TPS fails (singular matrix) | <4 landmarks or collinear | Add more, distribute across FOV |
| Residuals hitting search boundary | Search range too tight | Widen search range (maintain 1.5:1 ratio) |
