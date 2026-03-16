# File Naming Conventions

## Directory Structure

```
{base_path}/{mouse_name}/OUTPUT/
├── 2P/
│   ├── cellpose/           # 2P segmentation
│   ├── tile/stitched/      # Stitched images
│   ├── lowres_to_hires/    # Low-res to hi-res QA
│   └── registered/         # 2P-HCR registration + QA/
├── HCR/
│   ├── full_registered_stacks/  # Registered rounds
│   └── cellpose/           # HCR segmentation
└── MERGED/
    └── aligned_extracted_features/  # Feature tables
```

## File Patterns

### 2P Files
| Pattern | Created By |
|---------|-----------|
| `lowres_meanImg_C{ch}_plane{N}.tiff` | `functional.py` |
| `hires_{color}_run{X}_tile{Y}.tiff` | External acquisition |
| `hires_stitched_plane{N}_rotated.tiff` | `auto_stitching.py` |
| `lowres_meanImg_C{ch}_plane{N}_seg_rotated.tiff` | `segmentation.py` |
| `lowres_plane{N}_masks_in_hires_space.tiff` | `registrations.py` |

### Registration Files
| Pattern | Created By |
|---------|-----------|
| `twop_plane{N}_aligned_3d.tiff` | `registrations.py` — 3D labeled volume (uint16) in HCR space |
| `twop_plane{N}_registration_params.npz` | `registrations.py` — see fields below |
| `plane{N}_TO_HCR{R}_landmarks.csv` | User (BigWarp) |
| `plane{N}_TO_HCR{R}_landmarks_auto.csv` | `registrations.py` — auto-refined landmarks |
| `plane{N}_{BEFORE|AFTER}_registration_overlay.tiff` | `registrations.py` (QA) |

#### Registration Params NPZ Fields (`twop_plane{N}_registration_params.npz`)

| Field | Type | Description |
|-------|------|-------------|
| `global_theta` | float | Global rotation (always 0.0 in v8) |
| `global_dy`, `global_dx` | float | Global Y/X shift (HCR pixels) |
| `global_dz` | float | Global Z shift (HCR slices) |
| `cumulative_dy`, `cumulative_dx` | 2D array | Composed per-pixel Y/X shift fields (affine + tiles) |
| `cumulative_dz` | 2D array | Composed per-pixel Z shift field |
| `z_map_local` | 2D array | Final Z-coordinate map |
| `iou_baseline` | float | IoU before any refinement |
| `iou_global` | float | IoU after global search |
| `iou_local` | float | Final IoU after all refinement |
| `erosion` | int | Erosion parameter used |
| `hcr_shape` | 1D array | Shape of HCR reference volume |
| `algorithm_version` | int | Algorithm version (8) |
| `iou_affine_pass1` | float | IoU after first affine pass |
| `iou_affine_composed` | float | IoU after composed affine |
| `iou_local_300` | float | IoU after 300px local tiles |
| `iou_fine_100` | float | IoU after 100px fine tiles |
| `crop_offsets` | 1D array | [y0, x0, y1, x1] bounding box used during alignment |

### HCR Files
| Pattern | Created By |
|---------|-----------|
| `HCR{N}_to_HCR{R}.tiff` | `registrations.py` |
| `HCR{N}_to_HCR{R}_masks.tiff` | `segmentation.py` |

### Merged Files
| Pattern | Created By |
|---------|-----------|
| `full_table_{feature}_twop_plane{N}.pkl` | `segmentation.py` |

## Checking Pipeline State

| File Exists | Step Complete |
|-------------|---------------|
| `lowres_meanImg_*.tiff` | Suite2p extraction |
| `hires_stitched_*.tiff` | Tile stitching |
| `*_seg_rotated.tiff` | 2P segmentation |
| `HCR*_to_HCR*.tiff` | HCR registration |
| `*_masks.tiff` | HCR segmentation |
| `twop_*_aligned_3d.tiff` | 2P-HCR registration |
| `full_table_*.pkl` | Feature extraction |

## Version Suffixes
- `_auto` — automatically generated (vs manual)
- `_rotated` — rotation applied for alignment
