# Pipeline Architecture & Modes

## Stages

### Stage 1: Per-Plane Processing
- Unwarp/stitch tiles, extract Suite2p planes
- HCR round-to-round registration, Cellpose segmentation (HCR + 2P)
- Each plane independent → parallelizable

### Stage 2: Cross-Plane Registration
- Low-res to high-res alignment (SIFT or landmarks)
- 2P-to-HCR registration (TPS + global + local refinement)
- Requires Stage 1 outputs

### Stage 3: Mask Alignment & Merging
- Align 2P masks to HCR space, match across rounds
- Generate feature tables (`.pkl`)
- Requires all registrations complete

## Data Flow

```
2P SBX Files              HCR TIFF Files
     │                         │
     ▼                         ▼
[tiling.py]              [registrations.py]
Unwarp, Stitch           Register rounds
     │                         │
     ▼                         ▼
[functional.py]          [segmentation.py]
Suite2p extraction       Cellpose on HCR
     │                         │
     ▼                         ▼
[segmentation.py]              │
Cellpose on 2P                 │
     │                         │
     └────────┬────────────────┘
              ▼
      [registrations.py]
      2P → HCR alignment
              │
              ▼
      [segmentation.py]
      Align + match masks
              │
              ▼
      Feature Tables (.pkl)
```

---

## Execution Modes

```
Do you have 2P data?
├─ NO  → python master_pipeline.py manifest.hjson --only_hcr
└─ YES → Do you have multiple tiles?
         ├─ YES → use_automated_stitching: true  (Hi-Res Stitched)
         └─ NO  → use_automated_stitching: false  (Standard)
```

| Mode | What runs | What's generated |
|------|-----------|------------------|
| **Full Pipeline** | All stages | `OUTPUT/2P/`, `OUTPUT/HCR/`, `OUTPUT/MERGED/` |
| **HCR-Only** (`--only_hcr`) | HCR reg + seg only | `OUTPUT/HCR/` only, no 2P outputs |
| **Hi-Res Stitched** | Full + tile stitching | Tile files → stitched images → registration |
| **Standard** | Full, no stitching | Low-res images → registration directly |

### Mode Pitfalls
- **HCR-Only**: No 2P outputs, tables incomplete (no 2P cell matches)
- **Hi-Res Stitched**: Tile naming must follow `hires_{color}_run{X}_tile{Y}.tiff`; `overlap_fraction` must match acquisition
- **Switching modes**: Existing outputs preserved; add/remove `--only_hcr` or toggle `use_automated_stitching`

---

## Design Rationale

1. **Dependency ordering**: Can't align masks until registration complete
2. **Failure isolation**: Earlier outputs preserved if later stage fails
3. **Resume**: Can restart from any stage (checks output file existence)
4. **Memory**: Process one stage at a time
