# Auto-Stitching Guide

## Algorithm

1. **Shift Detection**: Phase correlation (FFT) on overlap regions of adjacent tiles
2. **Global Optimization**: Least-squares optimization of tile positions from pairwise shifts
3. **Blending**: Place tiles at computed positions with linear feathering in overlaps

**Key function**: `auto_stitch_tiles()` in `src/auto_stitching.py`

## Parameters

```hjson
"stitching": {
    "overlap_fraction": 0.50,  // Must match actual acquisition overlap
    "noise_floor": 15.0        // Signal threshold; increase for noisy images
}
```

## Output

```
OUTPUT/2P/tile/
├── shift_report.json           # Computed shifts and quality scores
└── stitched/
    ├── hires_stitched_plane0.tiff
    └── hires_stitched_plane0_rotated.tiff
```

## Failure Modes

| Issue | Cause | Fix |
|-------|-------|-----|
| Tiles misaligned | Wrong `overlap_fraction` | Verify matches acquisition settings |
| Poor shift detection | Low signal in overlap | Adjust `noise_floor`; try manual stitching |
| Wrong shift (repetitive patterns) | Phase correlation ambiguity | Use landmarks or manual stitching |
| Memory crash during blending | Too many/large tiles | Process planes separately |
| Visible seams | Insufficient feathering | Increase overlap or feathering region |

## Fallback

If auto-stitching fails:
1. Stitch externally (e.g., Fiji/ImageJ)
2. Save as `hires_stitched_plane{X}.tiff`
3. Set `use_automated_stitching: false`
4. Re-run pipeline

## QA Checks

Verify: no visible seams, consistent brightness, correct feature alignment across boundaries, no missing overlap regions.
