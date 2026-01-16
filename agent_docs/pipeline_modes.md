# Pipeline Modes & Branching

## Overview

The pipeline supports multiple execution modes to handle different experimental setups. Understanding which mode applies to your data is critical for successful processing.

## Mode Decision Tree

```
Start
  │
  ├─ Do you have 2P data?
  │   │
  │   ├─ NO → Use --only_hcr flag
  │   │
  │   └─ YES → Continue
  │       │
  │       ├─ Do you have multiple tiles that need stitching?
  │       │   │
  │       │   ├─ YES → Set use_automated_stitching: true (Hi-Res Stitched mode)
  │       │   │
  │       │   └─ NO → Set use_automated_stitching: false (Standard mode)
  │       │
  │       └─ Continue with full pipeline
```

## Mode Details

### Full Pipeline (Default)

**When to use:**
- You have both 2P and HCR data
- You want complete processing from raw data to feature tables

**Command:**
```bash
python master_pipeline.py examples/your_manifest.hjson
```

**What runs:**
1. All per-plane processing (2P + HCR)
2. All registrations
3. Mask alignment and merging

**Files generated:**
- `OUTPUT/2P/` - All 2P outputs
- `OUTPUT/HCR/` - All HCR outputs
- `OUTPUT/MERGED/` - Final feature tables

---

### HCR-Only Mode (`--only_hcr`)

**When to use:**
- No 2P data available
- Re-processing only HCR steps after fixing issues
- Testing HCR registration parameters

**Command:**
```bash
python master_pipeline.py examples/your_manifest.hjson --only_hcr
```

**What runs:**
1. HCR round registration
2. HCR segmentation
3. HCR mask matching (no 2P alignment)

**Files generated:**
- `OUTPUT/HCR/` - HCR outputs only
- No 2P-related outputs

**Important:**
- 2P-related parameters are ignored
- Some tables will be incomplete (no 2P cell matches)

---

### Hi-Res Stitched Workflow

**When to use:**
- Multiple tile acquisitions per plane
- Using automated or manual tile stitching

**Manifest setting:**
```hjson
"params": {
  "use_automated_stitching": true,
  "auto_stitch_params": {
    "overlap_fraction": 0.50,
    "noise_floor": 15.0
  }
}
```

**What changes:**
- Expects tile files: `hires_{color}_run{X}_tile{Y}.tiff`
- Runs `auto_stitch_tiles()` to combine tiles
- Generates stitched outputs: `hires_stitched_plane{X}.tiff`
- Uses stitched images for downstream registration

**Files expected:**
```
OUTPUT/2P/tile/
├── hires_green_run001_tile0.tiff
├── hires_green_run001_tile1.tiff
├── hires_red_run001_tile0.tiff
└── ...
```

**Files generated:**
```
OUTPUT/2P/tile/stitched/
├── hires_stitched_plane0.tiff
├── hires_stitched_plane0_rotated.tiff
└── ...
```

---

### Standard Workflow (No Stitching)

**When to use:**
- Single FOV acquisition
- No tiling needed

**Manifest setting:**
```hjson
"params": {
  "use_automated_stitching": false
}
```

**What changes:**
- Expects single image per plane
- Skips all stitching steps
- Uses low-res images directly

**Files expected:**
```
OUTPUT/2P/
├── lowres_meanImg_C0_plane0.tiff
└── ...
```

## File Expectations by Mode

| File Type | Full + Hi-Res | Full + Standard | HCR-Only |
|-----------|---------------|-----------------|----------|
| Tile TIFFs | Required | Not used | Not used |
| Stitched TIFFs | Generated | Not generated | Not used |
| Low-res 2P | Generated | Generated | Not used |
| HCR rounds | Required | Required | Required |
| 2P segmentation | Generated | Generated | Not generated |
| HCR segmentation | Generated | Generated | Generated |
| 2P→HCR landmarks | Required | Required | Not used |
| Feature tables | Complete | Complete | Partial |

## Mode-Specific Pitfalls

### HCR-Only Mode Pitfalls

1. **Don't expect 2P outputs**: Pipeline explicitly skips all 2P processing
2. **Tables are incomplete**: No `twop_plane*` columns in merged tables
3. **Landmarks ignored**: 2P→HCR landmarks not used

### Hi-Res Stitched Pitfalls

1. **Tile naming matters**: Must follow `hires_{color}_run{X}_tile{Y}.tiff` pattern
2. **Overlap fraction critical**: Incorrect value causes stitching failures
3. **Memory intensive**: Large stitched images require significant RAM

### Standard Workflow Pitfalls

1. **Don't set stitching params**: They're ignored but may cause confusion
2. **File naming different**: Uses `lowres_meanImg_*` not stitched files

## Switching Between Modes

### From Full to HCR-Only

Simply add `--only_hcr` flag. Existing 2P outputs are preserved but not used.

### From HCR-Only to Full

Remove `--only_hcr` flag. Pipeline will process 2P steps that were skipped.

### From Standard to Hi-Res

1. Set `use_automated_stitching: true`
2. Provide tile files
3. Delete existing low-res outputs if they should be replaced
4. Re-run pipeline

## Troubleshooting Mode Issues

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| "File not found" for tiles | Wrong mode selected | Check `use_automated_stitching` setting |
| Empty 2P columns in tables | Running in HCR-only mode | Remove `--only_hcr` flag |
| Stitching runs unexpectedly | `use_automated_stitching: true` when not needed | Set to `false` |
| Missing landmark files | HCR-only mode doesn't need them | Only required for full pipeline |
