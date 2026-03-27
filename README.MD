# EASI-FISH Pipeline

Automated image processing pipeline for **2-photon (2P) microscopy** and **HCR confocal imaging** of mouse brain experiments. Takes raw imaging data and a manifest file, and produces merged cell-identity tables matching 2P functional recordings to HCR molecular markers.

## Quickstart

1. Copy the appropriate template from `examples/` and fill in your values
2. Organize your data following the [expected folder structure](#folder-structure)
3. Run the pipeline on the cluster:

```bash
# Full pipeline (2P + HCR)
python master_pipeline.py --manifest examples/your_manifest.hjson

# HCR-only mode (no 2P data needed)
python master_pipeline.py --manifest examples/your_manifest.hjson --only_hcr
```

## Choosing a Manifest Template

| Template | `input_format` | When to use |
|----------|---------------|-------------|
| `demo_standard.hjson` | `sbx` | ScanBox 2P + HCR, low-res (Suite2p mean image directly) |
| `demo_hires.hjson` | `sbx` | ScanBox 2P + HCR, hi-res anatomical tiles (stitch + unwarp) |
| `demo_tiff.hjson` | `tiff` | Non-ScanBox microscope — bring your own pre-processed 2P mean images |
| `demo_hcr_only.hjson` | n/a | HCR confocal only, no 2P data (`--only_hcr` flag) |
| `demo.hjson` | `sbx` | General ScanBox template with all options documented |

**Quick guide:**
- **ScanBox user with .sbx files?** Use `demo_standard.hjson` or `demo_hires.hjson`
- **Different microscope, have TIFF mean images?** Use `demo_tiff.hjson`
- **Only HCR confocal, no 2P?** Use `demo_hcr_only.hjson`

## Execution Modes

| Mode | Flag | Template | When to use |
|------|------|----------|-------------|
| **HCR-only** | `--only_hcr` | `demo_hcr_only.hjson` | You only have HCR confocal data, no 2P |
| **Full + hi-res** | (none) | `demo_hires.hjson` | 2P + HCR, with high-res anatomical tile runs (ScanBox) |
| **Full + standard** | (none) | `demo_standard.hjson` | 2P + HCR, no high-res tiles (ScanBox, Suite2p mean image) |
| **Full + TIFF** | (none) | `demo_tiff.hjson` | 2P + HCR, pre-processed TIFF mean images (any microscope) |
| **General template** | (none) | `demo.hjson` | All options documented with comments |

## Pipeline Steps

### Full pipeline (2P + HCR)

```
Phase 1 — HCR processing (runs once for all rounds)
  1. HCR round-to-round registration    All rounds → reference round (BigStream)
  2. HCR cellpose segmentation          3D cell segmentation on registered stacks
  3. HCR probe intensity extraction      Per-cell fluorescence for each probe

Phase 2 — 2P processing (per imaging plane)
  4. Tile processing & stitching         (hi-res mode only) Unwarp + stitch tiles
  5. Suite2p extraction                  Mean images from functional recordings
  6. 2P cellpose segmentation            2D cell segmentation on mean images

Phase 3 — Cross-modal registration (per plane)
  7. Low-res → high-res registration     (hi-res mode only) SIFT or landmark-based
  8. 2P → HCR registration              Landmark-based, with cascade refinement

Phase 4 — Output (per plane)
  9. Mask alignment                      Match cells across HCR rounds + 2P
 10. Mask merging                        Final merged feature tables
```

### HCR-only mode (`--only_hcr`)

```
  1. HCR round-to-round registration
  2. HCR cellpose segmentation
  3. HCR probe intensity extraction
  4. Mask alignment across HCR rounds
  5. Merged HCR feature tables
```

## Manifest File (.hjson)

The manifest is an [HJSON](https://hjson.github.io/) file (JSON with comments and relaxed syntax) that configures everything about your experiment. It has two top-level sections:

### `data` — What you're processing

| Field | Description |
|-------|-------------|
| `base_path` | Root path on the cluster (e.g. `/mnt/nasquatch/data/2p/lab/pipeline`) |
| `mouse_name` | Folder name under base_path for this experiment |
| `two_photon_imaging.sessions` | 2P session config (date, runs, planes). Only 1 session supported. Omit for `--only_hcr` |
| `HCR_confocal_imaging.rounds` | List of HCR rounds, each with round number, channel names, and voxel resolution |
| `HCR_confocal_imaging.reference_round` | Which round all others register to. Pick the one with best signal |

#### 2P session fields

| Field | Required | Description |
|-------|----------|-------------|
| `input_format` | **Yes** | `"sbx"` (ScanBox) or `"tiff"` (pre-processed TIFFs) |
| `date` | Yes | Date string used in folder names: `{mouse}_{date}_{run}` |
| `functional_planes` | Yes | Planes to process. **First = reference plane** for HCR landmark alignment |
| `functional_run` | SBX only | Run number containing the `suite2p/` folder |
| `anatomical_hires_green_runs` | SBX only | Hi-res tile run numbers (green channel). Paired 1:1 with red |
| `anatomical_hires_red_runs` | SBX only | Hi-res tile run numbers (red channel) |
| `anatomical_lowres_green_runs` | SBX only | Low-res anatomical run numbers (alternative to hi-res) |
| `anatomical_lowres_red_runs` | SBX only | Low-res anatomical run numbers |
| `unwarp_config` | SBX only | Distortion correction file (in `{base_path}/Calibration_files_for_unwarping/`) |
| `unwarp_steps` | SBX only | Number of unwarp interpolation steps |
| `has_hires` | TIFF only | Set `true` if providing pre-stitched hi-res images (default: `false`) |

> **SBX mode:** Set either hi-res or low-res anatomical runs, not both. Leave both empty to use only the Suite2p mean image.
>
> **TIFF mode:** Place your pre-processed mean images in the OUTPUT directory before running. See `demo_tiff.hjson` for file placement instructions.

#### HCR round fields

| Field | Description |
|-------|-------------|
| `round` | Round identifier string (e.g. `"01"`, `"02"`) |
| `channels` | List of channel names. DAPI should always be first (used for registration) |
| `resolution` | Voxel size `[x, y, z]` in microns — get from your confocal acquisition metadata |

### `params` — How to process it

| Section | Required | Description |
|---------|----------|-------------|
| `HCR_to_HCR_registration` | Yes | BigStream downsampling factors `[x, y, z]`. Default: `[3, 3, 2]` |
| `rotation_2p_to_HCR` | For 2P | Coarse rotation/flip to align 2P orientation with HCR |
| `HCR_cellpose` | Yes | Cellpose model + parameters for HCR 3D segmentation. **Must include `cellpose_channel`** |
| `2p_cellpose` | For 2P | Cellpose model + parameters for 2P 2D segmentation |
| `intensity_extraction` | Yes | Neuropil settings for probe intensity measurement |
| `automation` | No | Override manual steps with automated alternatives (see below) |
| `HCR_selected_registrations` | Later | Populated during the interactive HCR registration step |

#### Automation options

All default to `"manual"`. Set to `"auto"` to use automated alternatives:

| Option | Manual | Auto |
|--------|--------|------|
| `twop_to_hcr` | User-provided BigWarp landmarks only | Automated refinement (still needs initial landmarks) |
| `lowres_to_hires` | BigWarp landmarks + TPS | SIFT feature matching + RANSAC affine |
| `stitching` | BigStitcher coordinates | SIFT + phase correlation |

## Folder Structure

### SBX mode (`input_format: "sbx"`)

```
{base_path}/
└── {mouse_name}/
    ├── 2P/
    │   ├── {mouse}_{date}_{run}/
    │   │   ├── {mouse}_{date}_{run}.sbx
    │   │   └── suite2p/          (for functional run)
    │   └── ...
    ├── HCR/
    │   ├── {mouse}_HCR01.tif     (reference round)
    │   ├── {mouse}_HCR02.tif     (additional rounds)
    │   └── ...
    └── OUTPUT/                   (created by pipeline)
        ├── HCR/
        │   ├── full_registered_stacks/
        │   ├── cellpose/
        │   ├── registrations/
        │   └── extract_intensities/
        ├── 2P/
        │   ├── cellpose/
        │   ├── suite2p/
        │   ├── tile/             (hi-res mode only)
        │   └── registered/
        └── MERGED/
            ├── aligned_masks/
            └── aligned_extracted_features/
                └── merged_table_plane{N}.csv   ← final output
```

### TIFF mode (`input_format: "tiff"`)

Place your pre-processed mean images **before** running the pipeline:

```
{base_path}/
└── {mouse_name}/
    ├── HCR/
    │   ├── {mouse}_HCR01.tif
    │   └── ...
    └── OUTPUT/
        └── 2P/
            └── cellpose/                        ← YOU CREATE THIS
                ├── lowres_meanImg_C0_plane0.tiff     (green only, 2D float32)
                └── lowres_meanImg_C0_plane1.tiff     (one per plane)
```

For green+red, use `lowres_meanImg_C01_plane{N}.tiff` (CYX, 2 channels) instead.

For hi-res (`has_hires: true`), pre-stitch your tiles and place them at:
```
OUTPUT/2P/tile/stitched/hires_stitched_plane{N}.tiff   (CYX, float32)
```

## Re-Entry / Resume

The pipeline skips steps when output files already exist. To re-run a specific step, delete its output files. To re-run everything, delete the entire `OUTPUT/` folder.

## Key Source Files

| File | Purpose |
|------|---------|
| `master_pipeline.py` | Main orchestrator |
| `src/meta.py` | Manifest parsing and validation |
| `src/registrations.py` | Registration orchestration (HCR-to-HCR, 2P-to-HCR) |
| `src/registrations_utils.py` | Registration algorithms |
| `src/segmentation.py` | Cellpose segmentation, mask alignment, merging |
| `src/tiling.py` | Tile processing and stitching |
| `src/auto_stitching.py` | Automated stitching |
| `src/functional.py` | Suite2p data extraction |

