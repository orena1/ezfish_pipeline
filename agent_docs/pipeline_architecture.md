# Pipeline Architecture

## Overview

The ezfish_pipeline processes 2-photon (2P) and HCR confocal imaging data through multiple stages. This document explains the **current** architecture and the **rationale** for each stage's existence.

## Current Stage Layout

### Stage 1: Per-Plane Processing

**What it does:**
- Process high-res tile SBX files (if available)
- Unwarp tiles using lens distortion calibration
- Stitch tiles together
- Extract registered Suite2p planes from functional imaging
- HCR round-to-round registration
- Cellpose segmentation on HCR rounds
- Cellpose segmentation on 2P images

**Why this stage exists:**
- Each plane can be processed independently → enables parallelization
- No cross-plane dependencies at this stage
- Segmentation must complete before masks can be aligned

**Key functions:**
- `process_plane()` in `master_pipeline.py`
- `stitch_tiles_and_rotate()` in `src/tiling.py`
- `run_cellpose()`, `run_cellpose_2p()` in `src/segmentation.py`
- `register_rounds()` in `src/registrations.py`

### Stage 2: Cross-Plane Registration

**What it does:**
- Low-res to high-res landmark registration (hybrid approach)
  - Reference plane: uses manual landmarks (TPS transformation)
  - Other planes: automated rigid alignment
- 2P to HCR registration per plane
  - Global alignment (rotation + XY + Z search)
  - Local tile-based refinement
  - QA overlay generation

**Why this stage exists:**
- Requires per-plane outputs from Stage 1
- Cross-modality alignment needs both 2P and HCR images processed
- Reference plane landmarks inform other plane alignments

**Key functions:**
- `register_lowres_to_hires()` in `src/registrations.py`
- `twop_to_hcr_registration()` in `src/registrations.py`
- `global_alignment()`, `compute_tile_shifts()` in `src/registrations_utils.py`

### Stage 3: Mask Alignment & Merging

**What it does:**
- Align 2P masks to HCR space
- Match HCR round masks to reference
- Merge all alignments into final feature tables

**Why this stage exists:**
- Requires all registrations to be complete
- Feature tables need aligned masks from all rounds/planes
- Final output generation depends on all upstream data

**Key functions:**
- `align_masks()` in `src/segmentation.py`
- `merge_masks()` in `src/segmentation.py`
- `match_masks()` in `src/segmentation.py`

## Data Flow Diagram

```
2P SBX Files                    HCR TIFF Files
     │                               │
     ▼                               ▼
[tiling.py]                    [registrations.py]
SBX → TIFF                     Register rounds
Unwarp, Stitch                 to reference
     │                               │
     ▼                               ▼
[functional.py]                [segmentation.py]
Extract Suite2p                Cellpose on HCR
mean images                          │
     │                               │
     ▼                               ▼
[segmentation.py]              Registered HCR
Cellpose on 2P                 round stacks
     │                               │
     └───────────┬───────────────────┘
                 ▼
         [registrations.py]
         2P → HCR registration
         (landmarks + automated)
                 │
                 ▼
         [segmentation.py]
         Align masks
         Match masks (IoU)
         Merge to tables
                 │
                 ▼
         Feature Tables
         (per-plane .pkl files)
```

## Dependency Graph

```
                    ┌─────────────────┐
                    │  Manifest Load  │
                    │    (meta.py)    │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
    ┌─────────────────┐           ┌─────────────────┐
    │   2P Processing │           │  HCR Processing │
    │   (tiling.py)   │           │ (registrations) │
    └────────┬────────┘           └────────┬────────┘
             │                             │
             ▼                             ▼
    ┌─────────────────┐           ┌─────────────────┐
    │ 2P Segmentation │           │ HCR Segmentation│
    │ (segmentation)  │           │ (segmentation)  │
    └────────┬────────┘           └────────┬────────┘
             │                             │
             └──────────────┬──────────────┘
                            ▼
                  ┌─────────────────┐
                  │ 2P → HCR Reg    │
                  │ (registrations) │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Mask Alignment │
                  │  & Merging      │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Feature Tables  │
                  └─────────────────┘
```

## Design Rationale

### Why Multiple Stages?

1. **Dependency ordering**: Cannot align masks until registration is complete
2. **Failure isolation**: If one stage fails, earlier outputs are preserved
3. **Resume capability**: Can restart from any stage if outputs exist
4. **Memory management**: Process one stage at a time to limit memory usage

### Why Per-Plane First?

1. Each plane is independent until cross-plane registration
2. Enables future parallelization across planes
3. Simplifies debugging (isolate issues to specific planes)

### Why Hybrid Landmark Approach?

1. Reference plane needs high precision → manual landmarks
2. Other planes can use reference landmarks + automated refinement
3. Reduces manual effort while maintaining quality

## Future Considerations

When modifying architecture, consider:
- Does the change break dependency ordering?
- Can the change be isolated to one stage?
- Does it affect resume behavior?
- Will it impact memory usage patterns?

Document any architectural changes in this file with rationale.
