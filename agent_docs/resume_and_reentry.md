# Resume & Re-Entry Guide

## Overview

Understanding how to stop, restart, and re-process specific steps is critical for efficient pipeline usage. This guide covers safe re-entry patterns.

## How the Pipeline Detects Completion

The pipeline checks for **output file existence** to determine which steps are complete:

```python
if output_path.exists():
    print(f"Skipping, already exists")
    continue
```

This means:
- Existing file → step skipped
- No file → step runs
- Partial/corrupted file → step skipped (problem!)

## Re-Entry Scenarios

### Scenario 1: Clean Re-Run (Start Fresh)

**Use when:** Want to completely reprocess all data.

**Steps:**
1. Delete the entire OUTPUT folder:
   ```bash
   rm -rf {base_path}/{mouse_name}/OUTPUT/
   ```
2. Run pipeline normally:
   ```bash
   python master_pipeline.py manifest.hjson
   ```

**Caution:** This deletes all intermediate results.

---

### Scenario 2: Re-Do Specific Step

**Use when:** One step needs reprocessing, but others are fine.

**Steps:**
1. Identify output file(s) for that step (see [file_conventions.md](file_conventions.md))
2. Delete only those files
3. Re-run pipeline

**Example - Re-do 2P-HCR registration for plane 0:**
```bash
# Delete registration outputs
rm OUTPUT/2P/registered/twop_plane0_aligned_3d.tiff
rm OUTPUT/2P/registered/QA/plane0_*.tiff
# Re-run
python master_pipeline.py manifest.hjson
```

---

### Scenario 3: Resume After Crash/Interrupt

**Use when:** Pipeline was interrupted mid-run.

**Risk:** Partial files may exist that are incomplete/corrupted.

**Steps:**
1. Check the last completed step (look at console output or file timestamps)
2. Delete any potentially partial files
3. Re-run pipeline

**Warning signs of partial files:**
- Unusually small file size
- Recent modification time during crash
- Errors reading file in downstream steps

---

### Scenario 4: Change Parameters and Re-Run

**Use when:** Adjusting parameters (e.g., erosion, tile_sizes).

**Steps:**
1. Update manifest with new parameters
2. Delete output files that depend on those parameters
3. Re-run pipeline

**Parameter → Files to delete:**

| Parameter Changed | Files to Delete |
|-------------------|-----------------|
| `twop_to_hcr_registration.*` | `twop_*_aligned_3d.tiff`, QA overlays |
| `HCR_cellpose.*` | `*_masks.tiff` in HCR/cellpose |
| `2p_cellpose.*` | `*_seg_rotated.tiff` |
| `auto_stitch_params.*` | `hires_stitched_*.tiff` |

---

### Scenario 5: Add New Plane

**Use when:** Processing additional planes after initial run.

**Steps:**
1. Update manifest to include new plane in `additional_functional_planes`
2. Run pipeline - existing plane outputs are preserved
3. Only new plane is processed

---

### Scenario 6: Switch Modes Mid-Processing

**Use when:** Switching from HCR-only to full pipeline (or vice versa).

**From HCR-only to Full:**
1. Remove `--only_hcr` flag
2. Run pipeline - 2P steps will process, HCR outputs preserved

**From Full to HCR-only:**
1. Add `--only_hcr` flag
2. Run pipeline - 2P outputs preserved but not used/updated

---

## Identifying What Ran

### Check File Timestamps

```bash
ls -lt OUTPUT/2P/registered/
```

Files are created in pipeline order. Recent timestamps show recent processing.

### Check Console Output

Pipeline prints status for each step:
- `"Processing..."` → Step running
- `"Skipping, already exists"` → Step skipped
- `"Error..."` → Step failed

### Check File Existence Table

See [file_conventions.md](file_conventions.md) for which files indicate which steps are complete.

## Safe Deletion Patterns

### Delete One Step's Outputs

```bash
# Registration only
rm OUTPUT/2P/registered/twop_plane*_aligned_3d.tiff
rm -rf OUTPUT/2P/registered/QA/

# Segmentation only (2P)
rm OUTPUT/2P/cellpose/*_seg_rotated.tiff

# Segmentation only (HCR)
rm OUTPUT/HCR/cellpose/*_masks.tiff

# Feature tables only
rm OUTPUT/MERGED/aligned_extracted_features/*.pkl
```

### Delete One Plane's Outputs

```bash
# Everything for plane 0
rm OUTPUT/2P/cellpose/*plane0*
rm OUTPUT/2P/registered/*plane0*
rm OUTPUT/2P/registered/QA/*plane0*
rm OUTPUT/MERGED/aligned_extracted_features/*plane0*
```

## Avoiding State Confusion

### Problem: Mixed Parameter Runs

Running with different parameters without cleaning outputs can leave inconsistent state.

**Solution:**
- Track parameter changes in manifest commits
- When parameters change significantly, do clean re-run
- Or systematically delete affected outputs

### Problem: Interrupted Multi-Plane Processing

If interrupted between planes, some planes complete, others don't.

**Solution:**
- Check which planes have complete outputs
- Delete partial plane outputs
- Re-run (completed planes skip, incomplete re-process)

### Problem: Corrupt Landmark Files

Manual landmark edits or crashed auto-refinement can corrupt CSVs.

**Solution:**
- Keep backups of working landmark files
- Delete `*_auto.csv` if auto-refinement failed
- Re-create manual landmarks if needed

## Automation Checkpoint Re-Entry

When automation checkpoints are enabled, the pipeline may pause for user input.

**If you exit during a checkpoint:**
1. Pipeline state is saved
2. Re-running resumes from checkpoint
3. Or delete checkpoint file to restart that step

**Checkpoint files:** Look for `*_checkpoint.json` or similar in OUTPUT.

## Best Practices

1. **Commit manifests**: Track parameter changes with git
2. **Document re-runs**: Note what was deleted and why
3. **Backup landmarks**: Keep copies of working landmark files
4. **Clean runs for major changes**: Don't mix outputs from different parameter sets
5. **Check file sizes**: Suspiciously small files may be corrupt
