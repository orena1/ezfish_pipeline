# CLAUDE.MD - ezfish_pipeline

> **CRITICAL**: Never ignore this file. Read it at the start of every session.

## Working Principles (For Claude)

### Before Making Changes
1. **Ask before large changes** - Don't implement significant refactors or new features without discussing first
2. **Understand the full picture** - Read related code before modifying; trace data flow through affected functions
3. **Consider all execution modes** - Changes must work for: full pipeline, HCR-only, hi-res stitched, standard

### Code Quality Standards
1. **Concise over clever** - Simple, readable solutions beat elegant complexity
2. **Robust over fast** - Handle edge cases; fail gracefully with clear errors
3. **User experience first** - Minimize console noise; provide actionable feedback when errors occur
4. **No orphaned code** - If you add something, make sure it's actually used

### Self-Validation
1. **Test your ideas** - When proposing changes, write a small test or check that validates the approach
2. **Check before/after** - For registration/alignment changes, compare outputs visually or numerically
3. **Trace through scenarios** - Mentally (or actually) run through: normal case, edge case, error case

### What NOT to Do
- Don't add features that weren't requested
- Don't refactor working code "while you're there"
- Don't add verbose logging/comments unless asked
- Don't assume parameters are wrong - ask first
- Don't make breaking changes to file formats without discussion

---

## Project Purpose

Automated image processing pipeline for **2-photon microscopy (2P)** and **HCR confocal imaging** of mouse brain experiments. Integrates:
- **2P imaging**: Functional/structural GCaMP recordings
- **HCR imaging**: Molecular in situ hybridization for gene expression

## Current Architecture (Evolving)

The pipeline currently uses a multi-stage design. Each stage exists for specific dependency reasons:

| Stage | Purpose | Why It Exists |
|-------|---------|---------------|
| Per-Plane Processing | Unwarp, stitch, segment each plane independently | No cross-plane dependencies; enables parallelization |
| Cross-Plane Registration | 2P-to-HCR alignment, landmark-based transforms | Requires completed per-plane outputs |
| Mask Alignment & Merging | Match masks, generate feature tables | Requires all registrations complete |

This architecture is **not fixed**. Document changes when modifying stage boundaries.

## Execution Modes

| Mode | Command | Use When |
|------|---------|----------|
| Full Pipeline | `python master_pipeline.py manifest.hjson` | Complete 2P + HCR processing |
| HCR Only | `python master_pipeline.py manifest.hjson --only_hcr` | No 2P data, or re-running HCR steps |
| Hi-Res Stitched | Set `use_automated_stitching: true` in manifest | Multiple tile acquisitions needing stitching |
| Standard | Set `use_automated_stitching: false` | Single FOV, no stitching needed |

See [agent_docs/pipeline_modes.md](agent_docs/pipeline_modes.md) for detailed mode branching.

## Quick Reference

```bash
# Run full pipeline
python master_pipeline.py examples/your_manifest.hjson

# Run HCR-only
python master_pipeline.py examples/your_manifest.hjson --only_hcr

# Common location for manifests
examples/
```

## Code Conventions

### Manifest-Driven Architecture
All functions receive `full_manifest` dict:
```python
def process_something(full_manifest: dict, session: dict):
    data = full_manifest['data']
    params = full_manifest['params']
```

### Configuration Extraction (with defaults)
```python
reg_params = params.get('twop_to_hcr_registration', {})
EROSION = reg_params.get('erosion', 2)  # Always provide sensible default
```

### Path Handling
Use `pathlib.Path` consistently:
```python
output_dir = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT'
```

### Output Verbosity
- Minimize repetitive console messages
- Use `rprint()` for user-facing messages with color
- Progress bars (`tqdm`) for long operations
- Skip "already exists" messages unless debugging

### Error Handling
- Raise specific exceptions with file paths in messages
- Fail fast with diagnostics rather than silent corruption
- Use fallbacks when automation fails (e.g., auto-stitch → manual)

## Key Parameters (Frequently Adjusted)

| Parameter | Location | Impact |
|-----------|----------|--------|
| `erosion` | `twop_to_hcr_registration` | Edge artifact removal; increase for noisy boundaries |
| `tile_sizes` | `twop_to_hcr_registration` | Local refinement granularity; smaller = finer but slower |
| `rotation_range` | `twop_to_hcr_registration` | Search range for rotation alignment |
| `z_range_global` | `twop_to_hcr_registration` | Z-offset search range |
| `overlap_fraction` | `auto_stitch_params` | Tile overlap for stitching; typically 0.3-0.5 |
| `diameter` | `HCR_cellpose`, `2p_cellpose` | Expected cell diameter in pixels |

See [agent_docs/parameter_tuning.md](agent_docs/parameter_tuning.md) for complete reference.

## Re-Entry & Resume Behavior

The pipeline detects completed steps by checking output file existence:
- **Skip existing**: If output file exists, step is skipped
- **Re-do a file**: Delete the specific output file, then re-run
- **Fresh re-run**: Delete the entire OUTPUT folder

**Important**: Mid-run exits may leave partial state. Check [agent_docs/resume_and_reentry.md](agent_docs/resume_and_reentry.md) for safe re-entry patterns.

## Common Pain Points

| Issue | Quick Solution |
|-------|---------------|
| Multi-mode confusion | Check mode decision tree in [pipeline_modes.md](agent_docs/pipeline_modes.md) |
| Stage dependency errors | Ensure previous stage outputs exist before running |
| Registration quality issues | Adjust erosion, tile_sizes; check QA overlays |
| Performance bottlenecks | Use tile-based processing; check mask matching optimization |
| Parameter sensitivity | Start conservative, adjust based on QA overlays |

## Documentation Maintenance

### When to Update CLAUDE.MD
- New execution modes added
- Core architecture changes
- New critical parameters
- Recurring pain points identified

### When to Update agent_docs/
- Detailed workflow changes
- New failure modes discovered
- Parameter tuning scenarios
- Troubleshooting additions

### Maintenance Protocol
1. After significant changes, review affected docs
2. Keep CLAUDE.MD under 300 lines (delegate details to agent_docs/)
3. Update troubleshooting when resolving new issues
4. Document *why* not just *what* for architecture decisions

## Detailed Documentation

| Topic | File |
|-------|------|
| Architecture & Stage Rationale | [agent_docs/pipeline_architecture.md](agent_docs/pipeline_architecture.md) |
| Execution Modes & Branching | [agent_docs/pipeline_modes.md](agent_docs/pipeline_modes.md) |
| Registration Workflows | [agent_docs/registration_guide.md](agent_docs/registration_guide.md) |
| Parameter Reference | [agent_docs/parameter_tuning.md](agent_docs/parameter_tuning.md) |
| Auto-Stitching | [agent_docs/auto_stitching.md](agent_docs/auto_stitching.md) |
| Segmentation | [agent_docs/segmentation.md](agent_docs/segmentation.md) |
| File Naming & Output | [agent_docs/file_conventions.md](agent_docs/file_conventions.md) |
| Resume & Re-Entry | [agent_docs/resume_and_reentry.md](agent_docs/resume_and_reentry.md) |
| Troubleshooting | [agent_docs/troubleshooting.md](agent_docs/troubleshooting.md) |
| **Figure Notebooks** | [FIGURE_NOTEBOOKS.md](FIGURE_NOTEBOOKS.md) |

## Key Source Files

| File | Purpose |
|------|---------|
| `master_pipeline.py` | Main orchestrator |
| `src/meta.py` | Manifest parsing & validation |
| `src/registrations.py` | Registration orchestration |
| `src/registrations_utils.py` | Registration algorithms |
| `src/segmentation.py` | Cellpose & mask operations |
| `src/tiling.py` | Tile processing & stitching |
| `src/auto_stitching.py` | Automated stitching |
| `src/functional.py` | Suite2p extraction |

## HCR-HCR Registration Notebook

The HCR round-to-round registration workflow uses a **single consolidated notebook**:
- `src/processing_notebooks/HCR_rounds/HCR_registration_consolidated.ipynb`

This replaces the previous separate low-res and hi-res notebooks (`1_scan_lowres_parameters.ipynb`, `2_scan_highres_parameters.ipynb`). The consolidated notebook handles both resolution modes in a unified workflow.

---

*Last updated: 2026-01-08*
