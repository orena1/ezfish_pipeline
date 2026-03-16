# CLAUDE.MD - ezfish_pipeline

> **MODEL**: Always use the latest Claude Opus with extended thinking enabled.

---

## Critical Warnings

### NEVER Use NotebookEdit on Jupyter Notebooks
Corrupts notebook JSON by converting cell `source` arrays to strings. Use `perl` with JSON module instead.
See [agent_docs/notebook_editing.md](agent_docs/notebook_editing.md) for safe editing instructions.

### NEVER Edit HJSON with JSON Parsers
HJSON files (`manifest.hjson`) have unquoted keys, comments, trailing commas. Standard `json.loads()`/`json.dumps()` destroys them. The `Edit` tool (text replacement) is safe for HJSON.

---

## Path Conventions (Linux vs Windows)

| Context | Format | Example |
|---------|--------|---------|
| **Jupyter notebooks / manifests** (run on Linux) | Linux | `/mnt/nasquatch/data/2p/jonna/...` |
| **Claude Code file operations** (Windows) | Windows | `Z:\2p\jonna\Code_Python\...` |

**Rules**: Manifests and notebooks use Linux paths (they execute on Linux). Claude Code file operations use Windows paths.

---

## Working Principles

1. **Ask before large changes** - Don't implement significant refactors without discussing first
2. **Consider all execution modes** - Full pipeline, HCR-only, hi-res stitched, standard
3. **Read before modifying** - Trace data flow through affected functions
4. **Don't assume parameters are wrong** - Ask first
5. **No breaking changes to file formats** without discussion

---

## Project Purpose

Automated image processing pipeline for **2-photon microscopy (2P)** and **HCR confocal imaging** of mouse brain experiments.

### Pipeline Stages
| Stage | Purpose |
|-------|---------|
| Per-Plane Processing | Unwarp, stitch, segment each plane independently |
| Cross-Plane Registration | 2P-to-HCR alignment, landmark-based transforms |
| Mask Alignment & Merging | Match masks, generate feature tables |

### Execution Modes
| Mode | How |
|------|-----|
| Full Pipeline | `python master_pipeline.py manifest.hjson` |
| HCR Only | `python master_pipeline.py manifest.hjson --only_hcr` |
| Hi-Res Stitched | `use_automated_stitching: true` in manifest |
| Standard | `use_automated_stitching: false` in manifest |

---

## Code Conventions

### Manifest-Driven Architecture
```python
def process_something(full_manifest: dict, session: dict):
    data = full_manifest['data']
    params = full_manifest['params']

# Config extraction with defaults
reg_params = params.get('twop_to_hcr_registration', {})
EROSION = reg_params.get('erosion', 2)
```

### Path Handling
Use `pathlib.Path` consistently:
```python
output_dir = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT'
```

### Output & Errors
- `rprint()` for user-facing messages; `tqdm` for progress bars
- Minimize repetitive console messages
- Raise specific exceptions with file paths; fail fast over silent corruption

---

## Re-Entry Behavior

Pipeline skips steps when output files exist. Delete specific outputs to re-run steps, or the entire OUTPUT folder for a fresh run. See [agent_docs/resume_and_reentry.md](agent_docs/resume_and_reentry.md).

---

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
| `src/analysis_utils.py` | Analysis notebook utilities |

### HCR-HCR Registration Notebook
Single consolidated notebook: `src/processing_notebooks/HCR_rounds/HCR_registration_consolidated.ipynb`

---

## Detailed Documentation

| Topic | File |
|-------|------|
| Architecture & Modes | [agent_docs/pipeline_architecture.md](agent_docs/pipeline_architecture.md) |
| Registration Workflows | [agent_docs/registration_guide.md](agent_docs/registration_guide.md) |
| Parameter Reference | [agent_docs/parameter_tuning.md](agent_docs/parameter_tuning.md) |
| Auto-Stitching | [agent_docs/auto_stitching.md](agent_docs/auto_stitching.md) |
| Segmentation | [agent_docs/segmentation.md](agent_docs/segmentation.md) |
| File Naming & Output | [agent_docs/file_conventions.md](agent_docs/file_conventions.md) |
| Resume & Re-Entry | [agent_docs/resume_and_reentry.md](agent_docs/resume_and_reentry.md) |
| Troubleshooting | [agent_docs/troubleshooting.md](agent_docs/troubleshooting.md) |
| Flexible Alignment Strategy | [agent_docs/flexible_alignment_strategy.md](agent_docs/flexible_alignment_strategy.md) |
| **Figure Notebooks** | [FIGURE_NOTEBOOKS.md](FIGURE_NOTEBOOKS.md) |

---

## Environment Constraints (Windows)

**Does NOT work**: `python`/`python3`/`py` (Windows Store stubs), `node`/`npm`, `NotebookEdit` tool

**Does work**: `Read`/`Edit`/`Glob`/`Grep` tools, `perl` (for JSON/notebook manipulation), `ls`/`cat`/`grep` via Bash

---

*Last updated: 2026-02-11*
