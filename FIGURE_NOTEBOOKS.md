# FIGURE_NOTEBOOKS.md - Publication Figure Generation

> Guidelines for creating Neuron NeuroResource publication-quality figures.
> For pipeline code guidelines, see [CLAUDE.md](CLAUDE.md).

---

## Working Principles

Figure notebooks have **different priorities** than pipeline code:
- **Aesthetics over robustness**: Visual polish matters; edge cases don't
- **Clarity over brevity**: Explicit is better than clever
- **Narrative-driven**: Every panel tells part of a story

### Before Creating a Panel
1. Understand the figure's purpose — what claim does this panel support?
2. Know the data source — where does the input come from in the pipeline?
3. Check existing panels — maintain visual consistency across the paper

### Quality Standards
1. **Reproducible**: Notebook runs top-to-bottom without manual intervention
2. **Self-contained**: All paths and parameters defined within the notebook
3. **Export-ready**: Outputs are vector PDFs suitable for Illustrator

---

## Directory Structure

```
figure_notebooks/
├── figure_1/
│   ├── panel_a.ipynb
│   ├── panel_bc.ipynb      # Combine related panels
│   └── outputs/            # Generated PDFs
├── figure_2/
├── figure_3/
├── supplementary/
└── shared/
    └── figure_style.py
```

**Naming**: Notebooks: `panel_X.ipynb` | Outputs: `figure_N_panel_X.pdf`

---

## Notebook Structure

```python
# Cell 1: Imports
# Cell 2: Style Configuration (ALL visual params here)
FONTSIZE_TITLE = 12
FONTSIZE_LABEL = 10
FONTSIZE_TICK = 8
COLOR_PRIMARY = '#2E86AB'
COLOR_SECONDARY = '#A23B72'

# Cell 3: Paths (all paths defined at top)
DATA_DIR = Path(r'Z:\...\OUTPUT')
OUTPUT_DIR = Path('./outputs')

# Cell 4+: Data Loading → Processing → Plotting → Export
```

---

## Visual Standards

| Element | Font | Size |
|---------|------|------|
| Figure title | Arial | 12pt Bold |
| Axis labels | Arial | 10pt |
| Tick/Legend/Scale bar | Arial | 8pt |

| Element | Line Width |
|---------|-----------|
| Axis spines | 1.0 pt |
| Data lines | 1.5 pt |
| Grid/Annotations | 0.5-0.75 pt |

- Use colorblind-friendly palettes: `viridis`, `plasma`, `cividis`, `RdBu`
- Always include scale bars on microscopy images

---

## Export

```python
fig.savefig(output_dir / f'{name}.pdf', format='pdf', bbox_inches='tight', dpi=300)
fig.savefig(output_dir / f'{name}.svg', format='svg', bbox_inches='tight')
```

**Figure sizing** (journal column widths):
- Single column: ~3.5" | 1.5 column: ~5" | Full width: ~7"

---

## Standard Plotting Setup

```python
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.linewidth': 1.0,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
})
```

---

## Helper Module

`figure_notebooks/figure_2/notebook_utils.py` provides:
- `load_registration_params()` — load `.npz` registration params
- `load_landmarks_from_manifest()` — landmarks with manifest-based conversion
- `apply_full_transform()` — TPS + global + local transforms
- `get_hcr_resolution_from_manifest()` — HCR resolution from manifest

**Resolution source**: All HCR resolution comes from the manifest. 2P coordinates are already in pixels (no conversion needed).

---

## Transform Chain for Overlays

The pipeline saves registration params to `twop_plane{N}_registration_params.npz`. When creating alignment overlays:

- **Auto landmarks** (`_auto.csv`): TPS already includes global transform → apply local only
- **Manual landmarks**: TPS gives baseline → apply global → apply local
- **DO NOT** use `twop_aligned_3d.tiff` with raw that only has TPS — transforms won't match

See `notebook_utils.py` helper functions for correct transform application.

---

*For pipeline code guidelines, see [CLAUDE.md](CLAUDE.md)*
