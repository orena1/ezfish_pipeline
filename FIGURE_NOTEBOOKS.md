# FIGURE_NOTEBOOKS.md - Publication Figure Generation

> Guidelines for creating Neuron NeuroResource publication-quality figures.

## Working Principles (For Claude)

### Mindset Shift from Pipeline Code
Figure notebooks have **different priorities** than pipeline code:
- **Aesthetics over robustness**: Visual polish matters; edge cases don't
- **Clarity over brevity**: Explicit is better than clever
- **Narrative-driven**: Every panel tells part of a story

### Before Creating a Panel
1. **Understand the figure's purpose** - What claim does this panel support?
2. **Know the data source** - Where does the input data come from in the pipeline?
3. **Check existing panels** - Maintain visual consistency across the paper

### Quality Standards
1. **Reproducible**: Notebook runs top-to-bottom without manual intervention
2. **Self-contained**: All paths and parameters defined within the notebook
3. **Export-ready**: Outputs are vector PDFs suitable for Illustrator
4. **Documented**: Markdown cells explain each major step

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
├── figure_4/
├── supplementary/          # Supp figures
└── shared/                 # Optional: style utilities
    └── figure_style.py
```

### Naming Conventions
- Notebooks: `panel_X.ipynb` or `panel_XY.ipynb` for combined panels
- Outputs: `figure_N_panel_X.pdf` (e.g., `figure_2_panel_a.pdf`)

---

## Notebook Structure

Every figure notebook should follow this organization:

```python
# Cell 1: Imports and Setup
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
# ... other imports

# Cell 2: Style Configuration (ALL visual params here)
FONTSIZE_TITLE = 12
FONTSIZE_LABEL = 10
FONTSIZE_TICK = 8
LINEWIDTH_AXIS = 1.0
LINEWIDTH_DATA = 1.5
FIGURE_DPI = 300

# Colors
COLOR_PRIMARY = '#2E86AB'
COLOR_SECONDARY = '#A23B72'

# Cell 3: Paths
DATA_DIR = Path(r'Z:\...\OUTPUT')
OUTPUT_DIR = Path('./outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

# Cell 4+: Data Loading
# Cell 5+: Processing (if needed)
# Cell 6+: Plotting
# Final Cell: Export
```

### Cell Guidelines
- **One logical operation per cell** - easier to debug and re-run
- **Markdown headers** - Use `## Section Name` to organize
- **No hardcoded paths in plotting cells** - Define all paths at top

---

## Visual Standards

### Typography
| Element | Font | Size | Weight |
|---------|------|------|--------|
| Figure title | Arial | 12pt | Bold |
| Axis labels | Arial | 10pt | Regular |
| Tick labels | Arial | 8pt | Regular |
| Legend | Arial | 8pt | Regular |
| Scale bar text | Arial | 8pt | Regular |

### Colors
Use colorblind-friendly palettes. Suggested options:
- **Sequential**: `viridis`, `plasma`, `cividis`
- **Diverging**: `RdBu`, `coolwarm`
- **Categorical**: Define explicit hex codes for consistency

### Line Weights
| Element | Width |
|---------|-------|
| Axis spines | 1.0 pt |
| Data lines | 1.5 pt |
| Grid lines | 0.5 pt |
| Annotations | 0.75 pt |

### Microscopy Images
- **Always include scale bar** - white or black depending on background
- **Consistent orientation** - anterior up, lateral right (or document if different)
- **Channel colors**: Document what each color represents

---

## Export Requirements

### Primary Format: PDF
```python
def save_figure(fig, name, output_dir=OUTPUT_DIR):
    """Save figure as PDF and SVG for Illustrator compatibility."""
    pdf_path = output_dir / f'{name}.pdf'
    svg_path = output_dir / f'{name}.svg'

    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    print(f'Saved: {pdf_path}')
```

### Figure Sizing
Set explicit figure size in inches based on journal column width:
- **Single column**: ~3.5 inches wide
- **1.5 column**: ~5 inches wide
- **Full width**: ~7 inches wide

```python
fig, ax = plt.subplots(figsize=(3.5, 3))  # Single column, square-ish
```

### Naming Convention
`figure_N_panel_X.pdf` where:
- N = figure number (1, 2, 3, 4, S1, S2, ...)
- X = panel letter (a, b, c, ...)

---

## Code Patterns

### Standard Plotting Setup
```python
# Apply publication style
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.linewidth': 1.0,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,  # For notebook display
})
```

### Image with Scale Bar
```python
def add_scale_bar(ax, pixel_size_um, bar_length_um=50, color='white'):
    """Add scale bar to microscopy image."""
    bar_length_px = bar_length_um / pixel_size_um
    # Position in bottom-right
    x = ax.get_xlim()[1] - bar_length_px - 20
    y = ax.get_ylim()[0] - 20  # Note: imshow has inverted y
    ax.plot([x, x + bar_length_px], [y, y], color=color, linewidth=2)
    ax.text(x + bar_length_px/2, y - 10, f'{bar_length_um} µm',
            ha='center', va='top', color=color, fontsize=8)
```

### Multi-Panel Figure
```python
fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))
# ... plot each panel
fig.tight_layout()
save_figure(fig, 'figure_2_panel_abc')
```

---

## What NOT to Do
- Don't leave default matplotlib styling (gray background, etc.)
- Don't use rainbow colormaps (`jet`, `hsv`) - not colorblind safe
- Don't hardcode paths mid-notebook - all paths at top
- Don't skip scale bars on microscopy images
- Don't forget to set explicit figure dimensions

---

*For pipeline code guidelines, see [CLAUDE.md](CLAUDE.md)*
