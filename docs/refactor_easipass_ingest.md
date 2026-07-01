# EASI-PASS ingest-mode refactor — design & plan

> Status: **plan / pre-implementation**. Branch `feature/easipass-ingest-refactor` off merged
> main (baseline PR #33). No code changes yet. This doc is the contract to review before step 1.
> Tied to the manuscript Methods section — the final code must match what the paper describes.

## 1. Motivation

The functional "input formats" (`sbx` default / `suite2p` / `tiff`) are a muddle:

- `sbx` and `suite2p` **both read Suite2p-registered output** (`ops['meanImg']`). `sbx`
  additionally does red-channel combine + hi-res SBX tile stitching, **and is the only path
  that extracts activity traces** (`master_pipeline.py:121` excludes `tiff`/`suite2p`).
- `tiff` gets **no traces** (`master_pipeline.py:121`). (It *can* use hi-res — a pre-stitched
  `plane_{N}_hires.tiff` already triggers `has_hires` + low→hi-res reg; it just can't stitch raw
  tiles, which is by design.)
- The default `_get_input_format` → `'sbx'` reads as "SBX unless proven TIFF", which is
  backwards for a tool that ships to many labs.

Net: today `suite2p` and `tiff` inputs get mean-image + masks + matching but **no activity
traces at all**. Raw-`.sbx` residue is actually narrow and structural, not a real "format".

## 2. Target design

### 2a. User-facing modes (3)

| Mode | Input | Functional output | In paper? |
|---|---|---|---|
| **Video / full** (default) | a registered **movie** (multipage TIFF *or* Suite2p) | masks → **computed traces** + full cross-modal + molecular join | ✅ |
| **Stills / image-align** (today's `--tiff_only`) | a single static 2P **image** (no movie) | align to HCR/ex-vivo volume + molecular join, **no traces** | ✅ (as a capability) |
| **HCR-only** (`--only_hcr`) | HCR rounds only, no 2P | — | ✅ |

The stills-vs-video split is the real axis (**does temporal data exist?**), *not* tiff-vs-suite2p.
Naming TBD — candidates: `--stills` / `--image_align` for the still path; video/full is the
default (no flag). `input_format` selects **only the movie importer** (`tiff|suite2p`); the
Scanbox hi-res adapters are a separate axis (see §2a-bis).

### 2a-bis. Two orthogonal axes (NOT one `input_format`)

The movie/trace source and the hi-res anatomical handling are **independent**. The old `sbx`
value conflated them and was misleading — the movie is **always Suite2p `data.bin`** in the
Andermann setup (no video TIFF is ever written), so there is no "sbx movie".

**Axis 1 — movie source (`input_format`)** = where traces come from:

| `input_format` | Movie | Mean image | Audience |
|---|---|---|---|
| `tiff` | multipage TIFF (memmap) | `movie.mean(0)` | standard / any lab (default) |
| `suite2p` | `data.bin` (`BinaryFile`) | `ops['meanImg']` | labs w/ Suite2p (incl. Andermann) |

**Axis 2 — Scanbox hi-res anatomical adapter** = the pipeline does the extra work itself
(register + stitch + calibrated unwarp of raw `.sbx` hi-res stacks, + red-from-`sbx_memmap`)
*because those stacks were never run through Suite2p*. Gated on the **presence of raw-`.sbx`
hi-res tile runs** (`anatomical_hires_*_runs`), NOT on a movie-format string.

**"SBX mode" = `input_format: suite2p` + Scanbox hi-res adapter ON** — the Andermann full
config. First-class and working; in the *paper* it is lab-internal Scanbox instrumentation, not
a headline mode. A plain Suite2p lab that supplies pre-stitched hi-res simply leaves Axis 2 off.

### 2b. One importer contract (`src/importers.py`)

Both importers return the same object; **`movie` is optional** (absent ⇒ stills ⇒ skip traces):

| field | purpose | TIFF importer | Suite2p importer |
|---|---|---|---|
| `mean_image` (2D) | cellpose + registration target | `movie.mean(0)`, or the still itself | `ops['meanImg']` |
| `movie` (lazy `T×[Z]×Y×X`, optional) | mask → trace extraction | tifffile **memmap** (never load whole) | `BinaryFile(data.bin)` |
| `red` (2D, optional) | 2P→HCR anatomical reference | **separate red TIFF** path | `sbx_memmap` (lab adapter) |
| `voxel` / resolution | µm scaling | TIFF metadata | manifest / ops |

Everything after ingest (cellpose, trace extraction, all registration, matching, molecular
join) consumes this object — format-agnostic. **EASI-PASS does not do motion correction**
(upstream Suite2p/CaImAn); the movie is assumed motion-registered. The hi-res SBX StackReg is
anatomical-tile-only and does not contradict this.

### 2c. The 3 SBX adapters (lab-only, Andermann; NOT in paper)

Each gated on the **presence of its raw-`.sbx` manifest field** (not on `input_format`), which
is already how the pipeline mostly works (`has_hires`, `has_red` are field-presence checks):

1. Hi-res tile register + stitch — `tl.process_session_sbx` → `_hires_sbx_channel_mean`,
   gated on `anatomical_hires_*_runs` present
2. Red anatomical channel — `sbx_memmap` (`combine_with_red`), gated on `anatomical_lowres_red_runs`
3. Scanner unwarp — `_maybe_apply_lowres_unwarp` / `unwarp_tiles`, gated on `unwarp_config`

These are inherently Scanbox: a `tiff`/`suite2p` user with hi-res supplies it **pre-stitched**,
so they never enter the stitch/unwarp path.

## 3. `input_format`-conditional branches to migrate (audit)

| Location | Today | Target |
|---|---|---|
| `functional.py:_get_input_format` | default `'sbx'` | reframe; positive selection, no misleading default |
| `master_pipeline.py:164` (process_plane) | `has_hires and input_format != 'tiff'` → SBX stitch | retarget gate to Axis-2 (raw-`.sbx` tiles present); TIFF hi-res already works via pre-stitched — no stitching parity needed |
| `master_pipeline.py:172-178` | tiff/suite2p/else lowres mean | route through importer `mean_image` |
| `master_pipeline.py:121` | traces **only** for `sbx` | compute traces whenever `movie` present (both `tiff` + `suite2p`) |
| `segmentation.py:extract_electrophysiology_intensities` | hardcodes `BinaryFile(data.bin)` | accept a **movie source**; **rename** (calcium, not ephys) |
| `tiling.py:246` | `input_format != 'tiff'` | gate on raw-`.sbx` hi-res tiles present (Axis 2) |
| `meta.py:89-152` | validation | positive input_format handling |

## 4. Migration order (each step independently shippable, behavior-preserving until step 4)

1. **Introduce `src/importers.py` + the contract, no behavior change.** Wrap the three existing
   paths behind `TiffImporter` / `Suite2pImporter`; assert outputs identical to today.
2. **Generalize trace extraction.** Give `extract_electrophysiology_intensities` a movie source,
   call it for suite2p *and* tiff (fix `:121`), rename it. Stills (no movie) skip it.
3. **TIFF hi-res — mostly already works; likely no change.** A TIFF user with a pre-stitched
   `plane_{N}_hires.tiff` ALREADY gets `has_hires=True` (`meta.py:110-112`, file-presence trigger,
   independent of `input_format`/sbx runs) and the full low→hi-res registration; the SBX stitch is
   correctly skipped. The only gap is *stitching raw non-sbx tiles*, which likely isn't a real use
   case (TIFF users pre-stitch). Descope unless a concrete need appears; keep `process_session_sbx`
   Scanbox-only.
4. **Flip default + collapse `input_format` to `tiff|suite2p`.** Drop the misleading `sbx`
   value; TIFF becomes default; the 3 Scanbox adapters gate on raw-`.sbx` field presence
   (Axis 2), independent of the movie source.
5. **Docs + Methods alignment + HCR-only "superior" output** (hybrid `align_somaprint_hcr` +
   clean standalone gene-expression table).

## 5. Constraints

- No manifest/output format breaks — add fields (e.g. red-TIFF path), keep existing keys valid.
- GitHub-clean: SBX adapters kept working + documented + flag-gated; no env band-aids; scratch
  gitignored.
- Follow conventions: manifest-driven config, pathlib, `rprint`/`track` from `src.meta`,
  fail-fast with file paths.

## 6. Segmentation source: compute vs accept (settled)

A second axis alongside the movie source. Defaults per importer; overridable.

- **`compute`** (cellpose masks + traces averaged over them) — default for TIFF; **the path that
  reproduces the paper figures**. Unchanged from today.
- **`accept`** (reuse the user's Suite2p output) — default for the Suite2p importer: rasterize
  `stat.npy` ROIs (filtered by `iscell.npy`) into a labeled mask image, take traces from `F.npy`.
  No cellpose, no re-extraction.

**Rasterize with single ownership (cellpose-style, no overlaps)** — a shared pixel goes to the
ROI with the higher `lam` (Suite2p's per-pixel footprint weight):

```python
mask_img = np.zeros((Ly, Lx), np.uint16); best = np.zeros((Ly, Lx), np.float32)
for i, roi in enumerate(stat):
    if not iscell[i, 0]: continue
    win = roi['lam'] > best[roi['ypix'], roi['xpix']]
    yy, xx = roi['ypix'][win], roi['xpix'][win]
    mask_img[yy, xx] = i + 1; best[yy, xx] = roi['lam'][win]
```

The accept path **writes `mask_img` to the exact artifact `extract_2p_cellpose_masks` produces**,
so matching/registration/merge are untouched.

## 7. Implementation approach (easiest per piece)

Two principles keep it low-risk: (a) accept writes to cellpose's own output path → downstream
blind to the source; (b) `FunctionalInput` carries **optionals** → every "has movie/traces/masks?"
is one `is not None` check, not a format branch.

| Piece | Easiest solution | Touches |
|---|---|---|
| Importer contract | `dataclass FunctionalInput{mean_image, movie?, masks?, traces?, red?, voxel}` + `load_functional_input()` factory; step 1 just wraps existing `prepare_*`. | new `src/importers.py` |
| Stills vs video | **Auto-detect from TIFF shape**: 2D/CYX ⇒ stills; `(T,Y,X)` ⇒ movie. No new flag. | `importers.py` |
| Default = stills | `_get_input_format` default `'sbx' → 'tiff'`. | `functional.py:62` |
| Movie source | `MovieSource.mask_trace(ys,xs)->(T,)`: Suite2p wraps `BinaryFile`, TIFF wraps `tifffile.memmap`. | `importers.py` |
| Accept (suite2p) | Rasterize ROIs (§6) → cellpose's output path; load `F.npy`. Skip cellpose. | `importers.py` + `process_plane` gate |
| Traces for all | Replace `:121` format check with `if traces: use; elif movie: compute; else: skip`. | `master_pipeline.py:120` |
| Hi-res | Already works; retarget stitch gate `!= 'tiff'` → raw-sbx-tiles-present. | `master_pipeline.py:164` |
| SBX pre-proc | No code — legacy stitcher stays gated on sbx tiles, *produces* the hires TIFF. | (none) |
| Drop `sbx` value | `verify_manifest` shim: `'sbx' → 'suite2p' + tiles-present`, deprecation notice. | `meta.py:92` |
| Non-interactive | `--yes` arg threaded into `user_input_missing` + Enter gates. | `master_pipeline.py`, `meta.py` |

## 8. Open questions (remaining)

- **Naming** for the stills/video/`--tiff_only` surface (functional, not blocking).
- **Multi-plane Suite2p**: pull ROIs/F per plane (`stat['iplane']` / per-plane folders).
- **Suite2p red channel**: separate red TIFF (as for TIFF mode) for 2P→HCR when desired.

Resolved: movie accessor = `mask_trace` helper · TIFF stills red = separate TIFF ·
`process_session_sbx` stays Scanbox-only (no generalization) · segmentation source = accept/compute axis.
