from collections import defaultdict
from pathlib import Path
import shutil
from cellpose import models
from cellpose import io
from cellpose import utils as _cp_utils

# Cellpose 3 (cyto/cyto3 U-Net) and cellpose 4 (cellpose-SAM, ViT) share most
# of the eval signature but differ on `channels` (cp4 is single-channel by
# design, no `channels` arg) and on `diameter` (cp4 auto-detects when omitted).
# Branching once at import time keeps the call sites clean. Use
# importlib.metadata since cellpose 4 doesn't expose `__version__` on the
# top-level module.
from importlib.metadata import version as _pkg_version
_CP_MAJOR = int(_pkg_version('cellpose').split('.')[0])
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.ndimage import median_filter as scipy_median_filter
import pickle as pkl
from suite2p.io import binary
from scipy.sparse import csr_matrix
from tifffile import imread as tif_imread
from tifffile import imwrite as tif_imsave
from tifffile import TiffFile
from tqdm.auto import tqdm
from scipy.spatial import ConvexHull
from scipy.interpolate import LinearNDInterpolator
try:
    from .registrations import verify_rounds  # Relative import (running as part of a package)
    from .functional import get_number_of_suite2p_planes
    from .meta import get_intensity_extraction_config, get_round_folder_name, output_root, rprint
except ImportError:
    from registrations import verify_rounds  # Absolute import (running in Jupyter notebook)
    from functional import get_number_of_suite2p_planes
    from meta import get_intensity_extraction_config, get_round_folder_name, output_root, rprint



# Module-level cache for Cellpose models (optimization: avoid reloading per function call)
# Key: (model_path, gpu), Value: CellposeModel instance
_cellpose_model_cache = {}


def _apply_neuropil_pooling(pooling_method: str, values: np.ndarray) -> np.ndarray:
    """Apply a pooling method to neuropil values.

    Args:
        pooling_method: One of 'mean', 'median', or 'percentile-N' (e.g., 'percentile-25')
        values: Array of neuropil values, shape (n_pixels, n_channels)

    Returns:
        Pooled values, shape (n_channels,)
    """
    if pooling_method == 'mean':
        return values.mean(axis=0)
    elif pooling_method == 'median':
        return np.median(values, axis=0)
    elif pooling_method.startswith('percentile-'):
        percentile = int(pooling_method.split('-')[1])
        return np.percentile(values, percentile, axis=0)
    else:
        raise ValueError(f"Unsupported pooling method: {pooling_method}")

def _get_cached_cellpose_model(model_path: str, gpu: bool):
    """Get or create a cached Cellpose model instance.

    Works for both cp3 (custom cyto-style checkpoint paths) and cp4
    (`pretrained_model='cpsam'` for the SAM generalist, or a path to a
    fine-tuned cpsam checkpoint).
    """
    cache_key = (model_path, gpu)
    if cache_key not in _cellpose_model_cache:
        _cellpose_model_cache[cache_key] = models.CellposeModel(
            pretrained_model=model_path,
            gpu=gpu
        )
    return _cellpose_model_cache[cache_key]


# 3D HCR stacks are segmented as 2D-per-z + IoU stitching (do_3D=False,
# stitch_threshold>0) rather than cellpose's do_3D=True orthogonal-flow merge.
# do_3D=True produced choppy per-slice cross-sections on anisotropic confocal
# data (z-step >> xy pixel). 0.3 is cellpose's default and was validated on JS078
# DAPI in figure_notebooks/figure_4_cortex/cellpos4_smallchunk_fixing.ipynb.
HCR_STITCH_THRESHOLD = 0.3


# cellpose's stitch3D allocates new cumulative labels at masks.dtype (uint16),
# which overflows once pre-merge candidates across slices exceed 65535. We wrap it
# to upcast to int32. This mutates a cellpose global, so it is applied lazily and
# idempotently from the 3D-stitch path (see _CellposeModelWrapper.eval) rather than
# as an import side effect — importing this module must not silently repatch cellpose
# for unrelated consumers (figure notebooks, other scripts).
_stitch3D_patched = False


def _ensure_stitch3D_int32_patch():
    """Idempotently wrap cellpose.utils.stitch3D to upcast masks to int32.

    No-op if already applied, or if this cellpose build has no stitch3D (e.g. a
    version that stitches through a different path), so it never crashes at import
    or on cp4.
    """
    global _stitch3D_patched
    if _stitch3D_patched or not hasattr(_cp_utils, 'stitch3D'):
        return
    _original_stitch3D = _cp_utils.stitch3D

    def _stitch3D_int32(masks, *args, **kwargs):
        if hasattr(masks, 'dtype') and masks.dtype.itemsize < 4:
            masks = masks.astype(np.int32)
        return _original_stitch3D(masks, *args, **kwargs)

    _cp_utils.stitch3D = _stitch3D_int32
    _stitch3D_patched = True


def _eval_kwargs(cellpose_params: dict, is_3d_stack: bool) -> dict:
    """Build kwargs for `model.eval()` that work on both cellpose 3 and 4.

    cp3 passes `channels=[0,0]` (DAPI-only grayscale); cp4 drops `channels`
    (single-channel by SAM design). `diameter` is passed only when set
    (None/null in manifest = cp4 auto-detect; cp3 falls back to its own
    default). For 3D HCR stacks we always run in stitch mode -- see
    HCR_STITCH_THRESHOLD above.
    """
    kw = dict(
        flow_threshold=cellpose_params['flow_threshold'],
        cellprob_threshold=cellpose_params['cellprob_threshold'],
        do_3D=False,
    )
    if is_3d_stack:
        kw['stitch_threshold'] = HCR_STITCH_THRESHOLD
        kw['z_axis'] = 0
    diameter = cellpose_params.get('diameter')
    if diameter is not None:
        kw['diameter'] = diameter
    # Passthrough to cellpose's normalize= API (e.g. {tile_norm_blocksize: 100}).
    if cellpose_params.get('normalize') is not None:
        kw['normalize'] = cellpose_params['normalize']
    if _CP_MAJOR < 4:
        kw['channels'] = [0, 0]
    return kw


# Outlier-size HCR mask filter. Thresholds are relative to the stack's
# median so the rule self-calibrates across mice and microscopes.
HCR_MASK_VOL_MIN_FRAC    = 0.1
HCR_MASK_VOL_MAX_FRAC    = 10.0
HCR_MASK_BBOX_MAX_FRAC   = 3.0
HCR_MASK_MIN_N_FOR_STATS = 10


def _filter_implausible_hcr_masks(masks: np.ndarray) -> tuple:
    """Drop labels with vol < 0.1x or > 10x median, or xy bbox > 3x median bbox,
    then relabel 1..N. Skipped if fewer than HCR_MASK_MIN_N_FOR_STATS labels.
    Returns (masks, counts) with total/kept/tiny/huge_vol/huge_xy/median_vol/median_bbox.
    """
    from skimage.segmentation import relabel_sequential
    from scipy.ndimage import find_objects

    counts = dict(total=0, kept=0, tiny=0, huge_vol=0, huge_xy=0,
                  median_vol=0.0, median_bbox=0.0)
    if masks.max() == 0:
        return masks, counts

    vols   = np.bincount(masks.ravel())
    slices = find_objects(masks)

    present, bbox_ext = [], []
    for lbl in range(1, len(vols)):
        if vols[lbl] == 0:
            continue
        present.append(lbl)
        sl = slices[lbl - 1]
        bbox_ext.append(max(sl[1].stop - sl[1].start, sl[2].stop - sl[2].start)
                        if sl is not None else 0)
    present  = np.array(present, dtype=np.int64)
    bbox_ext = np.array(bbox_ext, dtype=np.int64)
    label_vols = vols[present]

    counts['total'] = int(len(present))
    if len(present) < HCR_MASK_MIN_N_FOR_STATS:
        counts['kept'] = counts['total']
        return masks, counts

    med_vol  = float(np.median(label_vols))
    med_bbox = float(np.median(bbox_ext))
    counts['median_vol']  = med_vol
    counts['median_bbox'] = med_bbox

    min_vol      = HCR_MASK_VOL_MIN_FRAC  * med_vol
    max_vol      = HCR_MASK_VOL_MAX_FRAC  * med_vol
    max_bbox_ext = HCR_MASK_BBOX_MAX_FRAC * med_bbox

    drop_mask = np.zeros(len(vols), dtype=bool)
    for i, lbl in enumerate(present):
        v = label_vols[i]
        if v < min_vol:
            drop_mask[lbl] = True; counts['tiny']     += 1; continue
        if v > max_vol:
            drop_mask[lbl] = True; counts['huge_vol'] += 1; continue
        if bbox_ext[i] > max_bbox_ext:
            drop_mask[lbl] = True; counts['huge_xy']  += 1

    counts['kept'] = counts['total'] - counts['tiny'] - counts['huge_vol'] - counts['huge_xy']

    if drop_mask.any():
        zero_lookup = np.arange(len(vols), dtype=masks.dtype)
        zero_lookup[drop_mask] = 0
        filtered, _, _ = relabel_sequential(zero_lookup[masks])
        return filtered.astype(masks.dtype), counts
    return masks, counts


# CellposeModelWrapper class
# This class encapsulates the Cellpose model and its configuration.
# It's needed to:
# 1. Provide a consistent interface for Cellpose model initialization and evaluation
# 2. Allow lazy loading of the model, which can be computationally expensive
# 3. Centralize the configuration management for Cellpose parameters

class CellposeModelWrapper:
    def __init__(self, params):
        self.params = params
        self.model = None

    def eval(self, raw_image, progress=None):
        if self.model is None:
            # Use cached model (optimization: reuse across pipeline)
            self.model = _get_cached_cellpose_model(
                self.params['HCR_cellpose']['model_path'],
                self.params['HCR_cellpose']['gpu']
            )

        kw = _eval_kwargs(self.params['HCR_cellpose'], is_3d_stack=True)
        if 'stitch_threshold' in kw:
            _ensure_stitch3D_int32_patch()
        if progress is not None:
            kw['progress'] = progress
        return self.model.eval(raw_image, **kw)

def run_cellpose(full_manifest):
    params = full_manifest['params']

    # Segmentation precedes HCR-HCR registration, so enumerate the IMAGED rounds
    # (every non-reference round in the manifest), not the registered ones — the
    # registration may not be computed yet (or may have been moved aside for a
    # clean re-run). Each round is segmented on its own acquired (un-warped)
    # volume into cellpose/; align_masks_to_reference later warps those labels
    # into the HCR01 frame (cellpose_aligned/) for matching.
    round_to_rounds, reference_round, _ = verify_rounds(
        full_manifest, parse_registered=False,
        print_rounds=False, print_registered=False, func='cellpose')

    cellpose_channel_index = params['HCR_cellpose']['cellpose_channel']
    all_rounds = list(round_to_rounds.keys()) + [reference_round['round']]

    # Which rounds still need segmenting
    skipped = []
    to_process = []
    for HCR_round in all_rounds:
        round_folder_name = get_round_folder_name(HCR_round, reference_round['round'])
        output_path = output_root(full_manifest) / 'HCR' / 'cellpose' / f"{round_folder_name}_masks.tiff"
        mov = reference_round if HCR_round == reference_round['round'] else round_to_rounds[HCR_round]
        source_path = mov['image_path']
        if output_path.exists():
            skipped.append(round_folder_name)
        else:
            to_process.append((round_folder_name, source_path, output_path))

    # Print summary
    if not to_process:
        rprint(f"[dim]HCR cellpose: all {len(all_rounds)} rounds exist[/dim]")
        return

    rprint(f"HCR cellpose: processing {len(to_process)}/{len(all_rounds)} rounds")
    model_wrapper = CellposeModelWrapper(params)

    # Disable the outer round-counter bar when there's only one round to process
    # (the per-slice intra-round bar below is more useful in that case).
    iterator = tqdm(to_process, desc="HCR cellpose", disable=len(to_process) <= 1)
    for round_folder_name, source_path, output_path in iterator:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        raw_image = tif_imread(source_path)
        cellpose_input = raw_image[:, cellpose_channel_index, :, :]

        # Per-slice progress bar (cellpose ticks once per Z-slice during 3D
        # inference). Useful for big HCR stacks that take 10s of minutes.
        nz = cellpose_input.shape[0]
        with tqdm(total=nz, desc=f"  {round_folder_name} (cellpose-SAM)",
                  unit="slice", leave=False) as pb:
            masks, _, _ = model_wrapper.eval(cellpose_input, progress=pb)

        masks, fc = _filter_implausible_hcr_masks(masks)
        rprint(f"  {round_folder_name}: kept {fc['kept']}/{fc['total']} masks "
               f"(dropped {fc['tiny']} tiny + {fc['huge_vol']} huge_vol + {fc['huge_xy']} huge_xy; "
               f"median vol={fc['median_vol']:.0f} vox, bbox={fc['median_bbox']:.0f} px)")

        tif_imsave(output_path, masks)

def run_cellpose_2p(tiff_path: Path, output_path: Path, cellpose_params: dict):
    """
    Run cellpose segmentation on a single 2P plane (2D image).
    Saves in Cellpose 2-compatible format for GUI editing.
    """
    raw_image = tif_imread(tiff_path)
    assert raw_image.ndim == 2, f"Expected 2D image, got {raw_image.ndim}D"

    # Use cached model (optimization: reuse across planes)
    model = _get_cached_cellpose_model(
        cellpose_params['model_path'],
        cellpose_params['gpu']
    )

    masks, flows, styles = model.eval(
        raw_image,
        **_eval_kwargs(cellpose_params, is_3d_stack=False),
    )

    io.masks_flows_to_seg(raw_image, masks, flows, str(tiff_path.parent / tiff_path.stem), channels=[0,0])

    generated_file = tiff_path.parent / f'{tiff_path.stem}_seg.npy'
    if generated_file != output_path:
        shutil.move(str(generated_file), str(output_path))

    # Save masks as TIFF for user verification
    masks_tiff_path = tiff_path.parent / f'{tiff_path.stem}_masks.tiff'
    tif_imsave(masks_tiff_path, masks.astype(np.uint16))

def extract_2p_cellpose_masks(full_manifest: dict, session: dict):
    manifest = full_manifest['data']
    params = full_manifest['params']
    cellpose_path = output_root(full_manifest) / '2P' / 'cellpose'

    # Get the current plane being processed
    current_plane = session['functional_plane'][0]

    twop_cellpose_file = cellpose_path / f'lowres_meanImg_C0_plane{current_plane}_seg.npy'
    tiff_path = cellpose_path / f'lowres_meanImg_C0_plane{current_plane}.tiff'

    if twop_cellpose_file.exists():
        rprint(f"[dim]2P cellpose plane {current_plane}: exists[/dim]")
    else:
        if not tiff_path.exists():
            raise FileNotFoundError(f"2P tiff file not found: {tiff_path}")
        rprint(f"[bold]Running Cellpose on 2P plane {current_plane}[/bold]")
        run_cellpose_2p(tiff_path, twop_cellpose_file, params['2p_cellpose'])

    return twop_cellpose_file


def verify_2p_cellpose_segmentations(seg_files: list):
    """Single consolidated prompt for verifying 2P cellpose segmentations across all planes.

    Caller passes a list of (plane, seg_file) tuples produced by
    extract_2p_cellpose_masks; user opens whichever planes need editing in the
    Cellpose GUI, then presses Enter once to continue.
    """
    if not seg_files:
        return
    rprint("\n[bold]Verify 2P cellpose segmentations:[/bold]")
    for plane, path in seg_files:
        rprint(f"  plane {plane}: [yellow]{path}[/yellow]")
    rprint("\nOpen any plane that needs editing in the Cellpose GUI, then press [green]Enter[/green] to continue...")
    input()

def compute_M(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)),
                      shape=(data.max() + 1, data.size))

def get_indices_sparse(data):
    M = compute_M(data)
    inds = []
    for row in tqdm(M):
        inds.append(np.unravel_index(row.data, data.shape))
    return inds


def get_neuropil_mask_square(volume, radius, bound, inds):
    '''
    Get the neuropil mask square for each cell in the volume.

    For each cell, finds pixels within a square region (defined by `radius`) around
    the cell center that can be used for neuropil subtraction. Excludes:
    - Pixels that are part of any cell (volume > 0)
    - Pixels within `bound` distance of any cell edge (to avoid signal bleed)

    Parameters
    ----------
    volume : ndarray (Z, X, Y)
        3D labeled mask volume where each cell has a unique integer ID, 0 = background
    radius : int
        Half-width of the square neuropil sampling region around each cell center
    bound : int
        Exclusion boundary - pixels within this distance of any cell are excluded
        from neuropil to avoid picking up signal spillover from nearby cells
    inds : list
        Pre-computed cell indices from get_indices_sparse(volume)

    Returns
    -------
    dict
        {mask_id: ndarray of shape (3, N)} where each column is (z, x, y) coords
        of valid neuropil pixels for that cell
    '''
    from scipy.ndimage import binary_dilation

    all_masks_locs = {}
    x_max, y_max = volume.shape[1], volume.shape[2]

    # Pre-compute unique mask IDs excluding background (0)
    unique_masks = np.unique(volume)
    unique_masks = unique_masks[unique_masks != 0]

    # Pre-compute the exclusion zone: dilate all cells by `bound` pixels in XY only
    # This creates a mask of "too close to any cell" regions
    # Neuropil will only be sampled from pixels outside this dilated region
    # We dilate each z-plane independently (no dilation in z direction)
    if bound > 0:
        # Create 2D structuring element extended to 3D with no z-connectivity
        # Shape (1, 3, 3) means: 1 slice in z, 3x3 in XY
        # This is equivalent to per-plane 2D dilation but runs as single 3D operation
        from scipy.ndimage import generate_binary_structure, iterate_structure
        struct_2d = generate_binary_structure(2, 1)  # 2D cross
        struct_2d = iterate_structure(struct_2d, 1)  # Makes it a 3x3 square
        struct_3d = struct_2d[np.newaxis, :, :]  # Extend to 3D: shape (1, 3, 3)

        # Single vectorized dilation across entire volume (faster than per-z loop)
        exclusion_zone = binary_dilation(volume > 0, structure=struct_3d, iterations=bound)
    else:
        exclusion_zone = volume > 0

    def _coords_for_mask(mask_id, r):
        mask_coords = []
        z_planes = np.unique(inds[mask_id][0])
        for z in z_planes:
            mask_points = (inds[mask_id][0] == z)
            if not np.any(mask_points):
                continue
            mx = int(inds[mask_id][1][mask_points].mean())
            my = int(inds[mask_id][2][mask_points].mean())
            x_min, x_max_local = max(mx-r, 0), min(mx+r, x_max)
            y_min, y_max_local = max(my-r, 0), min(my+r, y_max)
            neuropil_points = ~exclusion_zone[z, x_min:x_max_local, y_min:y_max_local]
            if np.any(neuropil_points):
                x_coords, y_coords = np.where(neuropil_points)
                z_coords = np.full_like(x_coords, z)
                mask_coords.append(np.stack([z_coords, x_coords + x_min, y_coords + y_min]))
        if mask_coords:
            return np.hstack(mask_coords).astype(np.int32)
        return None

    pending = []
    for mask_id in tqdm(unique_masks, desc="Computing neuropil regions"):
        coords = _coords_for_mask(mask_id, radius)
        if coords is not None:
            all_masks_locs[mask_id] = coords
        else:
            pending.append(mask_id)

    # Expansion fallback: cells fully surrounded by other cells get the radius bumped
    # by `expand_step` until valid neuropil pixels are found. Column names downstream
    # still reflect the original `radius` — these are bookkept silently so the output
    # schema stays consistent.
    expand_step = 20
    max_expansions = 50  # radius can grow up to radius + 1000 px; effectively unbounded
    current_radius = radius
    expansion = 0
    while pending and expansion < max_expansions:
        expansion += 1
        current_radius += expand_step
        next_pending = []
        for mask_id in tqdm(pending, desc=f"Expanding neuropil to r={current_radius} for {len(pending)} cells"):
            coords = _coords_for_mask(mask_id, current_radius)
            if coords is not None:
                all_masks_locs[mask_id] = coords
            else:
                next_pending.append(mask_id)
        pending = next_pending

    if pending:
        # Should be effectively impossible — image is saturated with cells out to 1000+ px.
        rprint(f"[red]  {len(pending)} cells still have no neuropil after expansion to r={current_radius}; they will be omitted.[/red]")

    return all_masks_locs


def extract_probe_intensity(full_manifest):
    params = full_manifest['params']
    round_to_rounds, reference_round, register_rounds = verify_rounds(full_manifest, parse_registered = True,
                                                                      print_rounds = False, print_registered = False, func='intensities-extraction')

    all_rounds = register_rounds + [reference_round['round']]

    def _stack_source(HCR_round, round_folder_name):
        # Intensities are measured on the acquired (un-warped) round image, so the masks MUST be
        # that round's NATIVE-frame masks -- not the ref-warped {round_folder}_masks.tiff. On mice
        # whose rounds were acquired at different dims (e.g. cp4 SRC110/JS082: HCR02 3624x3215 vs
        # HCR01 1942x1944) the warped masks have the ref shape, which != the native image -> the
        # `raw_image[:,0].shape == masks.shape` assert below fires. Native masks are written as
        # HCR{N}_native_masks.tiff; fall back to the canonical name when absent (older mice where
        # native == ref shape, so the pairing was harmless).
        cellpose = output_root(full_manifest) / 'HCR' / 'cellpose'
        if HCR_round == reference_round['round']:
            mov = reference_round
            masks = cellpose / f"{round_folder_name}_masks.tiff"   # ref isn't warped: canonical IS native
        else:
            mov = round_to_rounds[HCR_round]
            native = cellpose / f"HCR{HCR_round}_native_masks.tiff"
            masks = native if native.exists() else cellpose / f"{round_folder_name}_masks.tiff"
        return Path(mov['image_path']), masks

    # Check which rounds need processing
    to_process = []
    for HCR_round_to_register in all_rounds:
        round_folder_name = get_round_folder_name(HCR_round_to_register, reference_round['round'])
        if HCR_round_to_register == reference_round['round']:
            channels_names = reference_round['channels']
        else:
            channels_names = round_to_rounds[HCR_round_to_register]['channels']
        output_folder = output_root(full_manifest) / 'HCR' / 'extract_intensities'
        pkl_output_path = output_folder / f"{round_folder_name}_probs_intensities.pkl"
        if not pkl_output_path.exists():
            to_process.append((HCR_round_to_register, round_folder_name, channels_names))

    # Print summary
    if not to_process:
        rprint(f"[dim]HCR intensities: all {len(all_rounds)} rounds exist[/dim]")
        return

    rprint(f"HCR intensities: extracting {len(to_process)}/{len(all_rounds)} rounds")

    # Pre-flight: check TIFF channel count vs manifest for all rounds (metadata-only, cheap)
    mismatches = []
    for HCR_round_to_register, round_folder_name, channels_names in to_process:
        stack_path, _ = _stack_source(HCR_round_to_register, round_folder_name)
        with TiffFile(stack_path) as tf:
            n_ch = tf.series[0].shape[1]
        if n_ch != len(channels_names):
            mismatches.append(f"  {round_folder_name}: TIFF has {n_ch} channels, manifest lists {len(channels_names)} ({channels_names})")
    if mismatches:
        raise ValueError("HCR channel-count mismatch (fix manifest channel list or re-register stack):\n" + "\n".join(mismatches))

    # neuropil parameters
    intensity_config = get_intensity_extraction_config(params)
    neuropil_radius = intensity_config['neuropil_radius']
    neuropil_boundary = intensity_config['neuropil_boundary']
    neuropil_pooling = intensity_config['neuropil_pooling']

    # FIXED: Create median filter label BEFORE converting to numpy array
    median_filter = None
    median_filter_label = None
    if intensity_config.get('stack_median_filter'):
        median_filter_config = intensity_config.get('stack_median_filter')
        assert isinstance(median_filter_config, (list, tuple)) and len(median_filter_config) == 3
        # Create label from original config values
        median_filter_label = f'medflt_{median_filter_config[0]}x{median_filter_config[1]}x{median_filter_config[2]}'
        # Then create numpy array
        median_filter = np.ones((median_filter_config[0], 1, median_filter_config[1], median_filter_config[2]))

    for HCR_round_to_register, round_folder_name, channels_names in to_process:
        full_stack_path, full_stack_masks_path = _stack_source(HCR_round_to_register, round_folder_name)
        output_folder = output_root(full_manifest) / 'HCR' / 'extract_intensities'
        output_folder.mkdir(parents=True, exist_ok=True)
        csv_output_path = output_folder / f"{round_folder_name}_probs_intensities.csv"
        pkl_output_path = output_folder / f"{round_folder_name}_probs_intensities.pkl"

        # Load images and verify sizes
        raw_image = tif_imread(full_stack_path)
        masks = tif_imread(full_stack_masks_path)
        assert raw_image[:,0,:,:].shape == masks.shape

        # Apply median filter if configured (with disk caching for re-runs)
        med_filter_stack = None
        if median_filter is not None:
            # Cache path for filtered stack (avoids recomputation on re-runs)
            medfilt_cache_path = output_folder / f"{round_folder_name}_{median_filter_label}_cache.npy"
            if medfilt_cache_path.exists():
                med_filter_stack = np.load(medfilt_cache_path, mmap_mode='r')
            else:
                # Use scipy directly (faster than skimage wrapper for simple footprints)
                # median_filter shape is (z, 1, y, x) - the 1 means channels are independent
                med_filter_stack = scipy_median_filter(raw_image, footprint=median_filter)
                np.save(medfilt_cache_path, med_filter_stack)
        # acceleration step
        inds = get_indices_sparse(masks)
        neuropil_masks_inds = get_neuropil_mask_square(masks, neuropil_radius, neuropil_boundary, inds)

        number_of_channels = int(raw_image.shape[1])

        # Pre-compute reusable arrays and column names (optimization: avoid recreation per mask)
        channel_indices = np.arange(number_of_channels)

        # Pre-compute neuropil column names (moved outside mask loop)
        neuropil_col_names = {}
        for pooling_method in neuropil_pooling:
            if pooling_method.startswith('percentile-'):
                percentile = pooling_method.split('-')[1]
                col_suffix = f'neuropil_{percentile}pct_nr{neuropil_radius}_nb_{neuropil_boundary}'
            else:
                col_suffix = f'neuropil_{pooling_method}_nr{neuropil_radius}_nb_{neuropil_boundary}'
            neuropil_col_names[pooling_method] = col_suffix

        to_pnd = defaultdict(list)
        skipped_no_neuropil = 0
        for mask_id, mask_inds in enumerate(tqdm(inds, desc=f"HCR {round_folder_name}")):
            if mask_id == 0:
                continue
            if len(mask_inds[0]) == 0:
                raise Exception("Mask with zero pixels")
            # Cells fully surrounded by other cells within neuropil_radius+neuropil_boundary
            # have no valid neuropil pixels; get_neuropil_mask_square omits them. Skip rather
            # than crash — these cells get no intensity row at all.
            if mask_id not in neuropil_masks_inds:
                skipped_no_neuropil += 1
                continue
            Z, Y, X = mask_inds

            vals_per_mask_per_channel = raw_image[Z, :, Y, X]

            # Compute means once and reuse
            mean_vals = vals_per_mask_per_channel.mean(axis=0)
            z_mean, x_mean, y_mean = Z.mean(), X.mean(), Y.mean()

            # Use numpy arrays directly with extend (avoids list() conversions)
            to_pnd['mask_id'].extend(np.full(number_of_channels, mask_id))
            to_pnd['channel'].extend(channel_indices)
            to_pnd['channel_name'].extend(channels_names)
            to_pnd['mean'].extend(mean_vals)

            # Add X,Y,Z (pre-computed means, extend with numpy full arrays)
            to_pnd['Z'].extend(np.full(number_of_channels, z_mean))
            to_pnd['X'].extend(np.full(number_of_channels, x_mean))
            to_pnd['Y'].extend(np.full(number_of_channels, y_mean))

            # Extract neuropil values
            neuropil_Z, neuropil_Y, neuropil_X = neuropil_masks_inds[mask_id]
            neuropil_vals_per_channel = raw_image[neuropil_Z, :, neuropil_Y, neuropil_X]

            if median_filter is not None:
                to_pnd['mean_' + median_filter_label].extend(med_filter_stack[Z, :, Y, X].mean(axis=0))
                neuropil_vals_per_channel_median = med_filter_stack[neuropil_Z, :, neuropil_Y, neuropil_X]

            # Apply neuropil pooling methods (using pre-computed column names)
            for pooling_method in neuropil_pooling:
                col_suffix = neuropil_col_names[pooling_method]
                to_pnd[col_suffix].extend(_apply_neuropil_pooling(pooling_method, neuropil_vals_per_channel))
                if median_filter is not None:
                    to_pnd[f'{median_filter_label}_{col_suffix}'].extend(
                        _apply_neuropil_pooling(pooling_method, neuropil_vals_per_channel_median)
                    )

        if skipped_no_neuropil:
            rprint(f"[yellow]  {round_folder_name}: skipped {skipped_no_neuropil} cells with no available neuropil region[/yellow]")

        #Save to `masks_path` folder
        df = pd.DataFrame(to_pnd)
        df.attrs['raw_image_path'] = str(full_stack_path)
        df.attrs['masks_path'] = str(full_stack_masks_path)
        df.attrs['HCR_round_number'] = HCR_round_to_register
        df.to_csv(csv_output_path)
        df.to_pickle(pkl_output_path)


def extract_electrophysiology_intensities(full_manifest: dict , session: dict):
    rprint("[bold]Extract 2P Intensities[/bold]")

    manifest = full_manifest['data'] 
    mouse_name = manifest['mouse_name']
    date = session['date']
    suite2p_run = session['functional_run'][0]

    suite2p_path = Path(manifest['base_path']) / manifest['mouse_name'] / '2P' /  f'{mouse_name}_{date}_{suite2p_run}' / 'suite2p'
    save_path = output_root(full_manifest) / '2P' / 'suite2p'
    cellpose_path = output_root(full_manifest) / '2P' / 'cellpose'
    save_path.mkdir(exist_ok=True, parents=True)
    functional_plane = session['functional_plane'][0]

    planes = get_number_of_suite2p_planes(suite2p_path)
    # Assuming `planes`, `suite2p_path`, `savepath`, `mouse`, `run`, and `ops` are defined
    for plane in [functional_plane]:
        pkl_save_path = save_path / f'lowres_meanImg_C0_plane{plane}.pkl'
        if pkl_save_path.exists():
            rprint(f"[dim]2P intensities: plane {plane} exists[/dim]")
            continue

        ops = np.load(suite2p_path / f'plane0/ops.npy', allow_pickle=True).item()
        # Set up binary file
        bin_file = binary.BinaryFile(filename=suite2p_path / f'plane{plane}' / 'data.bin', Lx=ops['Lx'], Ly=ops['Ly'])
        # Move data to a numpy array
        all_data = bin_file.data
        # Load masks
        stats = np.load(cellpose_path / f'lowres_meanImg_C0_plane{plane}_seg.npy', allow_pickle=True).item()
        masks_locs = get_indices_sparse(stats['masks'])  # Get (x, y) indices per mask

        # Process each mask to get mean values
        mean_frames = []
        for mask_loc in tqdm(masks_locs[1:], desc=f"2P plane {plane}"):
            mean_frames.append(all_data[:, mask_loc[0], mask_loc[1]].mean(axis=1))

        mean_frames = np.array(mean_frames)
        # Save masks_locs as a dictionary
        masks_locs_dict = {f'cell_{i}': masks_locs[i] for i in range(len(masks_locs))}

        # Save data to .mat files
        sio.savemat(save_path / f'lowres_meanImg_C0_plane{plane}_locs.mat', {'masks_locs': masks_locs_dict})
        sio.savemat(save_path / f'lowres_meanImg_C0_plane{plane}_traces.mat', {'mean_frames': mean_frames})

        # Save data to .pkl files
        pkl.dump({'masks_locs': masks_locs_dict,
                  'mean_frames': mean_frames},
                  open(pkl_save_path, 'wb'))

def bigwarp_pixel_adjustment(stack1, stack2, name1="Stack1", name2="Stack2"):
    """Pad smaller array to match larger dimensions when bigwarp causes size mismatch."""
    if stack1.shape == stack2.shape:
        return stack1, stack2

    rprint(f"[yellow]Dimension mismatch - {name1}: {stack1.shape}, {name2}: {stack2.shape}[/yellow]")
    # Check if padding exceeds 5 pixels in any dimension
    if any(abs(s1 - s2) > 5 for s1, s2 in zip(stack1.shape, stack2.shape)):
        raise ValueError(
            f"ERROR: Padding exceeds 5 pixels: {name1}={stack1.shape}, {name2}={stack2.shape}, "
            "this means that there is a serious issue with bigwarp, please re-do the registration"
        )

    max_dims = tuple(max(s1, s2) for s1, s2 in zip(stack1.shape, stack2.shape))
    
    def pad_if_needed(arr, target):
        if arr.shape != target:
            pad_width = [(0, t - s) for s, t in zip(arr.shape, target)]
            return np.pad(arr, pad_width, constant_values=0)
        return arr
    
    return pad_if_needed(stack1, max_dims), pad_if_needed(stack2, max_dims)


def _neighborhood_foreground_iou(stack1_masks, stack2_masks, mask1_inds, window_px):
    '''
    Neighborhood-agreement term for a source cell: the IoU of the binary
    any-cell foreground (all labels merged) of stack1 vs stack2 in a
    window_px-by-window_px XY window centered on the source cell's centroid,
    restricted in Z to the slices the source cell occupies (same fairness
    logic as iou_at_mask1_z). High when the local cellular landscape agrees
    across modalities (well-registered region); low when only the candidate
    cell coincidentally overlaps (partial alignment). Reported alongside the
    per-cell IoU as a registration-quality signal — it does NOT gate matches;
    thresholding is left to the user.
    '''
    src_3d = (len(mask1_inds) == 3)
    if src_3d:
        zc, yc, xc = mask1_inds
        zs = np.unique(zc)
    else:
        yc, xc = mask1_inds
        zs = None

    cy = int(round(float(yc.mean())))
    cx = int(round(float(xc.mean())))
    h = int(window_px) // 2
    sy, sx = stack1_masks.shape[-2], stack1_masks.shape[-1]
    y0, y1 = max(0, cy - h), min(sy, cy + h + 1)
    x0, x1 = max(0, cx - h), min(sx, cx + h + 1)

    if stack1_masks.ndim == 3:
        z1 = zs if zs is not None else np.arange(stack1_masks.shape[0])
        sub1 = stack1_masks[z1][:, y0:y1, x0:x1] > 0
    else:
        sub1 = stack1_masks[y0:y1, x0:x1] > 0

    if stack2_masks.ndim == 3:
        z2 = zs if zs is not None else np.arange(stack2_masks.shape[0])
        z2 = z2[z2 < stack2_masks.shape[0]]  # defensive; aligned stacks share depth
        sub2 = stack2_masks[z2][:, y0:y1, x0:x1] > 0
    else:
        sub2 = stack2_masks[y0:y1, x0:x1] > 0

    # If dimensionalities disagree (source 3D vs target 2D or vice versa),
    # collapse to a 2D projection so the boolean windows are comparable.
    if sub1.shape != sub2.shape:
        if sub1.ndim == 3:
            sub1 = sub1.any(axis=0)
        if sub2.ndim == 3:
            sub2 = sub2.any(axis=0)

    union = int(np.count_nonzero(sub1 | sub2))
    if union == 0:
        return 0.0
    return float(np.count_nonzero(sub1 & sub2)) / union


def match_masks(stack1_masks_path, stack2_masks_path, neighborhood_window_px=None):
    '''
    Match masks between two registered stacks and record several overlap
    metrics per pair.

    Motivation
    ----------
    For 2P→HCR matching the source (stack1) is essentially a 2D sheet draped
    in 3D space, while the target (stack2) is a full 3D HCR volume. A
    symmetric IoU divides by |mask1| + |mask2| − overlap, so the 3D HCR
    volume dominates the union and the IoU is structurally capped at
    ~area_2P / volume_HCR (often ≤ 0.3 even for clean matches). Filtering
    on that number is misleading. We additionally compute mask2's size
    restricted to the z-slices that mask1 occupies, which gives a fair
    apples-to-apples denominator. For HCR↔HCR (both 3D) the restriction is
    a near-no-op.

    Returns DataFrame with one row per overlapping (mask1, mask2) pair and
    columns:

      mask1, mask2
          Label IDs in stack1 and stack2.
      intersection
          Overlap voxel count.
      mask1_size
          Total voxels in mask1.
      mask2_size
          Total voxels in mask2 (full 3D for HCR).
      mask2_size_at_mask1_z
          Voxels of mask2 restricted to the z-slices that contain mask1.
          For 2P↔HCR: HCR cell's footprint at the 2P sheet's z-range.
          For HCR↔HCR: ≈ mask2_size.
      iou
          Legacy 3D-symmetric IoU: intersection /
          (mask1_size + mask2_size − intersection). Kept for backward
          compat; depressed for 2P→HCR.
      iou_at_mask1_z
          Fair IoU using mask2_size_at_mask1_z as the mask2 denominator.
          Drives is_best_match.
      containment_2p
          intersection / mask1_size. Fraction of mask1 explained by mask2.
          Bounded [0, 1] regardless of dimensionality.
      containment_hcr_at_z
          intersection / mask2_size_at_mask1_z. Fraction of mask2 (at
          mask1's z-range) explained by mask1.
      is_best_match
          True for the pair that wins the greedy 1:1 match, ranked by
          iou_at_mask1_z.
      neighborhood_iou, neighborhood_window_px
          Present only when ``neighborhood_window_px`` is passed. Per source
          cell (identical across all of that mask1's rows): the any-cell
          foreground IoU in a window around the cell (see
          _neighborhood_foreground_iou). A registration-quality signal
          REPORTED alongside the per-cell IoU; it never gates the match set,
          so is_best_match and all match counts are unchanged whether or not
          it is computed. Thresholding is left to the user.

    For 2D inputs mask2_size_at_mask1_z == mask2_size and iou_at_mask1_z
    == iou.

    Parameters
    ----------
    neighborhood_window_px : int or None
        XY window side length (px) for the neighborhood-IoU term. None
        (default) skips it entirely — output is byte-identical to the
        historical schema. Threaded from
        ``params.twop_to_hcr_registration.neighborhood_window_px`` for the
        cross-modal (2P→HCR) matcher; left None for HCR↔HCR.
    '''
    stack1_masks = tif_imread(stack1_masks_path)
    stack2_masks = tif_imread(stack2_masks_path)

    # Validate mask dtypes upfront (float masks cause indexing issues)
    if not np.issubdtype(stack1_masks.dtype, np.integer):
        raise ValueError(
            f"stack1_masks has non-integer dtype {stack1_masks.dtype}. "
            f"Masks must be stored as integers. Check file: {stack1_masks_path}"
        )

    # Handle dimension mismatches from bigwarp
    stack1_masks, stack2_masks = bigwarp_pixel_adjustment(stack1_masks, stack2_masks, "Stack1", "Stack2")

    stack1_masks_inds = get_indices_sparse(stack1_masks.astype(np.uint16))

    # Total mask2 voxels per ID (full volume).
    stack2_mask_ids, stack2_mask_sizes = np.unique(stack2_masks, return_counts=True)
    stack2_size_lookup = dict(zip(stack2_mask_ids, stack2_mask_sizes))

    # Per-z mask2 voxel counts for the z-restricted denominator. Built once
    # via np.bincount per z-slice; lookup per pair is then a small sum.
    is_3d = stack2_masks.ndim == 3
    if is_3d:
        max_id_2 = int(stack2_masks.max())
        nz = stack2_masks.shape[0]
        mask2_size_per_z = np.zeros((nz, max_id_2 + 1), dtype=np.int64)
        for z in range(nz):
            mask2_size_per_z[z] = np.bincount(stack2_masks[z].ravel(),
                                              minlength=max_id_2 + 1)

    to_pandas = {
        'mask1': [], 'mask2': [],
        'intersection': [], 'mask1_size': [], 'mask2_size': [],
        'mask2_size_at_mask1_z': [],
        'iou': [], 'iou_at_mask1_z': [],
        'containment_2p': [], 'containment_hcr_at_z': [],
    }

    # Enumerate masks starting from 1 (index 0 is background)
    for mask1_id, mask1_inds in enumerate(stack1_masks_inds[1:], start=1):
        if len(mask1_inds[0]) == 0:
            continue

        mask1_size = len(mask1_inds[0])

        mask2_at_mask1 = stack2_masks[mask1_inds]
        # Find ALL overlapping mask2 labels (excluding background)
        mask2_nonzero = mask2_at_mask1[mask2_at_mask1 > 0]
        if len(mask2_nonzero) == 0:
            continue  # No overlap with any mask2

        # z-slices the mask1 voxels occupy (used for the fair denominator).
        unique_zs = np.unique(mask1_inds[0]) if is_3d else None

        # All overlapping mask2 IDs AND their intersection counts in one pass — the counts
        # ARE the per-id intersections, so we avoid re-scanning mask2_at_mask1 with `== id`
        # for every overlapping label (was O(k*n) per mask1; now O(n)).
        overlapping_mask2_ids, overlap_counts = np.unique(mask2_nonzero, return_counts=True)

        for mask2_id, intersection in zip(overlapping_mask2_ids, overlap_counts):
            intersection = int(intersection)
            mask2_size = int(stack2_size_lookup.get(mask2_id, 0))
            if is_3d:
                mask2_size_at_mask1_z = int(mask2_size_per_z[unique_zs, mask2_id].sum())
            else:
                mask2_size_at_mask1_z = mask2_size

            union_3d = mask1_size + mask2_size - intersection
            union_at_z = mask1_size + mask2_size_at_mask1_z - intersection

            iou = intersection / union_3d if union_3d > 0 else 0.0
            iou_at_mask1_z = intersection / union_at_z if union_at_z > 0 else 0.0
            containment_2p = intersection / mask1_size if mask1_size > 0 else 0.0
            containment_hcr_at_z = (intersection / mask2_size_at_mask1_z
                                    if mask2_size_at_mask1_z > 0 else 0.0)

            to_pandas['mask1'].append(mask1_id)
            to_pandas['mask2'].append(int(mask2_id))
            to_pandas['intersection'].append(intersection)
            to_pandas['mask1_size'].append(mask1_size)
            to_pandas['mask2_size'].append(mask2_size)
            to_pandas['mask2_size_at_mask1_z'].append(mask2_size_at_mask1_z)
            to_pandas['iou'].append(iou)
            to_pandas['iou_at_mask1_z'].append(iou_at_mask1_z)
            to_pandas['containment_2p'].append(containment_2p)
            to_pandas['containment_hcr_at_z'].append(containment_hcr_at_z)

    df = pd.DataFrame(to_pandas)

    if df.empty:
        # No matches found - return empty DataFrame with expected columns
        df['is_best_match'] = pd.Series(dtype=bool)
        if neighborhood_window_px:
            df['neighborhood_iou'] = pd.Series(dtype=float)
            df['neighborhood_window_px'] = pd.Series(dtype=int)
        return df

    # Greedy 1:1 best-match, ranked by the fair (z-restricted) IoU.
    df = df.sort_values('iou_at_mask1_z', ascending=False).reset_index(drop=True)

    df_best = df.drop_duplicates('mask2', keep='first').copy()
    df_best = df_best.drop_duplicates('mask1', keep='first')

    df_best['is_best_match'] = True
    df = df.merge(df_best[['mask1', 'mask2', 'is_best_match']], on=['mask1', 'mask2'], how='left')
    df['is_best_match'] = df['is_best_match'].fillna(False)

    # Neighborhood-agreement term (reported, never gated). Computed once per
    # source cell — it depends only on the cell's location + the local
    # foreground — and mapped onto every one of that mask1's rows so the
    # value survives the mask2-keyed merge into the final table regardless of
    # which row wins. None window → column absent → historical schema.
    if neighborhood_window_px:
        nbhd = {}
        for m1 in df['mask1'].unique():
            m1 = int(m1)
            inds = stack1_masks_inds[m1]
            if len(inds[0]) == 0:
                continue
            nbhd[m1] = _neighborhood_foreground_iou(
                stack1_masks, stack2_masks, inds, int(neighborhood_window_px))
        df['neighborhood_iou'] = df['mask1'].map(nbhd)
        df['neighborhood_window_px'] = int(neighborhood_window_px)

    # Sort by mask1, then by fair IoU descending for readability
    df = df.sort_values(['mask1', 'iou_at_mask1_z'], ascending=[True, False])

    return df


## All dimensions must match, all must be 1 channel, all mask files must have 1 value per mask

def convex_mask(landmarks_path: str, stack_path: str, Ydist: int, full_manifest: dict):
    '''
    Use landmarks to create two boundary surfaces and mask out everything outside them.

    Args:
        landmarks_path: Path to CSV file containing landmarks used for High res -> HCR Round 1 registration
        stack_path: Path to masks that have been fully bigwarped (2x) to align with HCR Round 1
        Ydist: Distance in microns beyond which everything will be masked out
        
    Returns:
        numpy.ndarray: Masked image stack with regions outside boundary surfaces set to 0
    '''

    # Load landmark coordinates (X,Y,Z in microns) from CSV
    df = pd.read_csv(landmarks_path, header=None)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    x_values = df[5]  # X coordinates in microns
    y_values = df[6]  # Y coordinates in microns 
    z_values = df[7]  # Z coordinates in microns
    points = np.column_stack((x_values, y_values, z_values))

    # Create upper and lower boundary surfaces by offsetting landmark points
    top_points = points.copy()
    top_points[:, 2] += Ydist  # Shift up by Ydist microns
    bottom_points = points.copy()
    bottom_points[:, 2] -= Ydist  # Shift down by Ydist microns

    # Image resolution factors to convert microns to voxels
    resolution = full_manifest['data']['HCR_confocal_imaging']['rounds'][0]['resolution']

    # Convert point coordinates from microns to voxels
    top_points[:, 0] /= resolution[0]  # Scale X 
    top_points[:, 1] /= resolution[1]  # Scale Y
    top_points[:, 2] /= resolution[2]  # Scale Z
    bottom_points[:, 0] /= resolution[0]
    bottom_points[:, 1] /= resolution[1]
    bottom_points[:, 2] /= resolution[2]

    tiff_stack = tif_imread(stack_path)
    # Handle both single channel (3D) and multichannel (4D) images
    if tiff_stack.ndim < 4:
        tiff_stack_first_channel = tiff_stack
    elif tiff_stack.ndim == 4:
        tiff_stack_first_channel = tiff_stack[:, 0, :, :]
    else:
        raise ValueError(f"Unsupported number of dimensions: {tiff_stack.ndim}")

    z_slices, height, width = tiff_stack_first_channel.shape

    def extrapolate_surface_to_image_edges(points, height, width):
        """
        Extrapolate Z values across full image using Delaunay triangulation.
        
        Args:
            points: Landmark points
            height: Image height in pixels
            width: Image width in pixels
            
        Returns:
            numpy.ndarray: Extrapolated Z values for each X,Y position
        """
        X, Y = np.meshgrid(np.arange(width), np.arange(height))
        xy_grid = np.column_stack([X.ravel(), Y.ravel()])

        interpolator = LinearNDInterpolator(points[:, :2], points[:, 2])
        z_values = interpolator(xy_grid).reshape(height, width)

        # Fill NaN values using nearest convex hull points
        nan_mask = np.isnan(z_values)
        if np.any(nan_mask):
            convex_hull = ConvexHull(points[:, :2])
            hull_points = points[convex_hull.vertices]

            for i, j in zip(*np.where(nan_mask)):
                x, y = X[i, j], Y[i, j]
                nearest_point = hull_points[np.argmin(np.linalg.norm(hull_points[:, :2] - np.array([x, y]), axis=1))]
                z_values[i, j] = nearest_point[2]

        return z_values
    
    # Generate boundary surfaces
    top_z_values = extrapolate_surface_to_image_edges(top_points, height, width)
    bottom_z_values = extrapolate_surface_to_image_edges(bottom_points, height, width)

    def blackout_above_and_below(tiff_stack, top_z_values, bottom_z_values):
        """
        Mask out regions above top surface and below bottom surface.

        Args:
            tiff_stack: Input image stack
            top_z_values: Z coordinates of upper boundary surface  
            bottom_z_values: Z coordinates of lower boundary surface

        Returns:
            numpy.ndarray: Masked image stack
        """
        volume = np.copy(tiff_stack)

        for z in range(tiff_stack.shape[0]):
            mask = (z > top_z_values) | (z < bottom_z_values)
            volume[z, mask] = 0

        return volume

    # Apply masking to image stack
    blacked_out_stack_first_channel = blackout_above_and_below(tiff_stack_first_channel, top_z_values, bottom_z_values)
    return blacked_out_stack_first_channel



def adjust_landmarks_for_plane(reference_landmarks_path, new_landmarks_path, reference_optotune, target_optotune):
    """
    Adjust landmark Z-coordinates based on optotune difference.
    Note: 1 optotune unit ≈ 1 micron in 2P imaging ~ (calibrated)
    Tissue expansion factor: 1.5x for HCR (all dimensions)
    """
    TISSUE_EXPANSION_FACTOR = 1.5
    
    # Read the file line by line to preserve exact formatting
    with open(reference_landmarks_path, 'r') as f:
        lines = f.readlines()
    
    z_offset_2p = (target_optotune - reference_optotune) * 1.0  # microns in 2P space
    z_offset_hcr = z_offset_2p * TISSUE_EXPANSION_FACTOR  # convert to HCR space
    
    # Process each line
    adjusted_lines = []
    for line in lines:
        values = line.strip().split(',')
        
        # Only adjust column 7 if it's a finite number
        if len(values) > 7 and values[7] not in ['inf', '-inf']:
            try:
                z_value = float(values[7])
                if np.isfinite(z_value):
                    values[7] = str(z_value + z_offset_hcr)
            except ValueError:
                pass  # Keep original if can't parse
        
        adjusted_lines.append(','.join(values) + '\n')
    
    # Write back
    with open(new_landmarks_path, 'w') as f:
        f.writelines(adjusted_lines)

    rprint(f"[dim]Adjusted landmarks: Z offset = {z_offset_hcr:.1f} µm[/dim]")


def print_matching_summary(df: pd.DataFrame, source_name: str, target_name: str):
    """One-glance summary: how many source cells got a unique 1:1 match, how many didn't, and
    the median overlap of the matches (the metric that drives matching)."""
    if df.empty:
        rprint(f"[yellow]  {source_name} → {target_name}: no overlapping cells found[/yellow]")
        return

    # Old CSV format without is_best_match column
    if 'is_best_match' not in df.columns:
        rprint(f"  [dim]{source_name} → {target_name}: {len(df)} matches (old format)[/dim]")
        return

    best = df[df['is_best_match'] == True]
    n_best = len(best)
    n_src = int(df['mask1'].nunique())            # source cells overlapping >=1 target (denominator)
    n_unmatched = n_src - n_best                  # overlapped something but lost the greedy 1:1
    rate = n_best / n_src if n_src else 0.0

    # iou_at_mask1_z drives is_best_match: 3D IoU with mask2's denominator restricted to mask1's
    # z-slices (built for the 2P-sheet-vs-HCR-volume case; for HCR<->HCR it's ~= the full 3D IoU).
    iou_col = 'iou_at_mask1_z' if 'iou_at_mask1_z' in best.columns else 'iou'
    med_iou = best[iou_col].median() if n_best else 0.0

    rprint(f"  [cyan]{source_name} → {target_name}:[/cyan] [b]{n_best}[/b] cells matched 1:1 "
           f"([b]{rate*100:.0f}%[/b] of {n_src} overlapping) · {n_unmatched} unmatched (lost competition)")
    line = f"    median IoU {med_iou:.3f}"
    if n_best and 'containment_2p' in best.columns:
        line += f" · median containment {best['containment_2p'].median():.3f}"
    rprint(line)


def align_masks(full_manifest: dict,
                session: dict,
                only_hcr: bool = False,
                reference_plane: str = None
                ):

    rprint("\n" + "="*80)
    if reference_plane is not None:
        rprint(f"[bold green] Align Masks for Additional Plane (Reference: Plane {reference_plane})[bold green]")
    else:
        rprint("[bold green] Align Rounds Masks[bold green]")
    rprint("="*80)

    manifest = full_manifest['data']
    params = full_manifest['params']
    round_to_rounds, reference_round, register_rounds = verify_rounds(full_manifest, parse_registered = True, 
                                                                    print_rounds = False, print_registered = False)
    
    reference_round_tiff = output_root(full_manifest) / 'HCR' / 'full_registered_stacks' / f"HCR{reference_round['round']}.tiff"
    # Cross-round matching runs in the HCR01 frame, so read the reference-aligned masks.
    reference_round_masks = output_root(full_manifest) / 'HCR' / 'cellpose_aligned' / f"HCR{reference_round['round']}_masks.tiff"

    output_folder = output_root(full_manifest) / 'MERGED' / 'aligned_masks'
    output_folder.mkdir(parents=True, exist_ok=True)

    # Collect matching results for summary
    matching_results = []

    # Skip HCR round alignment for additional planes (already done with reference plane)
    if reference_plane is None:  # Only process HCR rounds for the reference plane

        for HCR_round_to_register in register_rounds:
            round_folder_name = get_round_folder_name(HCR_round_to_register, reference_round['round'])

            mov_stack_masks = output_root(full_manifest) / 'HCR' / 'cellpose_aligned' / f"{round_folder_name}_masks.tiff"

            save_path = output_folder / f"{round_folder_name}.csv"
            if save_path.exists():
                existing_df = pd.read_csv(save_path)
                # Check if file has all required columns; regenerate if stale
                if ({'iou', 'iou_at_mask1_z', 'is_best_match'}
                        .issubset(existing_df.columns)):
                    rprint(f"  [dim]{round_folder_name}: mask alignment exists[/dim]")
                    matching_results.append((f"HCR{HCR_round_to_register}", f"HCR{reference_round['round']}", existing_df))
                    continue
                else:
                    print(f"  {round_folder_name}: stale format, regenerating...")
                    save_path.unlink()
            # calculate the matching masks and the overlap
            mask1_to_mask2_df = match_masks(mov_stack_masks, reference_round_masks)
            mask1_to_mask2_df.to_csv(save_path)
            matching_results.append((f"HCR{HCR_round_to_register}", f"HCR{reference_round['round']}", mask1_to_mask2_df))

    if only_hcr:
        # Print summary for HCR-only mode
        if matching_results:
            rprint("\n[bold cyan]IoU overlap pre-pass[/bold cyan] "
                   "[dim](stage 1; the somaprint hybrid + consensus refine this next):[/dim]")
            for source, target, df in matching_results:
                print_matching_summary(df, source, target)
        rprint("[dim]Skipping 2P masks alignment (HCR-only mode)[/dim]")
        return

    if reference_plane is None:
        rprint("\n" + "="*80)
        rprint("[bold green] HCR Rounds Registrations COMPLETE[/bold green]")
        rprint("="*80 + "\n")

    rprint("\n" + "="*80)
    rprint("[bold green] Match 2P Masks to HCR[/bold green]")
    rprint("="*80)

    # Neighborhood-IoU reporting for the cross-modal (2P→HCR) matcher. On by
    # default (window 50px); set report_neighborhood_iou:false to skip. It only
    # adds a reported column — match counts are unchanged either way.
    _reg_params = params.get('twop_to_hcr_registration', {})
    _nbhd_win = (int(_reg_params.get('neighborhood_window_px', 50))
                 if _reg_params.get('report_neighborhood_iou', True) else None)

    plane = session['functional_plane'][0]
    reg_save_path = output_root(full_manifest) / '2P' / 'registered'

    # Use automated registration output (3D volume from twop_to_hcr_registration)
    masks_2p_aligned_3d_path = reg_save_path / f'twop_plane{plane}_aligned_3d.tiff'

    if not masks_2p_aligned_3d_path.exists():
        raise FileNotFoundError(
            f"Automated registration output not found: {masks_2p_aligned_3d_path}\n"
            f"Please ensure twop_to_hcr_registration() has completed successfully."
        )

    save_path = output_folder / f"twop_plane{plane}_to_HCR{reference_round['round']}.csv"
    if save_path.exists():
        existing_df = pd.read_csv(save_path)
        if ({'iou', 'iou_at_mask1_z', 'is_best_match'}
                .issubset(existing_df.columns)):
            rprint(f"[dim]2P masks alignment for plane {plane}: exists, skipping[/dim]")
            matching_results.append((f"2P plane {plane}", f"HCR{reference_round['round']}", existing_df))
        else:
            print(f"  2P plane {plane} mapping: stale format, regenerating...")
            save_path.unlink()
            rprint(f"[cyan]Matching 2P plane {plane} masks to HCR using automated registration...[/cyan]")
            mask1_to_mask2_df = match_masks(masks_2p_aligned_3d_path, reference_round_masks,
                                            neighborhood_window_px=_nbhd_win)
            mask1_to_mask2_df.to_csv(save_path)
            rprint(f"[green]✓ Saved mask matching to {save_path}[/green]")
            matching_results.append((f"2P plane {plane}", f"HCR{reference_round['round']}", mask1_to_mask2_df))
    else:
        # Match masks using automated registration output
        rprint(f"[cyan]Matching 2P plane {plane} masks to HCR using automated registration...[/cyan]")
        mask1_to_mask2_df = match_masks(masks_2p_aligned_3d_path, reference_round_masks,
                                        neighborhood_window_px=_nbhd_win)
        mask1_to_mask2_df.to_csv(save_path)
        rprint(f"[green]✓ Saved mask matching to {save_path}[/green]")
        matching_results.append((f"2P plane {plane}", f"HCR{reference_round['round']}", mask1_to_mask2_df))

    # Print summary of all matching results
    if matching_results:
        rprint("\n[bold cyan]Mask Matching Summary:[/bold cyan]")
        for source, target, df in matching_results:
            print_matching_summary(df, source, target)

    rprint("\n" + "="*80)
    rprint("[bold green] Align 2P Masks COMPLETE[/bold green]")
    rprint("="*80 + "\n")


def align_somaprint(full_manifest: dict, session: dict, only_hcr: bool = False):
    """Run somaprint on the current 2P plane and merge its picks into the
    existing IoU mask-matching CSV.

    Somaprint is a geometric (neighbor-vector) matcher independent of IoU.
    The result is denormalized as columns on the existing per-pair CSV:
    every IoU candidate row for a given mask1 carries that mask1's
    somaprint pick (somaprint_hcr_label, best/2nd score, confident bool).
    For the rare 2P cell that somaprint confidently matches but has no
    IoU overlap with any HCR mask, one synthetic row is appended (IoU=0,
    is_best_match=False) so the pick stays queryable.

    No-ops when only_hcr=True (no 2P plane), params.somaprint.enabled is
    False, or the target CSV already has the somaprint columns.
    """
    from . import somaprint as sp_lib

    if only_hcr:
        return

    sp_params = sp_lib.get_params(full_manifest)
    if not sp_params.get('enabled', True):
        rprint("[dim]Somaprint disabled (params.somaprint.enabled = false); skipping[/dim]")
        return

    _, reference_round, _ = verify_rounds(full_manifest, parse_registered=True,
                                          print_rounds=False, print_registered=False)
    plane = session['functional_plane'][0]
    hcr_round = reference_round['round']
    csv_path = (output_root(full_manifest) / 'MERGED' / 'aligned_masks'
                / f"twop_plane{plane}_to_HCR{hcr_round}.csv")

    if not csv_path.exists():
        rprint(f"[yellow]Somaprint: IoU CSV missing at {csv_path}; "
               f"align_masks must run first.[/yellow]")
        return

    df = pd.read_csv(csv_path)
    # Strip any pre-existing leading index column (legacy CSVs saved without
    # index=False); column-name access works regardless but accumulating
    # 'Unnamed: 0' columns on every re-save is ugly.
    df = df.loc[:, ~df.columns.str.match(r'^Unnamed: \d+$')]

    if 'somaprint_confident' in df.columns:
        rprint(f"[dim]Somaprint columns already present in {csv_path.name}; skipping[/dim]")
        return

    rprint("\n" + "="*80)
    rprint(f"[bold green] Somaprint Matching (2P plane {plane} -> HCR{hcr_round})[/bold green]")
    rprint("="*80)

    matches = sp_lib.run_for_plane(full_manifest, session, hcr_round)

    # Denormalize per-mask1: every row of a given mask1 carries that 2P
    # cell's somaprint pick. Cells somaprint didn't return at all stay NaN.
    hcr_label_by_m1 = {m1: v[0] for m1, v in matches.items()}
    best_by_m1 = {m1: v[1] for m1, v in matches.items()}
    second_by_m1 = {m1: v[2] for m1, v in matches.items()}
    conf_by_m1 = {m1: v[3] for m1, v in matches.items()}

    df['somaprint_hcr_label'] = df['mask1'].map(hcr_label_by_m1).astype('Int64')
    df['somaprint_best_score'] = df['mask1'].map(best_by_m1)
    df['somaprint_second_score'] = df['mask1'].map(second_by_m1)
    df['somaprint_confident'] = df['mask1'].map(conf_by_m1).fillna(False).astype(bool)

    # Edge case: 2P cells somaprint matched that don't appear in the IoU
    # CSV at all (zero overlap with every HCR mask). Append synthetic rows
    # so the pick propagates into the merged feature table. Typically 0-5
    # cells per plane on JS078.
    existing_mask1 = set(df['mask1'].astype(int).unique()) if not df.empty else set()
    orphan_m1s = [int(m1) for m1 in matches.keys() if int(m1) not in existing_mask1]

    if orphan_m1s:
        # 3D voxel counts in the registered volumes — match match_masks'
        # definition of mask1_size / mask2_size so the orphan rows are
        # informationally consistent with the rest of the CSV.
        twop_3d_path = (output_root(full_manifest) / '2P' / 'registered'
                        / f'twop_plane{plane}_aligned_3d.tiff')
        hcr_masks_path = (output_root(full_manifest) / 'HCR' / 'cellpose_aligned'
                          / f'HCR{hcr_round}_masks.tiff')
        twop_3d = tif_imread(str(twop_3d_path))
        hcr_3d = tif_imread(str(hcr_masks_path))
        tw_ids, tw_counts = np.unique(twop_3d, return_counts=True)
        hc_ids, hc_counts = np.unique(hcr_3d, return_counts=True)
        twop_size_by_lbl = dict(zip(map(int, tw_ids), map(int, tw_counts)))
        hcr_size_by_lbl = dict(zip(map(int, hc_ids), map(int, hc_counts)))

        orphan_rows = []
        for m1 in orphan_m1s:
            hcr_lbl, bs, ss, conf = matches[m1]
            orphan_rows.append({
                'mask1': int(m1),
                'mask2': int(hcr_lbl),
                'intersection': 0,
                'mask1_size': twop_size_by_lbl.get(int(m1), 0),
                'mask2_size': hcr_size_by_lbl.get(int(hcr_lbl), 0),
                'mask2_size_at_mask1_z': 0,
                'iou': 0.0,
                'iou_at_mask1_z': 0.0,
                'containment_2p': 0.0,
                'containment_hcr_at_z': 0.0,
                'is_best_match': False,
                'somaprint_hcr_label': int(hcr_lbl),
                'somaprint_best_score': float(bs) if np.isfinite(bs) else np.nan,
                'somaprint_second_score': float(ss) if np.isfinite(ss) else np.nan,
                'somaprint_confident': bool(conf),
            })
        df = pd.concat([df, pd.DataFrame(orphan_rows)], ignore_index=True)
        df['somaprint_hcr_label'] = df['somaprint_hcr_label'].astype('Int64')

    df = df.sort_values(['mask1', 'iou_at_mask1_z'], ascending=[True, False]).reset_index(drop=True)
    df.to_csv(csv_path, index=False)

    n_conf = int(df['somaprint_confident'].sum())
    extra = f" (+{len(orphan_m1s)} non-IoU picks)" if orphan_m1s else ""
    rprint(f"[green]✓ Somaprint integrated into {csv_path.name}: "
           f"{n_conf} confident matches{extra}[/green]")


def align_somaprint_hcr(full_manifest: dict, session: dict, only_hcr: bool = False):
    """Run the two-stage HCR round-to-round matcher and merge its picks into each
    round's existing IoU mask-matching CSV.

    The matcher (best-plane IoU overlap → pinned 2d_plane soma repair; see
    src/somaprint_hcr.py) re-pairs cells across HCR rounds in the HCR{reference}
    frame, recovering partners the plain IoU best-match misses in locally
    mis-registered patches. Its pick is denormalized as columns on each
    {round_folder}.csv (parallel to align_somaprint's 2P columns):

      somaprint_hcr_label    reference-round label the hybrid matcher picked
      somaprint_best_score   soma best score (NaN for stage-1 overlap anchors)
      somaprint_second_score soma within-plane null (separability companion)
      somaprint_confident    True for every hybrid match
      somaprint_source       'overlap' (stage-1 IoU) | 'somaprint' (stage-2 repair)

    For the rare moving cell the matcher picks but which has no IoU overlap with
    any reference mask (a recovered cell), one synthetic row is appended (IoU=0,
    is_best_match=False) so the pick propagates into the merged table.

    Runs in both full and only_hcr modes (HCR↔HCR has no 2P dependency). No-ops
    when params.somaprint_hcr.enabled is False, there are no non-reference rounds,
    or a CSV already carries the somaprint columns.
    """
    from . import somaprint_hcr as sph_lib
    from .registrations_utils import resolve_hcr_resolution

    params = sph_lib.get_params(full_manifest)
    if not params.get('enabled', True):
        rprint("[dim]Somaprint-HCR disabled (params.somaprint_hcr.enabled = false); skipping[/dim]")
        return

    _, reference_round, register_rounds = verify_rounds(
        full_manifest, parse_registered=True, print_rounds=False, print_registered=False)
    if not register_rounds:
        return

    ref = reference_round['round']
    res_xyz = resolve_hcr_resolution(reference_round['image_path'],
                                     reference_round.get('resolution'))

    aligned_dir = output_root(full_manifest) / 'MERGED' / 'aligned_masks'
    cellpose_aligned = output_root(full_manifest) / 'HCR' / 'cellpose_aligned'
    ref_masks_path = cellpose_aligned / f"HCR{ref}_masks.tiff"

    rprint("\n" + "="*80)
    rprint(f"[bold green] Somaprint-HCR Matching (HCR rounds -> HCR{ref})[/bold green]")
    rprint("="*80)

    for HCR_round in register_rounds:
        round_folder = get_round_folder_name(HCR_round, ref)
        csv_path = aligned_dir / f"{round_folder}.csv"
        if not csv_path.exists():
            rprint(f"  [yellow]{round_folder}: IoU CSV missing; align_masks must run first[/yellow]")
            continue

        df = pd.read_csv(csv_path)
        df = df.loc[:, ~df.columns.str.match(r'^Unnamed: \d+$')]
        if 'somaprint_confident' in df.columns:
            rprint(f"  [dim]{round_folder}: somaprint columns present; skipping[/dim]")
            continue

        mov_masks_path = cellpose_aligned / f"{round_folder}_masks.tiff"
        sph_stats = {}
        matches = sph_lib.run_for_round(mov_masks_path, ref_masks_path, res_xyz,
                                        params, round_label=round_folder, stats=sph_stats)
        if not matches:
            continue

        # Denormalize per-mask1 (moving cell). Cells the matcher didn't return stay NaN.
        df['somaprint_hcr_label'] = df['mask1'].map({m1: v[0] for m1, v in matches.items()}).astype('Int64')
        df['somaprint_best_score'] = df['mask1'].map({m1: v[1] for m1, v in matches.items()})
        df['somaprint_second_score'] = df['mask1'].map({m1: v[2] for m1, v in matches.items()})
        df['somaprint_confident'] = df['mask1'].map({m1: v[3] for m1, v in matches.items()}).fillna(False).astype(bool)
        df['somaprint_source'] = df['mask1'].map({m1: v[4] for m1, v in matches.items()})

        # Recovered cells with no IoU overlap row at all — append synthetic rows so
        # the pick carries into the merged table (mirror align_somaprint's orphans).
        existing_mask1 = set(df['mask1'].astype(int).unique()) if not df.empty else set()
        orphan_m1s = [int(m1) for m1 in matches.keys() if int(m1) not in existing_mask1]
        if orphan_m1s:
            mov_3d = tif_imread(str(mov_masks_path))
            ref_3d = tif_imread(str(ref_masks_path))
            mv_ids, mv_counts = np.unique(mov_3d, return_counts=True)
            rf_ids, rf_counts = np.unique(ref_3d, return_counts=True)
            mov_size = dict(zip(map(int, mv_ids), map(int, mv_counts)))
            ref_size = dict(zip(map(int, rf_ids), map(int, rf_counts)))
            orphan_rows = []
            for m1 in orphan_m1s:
                ref_lbl, bs, ss, conf, src = matches[m1]
                orphan_rows.append({
                    'mask1': int(m1), 'mask2': int(ref_lbl),
                    'intersection': 0,
                    'mask1_size': mov_size.get(int(m1), 0),
                    'mask2_size': ref_size.get(int(ref_lbl), 0),
                    'mask2_size_at_mask1_z': 0,
                    'iou': 0.0, 'iou_at_mask1_z': 0.0,
                    'containment_2p': 0.0, 'containment_hcr_at_z': 0.0,
                    'is_best_match': False,
                    'somaprint_hcr_label': int(ref_lbl),
                    'somaprint_best_score': float(bs) if np.isfinite(bs) else np.nan,
                    'somaprint_second_score': float(ss) if np.isfinite(ss) else np.nan,
                    'somaprint_confident': bool(conf),
                    'somaprint_source': src,
                })
            df = pd.concat([df, pd.DataFrame(orphan_rows)], ignore_index=True)
            df['somaprint_hcr_label'] = df['somaprint_hcr_label'].astype('Int64')

        df = df.sort_values(['mask1', 'iou_at_mask1_z'],
                            ascending=[True, False]).reset_index(drop=True)
        df.to_csv(csv_path, index=False)

        n_repair = sum(1 for v in matches.values() if v[4] == 'somaprint')
        extra = f", +{len(orphan_m1s)} non-IoU recoveries" if orphan_m1s else ""
        n_mov = sph_stats.get('n_mov', len(matches))      # total moving cells the matcher saw
        n_unmatched = max(0, n_mov - len(matches))
        rate = len(matches) / n_mov if n_mov else 0.0
        rprint(f"  [green]✓ {round_folder}:[/green] [b]{len(matches)}/{n_mov}[/b] cells matched "
               f"([b]{rate*100:.0f}%[/b]) · {n_unmatched} unmatched · "
               f"{len(matches) - n_repair} overlap + {n_repair} soma-repair{extra}")


def merge_masks(full_manifest: dict, session: dict, only_hcr: bool = False):
    rprint("\n" + "="*80)
    rprint("[bold green] Match Aligned Masks[bold green]")
    rprint("="*80)

    manifest = full_manifest['data']
    params = full_manifest['params']
    if only_hcr:
        plane = 0
    else:
        plane = session['functional_plane'][0]

    round_to_rounds, reference_round, register_rounds = verify_rounds(full_manifest, parse_registered = True, 
                                                                    print_rounds = False, print_registered = False)
    HCR_intensities_path = output_root(full_manifest) / 'HCR' / 'extract_intensities'
    HCR_mapping_path = output_root(full_manifest) / 'MERGED' / 'aligned_masks'
    merged_table_path = output_root(full_manifest) / 'MERGED' / 'aligned_extracted_features'
    merged_table_path.mkdir(parents=True, exist_ok=True)
    
    # Check if median filter was applied
    intensity_config = get_intensity_extraction_config(params)
    median_filter = intensity_config.get('stack_median_filter')
    median_filter_label = None
    if median_filter:
        median_filter_label = f'medflt_{median_filter[0]}x{median_filter[1]}x{median_filter[2]}'

    # get available features to create merged table for
    neuropil_radius = intensity_config['neuropil_radius']
    neuropil_boundary = intensity_config['neuropil_boundary']
    neuropil_pooling = intensity_config['neuropil_pooling']
    
    # Build complete list of ALL features (original + median filtered)
    features_to_extract = ['mean']
    
    # Add median filtered mean if available
    if median_filter_label:
        features_to_extract.append(f'mean_{median_filter_label}')
    
    # Add ALL neuropil features (both regular and median filtered versions)
    for pooling_method in neuropil_pooling:
        if pooling_method == 'mean':
            base_feature = f'neuropil_mean_nr{neuropil_radius}_nb_{neuropil_boundary}'
            features_to_extract.append(base_feature)
            # Add median filtered version
            if median_filter_label:
                features_to_extract.append(f'{median_filter_label}_{base_feature}')
                
        elif pooling_method == 'median':
            base_feature = f'neuropil_median_nr{neuropil_radius}_nb_{neuropil_boundary}'
            features_to_extract.append(base_feature)
            # Add median filtered version
            if median_filter_label:
                features_to_extract.append(f'{median_filter_label}_{base_feature}')
                
        elif pooling_method.startswith('percentile-'):
            percentile = int(pooling_method.split('-')[1])
            base_feature = f'neuropil_{percentile}pct_nr{neuropil_radius}_nb_{neuropil_boundary}'
            features_to_extract.append(base_feature)
            # Add median filtered version
            if median_filter_label:
                features_to_extract.append(f'{median_filter_label}_{base_feature}')
        else:
            raise ValueError(f"Unsupported pooling method: {pooling_method}")

    rprint(f"[bold]Building merged tables[/bold] ({len(features_to_extract)} features)")

    # Metrics propagated from each mask-matching CSV into the merged table.
    # Source of these columns is match_masks() in this file — see its
    # docstring for the definitions. The merged table uses at-plane / at-z
    # semantics throughout (plane-restricted for 2P↔HCR, ≈full-multiplane
    # for HCR↔HCR), so the legacy 3D `iou`, the per-pair `mask2_size*`
    # columns, and the derivable `intersection` are dropped on the way in;
    # the survivors are the four below.
    METRIC_COLUMNS = (
        'iou_at_mask1_z',
        'containment_2p', 'containment_hcr_at_z',
        'mask1_size',
    )

    def _build_lookup_dicts(matching_df):
        """Return (mask2 → mask1) dict plus a {metric: {mask2 → value}} map."""
        mapping = {m2: m1 for m1, m2 in matching_df[['mask1', 'mask2']].values}
        metrics = {}
        for col in METRIC_COLUMNS:
            if col in matching_df.columns:
                metrics[col] = dict(zip(matching_df['mask2'].values,
                                        matching_df[col].values))
            else:
                # Legacy CSV that predates a column — propagate None so the
                # merged table is schema-stable; users see missing data
                # rather than a hard failure.
                metrics[col] = {}
        # Neighborhood IoU is cross-modal-only (2P→HCR) and lives outside the
        # shared METRIC_COLUMNS so the HCR-round schema is untouched. Extract
        # it here where the 2P table is built; absent on legacy/HCR CSVs.
        if 'neighborhood_iou' in matching_df.columns:
            metrics['neighborhood_iou'] = dict(zip(matching_df['mask2'].values,
                                                   matching_df['neighborhood_iou'].values))
        return mapping, metrics

    def _build_somaprint_lookups(matching_df):
        """Return (somaprint_hcr_label → mask1) plus {metric: {label → value}}
        for every somaprint pick (confident and not). Keyed on the somaprint
        pick (parallel to the IoU lookup's mask2 key) so each HCR cell row
        in the reference intensities table can fetch its somaprint partner;
        confidence is exposed as a metric so downstream code can still
        filter on it.

        Somaprint denormalizes per-mask1 across IoU candidate rows; we
        first reduce to one row per 2P cell, then dedupe on the HCR-label
        key with a score-sorted tiebreak so non-confident collisions
        resolve deterministically. Returns empty dicts on legacy CSVs
        that predate the somaprint columns.
        """
        if not {'somaprint_confident', 'somaprint_hcr_label'}.issubset(matching_df.columns):
            return {}, {}
        soma = (matching_df.dropna(subset=['somaprint_hcr_label'])
                .drop_duplicates('mask1')
                .sort_values('somaprint_best_score', ascending=False)
                .drop_duplicates('somaprint_hcr_label'))
        if soma.empty:
            return {}, {}
        soma_lbls = soma['somaprint_hcr_label'].astype(int)
        mapping = dict(zip(soma_lbls, soma['mask1'].astype(int)))
        # mask1_size on a somaprint row is the 2P partner's size
        # (match_masks fills it on IoU rows; align_somaprint fills it on
        # orphan rows from a fresh tiff read — both definitions are 3D
        # voxel counts in the registered 2P volume, which equals the 2P
        # cell's 2D pixel count by sheet-into-3D construction).
        metrics = {
            'somaprint_best_score': dict(zip(soma_lbls, soma['somaprint_best_score'].astype(float))),
            'somaprint_second_score': dict(zip(soma_lbls, soma['somaprint_second_score'].astype(float))),
            'somaprint_mask_size': dict(zip(soma_lbls, soma['mask1_size'].astype(int))),
            'somaprint_confident': dict(zip(soma_lbls, soma['somaprint_confident'].astype(bool))),
        }
        return mapping, metrics

    def _build_consensus_lookups(matching_df):
        """HCR↔HCR partner per reference cell, consensus of the two matchers: the
        hybrid somaprint pick where present, else the IoU best-match. Returns

          mapping        {ref_label → mov_label}   (drives round_{R}_mask / gene join)
          metrics        {METRIC_COLUMN → {ref → value}}  IoU/containment/size of the
                         CHOSEN pair (not the IoU best-match row — so a soma-recovered
                         pair carries its own low overlap, the recovery signal)
          extras         {match_source / somaprint_best_score / somaprint_second_score
                          → {ref → value}}; empty when the CSV predates somaprint.

        Falls back to pure IoU best-match (empty extras) on legacy CSVs, so disabling
        params.somaprint_hcr reproduces the historical table exactly.
        """
        if 'is_best_match' in matching_df.columns:
            iou_best = matching_df[matching_df['is_best_match'] == True]
        else:
            iou_best = matching_df
        iou_map = {int(m2): int(m1) for m1, m2 in iou_best[['mask1', 'mask2']].values}

        has_soma = {'somaprint_confident', 'somaprint_hcr_label'}.issubset(matching_df.columns)
        extras = {}
        if not has_soma:
            mapping = dict(iou_map)
        else:
            # one row per moving cell, then dedupe to 1:1 on the reference label
            # (highest soma score wins; overlap anchors carry NaN and lose ties).
            soma = (matching_df.dropna(subset=['somaprint_hcr_label'])
                    .drop_duplicates('mask1')
                    .sort_values('somaprint_best_score', ascending=False)
                    .drop_duplicates('somaprint_hcr_label'))
            soma_keys = soma['somaprint_hcr_label'].astype(int)
            soma_map = dict(zip(soma_keys, soma['mask1'].astype(int)))
            soma_bs = dict(zip(soma_keys, soma['somaprint_best_score'].astype(float)))
            soma_ss = dict(zip(soma_keys, soma['somaprint_second_score'].astype(float)))
            soma_src = (dict(zip(soma_keys, soma['somaprint_source']))
                        if 'somaprint_source' in soma.columns else {})
            mapping, src_map = {}, {}
            for ref in set(iou_map) | set(soma_map):
                if ref in soma_map:
                    mapping[ref] = soma_map[ref]
                    src_map[ref] = soma_src.get(ref, 'somaprint')
                else:
                    mapping[ref] = iou_map[ref]
                    src_map[ref] = 'iou'
            extras = {'match_source': src_map,
                      'somaprint_best_score': soma_bs,
                      'somaprint_second_score': soma_ss}

        # IoU/containment/size of the CHOSEN (mov, ref) pair, looked up per-pair so a
        # consensus that differs from the IoU best-match still reports its own overlap.
        pair = list(zip(matching_df['mask1'].astype(int), matching_df['mask2'].astype(int)))
        metrics = {}
        for col in METRIC_COLUMNS:
            if col in matching_df.columns:
                pair_val = dict(zip(pair, matching_df[col].values))
                metrics[col] = {ref: pair_val.get((mov, ref)) for ref, mov in mapping.items()}
            else:
                metrics[col] = {}
        return mapping, metrics, extras

    # ========== PRE-LOAD ALL DATA ONCE (optimization: avoid reloading per feature) ==========

    # Pre-load 2P mapping (feature-independent)
    if only_hcr:
        twoP_mapping_dict = {}
        twoP_metrics_dict = {col: {} for col in METRIC_COLUMNS}
        twoP_soma_mapping_dict = {}
        twoP_soma_metrics_dict = {}
    else:
        twop_mapping_path = HCR_mapping_path / f"twop_plane{plane}_to_HCR{reference_round['round']}.csv"
        try:
            towP_to_reference_mapping = pd.read_csv(twop_mapping_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"2P-to-HCR mapping file not found: {twop_mapping_path}\n"
                f"Please ensure align_masks() has completed successfully for plane {plane}."
            )
        # Somaprint columns denormalize per-mask1; build lookups from the
        # unfiltered df *before* narrowing to IoU best-match rows.
        twoP_soma_mapping_dict, twoP_soma_metrics_dict = _build_somaprint_lookups(towP_to_reference_mapping)

        if 'is_best_match' in towP_to_reference_mapping.columns:
            towP_to_reference_mapping = towP_to_reference_mapping[towP_to_reference_mapping['is_best_match'] == True]
        if towP_to_reference_mapping.empty:
            rprint(f"[yellow]Warning: No 2P-to-HCR matches found for plane {plane}. All cells will have None for 2P mapping.[/yellow]")
            twoP_mapping_dict = {}
            twoP_metrics_dict = {col: {} for col in METRIC_COLUMNS}
        else:
            twoP_mapping_dict, twoP_metrics_dict = _build_lookup_dicts(towP_to_reference_mapping)

    # Pre-load reference round intensities (full DataFrame, pivot per feature)
    ref_intensities_path = HCR_intensities_path / f"HCR{reference_round['round']}_probs_intensities.pkl"
    try:
        reference_round_intensities = pd.read_pickle(ref_intensities_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Reference round intensities not found: {ref_intensities_path}\n"
            f"Please ensure extract_probe_intensity() has completed for HCR round {reference_round['round']}."
        )

    # Pre-load HCR round mappings and intensities (feature-independent)
    HCR_rounds_names = register_rounds
    HCR_round_mapping_dict = []
    HCR_round_metrics_dict = []
    HCR_round_extra_dict = []
    preloaded_round_intensities = {}

    for HCR_round_to_register in register_rounds:
        round_file_name = f"HCR{HCR_round_to_register}_to_HCR{reference_round['round']}"

        # Load mapping
        round_mapping_file_path = HCR_mapping_path / f"{round_file_name}.csv"
        try:
            round_mapping_df = pd.read_csv(round_mapping_file_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"HCR round mapping file not found: {round_mapping_file_path}\n"
                f"Please ensure align_masks() has completed for HCR round {HCR_round_to_register}."
            )
        # Consensus of the IoU and somaprint matchers: round_{R}_mask follows the
        # hybrid pick where somaprint matched, else the IoU best-match. Built from
        # the FULL df (somaprint rows include is_best_match=False recoveries).
        round_mapping, round_metrics, round_extra = _build_consensus_lookups(round_mapping_df)
        HCR_round_mapping_dict.append(round_mapping)
        HCR_round_metrics_dict.append(round_metrics)
        HCR_round_extra_dict.append(round_extra)

        # Load intensities
        round_intensities_path = HCR_intensities_path / f"{round_file_name}_probs_intensities.pkl"
        try:
            preloaded_round_intensities[HCR_round_to_register] = pd.read_pickle(round_intensities_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"HCR round intensities not found: {round_intensities_path}\n"
                f"Please ensure extract_probe_intensity() has completed for HCR round {HCR_round_to_register}."
            )

    # ========== PROCESS EACH FEATURE (now only pivots, no file I/O) ==========

    skipped_features = []
    for feature in tqdm(features_to_extract, desc="Merging features"):
        merged_table_file_path = merged_table_path / f'full_table_{feature}_twop_plane{plane}.pkl'
        if merged_table_file_path.exists():
            skipped_features.append(feature)
            continue

        # Pivot reference round for this feature
        reference_round_intensities_pivot = pd.pivot(reference_round_intensities, index='mask_id', columns=['channel_name'], values=[feature]).reset_index()
        reference_round_intensities_pivot.rename(columns={'mask_id':'mask_id_main'},inplace=True)

        # Pivot each HCR round for this feature
        HCR_rounds_intensities_pivot = []
        for HCR_round_to_register in register_rounds:
            HCR_round_intensities = preloaded_round_intensities[HCR_round_to_register]
            if feature not in HCR_round_intensities.columns:
                rprint(f"[yellow]Feature '{feature}' not found in round {HCR_round_to_register}[/yellow]")
                HCR_rounds_intensities_pivot.append(pd.DataFrame())
                continue
            HCR_rounds_intensities_pivot.append(pd.pivot(HCR_round_intensities, index='mask_id', columns=['channel_name'], values=[feature]).reset_index())

        ####       ####
        ###  MATCH  ###
        ####       ####
        # Column naming in the merged table. All size / IoU / containment
        # values are plane-restricted (at-plane for 2P↔HCR; ≈full-multiplane
        # for HCR↔HCR). Subject of `containment` is named first
        # (X_containment = fraction of X inside its partner). Unprefixed
        # twoP_* = IoU side; twoP_somaprint_* = soma side.
        #
        # 2P↔HCR (IoU best-match):
        #   twoP_mask, twoP_iou, twoP_containment, HCR_containment,
        #   twoP_mask_size
        # 2P↔HCR (somaprint, geometric matcher):
        #   twoP_somaprint_mask, twoP_somaprint_best_score,
        #   twoP_somaprint_second_score, twoP_somaprint_mask_size
        # HCR-level (matcher-independent):
        #   HCR_mask_size
        # HCR round-to-round, one set per non-reference round R:
        #   round_{R}_mask, round_{R}_iou, round_{R}_containment,
        #   main_containment_round_{R}, round_{R}_mask_size
        #
        # See match_masks() and align_somaprint() docstrings for the
        # precise definition of each metric. `mask_id_main` always refers
        # to mask2 in the matching CSVs (the reference round = stack2).
        TWOP_COL_RENAME = {
            'iou_at_mask1_z':         'twoP_iou',
            'containment_2p':         'twoP_containment',
            'containment_hcr_at_z':   'HCR_containment',
            'mask1_size':             'twoP_mask_size',
            # Reported neighborhood-agreement term (2P→HCR only; None on legacy
            # CSVs → column of None, schema-stable). User thresholds on this.
            'neighborhood_iou':       'twoP_neighborhood_iou',
        }

        def _round_col(metric, round_name):
            return {
                'iou_at_mask1_z':         f'round_{round_name}_iou',
                'containment_2p':         f'round_{round_name}_containment',
                'containment_hcr_at_z':   f'main_containment_round_{round_name}',
                'mask1_size':             f'round_{round_name}_mask_size',
            }[metric]

        def _lookup_column(mask_ids, dct):
            """Map an iterable of mask_id_main values through a dict (None if missing)."""
            return [dct.get(i) for i in mask_ids]

        ref_mask_ids = reference_round_intensities_pivot.mask_id_main

        # HCR cell size at the warped 2P plane — always populated (matcher-
        # independent). Drives downstream sliver filtering. In only_hcr mode
        # there is no 2P plane to project against; fall back to full 3D
        # voxel counts so the column has stable semantics within the table.
        from . import somaprint as sp_lib
        if only_hcr:
            hcr_size_lookup = sp_lib.compute_full_hcr_sizes(
                full_manifest, reference_round['round'])
        else:
            hcr_size_lookup = sp_lib.compute_plane_projected_hcr_sizes(
                full_manifest, session, reference_round['round'])
        reference_round_intensities_pivot['HCR_mask_size'] = _lookup_column(
            ref_mask_ids, hcr_size_lookup)

        # 2P columns — IoU best-match path
        reference_round_intensities_pivot['twoP_mask'] = _lookup_column(ref_mask_ids, twoP_mapping_dict)
        for src_col, dest_col in TWOP_COL_RENAME.items():
            reference_round_intensities_pivot[dest_col] = _lookup_column(
                ref_mask_ids, twoP_metrics_dict.get(src_col, {}))

        # 2P columns — somaprint side (geometric matcher, independent of
        # IoU). For each HCR cell, twoP_somaprint_mask is the 2P cell that
        # somaprint picked here, confident or not (None if no somaprint
        # match landed on this HCR cell). twoP_somaprint_confident exposes
        # the confidence flag so downstream code can filter; scores are
        # populated whether or not the pick was confident. twoP_mask and
        # twoP_somaprint_mask can disagree; downstream consumers pick
        # which matcher to filter on.
        reference_round_intensities_pivot['twoP_somaprint_mask'] = _lookup_column(
            ref_mask_ids, twoP_soma_mapping_dict)
        reference_round_intensities_pivot['twoP_somaprint_best_score'] = _lookup_column(
            ref_mask_ids, twoP_soma_metrics_dict.get('somaprint_best_score', {}))
        reference_round_intensities_pivot['twoP_somaprint_second_score'] = _lookup_column(
            ref_mask_ids, twoP_soma_metrics_dict.get('somaprint_second_score', {}))
        reference_round_intensities_pivot['twoP_somaprint_mask_size'] = _lookup_column(
            ref_mask_ids, twoP_soma_metrics_dict.get('somaprint_mask_size', {}))
        reference_round_intensities_pivot['twoP_somaprint_confident'] = _lookup_column(
            ref_mask_ids, twoP_soma_metrics_dict.get('somaprint_confident', {}))

        # HCR round columns
        for j in range(len(HCR_round_mapping_dict)):
            HCR_main_2_HCR_round = HCR_round_mapping_dict[j]
            round_metrics = HCR_round_metrics_dict[j]
            round_name = HCR_rounds_names[j]
            reference_round_intensities_pivot[f'round_{round_name}_mask'] = _lookup_column(
                ref_mask_ids, HCR_main_2_HCR_round)
            for src_col in METRIC_COLUMNS:
                reference_round_intensities_pivot[_round_col(src_col, round_name)] = _lookup_column(
                    ref_mask_ids, round_metrics.get(src_col, {}))
            # Hybrid-matcher provenance (only when somaprint_hcr ran for this round):
            # which matcher produced round_{R}_mask and the soma scores behind a repair.
            extra = HCR_round_extra_dict[j]
            if extra:
                reference_round_intensities_pivot[f'round_{round_name}_match_source'] = _lookup_column(
                    ref_mask_ids, extra.get('match_source', {}))
                reference_round_intensities_pivot[f'round_{round_name}_somaprint_score'] = _lookup_column(
                    ref_mask_ids, extra.get('somaprint_best_score', {}))
                reference_round_intensities_pivot[f'round_{round_name}_somaprint_second_score'] = _lookup_column(
                    ref_mask_ids, extra.get('somaprint_second_score', {}))

        ####       ####
        ###  MERGE  ###
        ####       ####
        HCR_main_pivot_merged = reference_round_intensities_pivot.copy().reset_index()

        for j in range(len(HCR_round_mapping_dict)):
            round_mask_name = f'round_{HCR_rounds_names[j]}_mask'

            # Skip merge if the intensities DataFrame is empty (feature was missing for this round)
            if HCR_rounds_intensities_pivot[j].empty:
                continue

            HCR_main_pivot_merged = pd.merge(HCR_main_pivot_merged,
                                        HCR_rounds_intensities_pivot[j],
                                        left_on=round_mask_name,
                                        right_on='mask_id',
                                        suffixes=['',f'_round_{HCR_rounds_names[j]}'],
                                        how='left').drop(columns=['mask_id'])

        # Add 'plane' column for clarity (each plane gets separate file)
        HCR_main_pivot_merged['plane'] = plane
        HCR_main_pivot_merged.to_pickle(merged_table_file_path)

    if skipped_features:
        rprint(f"[dim]Skipped {len(skipped_features)} existing features[/dim]")

    rprint("\n" + "="*80)
    rprint("[bold green] Match Aligned Masks COMPLETE[/bold green]")
    rprint("="*80 + "\n")


def print_match_summary(full_manifest: dict, all_planes: list):
    """One-line per-plane summary of 2P→HCR match counts.

    Reports somaprint matches (primary) and IoU matches (parenthetical),
    then how many of the somaprint matches carry through to each non-
    reference HCR round. No-op for planes whose merged table or 2P
    seg.npy is missing.
    """
    _, reference_round, register_rounds = verify_rounds(
        full_manifest, parse_registered=True,
        print_rounds=False, print_registered=False, func='match-summary')
    ref = reference_round['round']

    merged_dir = output_root(full_manifest) / 'MERGED' / 'aligned_extracted_features'
    cellpose_dir = output_root(full_manifest) / '2P' / 'cellpose'

    rprint("\n[bold cyan]2P→HCR match summary[/bold cyan]")
    for plane in all_planes:
        seg_path = cellpose_dir / f'lowres_meanImg_C0_plane{plane}_seg.npy'
        merged_files = sorted(merged_dir.glob(f'full_table_*_twop_plane{plane}.pkl'))
        if not merged_files:
            rprint(f"  [yellow]Plane {plane}: merged table not found[/yellow]")
            continue

        if seg_path.exists():
            stats = np.load(seg_path, allow_pickle=True).item()
            total_2p = int(len(np.unique(stats['masks'])) - 1)
            denom = str(total_2p)
        else:
            denom = "?"

        df = pd.read_pickle(merged_files[0])
        # twoP_somaprint_mask is populated for confident AND non-confident
        # picks (so non-confident scores are still visible in the table);
        # the summary count is the confident subset, matching the historical
        # meaning of this line.
        if 'twoP_somaprint_mask' in df.columns:
            if 'twoP_somaprint_confident' in df.columns:
                soma_matched = df['twoP_somaprint_mask'].notna() & df['twoP_somaprint_confident'].fillna(False).astype(bool)
            else:
                soma_matched = df['twoP_somaprint_mask'].notna()
        else:
            soma_matched = None
        n_soma = int(soma_matched.sum()) if soma_matched is not None else 0
        n_iou = int(df['twoP_mask'].notna().sum()) if 'twoP_mask' in df.columns else 0

        line = (f"  Plane {plane}: {n_soma}/{denom} 2P masks matched HCR{ref} "
                f"(somaprint; IoU: {n_iou})")
        if soma_matched is not None:
            for r in register_rounds:
                col = f'round_{r}_mask'
                if col in df.columns:
                    in_round = int((soma_matched & df[col].notna()).sum())
                    line += f" → HCR{r}: {in_round}"
        rprint(line)