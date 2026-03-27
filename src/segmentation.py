from collections import defaultdict
from pathlib import Path
import shutil
from cellpose import models
from cellpose import io
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.ndimage import median_filter as scipy_median_filter
import pickle as pkl
from scipy.sparse import csr_matrix
from tifffile import imread as tif_imread
from tifffile import imwrite as tif_imsave
from tqdm.auto import tqdm
from rich import print as rprint
from scipy.spatial import ConvexHull
from scipy.interpolate import LinearNDInterpolator
from .registrations import verify_rounds
from .meta import get_intensity_extraction_config, get_round_folder_name



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
    """Get or create a cached Cellpose model instance."""
    cache_key = (model_path, gpu)
    if cache_key not in _cellpose_model_cache:
        _cellpose_model_cache[cache_key] = models.CellposeModel(
            pretrained_model=model_path,
            gpu=gpu
        )
    return _cellpose_model_cache[cache_key]

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

    def eval(self, raw_image):
        if self.model is None:
            # Use cached model (optimization: reuse across pipeline)
            self.model = _get_cached_cellpose_model(
                self.params['HCR_cellpose']['model_path'],
                self.params['HCR_cellpose']['gpu']
            )

        return self.model.eval(
            raw_image,
            channels=[0,0],
            diameter=self.params['HCR_cellpose']['diameter'],
            flow_threshold=self.params['HCR_cellpose']['flow_threshold'],
            cellprob_threshold=self.params['HCR_cellpose']['cellprob_threshold'],
            do_3D=True,
        )

def run_cellpose(full_manifest):
    manifest = full_manifest['data']
    params = full_manifest['params']

    round_to_rounds, reference_round, register_rounds = verify_rounds(full_manifest, parse_registered=True,
                                                                     print_rounds=False, print_registered=False, func='cellpose')

    cellpose_channel_index = params['HCR_cellpose']['cellpose_channel']
    all_rounds = register_rounds + [reference_round['round']]

    # Check which rounds need processing
    skipped = []
    to_process = []
    for HCR_round_to_register in all_rounds:
        round_folder_name = get_round_folder_name(HCR_round_to_register, reference_round['round'])
        output_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'HCR' / 'cellpose' / f"{round_folder_name}_masks.tiff"
        if output_path.exists():
            skipped.append(round_folder_name)
        else:
            to_process.append((HCR_round_to_register, round_folder_name))

    # Print summary
    if not to_process:
        rprint(f"[dim]HCR cellpose: all {len(all_rounds)} rounds exist[/dim]")
        return

    rprint(f"HCR cellpose: processing {len(to_process)}/{len(all_rounds)} rounds")
    model_wrapper = CellposeModelWrapper(params)

    for HCR_round_to_register, round_folder_name in tqdm(to_process, desc="HCR cellpose"):
        full_stack_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'HCR' / 'full_registered_stacks' / f"{round_folder_name}.tiff"
        output_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'HCR' / 'cellpose' / f"{round_folder_name}_masks.tiff"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        raw_image = tif_imread(full_stack_path)
        cellpose_input = raw_image[:, cellpose_channel_index, :, :]

        masks, _, _ = model_wrapper.eval(cellpose_input)
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
        channels=[0, 0],
        diameter=cellpose_params['diameter'],
        flow_threshold=cellpose_params['flow_threshold'],
        cellprob_threshold=cellpose_params['cellprob_threshold'],
        do_3D=False,
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
    cellpose_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / '2P' / 'cellpose'

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

    rprint(f"\n[bold]Verify 2P cellpose segmentation for plane {current_plane}:[/bold]")
    rprint(f"  File: [yellow]{twop_cellpose_file}[/yellow]")
    rprint("\nPress [green]Enter[/green] to continue...")
    input()
    return 

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

    for mask_id in tqdm(unique_masks, desc="Computing neuropil regions"):
        mask_coords = []

        # Get all z-planes where this mask appears
        z_planes = np.unique(inds[mask_id][0])
        for z in z_planes:
            # Get mask center for this z-plane
            mask_points = (inds[mask_id][0] == z)
            if not np.any(mask_points):
                continue

            mx = int(inds[mask_id][1][mask_points].mean())
            my = int(inds[mask_id][2][mask_points].mean())

            # Define square region around cell center
            x_min, x_max_local = max(mx-radius, 0), min(mx+radius, x_max)
            y_min, y_max_local = max(my-radius, 0), min(my+radius, y_max)

            # Valid neuropil = within square AND outside exclusion zone
            # exclusion_zone includes all cells + `bound`-pixel buffer around them
            neuropil_points = ~exclusion_zone[z, x_min:x_max_local, y_min:y_max_local]

            if np.any(neuropil_points):
                x_coords, y_coords = np.where(neuropil_points)
                z_coords = np.full_like(x_coords, z)
                mask_coords.append(np.stack([z_coords, x_coords + x_min, y_coords + y_min]))

        if mask_coords:
            all_masks_locs[mask_id] = np.hstack(mask_coords).astype(np.int32)

    return all_masks_locs


def extract_probs_intensities(full_manifest):
    manifest = full_manifest['data']
    params = full_manifest['params']
    round_to_rounds, reference_round, register_rounds = verify_rounds(full_manifest, parse_registered = True,
                                                                      print_rounds = False, print_registered = False, func='intensities-extraction')

    all_rounds = register_rounds + [reference_round['round']]

    # Check which rounds need processing
    to_process = []
    for HCR_round_to_register in all_rounds:
        round_folder_name = get_round_folder_name(HCR_round_to_register, reference_round['round'])
        if HCR_round_to_register == reference_round['round']:
            channels_names = reference_round['channels']
        else:
            channels_names = round_to_rounds[HCR_round_to_register]['channels']
        output_folder = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'HCR' / 'extract_intensities'
        pkl_output_path = output_folder / f"{round_folder_name}_probs_intensities.pkl"
        if not pkl_output_path.exists():
            to_process.append((HCR_round_to_register, round_folder_name, channels_names))

    # Print summary
    if not to_process:
        rprint(f"[dim]HCR intensities: all {len(all_rounds)} rounds exist[/dim]")
        return

    rprint(f"HCR intensities: extracting {len(to_process)}/{len(all_rounds)} rounds")

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
        full_stack_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'HCR' / 'full_registered_stacks' / f"{round_folder_name}.tiff"
        full_stack_masks_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'HCR' / 'cellpose' / f"{round_folder_name}_masks.tiff"
        output_folder = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'HCR' / 'extract_intensities'
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
        for mask_id, mask_inds in enumerate(tqdm(inds, desc=f"HCR {round_folder_name}")):
            if mask_id == 0:
                continue
            if len(mask_inds[0]) == 0:
                raise Exception("Mask with zero pixels")
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

        #Save to `masks_path` folder
        df = pd.DataFrame(to_pnd)
        df.attrs['raw_image_path'] = str(full_stack_path)
        df.attrs['masks_path'] = str(full_stack_masks_path)
        df.attrs['HCR_round_number'] = HCR_round_to_register
        df.to_csv(csv_output_path)
        df.to_pickle(pkl_output_path)


def extract_electrophysiology_intensities(full_manifest: dict , session: dict):
    from suite2p.io import binary
    from .functional import get_number_of_suite2p_planes

    rprint("[bold]Extract 2P Intensities[/bold]")

    manifest = full_manifest['data']
    mouse_name = manifest['mouse_name']
    date = session['date']
    suite2p_run = session['functional_run'][0]

    suite2p_path = Path(manifest['base_path']) / manifest['mouse_name'] / '2P' /  f'{mouse_name}_{date}_{suite2p_run}' / 'suite2p'
    save_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / '2P' / 'suite2p'
    cellpose_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / '2P' / 'cellpose'
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


def match_masks(stack1_masks_path, stack2_masks_path):
    '''
    Match masks between two registered stacks using IoU (Intersection over Union).

    Returns DataFrame with columns: mask1, mask2, iou, is_best_match
    - Records ALL overlapping pairs with their IoU scores
    - is_best_match=True indicates the winning match (highest IoU, one-to-one)
    - Users can filter to is_best_match==True for backward compatible behavior
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

    # OPTIMIZATION: Precompute all mask sizes in stack2 (avoids repeated full-image scans)
    stack2_mask_ids, stack2_mask_sizes = np.unique(stack2_masks, return_counts=True)
    stack2_size_lookup = dict(zip(stack2_mask_ids, stack2_mask_sizes))

    to_pandas = {'mask1': [], 'mask2': [], 'iou': []}

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

        # Get all unique mask2 IDs that overlap with this mask1
        overlapping_mask2_ids = np.unique(mask2_nonzero)

        # Calculate IoU for EACH overlapping mask2
        for mask2_id in overlapping_mask2_ids:
            # Intersection: pixels where both mask1 and this mask2 are present
            intersection = np.sum(mask2_at_mask1 == mask2_id)

            # Union: all pixels in mask1 + all pixels in mask2 - intersection
            mask2_size = stack2_size_lookup.get(mask2_id, 0)
            union = mask1_size + mask2_size - intersection

            iou = intersection / union if union > 0 else 0.0

            to_pandas['mask1'].append(mask1_id)
            to_pandas['mask2'].append(mask2_id)
            to_pandas['iou'].append(iou)

    df = pd.DataFrame(to_pandas)

    if df.empty:
        # No matches found - return empty DataFrame with expected columns
        df['is_best_match'] = pd.Series(dtype=bool)
        return df

    # Mark best matches: apply greedy 1:1 matching (highest IoU wins)
    df = df.sort_values('iou', ascending=False).reset_index(drop=True)

    # Track which matches are "best" (survive deduplication)
    df_best = df.drop_duplicates('mask2', keep='first').copy()
    df_best = df_best.drop_duplicates('mask1', keep='first')

    # Use vectorized merge to mark best matches (much faster than df.apply for large DataFrames)
    df_best['is_best_match'] = True
    df = df.merge(df_best[['mask1', 'mask2', 'is_best_match']], on=['mask1', 'mask2'], how='left')
    df['is_best_match'] = df['is_best_match'].fillna(False)

    # Sort by mask1, then by iou descending for readability
    df = df.sort_values(['mask1', 'iou'], ascending=[True, False])

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
    """Print a summary of mask matching results to console."""
    if df.empty:
        rprint(f"[yellow]  {source_name} → {target_name}: No matches found[/yellow]")
        return

    # Handle old CSV format without is_best_match column
    if 'is_best_match' not in df.columns:
        rprint(f"  [dim]{source_name} → {target_name}: {len(df)} matches (old format, no competition data)[/dim]")
        return

    total_pairs = len(df)
    best_matches = df[df['is_best_match'] == True]
    n_best = len(best_matches)

    # Count masks that had competition (multiple candidates)
    mask1_counts = df.groupby('mask1').size()
    mask2_counts = df.groupby('mask2').size()
    mask1_with_competition = (mask1_counts > 1).sum()
    mask2_with_competition = (mask2_counts > 1).sum()

    # IoU statistics for best matches
    if n_best > 0:
        iou_min = best_matches['iou'].min()
        iou_max = best_matches['iou'].max()
        iou_median = best_matches['iou'].median()
    else:
        iou_min = iou_max = iou_median = 0

    # Count unique source and target masks
    n_source_masks = df['mask1'].nunique()
    n_target_masks_matched = df['mask2'].nunique()

    rprint(f"  [cyan]{source_name} → {target_name}:[/cyan]")
    rprint(f"    Best matches: {n_best}")
    rprint(f"    Source masks with matches: {n_source_masks}")
    if mask1_with_competition > 0:
        rprint(f"    [yellow]Source masks with competition: {mask1_with_competition} (multiple targets)[/yellow]")
    if mask2_with_competition > 0:
        rprint(f"    [yellow]Target masks with competition: {mask2_with_competition} (multiple sources)[/yellow]")
    rprint(f"    IoU range: {iou_min:.3f} - {iou_max:.3f} (median: {iou_median:.3f})")


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
    
    reference_round_tiff = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'HCR' / 'full_registered_stacks' / f"HCR{reference_round['round']}.tiff"
    reference_round_masks = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'HCR' / 'cellpose' / f"HCR{reference_round['round']}_masks.tiff"

    output_folder = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'MERGED' / 'aligned_masks'
    output_folder.mkdir(parents=True, exist_ok=True)

    # Collect matching results for summary
    matching_results = []

    # Skip HCR round alignment for additional planes (already done with reference plane)
    if reference_plane is None:  # Only process HCR rounds for the reference plane

        for HCR_round_to_register in register_rounds:
            round_folder_name = get_round_folder_name(HCR_round_to_register, reference_round['round'])

            mov_stack_masks = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'HCR' / 'cellpose' / f"{round_folder_name}_masks.tiff"

            save_path = output_folder / f"{round_folder_name}.csv"
            if save_path.exists():
                existing_df = pd.read_csv(save_path)
                # Check if file has all required columns; regenerate if stale
                if 'iou' in existing_df.columns and 'is_best_match' in existing_df.columns:
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
            rprint("\n[bold cyan]Mask Matching Summary:[/bold cyan]")
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

    plane = session['functional_plane'][0]
    reg_save_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / '2P' / 'registered'

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
        if 'iou' in existing_df.columns and 'is_best_match' in existing_df.columns:
            rprint(f"[dim]2P masks alignment for plane {plane}: exists, skipping[/dim]")
            matching_results.append((f"2P plane {plane}", f"HCR{reference_round['round']}", existing_df))
        else:
            print(f"  2P plane {plane} mapping: stale format, regenerating...")
            save_path.unlink()
            rprint(f"[cyan]Matching 2P plane {plane} masks to HCR using automated registration...[/cyan]")
            mask1_to_mask2_df = match_masks(masks_2p_aligned_3d_path, reference_round_masks)
            mask1_to_mask2_df.to_csv(save_path)
            rprint(f"[green]✓ Saved mask matching to {save_path}[/green]")
            matching_results.append((f"2P plane {plane}", f"HCR{reference_round['round']}", mask1_to_mask2_df))
    else:
        # Match masks using automated registration output
        rprint(f"[cyan]Matching 2P plane {plane} masks to HCR using automated registration...[/cyan]")
        mask1_to_mask2_df = match_masks(masks_2p_aligned_3d_path, reference_round_masks)
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
    HCR_intensities_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'HCR' / 'extract_intensities'
    HCR_mapping_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'MERGED' / 'aligned_masks'
    merged_table_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'MERGED' / 'aligned_extracted_features'
    merged_table_path.mkdir(parents=True, exist_ok=True)
    
    # Check if median filter was applied
    intensity_config = get_intensity_extraction_config(params)
    median_filter = intensity_config.get('stack_median_filter')
    median_filter_label = None
    if median_filter is not None:
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

    # ========== PRE-LOAD ALL DATA ONCE (optimization: avoid reloading per feature) ==========

    # Pre-load 2P mapping (feature-independent)
    if only_hcr:
        twoP_mapping_dict = {}
        twoP_iou_dict = {}
    else:
        twop_mapping_path = HCR_mapping_path / f"twop_plane{plane}_to_HCR{reference_round['round']}.csv"
        try:
            towP_to_reference_mapping = pd.read_csv(twop_mapping_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"2P-to-HCR mapping file not found: {twop_mapping_path}\n"
                f"Please ensure align_masks() has completed successfully for plane {plane}."
            )
        if 'is_best_match' in towP_to_reference_mapping.columns:
            towP_to_reference_mapping = towP_to_reference_mapping[towP_to_reference_mapping['is_best_match'] == True]
        if towP_to_reference_mapping.empty:
            rprint(f"[yellow]Warning: No 2P-to-HCR matches found for plane {plane}. All cells will have None for 2P mapping.[/yellow]")
            twoP_mapping_dict = {}
            twoP_iou_dict = {}
        else:
            twoP_mapping_dict = {mask_2:mask_1 for mask_1,mask_2 in towP_to_reference_mapping[['mask1','mask2']].values}
            twoP_iou_dict = {mask_2:iou for mask_2,iou in towP_to_reference_mapping[['mask2','iou']].values}

    # Pre-load reference round intensities (full DataFrame, pivot per feature)
    ref_intensities_path = HCR_intensities_path / f"HCR{reference_round['round']}_probs_intensities.pkl"
    try:
        reference_round_intensities = pd.read_pickle(ref_intensities_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Reference round intensities not found: {ref_intensities_path}\n"
            f"Please ensure extract_probs_intensities() has completed for HCR round {reference_round['round']}."
        )

    # Pre-load HCR round mappings and intensities (feature-independent)
    HCR_rounds_names = register_rounds
    HCR_round_mapping_dict = []
    HCR_round_iou_dict = []
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
        if 'is_best_match' in round_mapping_df.columns:
            round_mapping_df = round_mapping_df[round_mapping_df['is_best_match'] == True]
        HCR_round_mapping_dict.append({mask_2:mask_1 for mask_1,mask_2 in round_mapping_df[['mask1','mask2']].values})
        HCR_round_iou_dict.append({mask_2:iou for mask_2,iou in round_mapping_df[['mask2','iou']].values})

        # Load intensities
        round_intensities_path = HCR_intensities_path / f"{round_file_name}_probs_intensities.pkl"
        try:
            preloaded_round_intensities[HCR_round_to_register] = pd.read_pickle(round_intensities_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"HCR round intensities not found: {round_intensities_path}\n"
                f"Please ensure extract_probs_intensities() has completed for HCR round {HCR_round_to_register}."
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
        # first let's match the 2p
        mask_tp_matched = []
        mask_tp_iou = []
        for i in reference_round_intensities_pivot.mask_id_main:
            if i in twoP_mapping_dict:
                mask_tp_matched.append(twoP_mapping_dict[i])
                mask_tp_iou.append(twoP_iou_dict[i])
            else:
                mask_tp_matched.append(None)
                mask_tp_iou.append(None)
        reference_round_intensities_pivot['twoP_mask'] = mask_tp_matched
        reference_round_intensities_pivot['twoP_iou'] = mask_tp_iou

        for j in range(len(HCR_round_mapping_dict)):
            HCR_main_2_HCR_round = HCR_round_mapping_dict[j]
            HCR_main_2_HCR_round_iou = HCR_round_iou_dict[j]
            mask_matched = []
            mask_iou = []
            for i in reference_round_intensities_pivot.mask_id_main:
                if i in HCR_main_2_HCR_round:
                    mask_matched.append(HCR_main_2_HCR_round[i])
                    mask_iou.append(HCR_main_2_HCR_round_iou[i])
                else:
                    mask_matched.append(None)
                    mask_iou.append(None)
            reference_round_intensities_pivot[f'mask_round_{HCR_rounds_names[j]}'] = mask_matched
            reference_round_intensities_pivot[f'iou_round_{HCR_rounds_names[j]}'] = mask_iou

        ####       ####
        ###  MERGE  ###
        ####       ####
        HCR_main_pivot_merged = reference_round_intensities_pivot.copy().reset_index()

        for j in range(len(HCR_round_mapping_dict)):
            round_mask_name = f'mask_round_{HCR_rounds_names[j]}'

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