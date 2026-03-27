import time
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm.auto import tqdm
from tifffile import TiffFile
from scipy.fft import rfft2, irfft2
from scipy.interpolate import RBFInterpolator, griddata, LinearNDInterpolator, NearestNDInterpolator
from scipy.ndimage import map_coordinates, rotate, binary_dilation, binary_erosion, distance_transform_edt
from scipy.spatial import ConvexHull
from skimage.transform import resize, AffineTransform, warp
from skimage.registration import phase_cross_correlation
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
from scipy.optimize import minimize
import scipy.io as sio
from pathlib import Path


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def prompt_overwrite_per_plane(plane_idx: int, output_path: Path, overwrite_state: list) -> bool:
    """
    Ask user whether to overwrite existing output files.

    On first call with an existing file, prompts once with [y/n].
    The answer applies to all planes for the rest of the session.

    Parameters
    ----------
    plane_idx : int
        The plane being processed (for display only)
    output_path : Path
        Path to the output file to check
    overwrite_state : list
        Single-element list holding state: [None, True, False]
        Mutated by this function to remember user's choice across calls

    Returns
    -------
    bool
        True if should process (file doesn't exist or user chose to overwrite),
        False if should skip
    """
    if not output_path.exists():
        return True  # File doesn't exist, proceed

    # Already have user's answer from previous prompt
    if overwrite_state[0] is not None:
        return overwrite_state[0]

    # Prompt user once - applies to all planes
    while True:
        response = input(f"Output files exist. Overwrite all? [y/n]: ").strip().lower()
        if response == 'y':
            overwrite_state[0] = True
            return True
        elif response == 'n':
            overwrite_state[0] = False
            return False
        else:
            print("  Please enter y or n")


def get_magnification_from_sbx(sbx_path):
    """
    Read magnification from SBX .mat file.

    Parameters
    ----------
    sbx_path : str or Path
        Path to .sbx file (will look for corresponding .mat file)

    Returns
    -------
    magnification : float
        Actual magnification value (e.g., 1.7, 4.0)
    """
    mat_path = Path(sbx_path).with_suffix('.mat')

    if not mat_path.exists():
        raise FileNotFoundError(f"Mat file not found: {mat_path}")

    # Load .mat file
    mat_data = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)

    # Get magnification info
    info = mat_data['info']
    magnification_list = info.config.magnification_list
    magnification_idx = int(info.config.magnification)

    # magnification_idx is 1-indexed in MATLAB, so subtract 1 for Python
    actual_magnification = float(magnification_list[magnification_idx - 1])

    return actual_magnification


def _rational_to_float(val):
    # tifffile may give (num, den) tuple-like or a Rational object
    try:
        return float(val[0]) / float(val[1])
    except Exception:
        return float(val)

def read_spacing_xyz_from_tiff(path):
    """
    Returns (sx, sy, sz, unit) where s* are in 'unit' per pixel (e.g. micrometer / pixel).
    If metadata is missing, falls back to 1.0.
    """
    sx = sy = sz = 1.0
    unit = None

    with TiffFile(path) as tif:
        page = tif.pages[0]

        # --- 1) ImageJ metadata (what Fiji commonly uses)
        ij = tif.imagej_metadata or {}
        # ImageJ 'unit' (e.g. 'micron', 'um')
        unit = ij.get("unit", None)

        # ImageJ 'spacing' is typically Z step in the same unit
        # (often present for stacks)
        if "spacing" in ij and ij["spacing"] is not None:
            try:
                sz = float(ij["spacing"])
            except Exception:
                pass

        # --- 2) TIFF resolution tags: usually pixels per unit
        # Fiji screenshot: "Resolution: 0.748 pixels per micron"
        # so micron_per_pixel = 1 / 0.748 = 1.3369
        tags = page.tags

        xres = tags.get("XResolution")
        yres = tags.get("YResolution")
        if xres and yres:
            x_ppu = _rational_to_float(xres.value)  # pixels per unit
            y_ppu = _rational_to_float(yres.value)
            if x_ppu and x_ppu > 0:
                sx = 1.0 / x_ppu
            if y_ppu and y_ppu > 0:
                sy = 1.0 / y_ppu

    return sx, sy, sz, unit


def tps_warp_2p_to_hcr(twop_2d, landmarks_df, hcr_shape, order=0, downsample=10):
    """Apply TPS warp to transform 2P image into HCR coordinate space.

    Parameters
    ----------
    order : int
        Interpolation order (0=nearest for labels, 1=bilinear for intensity).
    downsample : int
        Evaluate TPS on a grid coarsened by this factor, then upscale.
        TPS from ~30 landmarks is very smooth so coarse evaluation is exact.
        Set to 1 to disable downsampling.
    """
    from scipy.ndimage import zoom  # zoom not used elsewhere, keep local

    hcr_ny, hcr_nx = hcr_shape[1], hcr_shape[2]

    src_points = landmarks_df[['hcr_x_px', 'hcr_y_px']].values
    tgt_x = landmarks_df['2p_x_px'].values
    tgt_y = landmarks_df['2p_y_px'].values

    tps_to_2p_x = RBFInterpolator(src_points, tgt_x, kernel='thin_plate_spline')
    tps_to_2p_y = RBFInterpolator(src_points, tgt_y, kernel='thin_plate_spline')

    # Evaluate TPS on downsampled grid, then upscale (TPS is smooth)
    ds = max(1, int(downsample))
    small_ny = int(np.ceil(hcr_ny / ds))
    small_nx = int(np.ceil(hcr_nx / ds))
    small_yy, small_xx = np.mgrid[0:small_ny, 0:small_nx]
    small_coords = np.column_stack([(small_xx.ravel() * ds).astype(np.float64),
                                     (small_yy.ravel() * ds).astype(np.float64)])

    small_x = tps_to_2p_x(small_coords).reshape(small_ny, small_nx)
    small_y = tps_to_2p_y(small_coords).reshape(small_ny, small_nx)

    if ds > 1:
        zoom_y = hcr_ny / small_ny
        zoom_x = hcr_nx / small_nx
        twop_x_coords = zoom(small_x, (zoom_y, zoom_x), order=3)[:hcr_ny, :hcr_nx]
        twop_y_coords = zoom(small_y, (zoom_y, zoom_x), order=3)[:hcr_ny, :hcr_nx]
    else:
        twop_x_coords = small_x
        twop_y_coords = small_y

    twop_warped = map_coordinates(
        twop_2d.astype(np.float32),
        [twop_y_coords, twop_x_coords],
        order=order, mode='constant', cval=0
    ).astype(twop_2d.dtype)

    return twop_warped

def erode_labels(labels_2d, iterations):
    """Erode each labeled cell independently, including cell-cell boundaries.

    First removes inner boundaries between touching cells (1px), then
    applies distance-based erosion for the remaining iterations.
    This ensures touching cells are properly separated before erosion.
    """
    if iterations <= 0:
        return labels_2d

    result = labels_2d.copy()
    bounds = find_boundaries(labels_2d, mode='inner')
    result[bounds] = 0  # treat cell-cell boundaries as background
    if iterations > 1:
        dist = distance_transform_edt(result > 0)
        result[dist <= (iterations - 1)] = 0
    return result

def sample_hcr_at_zmap(hcr_3d_local, z_map_local, z_offset=0, z_expand=1):
    """Sample HCR volume along z-surface."""
    nz_local, ny_local, nx_local = hcr_3d_local.shape
    result = np.zeros((ny_local, nx_local), dtype=hcr_3d_local.dtype)
    z_center = np.round(z_map_local + z_offset).astype(int)
    y_idx, x_idx = np.mgrid[0:ny_local, 0:nx_local]
    
    for dz in range(-z_expand, z_expand + 1):
        z_sample = np.clip(z_center + dz, 0, nz_local - 1)
        sampled = hcr_3d_local[z_sample, y_idx, x_idx]
        # Take max label (preserve label IDs)
        result = np.where(sampled > result, sampled, result)
    
    return result

def sample_hcr_binary_at_zmap(hcr_3d_local, z_map_local, z_offset=0, z_expand=1):
    """Sample HCR volume as binary mask."""
    return sample_hcr_at_zmap(hcr_3d_local, z_map_local, z_offset, z_expand) > 0

def compute_iou(mask1, mask2, fov_mask=None):
    """
    Compute IoU (Intersection over Union) between two binary masks.

    Parameters
    ----------
    mask1 : ndarray
        First binary mask (typically 2P masks)
    mask2 : ndarray
        Second binary mask (typically HCR masks)
    fov_mask : ndarray, optional
        Field-of-view mask to restrict calculation. If provided, mask2 is
        masked to only include pixels within the FOV. This is critical when
        mask2 (HCR) covers a much larger area than mask1 (2P).

    Returns
    -------
    float
        IoU score in [0, 1]

    Notes
    -----
    When 2P covers a small FOV within a larger HCR image, the standard IoU
    will be artificially low because the union includes HCR cells outside
    the 2P region. Use fov_mask to restrict to the relevant region.
    """
    mask1 = np.asarray(mask1, dtype=bool)
    mask2 = np.asarray(mask2, dtype=bool)
    if fov_mask is not None:
        mask2 = mask2 & np.asarray(fov_mask, dtype=bool)
    intersection = (mask1 & mask2).sum()
    union = (mask1 | mask2).sum()
    return intersection / union if union > 0 else 0.0


def create_fov_mask(twop_mask, dilation_iterations=30):
    """
    Create a field-of-view mask from 2P data extent.

    Parameters
    ----------
    twop_mask : ndarray
        Binary mask of 2P cells (after TPS warp to HCR space)
    dilation_iterations : int
        Number of dilation iterations to expand FOV boundary.
        Default 30 pixels captures nearby HCR cells.

    Returns
    -------
    ndarray
        Binary FOV mask covering 2P data + margin
    """
    return binary_dilation(twop_mask > 0, iterations=dilation_iterations)

def compute_bidirectional_overlap(mask1, mask2):
    """
    Compute bidirectional overlap between two binary masks.

    overlap = sqrt(overlap_1² + overlap_2²) / sqrt(2)

    Where:
    - overlap_1 = fraction of mask1 pixels overlapping with mask2
    - overlap_2 = fraction of mask2 pixels overlapping with mask1

    This metric requires BOTH masks to substantially overlap each other,
    unlike IoU which harshly penalizes size mismatches, or one-sided overlap
    which ignores asymmetric coverage.

    Reference: Bhattarai et al. used similar bidirectional measure for
    cross-modality ROI matching (in vivo to ex vivo).

    Returns normalized value in [0, 1] where 1 = perfect overlap.
    """
    intersection = (mask1 & mask2).sum()
    sum1, sum2 = mask1.sum(), mask2.sum()

    if sum1 == 0 or sum2 == 0:
        return 0.0

    overlap_1 = intersection / sum1  # Fraction of mask1 in mask2
    overlap_2 = intersection / sum2  # Fraction of mask2 in mask1

    # Geometric combination normalized to [0, 1]
    return np.sqrt(overlap_1**2 + overlap_2**2) / np.sqrt(2)

def fft_iou_all_shifts(mask1, mask2, max_shift):
    """Find best XY shift using FFT-accelerated IoU search."""
    m1 = mask1.astype(np.float64)
    m2 = mask2.astype(np.float64)
    sum1, sum2 = m1.sum(), m2.sum()
    if sum1 == 0 or sum2 == 0:
        return 0, 0, 0

    h, w = m1.shape
    cross_corr = irfft2(np.conj(rfft2(m1)) * rfft2(m2), s=(h, w))
    iou_map = np.where((sum1 + sum2 - cross_corr) > 0,
                       cross_corr / (sum1 + sum2 - cross_corr), 0)

    return _best_shift_from_map(iou_map, max_shift)


def _best_shift_from_map(score_map, max_shift):
    """Extract best (dy, dx, score) from a zero-centered FFT score map."""
    h, w = score_map.shape
    shifted = np.fft.fftshift(score_map)
    cy, cx = h // 2, w // 2
    region = shifted[cy - max_shift:cy + max_shift + 1,
                     cx - max_shift:cx + max_shift + 1]
    flat_idx = np.argmax(region)
    best_i, best_j = np.unravel_index(flat_idx, region.shape)
    return best_i - max_shift, best_j - max_shift, region[best_i, best_j]

def fft_bidirectional_all_shifts(mask1, mask2, max_shift):
    """Find best XY shift using FFT-accelerated bidirectional overlap search.

    Bidirectional overlap = sqrt((intersection/sum1)² + (intersection/sum2)²) / sqrt(2)
    """
    m1 = mask1.astype(np.float64)
    m2 = mask2.astype(np.float64)
    sum1, sum2 = m1.sum(), m2.sum()
    if sum1 == 0 or sum2 == 0:
        return 0, 0, 0

    h, w = m1.shape
    cross_corr = irfft2(np.conj(rfft2(m1)) * rfft2(m2), s=(h, w))
    bidirectional_map = np.sqrt((cross_corr / sum1)**2 + (cross_corr / sum2)**2) / np.sqrt(2)

    return _best_shift_from_map(bidirectional_map, max_shift)

def shift_2d(mask, dy, dx):
    """Shift 2D mask with zero-padding."""
    if dy == 0 and dx == 0:
        return mask.copy()
    shifted = np.roll(np.roll(mask, int(dy), axis=0), int(dx), axis=1)
    dy, dx = int(dy), int(dx)
    if dy > 0: shifted[:dy, :] = 0
    elif dy < 0: shifted[dy:, :] = 0
    if dx > 0: shifted[:, :dx] = 0
    elif dx < 0: shifted[:, dx:] = 0
    return shifted

def rotate_2d(mask, angle_deg, order=0):
    """Rotate 2D image around center.

    Parameters
    ----------
    order : int
        Interpolation order (0=nearest for labels, 1=bilinear for intensity).
    """
    if angle_deg == 0:
        return mask.copy()
    return rotate(mask, angle_deg, reshape=False, order=order, mode='constant', cval=0)

def apply_shift_fields(mask_2d, dy_field, dx_field, order=0, return_labels=False):
    """
    Apply per-pixel shift fields.

    Parameters
    ----------
    mask_2d : ndarray
        2D mask to shift (can be binary or labeled)
    dy_field : ndarray
        Per-pixel Y shift field
    dx_field : ndarray
        Per-pixel X shift field
    order : int, optional
        Interpolation order (0=nearest-neighbor, 1=linear). Default 0.
    return_labels : bool, optional
        If True, return integer labels (for labeled masks).
        If False, return binary mask (for IoU calculation). Default False.

    Returns
    -------
    shifted : ndarray
        Shifted mask (binary bool if return_labels=False, else float/int)
    """
    h, w = mask_2d.shape
    yy, xx = np.mgrid[0:h, 0:w]
    src_y = (yy - dy_field).astype(np.float64)
    src_x = (xx - dx_field).astype(np.float64)
    shifted = map_coordinates(mask_2d.astype(float), [src_y, src_x],
                              order=order, mode='constant', cval=0)

    if return_labels:
        # Return labels (round to nearest integer to preserve label IDs)
        return np.round(shifted).astype(mask_2d.dtype)
    else:
        # Return binary (for IoU calculation)
        return shifted > 0.5

def add_px_columns(landmarks_df, hcr_resolution):
    """
    Convert landmark coordinates to pixel units.

    2P coordinates are already in pixels (uncalibrated 1:1 in BigWarp).
    HCR coordinates are in micrometers and need conversion.

    Parameters
    ----------
    landmarks_df : pd.DataFrame
        Landmarks with columns: 2p_x, 2p_y, 2p_z, hcr_x, hcr_y, hcr_z
    hcr_resolution : list
        HCR resolution as [X, Y, Z] in μm/pixel (from manifest)
    """
    HCR_RES_X, HCR_RES_Y, HCR_RES_Z = hcr_resolution[0], hcr_resolution[1], hcr_resolution[2]

    # HCR coordinates: convert from μm to pixels
    landmarks_df['hcr_x_px'] = landmarks_df['hcr_x'] / HCR_RES_X
    landmarks_df['hcr_y_px'] = landmarks_df['hcr_y'] / HCR_RES_Y
    landmarks_df['hcr_z_px'] = landmarks_df['hcr_z'] / HCR_RES_Z

    # 2P coordinates: already in pixels (uncalibrated 1:1)
    landmarks_df['2p_x_px'] = landmarks_df['2p_x']
    landmarks_df['2p_y_px'] = landmarks_df['2p_y']
    landmarks_df['2p_z_px'] = landmarks_df['2p_z']

    return landmarks_df

def load_landmarks(landmarks_path, hcr_resolution):
    """
    Load and prepare landmark data for registration.

    Loads landmarks from CSV, filters enabled ones within bounds,
    and converts physical units to pixel coordinates.

    Handles both 2D BigWarp exports (6 columns: name, enabled, 2p_x, 2p_y, hcr_x, hcr_y)
    and 3D exports (8 columns: name, enabled, 2p_x, 2p_y, 2p_z, hcr_x, hcr_y, hcr_z).

    Parameters
    ----------
    landmarks_path : Path
        Path to landmarks CSV file
    hcr_resolution : list
        HCR resolution as [X, Y, Z] in μm/pixel (from manifest)

    Returns
    -------
    pd.DataFrame
        Filtered landmarks with pixel coordinate columns added
    """
    # Read without column names first to detect format
    raw_df = pd.read_csv(landmarks_path, header=None)
    ncols = raw_df.shape[1]

    if ncols == 6:
        # 2D BigWarp export: name, enabled, 2p_x, 2p_y, hcr_x, hcr_y
        landmarks_df = raw_df.copy()
        landmarks_df.columns = ['name', 'enabled', '2p_x', '2p_y', 'hcr_x', 'hcr_y']
        landmarks_df['2p_z'] = 0.0
        landmarks_df['hcr_z'] = 0.0
        print(f"  Landmark CSV: 6 columns (2D format, z=0)")
    elif ncols >= 8:
        # 3D BigWarp export: name, enabled, 2p_x, 2p_y, 2p_z, hcr_x, hcr_y, hcr_z
        landmarks_df = raw_df.iloc[:, :8].copy()
        landmarks_df.columns = ['name', 'enabled', '2p_x', '2p_y', '2p_z', 'hcr_x', 'hcr_y', 'hcr_z']
        print(f"  Landmark CSV: {ncols} columns (3D format)")
    else:
        raise ValueError(
            f"Landmark CSV has {ncols} columns (expected 6 for 2D or 8 for 3D): {landmarks_path}"
        )

    # Filter only enabled landmarks and those within bounds
    landmarks_df = landmarks_df.query("enabled==True and hcr_x<9e5").copy()

    # Convert to pixel coordinates
    landmarks_df = add_px_columns(landmarks_df, hcr_resolution)

    return landmarks_df

def build_z_map(landmarks_df, output_shape):
    """
    Build a z-coordinate map from reference landmarks using interpolation.

    Uses linear interpolation where possible, falling back to nearest neighbor
    interpolation for regions without nearby landmarks.

    Parameters
    ----------
    landmarks_df : pd.DataFrame
        DataFrame containing landmark coordinates with columns:
        'hcr_y_px', 'hcr_x_px', 'hcr_z_px'
    output_shape : tuple of int
        Shape (height, width) of the output z-map

    Returns
    -------
    np.ndarray
        2D array of z-coordinates with shape output_shape
    """
    y_size, x_size = output_shape

    # Extract landmark coordinates
    landmark_coords = landmarks_df[['hcr_y_px', 'hcr_x_px']].values
    landmark_z_values = landmarks_df['hcr_z_px'].values

    # Create interpolators
    interp_linear = LinearNDInterpolator(landmark_coords, landmark_z_values)
    interp_nearest = NearestNDInterpolator(landmark_coords, landmark_z_values)

    # Build coordinate grid
    yy, xx = np.mgrid[0:y_size, 0:x_size]

    # Interpolate z-values: use linear where available, nearest neighbor for gaps
    z_map_linear = interp_linear(yy, xx)
    z_map = np.where(np.isnan(z_map_linear), interp_nearest(yy, xx), z_map_linear)

    return z_map

# =============================================================================
# ALIGNMENT FUNCTIONS
# =============================================================================

def _fft_iou_precomputed(twop_rfft, sum1, mask2, max_shift):
    """IoU search with pre-computed rfft2 of mask1 (avoids redundant FFTs)."""
    m2 = mask2.astype(np.float64)
    sum2 = m2.sum()
    if sum1 == 0 or sum2 == 0:
        return 0, 0, 0

    h, w = m2.shape
    cross_corr = irfft2(np.conj(twop_rfft) * rfft2(m2), s=(h, w))
    iou_map = np.where((sum1 + sum2 - cross_corr) > 0,
                       cross_corr / (sum1 + sum2 - cross_corr), 0)
    return _best_shift_from_map(iou_map, max_shift)


def global_alignment(twop_binary, hcr_3d_local, z_map, rotation_range, rotation_step,
                     z_range, xy_max, fov_mask=None):
    """Find best global (theta, dz, dy, dx).

    Parameters
    ----------
    fov_mask : ndarray, optional
        Binary mask of the 2P field of view (dilated). When provided, HCR is
        masked to this region before IOU computation so that HCR cells outside
        the 2P FOV don't bias the search.
    """
    rotation_angles = np.arange(rotation_range[0], rotation_range[1] + rotation_step, rotation_step)
    dz_vals = list(range(z_range[0], z_range[1] + 1))

    # Pre-compute HCR z-slices (only depend on dz, not theta)
    hcr_slices = {}
    for dz in dz_vals:
        hcr_slices[dz] = sample_hcr_binary_at_zmap(hcr_3d_local, z_map, z_offset=dz)

    best = {'iou': -1, 'theta': 0, 'dz': 0, 'dy': 0, 'dx': 0}

    for theta in rotation_angles:
        twop_rotated = rotate_2d(twop_binary, theta)

        # Pre-compute FFT of rotated 2P mask (reused across all dz)
        twop_f64 = twop_rotated.astype(np.float64)
        twop_rfft = rfft2(twop_f64)
        sum1 = twop_f64.sum()

        # Rotate FOV mask to match 2P rotation
        fov_rotated = None
        if fov_mask is not None:
            fov_rotated = rotate_2d(fov_mask.astype(np.float64), theta) > 0.5

        for dz in dz_vals:
            hcr_2d = hcr_slices[dz]

            if fov_rotated is not None:
                hcr_2d = hcr_2d & fov_rotated

            dy, dx, iou = _fft_iou_precomputed(twop_rfft, sum1, hcr_2d, xy_max)

            if iou > best['iou']:
                best = {'iou': iou, 'theta': theta, 'dz': dz, 'dy': dy, 'dx': dx}

    return best

def _process_single_tile(args):
    """Worker for a single tile shift computation (used by ThreadPoolExecutor)."""
    twop_tile, hcr_tile, z_map_tile, base_dz, z_vals, xy_max, cy, cx = args

    best_tile = {'iou': -1, 'dz': 0, 'dy': 0, 'dx': 0}
    for dz in z_vals:
        hcr_2d_tile = sample_hcr_binary_at_zmap(hcr_tile, z_map_tile, z_offset=base_dz + dz)
        dy, dx, iou = fft_iou_all_shifts(twop_tile, hcr_2d_tile, xy_max)
        if iou > best_tile['iou']:
            best_tile = {'iou': iou, 'dz': dz, 'dy': dy, 'dx': dx}

    return {'cy': cy, 'cx': cx, **best_tile}


def compute_tile_shifts(twop_2d_in, hcr_3d_local, z_map_local, base_dz,
                        tile_size, overlap, xy_max, z_range, min_pixels,
                        max_workers=None):
    """Compute optimal shift for each tile (parallelized across tiles)."""
    from concurrent.futures import ThreadPoolExecutor
    import os

    ny_local, nx_local = twop_2d_in.shape
    step = max(1, int(tile_size * (1 - overlap)))
    z_vals = list(range(z_range[0], z_range[1] + 1))

    # Build list of tile work items
    tasks = []
    for ty in range(0, ny_local - tile_size // 2, step):
        for tx in range(0, nx_local - tile_size // 2, step):
            y0_t, y1_t = max(0, ty), min(ny_local, ty + tile_size)
            x0_t, x1_t = max(0, tx), min(nx_local, tx + tile_size)

            if (y1_t - y0_t) < tile_size // 2 or (x1_t - x0_t) < tile_size // 2:
                continue

            twop_tile = twop_2d_in[y0_t:y1_t, x0_t:x1_t]
            if twop_tile.sum() < min_pixels:
                continue

            z_map_tile = z_map_local[y0_t:y1_t, x0_t:x1_t]
            hcr_tile = hcr_3d_local[:, y0_t:y1_t, x0_t:x1_t]
            cy, cx = (y0_t + y1_t) / 2, (x0_t + x1_t) / 2
            tasks.append((twop_tile, hcr_tile, z_map_tile, base_dz, z_vals, xy_max, cy, cx))

    if not tasks:
        return []

    # NumPy/SciPy FFT releases the GIL, so threads give real parallelism
    n_workers = max_workers or min(os.cpu_count() or 4, len(tasks))
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        tile_results = list(pool.map(_process_single_tile, tasks))

    return tile_results

def interpolate_shift_field(tile_results, shape):
    """Interpolate tile shifts to smooth per-pixel field."""
    ny_local, nx_local = shape
    
    if len(tile_results) < 3:
        return np.zeros(shape), np.zeros(shape), np.zeros(shape)
    
    centers = np.array([[t['cy'], t['cx']] for t in tile_results])
    dz_vals = np.array([t['dz'] for t in tile_results])
    dy_vals = np.array([t['dy'] for t in tile_results])
    dx_vals = np.array([t['dx'] for t in tile_results])
    
    yy, xx = np.mgrid[0:ny_local, 0:nx_local]
    grid_points = np.column_stack([yy.ravel(), xx.ravel()])
    
    dz_field = griddata(centers, dz_vals, grid_points, method='linear', fill_value=0).reshape(shape)
    dy_field = griddata(centers, dy_vals, grid_points, method='linear', fill_value=0).reshape(shape)
    dx_field = griddata(centers, dx_vals, grid_points, method='linear', fill_value=0).reshape(shape)
    
    # Fill NaNs
    if np.any(np.isnan(dz_field)):
        dz_nn = griddata(centers, dz_vals, grid_points, method='nearest').reshape(shape)
        dy_nn = griddata(centers, dy_vals, grid_points, method='nearest').reshape(shape)
        dx_nn = griddata(centers, dx_vals, grid_points, method='nearest').reshape(shape)
        dz_field = np.where(np.isnan(dz_field), dz_nn, dz_field)
        dy_field = np.where(np.isnan(dy_field), dy_nn, dy_field)
        dx_field = np.where(np.isnan(dx_field), dx_nn, dx_field)
    
    return dz_field, dy_field, dx_field


# =============================================================================
# LOW-RES TO HIGH-RES REGISTRATION FUNCTIONS
# =============================================================================

def compute_normalized_cross_correlation(img1, img2):
    """
    Compute normalized cross-correlation between two images.
    Returns value in range [-1, 1] where 1 = perfect match.
    """
    img1_norm = (img1 - np.mean(img1)) / (np.std(img1) + 1e-8)
    img2_norm = (img2 - np.mean(img2)) / (np.std(img2) + 1e-8)
    return np.mean(img1_norm * img2_norm)


def _sift_affine_registration(lowres_upsampled, hires_2d, config):
    """
    SIFT feature matching + RANSAC affine transform.

    Parameters
    ----------
    lowres_upsampled : ndarray
        Low-res image already upsampled to match high-res dimensions
    hires_2d : ndarray
        High-res target image
    config : dict
        Configuration parameters:
        - n_features: Max SIFT keypoints (default 5000)
        - ratio_threshold: Lowe's ratio test threshold (default 0.75)
        - ransac_reproj_threshold: RANSAC reprojection threshold (default 5.0)
        - min_matches: Minimum matches required (default 10)
        - min_feature_size: Minimum keypoint size to keep (default 0 = no filtering)
        - max_spatial_distance: Max distance between matched keypoint positions (default 0 = no limit)

    Returns
    -------
    affine_matrix : ndarray (2, 3) or None
        OpenCV-format affine matrix, or None if failed
    n_inliers : int
        Number of RANSAC inliers
    n_matches : int
        Number of good matches found
    error_msg : str or None
        Error message if failed, None if successful
    """
    import cv2

    config = config or {}
    n_features = config.get('n_features', 5000)
    ratio_thresh = config.get('ratio_threshold', 0.75)
    ransac_thresh = config.get('ransac_reproj_threshold', 5.0)
    min_matches = config.get('min_matches', 10)
    min_feature_size = config.get('min_feature_size', 0)  # 0 = no filtering
    max_spatial_distance = config.get('max_spatial_distance', 0)  # 0 = no limit
    percentile_norm = config.get('percentile_norm', (2, 98))  # Percentile-based normalization

    # Convert to 8-bit for OpenCV using percentile-based normalization (robust to outliers)
    def to_uint8(img, percentiles):
        img = img.astype(np.float64)
        p_low, p_high = np.percentile(img, percentiles)
        if p_high > p_low:
            img = np.clip((img - p_low) / (p_high - p_low + 1e-8), 0, 1)
        else:
            img = img - img.min()
            if img.max() > 0:
                img = img / img.max()
        return (img * 255).astype(np.uint8)

    lowres_8bit = to_uint8(lowres_upsampled, percentile_norm)
    hires_8bit = to_uint8(hires_2d, percentile_norm)

    # SIFT detection
    sift = cv2.SIFT_create(nfeatures=n_features)
    kp1, desc1 = sift.detectAndCompute(lowres_8bit, None)
    kp2, desc2 = sift.detectAndCompute(hires_8bit, None)

    if desc1 is None or desc2 is None:
        return None, 0, 0, "No features detected in one or both images"

    # Filter keypoints by minimum size (keeps cell-like features, filters noise)
    if min_feature_size > 0 and kp1 and kp2:
        mask1 = np.array([kp.size >= min_feature_size for kp in kp1])
        mask2 = np.array([kp.size >= min_feature_size for kp in kp2])
        kp1 = [kp for kp, keep in zip(kp1, mask1) if keep]
        kp2 = [kp for kp, keep in zip(kp2, mask2) if keep]
        desc1 = desc1[mask1] if desc1 is not None and mask1.sum() > 0 else None
        desc2 = desc2[mask2] if desc2 is not None and mask2.sum() > 0 else None

        if desc1 is None or desc2 is None:
            return None, 0, 0, f"No features after size filtering (min_feature_size={min_feature_size})"

    if len(kp1) < 4 or len(kp2) < 4:
        return None, 0, 0, f"Too few keypoints: {len(kp1)} in low-res, {len(kp2)} in high-res (need >= 4)"

    # FLANN matching
    FLANN_INDEX_KDTREE = 1
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=FLANN_INDEX_KDTREE, trees=5),
        dict(checks=50)
    )

    try:
        matches = flann.knnMatch(desc1, desc2, k=2)
    except Exception as e:
        return None, 0, 0, f"Feature matching failed: {e}"

    # Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

    # Spatial distance filter (for tile refinement, matched points should be close)
    if max_spatial_distance > 0:
        spatially_filtered = []
        for m in good_matches:
            pt1 = np.array(kp1[m.queryIdx].pt)
            pt2 = np.array(kp2[m.trainIdx].pt)
            if np.linalg.norm(pt1 - pt2) <= max_spatial_distance:
                spatially_filtered.append(m)
        good_matches = spatially_filtered

    n_matches = len(good_matches)

    if n_matches < min_matches:
        return None, 0, n_matches, f"Only {n_matches} matches found (need {min_matches}). Try increasing ratio_threshold."

    # RANSAC affine estimation
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    affine_matrix, inliers = cv2.estimateAffine2D(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=2000,
        confidence=0.99
    )

    if affine_matrix is None:
        return None, 0, n_matches, "RANSAC failed to find valid affine transform"

    n_inliers = int(inliers.sum()) if inliers is not None else 0
    return affine_matrix, n_inliers, n_matches, None


def register_lowres_to_hires_single_plane(lowres_2d, hires_2d, config=None):
    """
    Register a single low-res 2D image to a high-res 2D image using SIFT + RANSAC affine.

    This function:
    1. Upsamples low-res to match high-res dimensions
    2. Detects SIFT features in both images
    3. Matches features and computes affine transform via RANSAC
    4. Returns transform parameters and aligned image for QA

    Parameters
    ----------
    lowres_2d : ndarray (Y, X)
        Low-resolution 2P mean image (single plane)
    hires_2d : ndarray (Y, X)
        High-resolution stitched image (single plane)
    config : dict, optional
        Configuration from manifest params.lowres_to_hires_registration:
        - n_features: Max SIFT keypoints (default 5000)
        - ratio_threshold: Lowe's ratio test threshold (default 0.75)
        - ransac_reproj_threshold: RANSAC threshold in pixels (default 5.0)
        - min_matches: Minimum matches required (default 10)

    Returns
    -------
    transform_params : dict
        - 'scale_x': X scaling factor
        - 'scale_y': Y scaling factor
        - 'rotation': rotation angle in degrees
        - 'shear': shear parameter
        - 'shift_y': Y translation in pixels
        - 'shift_x': X translation in pixels
        - 'similarity': normalized cross-correlation score
        - 'method': 'sift'
        - 'n_inliers': number of RANSAC inliers
        - 'n_matches': number of good feature matches
        - 'affine_matrix': raw 2x3 OpenCV affine matrix

    lowres_aligned : ndarray
        Low-res image transformed to high-res space (for QA visualization)

    Raises
    ------
    RuntimeError
        If SIFT registration fails (not enough matches, RANSAC failure, etc.)
        Error message includes guidance on using BigWarp for manual landmarks.
    """
    import cv2

    config = config or {}

    # Calculate scale from image dimensions
    scale_y = hires_2d.shape[0] / lowres_2d.shape[0]
    scale_x = hires_2d.shape[1] / lowres_2d.shape[1]

    # Upsample low-res to match high-res dimensions
    lowres_upsampled = resize(
        lowres_2d,
        hires_2d.shape,
        order=1,  # Linear interpolation
        preserve_range=True,
        anti_aliasing=True
    )

    # Run SIFT registration
    affine_matrix, n_inliers, n_matches, error_msg = _sift_affine_registration(
        lowres_upsampled, hires_2d, config
    )

    if affine_matrix is None:
        raise RuntimeError(
            f"SIFT registration failed: {error_msg}\n\n"
            f"Recommended: Use BigWarp to create manual landmarks, then run with:\n"
            f"  automation.lowres_to_hires = 'landmarks'"
        )

    # Apply transform for QA visualization
    lowres_aligned = cv2.warpAffine(
        lowres_upsampled.astype(np.float32),
        affine_matrix,
        (hires_2d.shape[1], hires_2d.shape[0]),  # (width, height)
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # Decompose OpenCV affine matrix to parameter format
    # affine_matrix is [[a, b, tx], [c, d, ty]]
    a, b, tx = affine_matrix[0]
    c, d, ty = affine_matrix[1]

    # Extract scale, rotation, shear from matrix
    extracted_scale_x = np.sqrt(a**2 + c**2)
    extracted_scale_y = np.sqrt(b**2 + d**2)
    rotation = np.rad2deg(np.arctan2(c, a))
    shear = (a*b + c*d) / (extracted_scale_x * extracted_scale_y) if extracted_scale_x * extracted_scale_y > 0 else 0.0

    # Compute similarity score
    similarity = compute_normalized_cross_correlation(hires_2d, lowres_aligned)

    transform_params = {
        'scale_x': float(extracted_scale_x),
        'scale_y': float(extracted_scale_y),
        'rotation': float(rotation),
        'shear': float(shear),
        'shift_y': float(ty),
        'shift_x': float(tx),
        'similarity': float(similarity),
        'method': 'sift',
        'n_inliers': n_inliers,
        'n_matches': n_matches,
        'affine_matrix': affine_matrix.tolist()  # For JSON serialization
    }

    return transform_params, lowres_aligned


def refine_lowres_to_hires_with_tiles(lowres_aligned, hires_2d, masks, config=None):
    """
    Refine low-res to high-res alignment using direct per-tile SIFT corrections.

    This function applies each tile's affine correction DIRECTLY to its region
    (no griddata interpolation), with weighted blending at tile boundaries.

    Key features (from notebook optimization):
    1. Direct per-tile application - significantly better than griddata/RBF (+7% NCC)
    2. NCC quality gating - rejects tiles that worsen alignment
    3. Feathered blending at tile boundaries (blend_width parameter)
    4. Feature size and spatial distance filtering for robust matching

    Parameters
    ----------
    lowres_aligned : ndarray (Y, X)
        Low-res image after global SIFT alignment (same shape as hires)
    hires_2d : ndarray (Y, X)
        High-res target image
    masks : ndarray (Y, X)
        Masks after global SIFT transform (to be refined)
    config : dict, optional
        Configuration parameters:
        - tile_size: Tile size in pixels (default 600, ~300 in lowres space)
        - tile_overlap: Overlap fraction (default 0.3)
        - n_features: SIFT features per tile (default 100, quality over quantity)
        - min_matches: Minimum matches for valid tile (default 4)
        - max_shift: Maximum allowed shift per tile in pixels (default 40)
        - blend_width: Width of blending region at tile edges (default 40)
        - min_feature_size: Minimum keypoint size to keep (default 8)
        - max_spatial_distance: Max distance between matched positions (default 40)
        - require_ncc_improvement: Reject tiles that worsen NCC (default True)
        - min_overlap_fraction: Min valid content overlap in tile (default 0.5)

    Returns
    -------
    masks_refined : ndarray (Y, X)
        Masks with tile-based refinement applied
    lowres_refined : ndarray (Y, X)
        Low-res image with refinement applied (for QA)
    tile_info : list of dict
        Per-tile results for diagnostics
    """
    import cv2
    from tqdm import tqdm

    config = config or {}

    # Tile parameters (optimized defaults from notebook)
    tile_size = config.get('tile_size', 600)
    tile_overlap = config.get('tile_overlap', 0.3)
    blend_width = config.get('blend_width', 40)
    max_shift = config.get('max_shift', 40)
    min_overlap_fraction = config.get('min_overlap_fraction', 0.5)
    require_ncc_improvement = config.get('require_ncc_improvement', True)

    # SIFT parameters (optimized for cell-like features)
    n_features = config.get('n_features', 100)
    min_matches = config.get('min_matches', 4)
    min_feature_size = config.get('min_feature_size', 8)
    max_spatial_distance = config.get('max_spatial_distance', 40)
    ratio_threshold = config.get('ratio_threshold', 0.7)
    ransac_reproj_threshold = config.get('ransac_reproj_threshold', 3.0)
    percentile_norm = config.get('percentile_norm', (2, 98))  # Percentile-based normalization
    verbose = config.get('verbose', False)

    h, w = hires_2d.shape
    step = max(1, int(tile_size * (1 - tile_overlap)))

    # Generate tile positions - ensure FULL coverage including edges
    tile_positions = []
    y = 0
    while y < h:
        x = 0
        while x < w:
            y0, y1 = y, min(y + tile_size, h)
            x0, x1 = x, min(x + tile_size, w)
            # Relaxed minimum size (tile_size // 4) to include edge tiles
            if (y1 - y0) >= tile_size // 4 and (x1 - x0) >= tile_size // 4:
                tile_positions.append((y0, y1, x0, x1))
            x += step
        y += step

    # SIFT config for tiles
    tile_sift_config = {
        'n_features': n_features,
        'ratio_threshold': ratio_threshold,
        'ransac_reproj_threshold': ransac_reproj_threshold,
        'min_matches': min_matches,
        'min_feature_size': min_feature_size,
        'max_spatial_distance': max_spatial_distance,
        'percentile_norm': percentile_norm,
    }

    # Accumulators for direct tile application with weighted blending
    warped_img_sum = np.zeros((h, w), dtype=np.float64)
    weight_sum = np.zeros((h, w), dtype=np.float64)
    # For masks: use winner-takes-all (highest weight wins) instead of weighted average
    # Weighted averaging corrupts label IDs (e.g., averaging label 5 and 10 gives 7.5 -> wrong cell!)
    masks_result = masks.copy()  # Start with original masks
    masks_max_weight = np.zeros((h, w), dtype=np.float64)  # Track highest weight per pixel

    tile_results = []
    # Track failure reasons for diagnostics
    failure_reasons = {'overlap': 0, 'no_features': 0, 'few_inliers': 0, 'bad_scale': 0, 'bad_shift': 0, 'ncc_worse': 0, 'exception': 0}

    for y0, y1, x0, x1 in tqdm(tile_positions, desc="Tile SIFT refinement", leave=False):
        tile_h, tile_w = y1 - y0, x1 - x0

        # Extract tile regions
        lowres_tile = lowres_aligned[y0:y1, x0:x1].astype(np.float32)
        hires_tile = hires_2d[y0:y1, x0:x1].astype(np.float32)
        # Keep masks as original dtype for warpAffine - INTER_NEAREST preserves label values
        masks_tile = masks[y0:y1, x0:x1]

        # Tile center (for diagnostic output)
        cy = (y0 + y1) / 2
        cx = (x0 + x1) / 2

        # Check for sufficient valid content overlap (use normalized threshold like notebook)
        lr_norm = (lowres_tile - lowres_tile.min()) / max(lowres_tile.max() - lowres_tile.min(), 1e-6)
        hr_norm = (hires_tile - hires_tile.min()) / max(hires_tile.max() - hires_tile.min(), 1e-6)
        lowres_valid = lr_norm > 0.01
        hires_valid = hr_norm > 0.01
        overlap_mask = lowres_valid & hires_valid
        overlap_fraction = overlap_mask.sum() / overlap_mask.size

        if overlap_fraction < min_overlap_fraction:
            tile_results.append({'cy': cy, 'success': False, 'reason': 'overlap'})
            failure_reasons['overlap'] += 1
            continue

        # Compute NCC before correction
        ncc_before = compute_normalized_cross_correlation(lowres_tile, hires_tile)

        # For smaller edge tiles, scale up n_features to maintain feature density
        tile_area = tile_h * tile_w
        full_tile_area = tile_size * tile_size
        area_ratio = tile_area / full_tile_area
        if area_ratio < 0.5:
            adaptive_n_features = max(n_features, int(n_features / area_ratio))
            tile_sift_config_local = tile_sift_config.copy()
            tile_sift_config_local['n_features'] = adaptive_n_features
        else:
            tile_sift_config_local = tile_sift_config

        # Try local SIFT registration
        try:
            affine_matrix, n_inliers, n_matches, error_msg = _sift_affine_registration(
                lowres_tile, hires_tile, tile_sift_config_local
            )

            if affine_matrix is None:
                tile_results.append({'cy': cy, 'success': False, 'reason': 'no_features', 'error': error_msg})
                failure_reasons['no_features'] += 1
                continue

            if n_inliers < min_matches:
                tile_results.append({'cy': cy, 'success': False, 'reason': 'few_inliers', 'n_inliers': n_inliers})
                failure_reasons['few_inliers'] += 1
                continue

            # Check for unreasonable scale/rotation (should be ~identity for refinement)
            a, b = affine_matrix[0, 0], affine_matrix[0, 1]
            c, d = affine_matrix[1, 0], affine_matrix[1, 1]
            scale_x = np.sqrt(a**2 + c**2)
            scale_y = np.sqrt(b**2 + d**2)
            dx, dy = affine_matrix[0, 2], affine_matrix[1, 2]

            # Reject if scale deviates more than 5% from identity or shift exceeds max
            scale_ok = (0.95 <= scale_x <= 1.05) and (0.95 <= scale_y <= 1.05)
            shift_ok = (abs(dx) <= max_shift) and (abs(dy) <= max_shift)

            if not scale_ok:
                tile_results.append({'cy': cy, 'success': False, 'reason': 'bad_scale', 'scale': (scale_x, scale_y)})
                failure_reasons['bad_scale'] += 1
                continue

            if not shift_ok:
                tile_results.append({'cy': cy, 'success': False, 'reason': 'bad_shift', 'shift': (dx, dy)})
                failure_reasons['bad_shift'] += 1
                continue

            # Apply affine correction to get warped tile
            warped_lowres_tile = cv2.warpAffine(
                lowres_tile, affine_matrix, (tile_w, tile_h),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )

            # NCC quality gating: reject if correction worsens alignment
            ncc_after = compute_normalized_cross_correlation(warped_lowres_tile, hires_tile)
            if require_ncc_improvement and ncc_after < ncc_before:  # Changed <= to < (match notebook)
                tile_results.append({'cy': cy, 'success': False, 'reason': 'ncc_worse', 'ncc_before': ncc_before, 'ncc_after': ncc_after})
                failure_reasons['ncc_worse'] += 1
                continue

            # Apply same affine to masks (nearest-neighbor for labels)
            warped_masks_tile = cv2.warpAffine(
                masks_tile, affine_matrix, (tile_w, tile_h),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE
            )

            # Create feathered weight mask (1 in center, tapered to 0 at edges)
            weight = np.ones((tile_h, tile_w), dtype=np.float64)
            for i in range(min(blend_width, tile_h // 2)):
                factor = i / blend_width
                weight[i, :] *= factor
                weight[tile_h - 1 - i, :] *= factor
            for j in range(min(blend_width, tile_w // 2)):
                factor = j / blend_width
                weight[:, j] *= factor
                weight[:, tile_w - 1 - j] *= factor

            # Accumulate weighted contributions for image (blending is fine for images)
            warped_img_sum[y0:y1, x0:x1] += warped_lowres_tile * weight
            weight_sum[y0:y1, x0:x1] += weight

            # For masks: winner-takes-all (update only where this tile has higher weight)
            # This preserves label IDs instead of corrupting them via averaging
            update_mask = weight > masks_max_weight[y0:y1, x0:x1]
            masks_result[y0:y1, x0:x1] = np.where(update_mask, warped_masks_tile, masks_result[y0:y1, x0:x1])
            masks_max_weight[y0:y1, x0:x1] = np.maximum(masks_max_weight[y0:y1, x0:x1], weight)

            tile_results.append({'cy': cy, 'success': True, 'ncc_before': ncc_before, 'ncc_after': ncc_after, 'shift': (dx, dy)})

        except Exception as e:
            tile_results.append({'cy': cy, 'success': False, 'reason': 'exception', 'error': str(e)})
            failure_reasons['exception'] += 1

    # Print failure summary
    if verbose:
        print(f"  Tile failure breakdown: {failure_reasons}")

    # If no successful tiles, return original
    n_success = sum(1 for t in tile_results if t.get('success'))
    if n_success < 1:
        return masks.copy(), lowres_aligned.copy(), tile_results

    # Compute final result: divide by total weight for image, use winner-takes-all for masks
    no_coverage_mask = weight_sum < 0.01
    weight_sum_safe = np.maximum(weight_sum, 1e-6)

    lowres_refined = warped_img_sum / weight_sum_safe

    # Fill uncovered regions with original values
    lowres_refined[no_coverage_mask] = lowres_aligned[no_coverage_mask]
    # masks_result already has original values where no tile was applied (initialized from masks.copy())

    # Ensure masks are integer type
    masks_refined = masks_result.astype(masks.dtype)

    return masks_refined, lowres_refined, tile_results


def apply_lowres_to_hires_transform(lowres_masks, transform_params, hires_shape):
    """
    Apply low-res to high-res transformation to masks.

    Uses nearest-neighbor interpolation to preserve label IDs.
    Supports both rigid (scale + rotation + translation) and affine transforms.

    Parameters
    ----------
    lowres_masks : ndarray (Y, X)
        Low-resolution masks (integer labels)
    transform_params : dict
        Transform parameters from register_lowres_to_hires_single_plane()
        For rigid: 'scale', 'rotation', 'shift_y', 'shift_x', 'method'='rigid'
        For affine: 'scale_x', 'scale_y', 'rotation', 'shear', 'shift_y', 'shift_x', 'method'='affine'
    hires_shape : tuple
        Target high-res shape (Y, X)

    Returns
    -------
    hires_masks : ndarray (Y, X)
        Masks transformed to high-res space (integer labels preserved)
    """

    method = transform_params.get('method', 'sift')

    if method in ('affine', 'sift'):
        # Use affine transformation
        from skimage.transform import AffineTransform, warp
        import cv2

        # Step 1: Upsample masks to hires dimensions first (nearest-neighbor to preserve labels)
        # The SIFT affine was computed on upsampled images, so we need masks in that space
        masks_upsampled = resize(lowres_masks, hires_shape,
                                 order=0,  # Nearest-neighbor
                                 preserve_range=True,
                                 anti_aliasing=False).astype(lowres_masks.dtype)

        # Step 2: Apply the SIFT affine correction (operates in hires space)
        # Use the raw affine matrix directly if available
        if 'affine_matrix' in transform_params:
            affine_matrix_2x3 = np.array(transform_params['affine_matrix'])
            masks_transformed = cv2.warpAffine(
                masks_upsampled.astype(np.float32),
                affine_matrix_2x3,
                (hires_shape[1], hires_shape[0]),  # (width, height)
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            ).astype(lowres_masks.dtype)
        else:
            # Fallback: build affine from decomposed params
            theta_rad = np.deg2rad(transform_params['rotation'])
            cos_theta = np.cos(theta_rad)
            sin_theta = np.sin(theta_rad)

            affine_matrix = np.array([
                [transform_params['scale_x'] * cos_theta,
                 -transform_params['scale_x'] * sin_theta + transform_params['shear']],
                [transform_params['scale_y'] * sin_theta,
                 transform_params['scale_y'] * cos_theta]
            ])

            tform = AffineTransform(
                matrix=np.vstack([
                    np.hstack([affine_matrix,
                              [[transform_params['shift_x']],
                               [transform_params['shift_y']]]]),
                    [0, 0, 1]
                ])
            )

            masks_transformed = warp(masks_upsampled, tform.inverse,
                                    output_shape=hires_shape,
                                    order=0,  # Nearest-neighbor
                                    preserve_range=True).astype(lowres_masks.dtype)

        return masks_transformed

    else:
        # Original rigid transformation (backward compatible)
        # Step 1: Upsample by true physical scale (nearest-neighbor)
        true_scale = transform_params['scale']
        upsampled_shape_y = int(lowres_masks.shape[0] * true_scale)
        upsampled_shape_x = int(lowres_masks.shape[1] * true_scale)

        masks_upsampled = resize(lowres_masks, (upsampled_shape_y, upsampled_shape_x),
                                order=0,  # Nearest-neighbor
                                preserve_range=True,
                                anti_aliasing=False).astype(lowres_masks.dtype)

        # Step 2: Rotate (nearest-neighbor)
        theta = transform_params['rotation']
        if theta != 0:
            masks_rotated = rotate(masks_upsampled, theta,
                                  reshape=False,
                                  order=0,  # Nearest-neighbor
                                  mode='constant',
                                  cval=0).astype(lowres_masks.dtype)
        else:
            masks_rotated = masks_upsampled

        # Step 3: Handle size mismatch - pad or crop to match hires_shape
        if masks_rotated.shape != hires_shape:
            # Create output array matching high-res shape
            masks_padded = np.zeros(hires_shape, dtype=lowres_masks.dtype)

            # Calculate overlap region
            min_y = min(masks_rotated.shape[0], hires_shape[0])
            min_x = min(masks_rotated.shape[1], hires_shape[1])

            # Copy overlapping region (top-left alignment)
            masks_padded[:min_y, :min_x] = masks_rotated[:min_y, :min_x]
        else:
            masks_padded = masks_rotated

        # Step 4: Shift (nearest-neighbor via roll)
        shift_y = int(np.round(transform_params['shift_y']))
        shift_x = int(np.round(transform_params['shift_x']))

        masks_shifted = np.roll(masks_padded, shift_y, axis=0)
        masks_shifted = np.roll(masks_shifted, shift_x, axis=1)

        return masks_shifted.astype(lowres_masks.dtype)


# =============================================================================
# V8 REGISTRATION: 5-TIER PROGRESSIVE REFINEMENT
# =============================================================================
# Ported from flexible_local_alignment_v8_CIM132-Copy1.ipynb
# Strategy: global grid → double affine → double local tiles


# ---- Z-Map Extrapolation ----

def _fit_plane(coords, z, yy, xx):
    """Least-squares plane: z = a*y + b*x + c."""
    M = np.column_stack([coords, np.ones(len(z))])
    c = np.linalg.lstsq(M, z, rcond=None)[0]
    return c[0] * yy + c[1] * xx + c[2]


def _fit_quad(coords, z, yy, xx):
    """Quadratic surface: z = a*y^2 + b*x^2 + c*y*x + d*y + e*x + f."""
    M = np.column_stack([coords[:, 0]**2, coords[:, 1]**2,
                         coords[:, 0] * coords[:, 1],
                         coords, np.ones(len(z))])
    c = np.linalg.lstsq(M, z, rcond=None)[0]
    return c[0] * yy**2 + c[1] * xx**2 + c[2] * yy * xx + c[3] * yy + c[4] * xx + c[5]


def build_z_variant(coords, z_vals, shape, extrap='plane'):
    """Linear inside convex hull, *extrap* outside.

    Parameters
    ----------
    coords : ndarray (N, 2)
        Landmark coordinates (y, x).
    z_vals : ndarray (N,)
        Z values at each landmark.
    shape : tuple (ny, nx)
        Output shape.
    extrap : str
        Extrapolation method: 'nearest', 'plane', or 'quadratic'.

    Returns
    -------
    z_map : ndarray (ny, nx)
    outside_mask : ndarray bool (ny, nx)
        True where linear interpolation was undefined (extrapolated).
    """
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    z_lin = LinearNDInterpolator(coords, z_vals)(yy, xx)
    outside = np.isnan(z_lin)
    if extrap == 'nearest':
        z_fill = NearestNDInterpolator(coords, z_vals)(yy, xx)
    elif extrap == 'plane':
        z_fill = _fit_plane(coords, z_vals, yy, xx)
    elif extrap == 'quadratic':
        z_fill = _fit_quad(coords, z_vals, yy, xx)
    else:
        raise ValueError(f"Unknown extrap: {extrap}")
    return np.where(outside, z_fill, z_lin), outside


def build_z_quad_blend(coords, z_vals, shape, max_dist=300):
    """Quadratic near hull, blending smoothly to plane beyond max_dist px.

    Uses linear interpolation inside the convex hull of landmarks,
    quadratic extrapolation near the hull boundary, and blends to
    a plane fit at distances beyond max_dist. This prevents the
    quadratic from diverging far from landmarks.
    """
    z_quad, outside = build_z_variant(coords, z_vals, shape, 'quadratic')
    z_plane, _ = build_z_variant(coords, z_vals, shape, 'plane')
    dist = distance_transform_edt(outside.astype(np.float64))
    alpha = np.clip(dist / max_dist, 0, 1)
    return z_quad * (1 - alpha) + z_plane * alpha


# ---- Hull Mask Functions ----

def create_fov_mask_convex_hull(twop_binary, margin=30):
    """FOV mask from convex hull of 2P cells + dilation.

    More accurate than binary dilation (create_fov_mask) because it
    creates a convex boundary around all cells rather than dilating
    each cell individually.
    """
    from matplotlib.path import Path as MplPath
    ys, xs = np.where(twop_binary)
    if len(ys) < 3:
        return binary_dilation(twop_binary, iterations=max(margin, 1))
    step = max(1, len(ys) // 5000)
    pts = np.column_stack([xs[::step], ys[::step]])
    hull = ConvexHull(pts)
    path = MplPath(pts[hull.vertices])
    yy, xx = np.mgrid[:twop_binary.shape[0], :twop_binary.shape[1]]
    mask = path.contains_points(np.column_stack([xx.ravel(), yy.ravel()])).reshape(twop_binary.shape)
    return binary_dilation(mask, iterations=margin) if margin > 0 else mask


def create_landmark_hull_mask(landmarks_df, shape, margin=0):
    """Convex hull of TPS landmarks with optional erosion/dilation.

    Parameters
    ----------
    landmarks_df : DataFrame
        Must have 'hcr_x_px' and 'hcr_y_px' columns.
    shape : tuple (ny, nx)
        Output mask shape.
    margin : int
        Positive = dilate hull, negative = erode inward.
        Negative values create an "inward hull" that restricts the
        global search to the region where landmarks provide reliable
        TPS coverage.
    """
    from matplotlib.path import Path as MplPath
    coords = landmarks_df[['hcr_x_px', 'hcr_y_px']].values
    if len(coords) < 3:
        raise ValueError(f"Need >= 3 landmarks, got {len(coords)}")
    hull = ConvexHull(coords)
    path = MplPath(coords[hull.vertices])
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    mask = path.contains_points(np.column_stack([xx.ravel(), yy.ravel()])).reshape(shape)
    if margin > 0:
        mask = binary_dilation(mask, iterations=margin)
    elif margin < 0:
        mask = binary_erosion(mask, iterations=abs(margin))
    return mask


# ---- FFT-IoU with Moving Mask ----

def _extract_shift_region(full_map, xy_max):
    """Extract +/-xy_max shift region from FFT cross-correlation output.

    Index (i, j) in result corresponds to shift (i - xy_max, j - xy_max).
    """
    top = full_map[:xy_max + 1, :]
    bot = full_map[-xy_max:, :]
    reg_y = np.vstack([bot, top])
    left = reg_y[:, -xy_max:]
    right = reg_y[:, :xy_max + 1]
    return np.hstack([left, right])


def fft_iou_fov_precomputed(twop_rfft, fov_rfft, sum_twop, hcr_patch, max_shift,
                             fft_shape, peak_mask_radius=5):
    """FFT-IoU with pre-computed 2P and FOV transforms.

    The FOV mask moves with the 2P template, so the union correctly
    adapts per-shift. Also computes peak_ratio for match confidence.

    Returns
    -------
    dy, dx : int
        Best shift.
    iou_best : float
        IoU at best shift.
    peak_ratio : float
        Ratio of best IoU to second-best (sharpness measure).
    """
    m2 = hcr_patch.astype(np.float64)
    sum2 = m2.sum()
    if sum_twop == 0 or sum2 == 0:
        return 0, 0, 0.0, 0.0

    fft_h, fft_w = fft_shape
    hcr_rfft = rfft2(m2, s=fft_shape)

    inter = np.clip(irfft2(np.conj(twop_rfft) * hcr_rfft, s=fft_shape), 0, None)
    fov_hcr = np.clip(irfft2(np.conj(fov_rfft) * hcr_rfft, s=fft_shape), 0, None)
    union = sum_twop + fov_hcr - inter
    iou_map = np.where(union > 0, inter / union, 0.0)

    # Extract best shift from search region
    sr = np.fft.fftshift(iou_map)
    cy, cx = fft_h // 2, fft_w // 2
    region = sr[cy - max_shift:cy + max_shift + 1, cx - max_shift:cx + max_shift + 1]
    bi, bj = np.unravel_index(np.argmax(region), region.shape)
    peak_val = float(region[bi, bj])

    # Peak ratio: mask around peak, find second-highest
    region_m = region.copy()
    r0m = max(0, bi - peak_mask_radius)
    r1m = min(region_m.shape[0], bi + peak_mask_radius + 1)
    c0m = max(0, bj - peak_mask_radius)
    c1m = min(region_m.shape[1], bj + peak_mask_radius + 1)
    region_m[r0m:r1m, c0m:c1m] = 0
    second_peak = float(region_m.max())
    peak_ratio = peak_val / (second_peak + 1e-8)

    return bi - max_shift, bj - max_shift, peak_val, peak_ratio


# ---- Global Search (Moving-Mask IoU) ----

def global_search_moving_iou(twop_binary, hcr_3d_binary, z_map, mask,
                              z_range=(-15, 15), xy_max=500):
    """FFT-based global grid search using moving-mask IoU.

    The mask travels with the 2P template:
      I(dy,dx) = xcorr(2P * mask, HCR)   -- intersection
      T = sum(2P * mask)                  -- constant (template sum)
      H(dy,dx) = xcorr(mask, HCR)         -- HCR under mask (varies)
      IoU = I / (T + H - I)

    Parameters
    ----------
    twop_binary : ndarray (ny, nx)
        2P binary mask.
    hcr_3d_binary : ndarray (nz, ny, nx)
        HCR 3D binary volume.
    z_map : ndarray (ny, nx)
        Baseline Z-coordinate map.
    mask : ndarray (ny, nx)
        Search mask (e.g., inward hull of landmarks).
    z_range : tuple (min, max)
        Z-offset search range.
    xy_max : int
        Maximum XY shift to search.

    Returns
    -------
    dict with keys: 'iou', 'theta', 'dz', 'dy', 'dx'.
    theta is always 0.0 (TPS handles orientation).
    """
    h, w = twop_binary.shape
    fft_h = 1 << int(np.ceil(np.log2(h + xy_max)))
    fft_w = 1 << int(np.ceil(np.log2(w + xy_max)))
    fft_shape = (fft_h, fft_w)

    mask_f = mask.astype(np.float64)
    mask_rfft = rfft2(mask_f, s=fft_shape)

    # Moving-mask template: 2P * mask
    twop_masked = twop_binary.astype(np.float64) * mask_f
    twop_masked_rfft = rfft2(twop_masked, s=fft_shape)
    T = max(twop_masked.sum(), 1.0)

    dz_vals = list(range(z_range[0], z_range[1] + 1))
    best = {'iou': -1.0, 'theta': 0.0, 'dz': 0, 'dy': 0, 'dx': 0}

    for i, dz in enumerate(dz_vals):
        if i % max(1, len(dz_vals) // 5) == 0:
            print(f"  global search dz {i + 1}/{len(dz_vals)} (dz={dz:+d})...", flush=True)

        hcr_2d = sample_hcr_binary_at_zmap(hcr_3d_binary, z_map, z_offset=dz)
        hcr_f = hcr_2d.astype(np.float64)
        hcr_rfft = rfft2(hcr_f, s=fft_shape)

        # I(dy,dx) = xcorr(2P*mask, HCR)
        I_full = irfft2(np.conj(twop_masked_rfft) * hcr_rfft, s=fft_shape)
        # H(dy,dx) = xcorr(mask, HCR)
        H_full = irfft2(np.conj(mask_rfft) * hcr_rfft, s=fft_shape)

        # Extract +/-xy_max region
        I_reg = np.maximum(_extract_shift_region(I_full, xy_max), 0)
        H_reg = np.maximum(_extract_shift_region(H_full, xy_max), 1)

        # IoU = I / (T + H - I)
        iou_map = I_reg / np.maximum(T + H_reg - I_reg, 1)

        # Find peak
        idx = np.unravel_index(iou_map.argmax(), iou_map.shape)
        dy = int(idx[0]) - xy_max
        dx = int(idx[1]) - xy_max
        score = float(iou_map[idx])

        if score > best['iou']:
            best = {'iou': score, 'theta': 0.0, 'dz': int(dz), 'dy': dy, 'dx': dx}

    print(f"  Best: IoU={best['iou']:.4f} dy={best['dy']:+d} dx={best['dx']:+d} dz={best['dz']:+d}")
    return best


# ---- Cell-Level Displacement Extraction ----

def extract_centroids(labels_2d):
    """Return (N, 2) array of (cy, cx) from label image."""
    props = regionprops(labels_2d.astype(np.int32))
    return np.array([[p.centroid[0], p.centroid[1]] for p in props])


def compute_per_cell_ious(twop_labels, hcr_sampled, margin=5):
    """Compute per-cell IoU between labeled 2P cells and HCR binary mask.

    Returns dict {label_id: iou}.
    """
    hcr_bin = hcr_sampled > 0
    ny, nx = twop_labels.shape
    props = regionprops(twop_labels.astype(np.int32))
    result = {}
    for p in props:
        r0, c0, r1, c1 = p.bbox
        r0, c0 = max(0, r0 - margin), max(0, c0 - margin)
        r1, c1 = min(ny, r1 + margin), min(nx, c1 + margin)
        cell_crop = (twop_labels[r0:r1, c0:c1] == p.label)
        hcr_crop = hcr_bin[r0:r1, c0:c1]
        inter = (cell_crop & hcr_crop).sum()
        union = (cell_crop | hcr_crop).sum()
        result[p.label] = float(inter / union) if union > 0 else 0.0
    return result


def compute_tile_defaults(tile_size):
    """Auto-compute per-tile-size defaults for cascade stages.

    For known tile sizes (300, 100, 50), returns v13-validated values.
    For arbitrary sizes, interpolates from observed ratios.
    """
    KNOWN = {
        300: dict(overlap=0.5, min_cells=8, min_cells_affine=3, rbf_smoothing=50,
                  clamp_mult=2.0, border_spacing=120),
        100: dict(overlap=0.5, min_cells=4, min_cells_affine=3, rbf_smoothing=100,
                  clamp_mult=1.5, border_spacing=100),
        50:  dict(overlap=0.5, min_cells=2, min_cells_affine=3, rbf_smoothing=250,
                  clamp_mult=1.2, border_spacing=50),
    }
    if tile_size in KNOWN:
        return KNOWN[tile_size].copy()

    # Auto-compute from ratios observed across known sizes
    min_cells = max(2, min(12, int(tile_size / 37)))
    rbf_smoothing = max(30, min(500, int(15000 / tile_size)))
    clamp_mult = max(1.1, min(2.5, 1.0 + tile_size / 300))
    border_spacing = max(30, min(200, int(tile_size * 0.35)))

    return dict(overlap=0.5, min_cells=min_cells, min_cells_affine=3,
                rbf_smoothing=rbf_smoothing, clamp_mult=clamp_mult,
                border_spacing=border_spacing)


def compute_adaptive_matching_params(cell_df, tile_size, stage_idx=0,
                                     percentile=95, multiplier=1.5,
                                     tile_ratio=0.3, patch_ratio=2.0,
                                     patch_tile_ratio=0.6,
                                     search_xy_min=3, search_xy_max=50,
                                     search_z_default=2, search_z_fine=1):
    """Compute adaptive search/patch sizes from current residual magnitudes.

    For stage 0 (first cascade stage): uses tile-proportional defaults.
    For subsequent stages: measures residuals from post-correction re-match.

    Returns (patch_radius, search_xy, search_z).
    """
    if stage_idx == 0 or len(cell_df) == 0:
        # First stage or no data: tile-proportional defaults
        patch_r = int(tile_size * patch_tile_ratio)
        search_xy = int(tile_size * tile_ratio)
        search_z = search_z_default
        return patch_r, search_xy, search_z

    resid_mag = np.sqrt(cell_df.dy.values**2 + cell_df.dx.values**2)
    p = np.percentile(resid_mag, percentile)

    # Search range: enough to capture residuals with margin
    search_xy = max(search_xy_min, int(np.ceil(p * multiplier)))
    # Cap at fraction of tile size
    search_xy = min(search_xy, int(tile_size * tile_ratio), search_xy_max)
    # Patch must be at least patch_ratio * search
    patch_radius = max(search_xy, int(search_xy * patch_ratio))
    # Cap at tile size
    patch_radius = min(patch_radius, tile_size)

    search_z = search_z_fine

    return patch_radius, search_xy, search_z


def find_cell_displacements(twop_binary, twop_labels, hcr_3d_bin, z_map, centroids,
                            patch_radius=60, search_xy=30, search_z=3,
                            fov_dilation=10, min_peak_ratio=1.3):
    """Find (dy, dx, dz) per 2P cell via FFT-IoU on binary patches.

    For each cell:
    1. Extract patch around centroid (clamped to image bounds)
    2. Pre-compute 2P and FOV FFTs (reused across Z)
    3. Search across Z-offsets, finding best (dy, dx, dz) via FFT-IoU
    4. Compute single-cell IoU at found shift
    5. Record peak_ratio for match confidence

    Parameters
    ----------
    fov_dilation : int
        Number of binary dilation iterations for FOV mask (default=10).

    Returns list of dicts with keys: cy, cx, label, dy, dx, dz,
    iou_best, iou_zero, iou_gain, cell_iou, cell_iou_zero, cell_iou_gain,
    peak_ratio.
    """
    ny, nx = twop_binary.shape
    r = patch_radius
    full_size = 2 * r

    if full_size > ny or full_size > nx:
        print(f"  WARNING: patch size {full_size} > image ({ny}x{nx}), reducing patch_radius")
        r = min(ny, nx) // 4
        full_size = 2 * r
        print(f"  New patch_radius={r}, patch_size={full_size}")

    # Pre-sample HCR binary at all Z offsets
    t0 = time.time()
    hcr_slices = {}
    for dz in range(-search_z, search_z + 1):
        hcr_slices[dz] = sample_hcr_binary_at_zmap(hcr_3d_bin, z_map, z_offset=dz)
    print(f"  Pre-sampled {len(hcr_slices)} HCR Z-slices in {time.time() - t0:.1f}s")

    fov_dilated = binary_dilation(twop_binary, iterations=fov_dilation)

    # Search radius: must fit within patch
    cell_search_xy = min(search_xy, (full_size - 10) // 2)
    print(f"  Patch: {full_size}x{full_size}, search: +/-{cell_search_xy}px")
    print(f"  Patch:search ratio: {full_size / max(1, 2 * cell_search_xy):.1f}:1")

    # Pre-compute FFT shape (same for all cells with same patch size)
    fft_h = 1 << int(np.ceil(np.log2(full_size)))
    fft_shape = (fft_h, fft_h)

    results = []
    n_shifted = 0
    n_ambiguous = 0
    for cy, cx in tqdm(centroids, desc="Per-cell matching"):
        iy, ix = int(round(cy)), int(round(cx))

        # Find the label at this centroid
        cell_label = int(twop_labels[min(iy, ny - 1), min(ix, nx - 1)])

        # Extraction bounds: keep full patch size, shift to stay in image
        py0 = max(0, min(iy - r, ny - full_size))
        py1 = py0 + full_size
        px0 = max(0, min(ix - r, nx - full_size))
        px1 = px0 + full_size

        is_shifted = (py0 != iy - r) or (px0 != ix - r)
        if is_shifted:
            n_shifted += 1

        twop_patch = twop_binary[py0:py1, px0:px1]
        if twop_patch.sum() < 20:
            continue

        fov_local = fov_dilated[py0:py1, px0:px1]
        labels_patch = twop_labels[py0:py1, px0:px1]
        target_mask = (labels_patch == cell_label) if cell_label > 0 else None

        # Baseline: IoU at zero shift, dz=0
        hcr_zero = hcr_slices[0][py0:py1, px0:px1]
        iou_zero = compute_iou(twop_patch, hcr_zero, fov_mask=fov_local)

        # Baseline single-cell IoU
        cell_iou_zero = 0.0
        if target_mask is not None and target_mask.sum() > 0:
            ci = (target_mask & hcr_zero).sum()
            cu = (target_mask | (hcr_zero & fov_local)).sum()
            cell_iou_zero = float(ci / cu) if cu > 0 else 0.0

        # PRE-COMPUTE: 2P and FOV FFTs (once per cell, reused across all Z)
        m1 = twop_patch.astype(np.float64)
        fov_f = fov_local.astype(np.float64)
        twop_rfft = rfft2(m1, s=fft_shape)
        fov_rfft = rfft2(fov_f, s=fft_shape)
        sum_twop = m1.sum()

        best_iou, best_dy, best_dx, best_dz = -1, 0, 0, 0
        best_peak_ratio = 0.0
        for dz in range(-search_z, search_z + 1):
            hcr_patch = hcr_slices[dz][py0:py1, px0:px1]
            if hcr_patch.sum() < 10:
                continue

            dy, dx, iou, peak_ratio = fft_iou_fov_precomputed(
                twop_rfft, fov_rfft, sum_twop, hcr_patch, cell_search_xy,
                fft_shape)

            if iou > best_iou:
                best_iou, best_dy, best_dx, best_dz = iou, dy, dx, dz
                best_peak_ratio = peak_ratio

        # Compute single-cell IoU at the found shift
        cell_iou_best = 0.0
        if target_mask is not None and target_mask.sum() > 0 and best_iou > 0:
            hcr_at_best = hcr_slices[best_dz][py0:py1, px0:px1]
            hcr_shifted = shift_2d(hcr_at_best, -best_dy, -best_dx)
            ci = (target_mask & (hcr_shifted > 0)).sum()
            cu = (target_mask | ((hcr_shifted > 0) & fov_local)).sum()
            cell_iou_best = float(ci / cu) if cu > 0 else 0.0

        results.append({
            "cy": cy, "cx": cx, "label": cell_label,
            "dy": best_dy, "dx": best_dx, "dz": best_dz,
            "iou_best": best_iou, "iou_zero": iou_zero,
            "iou_gain": best_iou - iou_zero,
            "cell_iou": cell_iou_best, "cell_iou_zero": cell_iou_zero,
            "cell_iou_gain": cell_iou_best - cell_iou_zero,
            "peak_ratio": best_peak_ratio,
        })
        if best_peak_ratio < min_peak_ratio:
            n_ambiguous += 1

    print(f"  Edge cells (shifted extraction): {n_shifted}")
    print(f"  Ambiguous matches (peak_ratio < {min_peak_ratio}): {n_ambiguous}")
    return results


# ---- RANSAC Affine Fitting ----

def fit_affine_ransac(df, min_iou, min_gain, residual_thresh,
                      min_samples=6, max_trials=2000, max_scale_dev=0.2):
    """Fit 2D affine + linear Z via RANSAC with singular value constraints.

    Displacement model: [dy, dx] = J @ [y, x] + [c_y, c_x]
    Deformation matrix T = I - J must have SVs in [1-dev, 1+dev].

    Parameters
    ----------
    df : DataFrame
        Cell displacement results from find_cell_displacements().
    min_iou : float
        Minimum IoU for a match to be considered.
    min_gain : float
        Minimum IoU gain over zero-shift.
    residual_thresh : float
        RANSAC inlier threshold (pixels).
    min_samples : int
        Minimum cells per RANSAC sample.
    max_trials : int
        RANSAC iterations.
    max_scale_dev : float
        Maximum deviation from identity scaling.

    Returns
    -------
    affine : dict or None
        {'A_dy': array(3,), 'A_dx': array(3,)} -- coefficients [y, x, 1]
    z_model : array(3,) or None
        Z displacement model [y, x, 1]
    df_annotated : DataFrame
        Input df with 'inlier' column added.
    """
    good = df[(df.iou_best >= min_iou) & (df.iou_gain >= min_gain)].copy()
    print(f"  High-confidence matches: {len(good)} / {len(df)}")
    if len(good) < min_samples:
        print("  WARNING: Too few matches for RANSAC")
        return None, None, good

    src_pts = good[['cy', 'cx']].values
    dst_dy = good['dy'].values
    dst_dx = good['dx'].values
    dst_dz = good['dz'].values

    best_n_inliers = 0
    best_affine = None
    best_z_model = None
    best_inliers = None
    n_rejected_sv = 0
    N = len(good)
    rng = np.random.default_rng(42)

    def check_svs(A_dy_coeffs, A_dx_coeffs):
        """Check if deformation matrix has valid singular values."""
        J = np.array([[A_dy_coeffs[0], A_dy_coeffs[1]],
                       [A_dx_coeffs[0], A_dx_coeffs[1]]])
        T = np.eye(2) - J
        svs = np.linalg.svd(T, compute_uv=False)
        return svs, np.all(svs >= 1 - max_scale_dev) and np.all(svs <= 1 + max_scale_dev)

    for trial in range(max_trials):
        idx = rng.choice(N, size=min_samples, replace=False)
        M = np.hstack([src_pts[idx], np.ones((min_samples, 1))])
        try:
            A_dy = np.linalg.lstsq(M, dst_dy[idx], rcond=None)[0]
            A_dx = np.linalg.lstsq(M, dst_dx[idx], rcond=None)[0]
        except np.linalg.LinAlgError:
            continue

        # Check proposal SVs
        _, ok = check_svs(A_dy, A_dx)
        if not ok:
            n_rejected_sv += 1
            continue

        # Score on all points
        M_all = np.hstack([src_pts, np.ones((N, 1))])
        pred_dy = M_all @ A_dy
        pred_dx = M_all @ A_dx
        try:
            A_dz = np.linalg.lstsq(M, dst_dz[idx], rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        pred_dz = M_all @ A_dz
        residuals = np.sqrt((dst_dy - pred_dy)**2 + (dst_dx - pred_dx)**2
                            + (dst_dz - pred_dz)**2)
        inliers = residuals < residual_thresh
        n_in = inliers.sum()

        if n_in > best_n_inliers:
            # Refit on all inliers and validate
            M_in = M_all[inliers]
            cand_dy = np.linalg.lstsq(M_in, dst_dy[inliers], rcond=None)[0]
            cand_dx = np.linalg.lstsq(M_in, dst_dx[inliers], rcond=None)[0]
            _, ok_refit = check_svs(cand_dy, cand_dx)
            if not ok_refit:
                continue
            best_n_inliers = n_in
            best_inliers = inliers
            best_affine = {'A_dy': cand_dy, 'A_dx': cand_dx}
            best_z_model = np.linalg.lstsq(M_in, dst_dz[inliers], rcond=None)[0]

    print(f"  RANSAC: {n_rejected_sv} / {max_trials} proposals rejected by SV constraint")

    if best_affine is None:
        print("  WARNING: No valid affine found")
        return None, None, good

    # Final diagnostics
    svs_final, _ = check_svs(best_affine['A_dy'], best_affine['A_dx'])
    good = good.copy()
    good['inlier'] = best_inliers
    print(f"  RANSAC inliers: {best_n_inliers} / {len(good)}")
    print(f"  Affine dy [y, x, 1]: {best_affine['A_dy']}")
    print(f"  Affine dx [y, x, 1]: {best_affine['A_dx']}")
    print(f"  Z model  [y, x, 1]: {best_z_model}")
    print(f"  Deformation SVs: {svs_final}  (valid: [{1 - max_scale_dev:.1f}, {1 + max_scale_dev:.1f}])")
    return best_affine, best_z_model, good


# ---- Local Tile RANSAC with RBF Smoothing ----

def run_local_tile_ransac(twop_binary, hcr_3d_bin, z_map, centroids, cell_df,
                          tile_size=200, overlap=0.5, min_cells=5,
                          min_cells_affine=3,
                          min_iou=0.10, min_gain=0.04,
                          border_anchor_spacing=100,
                          smoothing=50,
                          max_tile_scale_dev=0.15):
    """Fit per-tile affines from cell matches, return smooth RBF displacement fields.

    1. Divide image into overlapping tiles
    2. For each tile with enough cells: fit least-squares affine
       (or median fallback if min_cells_affine <= n < min_cells)
       Affines are validated via SV constraint — tiles with excessive
       shear/scale fall back to median displacement.
    3. Evaluate affine at tile center → (dy, dx, dz)
    4. Add border anchor points (nearest-neighbor from accepted tiles)
    5. Fit RBF (thin_plate_spline) through tile+anchor points

    Returns (dy_field, dx_field, dz_field, tile_results).
    """
    ny, nx = twop_binary.shape
    step = int(tile_size * (1 - overlap))
    good = cell_df[(cell_df.iou_best >= min_iou) & (cell_df.iou_gain >= min_gain)]
    print(f"  Cells passing local threshold: {len(good)} / {len(cell_df)}")

    tile_results = []
    n_sv_rejected = 0
    for ty in range(0, ny - tile_size // 2, step):
        for tx in range(0, nx - tile_size // 2, step):
            ty1, tx1 = min(ny, ty + tile_size), min(nx, tx + tile_size)
            in_tile = good[(good.cy >= ty) & (good.cy < ty1) &
                           (good.cx >= tx) & (good.cx < tx1)]
            tc_y, tc_x = (ty + ty1) / 2, (tx + tx1) / 2
            if len(in_tile) < min_cells:
                tile_results.append({'cy': tc_y, 'cx': tc_x,
                    'dy': 0, 'dx': 0, 'dz': 0, 'accepted': False})
                continue
            if len(in_tile) < min_cells_affine:
                # Median fallback: too few cells for stable affine fit
                tile_results.append({
                    'cy': tc_y, 'cx': tc_x,
                    'dy': float(np.median(in_tile.dy.values)),
                    'dx': float(np.median(in_tile.dx.values)),
                    'dz': float(np.median(in_tile.dz.values)),
                    'accepted': True})
                continue
            src = in_tile[['cy', 'cx']].values
            M = np.hstack([src, np.ones((len(in_tile), 1))])
            try:
                A_dy = np.linalg.lstsq(M, in_tile.dy.values, rcond=None)[0]
                A_dx = np.linalg.lstsq(M, in_tile.dx.values, rcond=None)[0]
                A_dz = np.linalg.lstsq(M, in_tile.dz.values, rcond=None)[0]
            except np.linalg.LinAlgError:
                tile_results.append({'cy': tc_y, 'cx': tc_x,
                    'dy': 0, 'dx': 0, 'dz': 0, 'accepted': False})
                continue
            # SV constraint: reject per-tile affines with excessive shear/scale
            _J = np.array([[A_dy[0], A_dy[1]], [A_dx[0], A_dx[1]]])
            _T = np.eye(2) - _J
            _svs = np.linalg.svd(_T, compute_uv=False)
            if not (np.all(_svs >= 1 - max_tile_scale_dev) and
                    np.all(_svs <= 1 + max_tile_scale_dev)):
                # Affine has too much shear/scale — fall back to median
                tile_results.append({
                    'cy': tc_y, 'cx': tc_x,
                    'dy': float(np.median(in_tile.dy.values)),
                    'dx': float(np.median(in_tile.dx.values)),
                    'dz': float(np.median(in_tile.dz.values)),
                    'accepted': True, 'sv_rejected': True})
                n_sv_rejected += 1
                continue
            mc = np.array([[tc_y, tc_x, 1]])
            tile_results.append({
                'cy': tc_y, 'cx': tc_x,
                'dy': float(mc @ A_dy), 'dx': float(mc @ A_dx),
                'dz': float(mc @ A_dz), 'accepted': True})

    accepted = [t for t in tile_results if t['accepted']]
    if n_sv_rejected > 0:
        print(f"  SV-rejected tiles (→median fallback): {n_sv_rejected} "
              f"(max_scale_dev={max_tile_scale_dev:.2f})")
    print(f"  Accepted tiles: {len(accepted)} / {len(tile_results)}")
    if len(accepted) < 3:
        return np.zeros((ny, nx)), np.zeros((ny, nx)), np.zeros((ny, nx)), tile_results

    centers = np.array([[t['cy'], t['cx']] for t in accepted])
    dy_vals = np.array([t['dy'] for t in accepted])
    dx_vals = np.array([t['dx'] for t in accepted])
    dz_vals = np.array([t['dz'] for t in accepted])

    # Border anchor points: pin at edges using nearest accepted tile
    s = border_anchor_spacing
    border_pts = []
    for bx in np.arange(0, nx, s):
        border_pts.append([0.0, float(bx)])
        border_pts.append([float(ny - 1), float(bx)])
    for by in np.arange(s, ny - 1, s):
        border_pts.append([float(by), 0.0])
        border_pts.append([float(by), float(nx - 1)])
    border_pts = np.array(border_pts)
    n_border = len(border_pts)

    nn_dy = NearestNDInterpolator(centers, dy_vals)
    nn_dx = NearestNDInterpolator(centers, dx_vals)
    nn_dz = NearestNDInterpolator(centers, dz_vals)
    dy_vals = np.concatenate([dy_vals, nn_dy(border_pts)])
    dx_vals = np.concatenate([dx_vals, nn_dx(border_pts)])
    dz_vals = np.concatenate([dz_vals, nn_dz(border_pts)])
    centers = np.vstack([centers, border_pts])
    print(f"  Border anchors: {n_border} points (spacing={s}px)")

    yy, xx = np.mgrid[:ny, :nx]
    pts = np.column_stack([yy.ravel(), xx.ravel()])
    dy_f = RBFInterpolator(centers, dy_vals,
        kernel='thin_plate_spline', smoothing=smoothing)(pts).reshape(ny, nx)
    dx_f = RBFInterpolator(centers, dx_vals,
        kernel='thin_plate_spline', smoothing=smoothing)(pts).reshape(ny, nx)
    dz_f = RBFInterpolator(centers, dz_vals,
        kernel='thin_plate_spline', smoothing=smoothing)(pts).reshape(ny, nx)
    return dy_f, dx_f, dz_f, tile_results