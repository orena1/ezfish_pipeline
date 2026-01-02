import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm.auto import tqdm
from tifffile import TiffFile
from scipy.fft import fft2, ifft2
from scipy.interpolate import RBFInterpolator, griddata, LinearNDInterpolator, NearestNDInterpolator
from scipy.ndimage import binary_erosion, map_coordinates, rotate

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


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

        # --- 3) Some files put extra hints in ImageDescription (optional)
        # You can inspect it if needed:
        # desc = tags.get("ImageDescription")
        # print(desc.value[:500])

    return sx, sy, sz, unit


def tps_warp_2p_to_hcr(twop_2d, landmarks_df, hcr_shape):
    """Apply TPS warp to transform 2P masks into HCR coordinate space."""
    hcr_ny, hcr_nx = hcr_shape[1], hcr_shape[2]
    
    src_points = landmarks_df[['hcr_x_px', 'hcr_y_px']].values
    tgt_x = landmarks_df['2p_x'].values
    tgt_y = landmarks_df['2p_y'].values
    
    tps_to_2p_x = RBFInterpolator(src_points, tgt_x, kernel='thin_plate_spline')
    tps_to_2p_y = RBFInterpolator(src_points, tgt_y, kernel='thin_plate_spline')
    
    hcr_yy, hcr_xx = np.mgrid[0:hcr_ny, 0:hcr_nx]
    hcr_coords = np.column_stack([hcr_xx.ravel(), hcr_yy.ravel()])
    
    twop_x_coords = tps_to_2p_x(hcr_coords).reshape(hcr_ny, hcr_nx)
    twop_y_coords = tps_to_2p_y(hcr_coords).reshape(hcr_ny, hcr_nx)
    
    twop_warped = map_coordinates(
        twop_2d.astype(np.float32),
        [twop_y_coords, twop_x_coords],
        order=0, mode='constant', cval=0
    ).astype(twop_2d.dtype)
    
    return twop_warped

def erode_labels(labels_2d, iterations):
    """Erode label mask while preserving label IDs."""
    if iterations <= 0:
        return labels_2d
    binary = labels_2d > 0
    eroded_binary = binary_erosion(binary, iterations=iterations)
    return labels_2d * eroded_binary

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

def compute_iou(mask1, mask2):
    """Compute IoU (Intersection over Union) between two binary masks."""
    intersection = (mask1 & mask2).sum()
    union = (mask1 | mask2).sum()
    return intersection / union if union > 0 else 0.0

def fft_iou_all_shifts(mask1, mask2, max_shift):
    """Find best XY shift using FFT-accelerated IoU search."""
    m1 = mask1.astype(np.float64)
    m2 = mask2.astype(np.float64)
    sum1, sum2 = m1.sum(), m2.sum()
    if sum1 == 0 or sum2 == 0:
        return 0, 0, 0
    
    h, w = m1.shape
    cross_corr = np.real(ifft2(np.conj(fft2(m1)) * fft2(m2)))
    iou_map = np.where((sum1 + sum2 - cross_corr) > 0,
                       cross_corr / (sum1 + sum2 - cross_corr), 0)
    
    best_iou = -1
    best_dy, best_dx = 0, 0
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            iou = iou_map[dy % h, dx % w]
            if iou > best_iou:
                best_iou = iou
                best_dy, best_dx = dy, dx
    
    return best_dy, best_dx, best_iou

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

def rotate_2d(mask, angle_deg):
    """Rotate 2D mask around center."""
    if angle_deg == 0:
        return mask.copy()
    return rotate(mask, angle_deg, reshape=False, order=0, mode='constant', cval=0)

def apply_shift_fields(mask_2d, dy_field, dx_field):
    """Apply per-pixel shift fields."""
    h, w = mask_2d.shape
    yy, xx = np.mgrid[0:h, 0:w]
    src_y = (yy - dy_field).astype(np.float64)
    src_x = (xx - dx_field).astype(np.float64)
    shifted = map_coordinates(mask_2d.astype(float), [src_y, src_x],
                              order=0, mode='constant', cval=0)
    return shifted > 0.5

def add_px_columns(landmarks_df, path_reference_round_tiff, path_twop_reference_plane_tiff):
    # Get HCR resolutions using tifffile (more reliable than SimpleITK)
    HCR_RES_X, HCR_RES_Y, HCR_RES_Z, _ = read_spacing_xyz_from_tiff(path_reference_round_tiff)
    print(f"HCR resolutions: {HCR_RES_X} x {HCR_RES_Y} x {HCR_RES_Z} unit/pixel, please verify!")
    # Get 2P resolutions
    TWOP_RES_X, TWOP_RES_Y, _, _ = read_spacing_xyz_from_tiff(path_twop_reference_plane_tiff)
    print(f"2P resolutions: {TWOP_RES_X} x {TWOP_RES_Y} unit/pixel, please verify!")

    # Convert to pixels
    landmarks_df['hcr_x_px'] = landmarks_df['hcr_x'] / HCR_RES_X
    landmarks_df['hcr_y_px'] = landmarks_df['hcr_y'] / HCR_RES_Y
    landmarks_df['hcr_z_px'] = landmarks_df['hcr_z'] / HCR_RES_Z
    landmarks_df['2p_x_px'] = landmarks_df['2p_x'] / TWOP_RES_X
    landmarks_df['2p_y_px'] = landmarks_df['2p_y'] / TWOP_RES_Y
    landmarks_df['2p_z_px'] = landmarks_df['2p_z'] / 1  # 2P is single plane
    return landmarks_df

def load_landmarks(landmarks_path, hcr_ref_path, twop_ref_path):
    """
    Load and prepare landmark data for registration.

    Loads landmarks from CSV, filters enabled ones within bounds,
    and converts physical units to pixel coordinates.

    Parameters
    ----------
    landmarks_path : Path
        Path to landmarks CSV file
    hcr_ref_path : Path
        Path to HCR reference image (for resolution metadata)
    twop_ref_path : Path
        Path to 2P reference image (for resolution metadata)

    Returns
    -------
    pd.DataFrame
        Filtered landmarks with pixel coordinate columns added
    """
    landmarks_df = pd.read_csv(
        landmarks_path,
        header=None,
        names=['name', 'enabled', '2p_x', '2p_y', '2p_z', 'hcr_x', 'hcr_y', 'hcr_z']
    )

    # Filter only enabled landmarks and those within bounds
    landmarks_df = landmarks_df.query("enabled==True and hcr_x<9e5").copy()

    # Convert physical units to pixel coordinates
    landmarks_df = add_px_columns(landmarks_df, hcr_ref_path, twop_ref_path)

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

def global_alignment(twop_binary, hcr_3d_local, z_map, rotation_range, rotation_step,
                     z_range, xy_max, desc=""):
    """Find best global (theta, dz, dy, dx)."""
    rotation_angles = np.arange(rotation_range[0], rotation_range[1] + rotation_step, rotation_step)
    dz_vals = list(range(z_range[0], z_range[1] + 1))
    
    best = {'iou': -1, 'theta': 0, 'dz': 0, 'dy': 0, 'dx': 0}
    
    for theta in tqdm(rotation_angles, desc=f"Global {desc}", leave=False):
        twop_rotated = rotate_2d(twop_binary, theta)
        
        for dz in dz_vals:
            hcr_2d = sample_hcr_binary_at_zmap(hcr_3d_local, z_map, z_offset=dz)
            dy, dx, iou = fft_iou_all_shifts(twop_rotated, hcr_2d, xy_max)
            
            if iou > best['iou']:
                best = {'iou': iou, 'theta': theta, 'dz': dz, 'dy': dy, 'dx': dx}
    
    return best

def compute_tile_shifts(twop_2d_in, hcr_3d_local, z_map_local, base_dz,
                        tile_size, overlap, xy_max, z_range, min_pixels):
    """Compute optimal shift for each tile."""
    ny_local, nx_local = twop_2d_in.shape
    step = max(1, int(tile_size * (1 - overlap)))
    
    tile_results = []
    z_vals = list(range(z_range[0], z_range[1] + 1))

    pbar = tqdm(total=((ny_local - tile_size // 2) // step + 1) *
                    ((nx_local - tile_size // 2) // step + 1),
                     desc="Tile shifts", leave=False)

    for ty in range(0, ny_local - tile_size // 2, step):
        for tx in range(0, nx_local - tile_size // 2, step):
            y0_t, y1_t = max(0, ty), min(ny_local, ty + tile_size)
            x0_t, x1_t = max(0, tx), min(nx_local, tx + tile_size)

            if (y1_t - y0_t) < tile_size // 2 or (x1_t - x0_t) < tile_size // 2:
                continue

            twop_tile = twop_2d_in[y0_t:y1_t, x0_t:x1_t]
            z_map_tile = z_map_local[y0_t:y1_t, x0_t:x1_t]
            hcr_tile = hcr_3d_local[:, y0_t:y1_t, x0_t:x1_t]

            if twop_tile.sum() < min_pixels:
                continue

            best_tile = {'iou': -1, 'dz': 0, 'dy': 0, 'dx': 0}
            
            for dz in z_vals:
                hcr_2d_tile = sample_hcr_binary_at_zmap(hcr_tile, z_map_tile, z_offset=base_dz + dz)
                dy, dx, iou = fft_iou_all_shifts(twop_tile, hcr_2d_tile, xy_max)
                if iou > best_tile['iou']:
                    best_tile = {'iou': iou, 'dz': dz, 'dy': dy, 'dx': dx}

            tile_results.append({
                'cy': (y0_t + y1_t) / 2, 'cx': (x0_t + x1_t) / 2,
                'dz': best_tile['dz'], 'dy': best_tile['dy'], 'dx': best_tile['dx'],
                'iou': best_tile['iou']
            })
            pbar.update(1)

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

print("Alignment functions defined")

print("Helper functions defined")