import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm.auto import tqdm
from tifffile import TiffFile
from scipy.fft import fft2, ifft2
from scipy.interpolate import RBFInterpolator, griddata, LinearNDInterpolator, NearestNDInterpolator
from scipy.ndimage import binary_erosion, map_coordinates, rotate
from skimage.transform import resize, AffineTransform, warp
from skimage.registration import phase_cross_correlation
from scipy.optimize import minimize
import scipy.io as sio
from pathlib import Path


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def prompt_overwrite_per_plane(plane_idx: int, output_path: Path, overwrite_state: list) -> bool:
    """
    Ask user whether to overwrite existing output file for a plane.

    Parameters
    ----------
    plane_idx : int
        The plane being processed
    output_path : Path
        Path to the output file to check
    overwrite_state : list
        Single-element list holding state: [None, 'all', 'none']
        Mutated by this function to remember user's choice across calls

    Returns
    -------
    bool
        True if should process (file doesn't exist or user chose to overwrite),
        False if should skip
    """
    if not output_path.exists():
        return True  # File doesn't exist, proceed

    # Check if user already chose 'all' or 'none'
    if overwrite_state[0] == 'all':
        return True
    if overwrite_state[0] == 'none':
        return False

    # Prompt user
    while True:
        response = input(f"Plane {plane_idx}: {output_path.name} exists. Overwrite? [y/n/all/none]: ").strip().lower()
        if response == 'y':
            return True
        elif response == 'n':
            return False
        elif response == 'all':
            overwrite_state[0] = 'all'
            return True
        elif response == 'none':
            overwrite_state[0] = 'none'
            return False
        else:
            print("  Please enter y, n, all, or none")


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

        # --- 3) Some files put extra hints in ImageDescription (optional)
        # You can inspect it if needed:
        # desc = tags.get("ImageDescription")
        # print(desc.value[:500])

    return sx, sy, sz, unit


def tps_warp_2p_to_hcr(twop_2d, landmarks_df, hcr_shape):
    """Apply TPS warp to transform 2P masks into HCR coordinate space."""
    hcr_ny, hcr_nx = hcr_shape[1], hcr_shape[2]
    
    src_points = landmarks_df[['hcr_x_px', 'hcr_y_px']].values
    tgt_x = landmarks_df['2p_x_px'].values
    tgt_y = landmarks_df['2p_y_px'].values
    
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
    """Erode label mask while preserving label IDs.

    Each label is eroded independently to avoid artifacts when cells touch.
    """
    if iterations <= 0:
        return labels_2d

    result = np.zeros_like(labels_2d)
    unique_labels = np.unique(labels_2d)

    for label in unique_labels:
        if label == 0:  # Skip background
            continue
        # Erode each cell independently
        cell_mask = labels_2d == label
        eroded_mask = binary_erosion(cell_mask, iterations=iterations)
        result[eroded_mask] = label

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

def add_px_columns(landmarks_df, path_reference_round_tiff, path_twop_reference_plane_tiff):
    # Get HCR resolutions using tifffile (more reliable than SimpleITK)
    HCR_RES_X, HCR_RES_Y, HCR_RES_Z, _ = read_spacing_xyz_from_tiff(path_reference_round_tiff)
    # Get 2P resolutions
    TWOP_RES_X, TWOP_RES_Y, _, _ = read_spacing_xyz_from_tiff(path_twop_reference_plane_tiff)

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