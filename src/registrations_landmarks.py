"""
Landmark-based registration for low-res to high-res transformation.

Uses thin-plate spline (TPS) transformation based on user-provided landmarks
to handle non-linear distortions between low-res and high-res images.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from rich import print as rprint
from skimage.transform import PiecewiseAffineTransform, warp
from scipy.interpolate import Rbf
from tifffile import imread as tif_imread, imwrite as tif_imwrite

try:
    from . import automation as auto
    from .meta import get_rotation_config
except ImportError:
    import automation as auto
    from meta import get_rotation_config


def apply_tps_2d(src_pts, dst_pts, image, output_shape, order=1):
    """
    Apply 2D thin-plate spline transformation to an image.

    Parameters
    ----------
    src_pts : ndarray (N, 2)
        Source landmarks [x, y]
    dst_pts : ndarray (N, 2)
        Destination landmarks [x, y]
    image : ndarray (Y, X)
        Image to transform
    output_shape : tuple
        Output shape (Y, X)
    order : int
        Interpolation order (0=nearest neighbor for masks, 1=linear for images)

    Returns
    -------
    warped : ndarray
        Transformed image
    """
    # Use RBF (Radial Basis Function) for thin-plate spline interpolation
    # Build inverse mapping (from destination to source)

    rbf_x = Rbf(dst_pts[:, 0], dst_pts[:, 1], src_pts[:, 0], function='thin_plate', smooth=0)
    rbf_y = Rbf(dst_pts[:, 0], dst_pts[:, 1], src_pts[:, 1], function='thin_plate', smooth=0)

    # Create destination grid
    dst_y, dst_x = np.mgrid[0:output_shape[0], 0:output_shape[1]]
    dst_coords = np.column_stack([dst_x.ravel(), dst_y.ravel()])

    # Map to source coordinates
    src_x_mapped = rbf_x(dst_coords[:, 0], dst_coords[:, 1]).reshape(output_shape)
    src_y_mapped = rbf_y(dst_coords[:, 0], dst_coords[:, 1]).reshape(output_shape)

    # Interpolate image values
    from scipy.ndimage import map_coordinates
    coords = np.array([src_y_mapped, src_x_mapped])
    warped = map_coordinates(image, coords, order=order, mode='constant', cval=0)

    return warped


def register_lowres_to_hires_landmarks(full_manifest, session):
    """
    Transform low-res Suite2p masks to high-res stitched tile space using user-provided landmarks.

    This function uses thin-plate spline (TPS) transformation based on manual landmarks
    to handle non-linear distortions between low-res and high-res images.

    **Workflow**:
    1. User provides landmarks for ONE reference plane (usually the first plane)
    2. Pipeline uses those landmarks to register the reference plane
    3. Pipeline automatically registers other planes using rigid alignment (phase correlation)

    User should provide landmarks ONLY for the reference plane:
    OUTPUT/2P/registered/lowres_to_hires_plane{REFERENCE}_landmarks.csv

    Landmark CSV format (BigWarp, no header):
    name,enabled,lowres_x,lowres_y,hires_x,hires_y  (6 cols for 2D)

    Parameters
    ----------
    full_manifest : dict
        Pipeline manifest
    session : dict
        Session metadata with functional_plane list

    Outputs
    -------
    For each plane:
    - OUTPUT/2P/registered/lowres_plane{X}_masks_in_hires_space.tiff
    - OUTPUT/2P/registered/QualityCheck/lowres_to_hires/plane{X}_lowres_to_hires_overlay.tiff
    """

    rprint("\n[bold]Low-Res to High-Res Registration (Landmark-Based)[/bold]")

    base_path = Path(full_manifest['data']['base_path'])
    mouse = full_manifest['data']['mouse_name']

    # Support both old and new manifest formats
    if 'functional_planes' in session:
        TARGET_PLANES = session['functional_planes']
        REFERENCE_PLANE = TARGET_PLANES[0]
    else:
        REFERENCE_PLANE = session['functional_plane'][0]
        ADDITIONAL_PLANES = session.get('additional_functional_planes', [])
        TARGET_PLANES = [REFERENCE_PLANE] + ADDITIONAL_PLANES

    print(f"Planes: {TARGET_PLANES} (ref: {REFERENCE_PLANE})")

    # Output directories
    output_dir = base_path / mouse / 'OUTPUT' / '2P' / 'registered'
    qa_dir = output_dir / 'QualityCheck' / 'lowres_to_hires'

    # Create QualityCheck directory - handle case where 'QualityCheck' exists as a file
    try:
        qa_dir.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        qa_parent = output_dir / 'QualityCheck'
        if qa_parent.exists() and qa_parent.is_file():
            qa_parent.unlink()
            qa_dir.mkdir(parents=True, exist_ok=True)
        else:
            raise

    # Store reference landmarks for propagation to other planes
    reference_landmarks = None

    # Process each plane
    for plane_idx in TARGET_PLANES:
        is_reference = (plane_idx == REFERENCE_PLANE)
        ref_marker = " (ref)" if is_reference else ""
        print(f"\nPlane {plane_idx}{ref_marker}:")

        # --- LOAD FILES ---
        # Low-res: rotated mean image from Suite2p
        lowres_img_path = base_path / mouse / 'OUTPUT' / '2P' / 'registered' / f'lowres_meanImg_C0_plane{plane_idx}_rotated.tiff'

        # High-res: rotated stitched image
        hires_img_path = base_path / mouse / 'OUTPUT' / '2P' / 'registered' / f'hires_stitched_plane{plane_idx}_rotated.tiff'

        # Low-res masks: cellpose output (will rotate on-the-fly)
        lowres_masks_path = base_path / mouse / 'OUTPUT' / '2P' / 'cellpose' / f'lowres_meanImg_C0_plane{plane_idx}_seg.npy'

        # User-provided landmarks (only for reference plane)
        landmarks_path = output_dir / f'lowres_to_hires_plane{REFERENCE_PLANE}_landmarks.csv'

        # Check if high-res exists for this plane
        if not hires_img_path.exists():
            rprint(f"  [yellow]Skipped: no high-res image[/yellow]")
            continue

        # Check other required files
        missing_files = []
        if not lowres_img_path.exists():
            missing_files.append(f"Low-res image: {lowres_img_path}")
        if not lowres_masks_path.exists():
            missing_files.append(f"Low-res masks: {lowres_masks_path}")

        # Only check landmarks for reference plane
        if is_reference and not landmarks_path.exists():
            missing_files.append(f"Landmarks (REQUIRED): {landmarks_path}")

        if missing_files:
            # Reference plane landmarks are REQUIRED - prompt user and wait
            if is_reference and not landmarks_path.exists():
                instructions = (
                    f"In BigWarp, open these two images:\n"
                    f"  Moving: {lowres_img_path}\n"
                    f"  Target: {hires_img_path}\n\n"
                    f"  Place landmarks mapping the LOW-RES mean image to the HIGH-RES stitched image."
                )
                auto.prompt_for_missing_file(
                    landmarks_path,
                    f"Low-res to high-res landmarks for Plane {REFERENCE_PLANE}",
                    instructions=instructions
                )
                # Remove landmarks from missing list after user creates it
                missing_files = [f for f in missing_files if "Landmarks" not in f]

            # For other missing files (not landmarks), skip plane
            if missing_files:
                rprint(f"  [yellow]Missing: {', '.join([f.split(':')[0] for f in missing_files])}[/yellow]")
                continue

        # Load images and masks
        lowres_img = tif_imread(str(lowres_img_path))
        hires_img = tif_imread(str(hires_img_path))

        # Load cellpose masks from .npy file
        lowres_masks_dict = np.load(str(lowres_masks_path), allow_pickle=True).item()
        lowres_masks = lowres_masks_dict['masks']

        # Apply rotation to match rotated low-res image
        # Get rotation from manifest
        rotation_params = get_rotation_config(full_manifest.get('params', {}))
        rotation_angle = rotation_params.get('rotation', 0)
        flip_lr = rotation_params.get('fliplr', False)
        flip_ud = rotation_params.get('flipud', False)

        # Apply same rotation as was applied to images
        from scipy.ndimage import rotate as ndimage_rotate
        if rotation_angle != 0:
            lowres_masks = ndimage_rotate(lowres_masks, rotation_angle, reshape=True, order=0, mode='constant', cval=0).astype(lowres_masks.dtype)
        if flip_lr:
            lowres_masks = np.fliplr(lowres_masks)
        if flip_ud:
            lowres_masks = np.flipud(lowres_masks)

        # Save rotated masks as TIFF for use by rest of pipeline
        rotated_masks_path = base_path / mouse / 'OUTPUT' / '2P' / 'cellpose' / f'lowres_meanImg_C0_plane{plane_idx}_seg_rotated.tiff'
        tif_imwrite(str(rotated_masks_path), lowres_masks.astype(np.uint16))

        # Extract 2D from high-res (could be ZCYX or CYX)
        if hires_img.ndim == 4:  # ZCYX
            hires_2d = hires_img[:, 0].mean(axis=0)
        elif hires_img.ndim == 3:  # CYX
            hires_2d = hires_img[0]
        else:  # YX
            hires_2d = hires_img

        # Extract 2D from low-res (should be YX already after rotation)
        if lowres_img.ndim == 2:
            lowres_2d = lowres_img
        else:
            lowres_2d = lowres_img[0] if lowres_img.ndim == 3 else lowres_img

        # --- REGISTRATION STRATEGY ---
        if is_reference:
            # REFERENCE PLANE: Use landmarks + TPS
            try:
                # Load BigWarp landmarks (CSV without header, specific column format)
                landmarks_df = pd.read_csv(landmarks_path, header=None)
                landmarks_df = landmarks_df.replace([np.inf, -np.inf], np.nan).dropna()

                # BigWarp formats: 6 cols (2D) or 8+ cols (3D)
                if landmarks_df.shape[1] == 6:
                    src_pts = landmarks_df[[2, 3]].values
                    dst_pts = landmarks_df[[4, 5]].values
                elif landmarks_df.shape[1] >= 8:
                    src_pts = landmarks_df[[2, 3]].values
                    dst_pts = landmarks_df[[5, 6]].values
                else:
                    print(f"  Error: BigWarp CSV has {landmarks_df.shape[1]} cols (expected 6 or 8+)")
                    continue

                n_landmarks = len(src_pts)
                if n_landmarks < 3:
                    print(f"  Error: Need >= 3 landmarks, found {n_landmarks}")
                    continue

                reference_landmarks = (src_pts, dst_pts)

            except Exception as e:
                print(f"  Error loading landmarks: {e}")
                continue

            try:
                # Transform masks using TPS with nearest-neighbor interpolation
                hires_masks = apply_tps_2d(
                    src_pts, dst_pts,
                    lowres_masks.astype(float),
                    output_shape=hires_2d.shape,
                    order=0
                )
                hires_masks = hires_masks.astype(lowres_masks.dtype)
                n_cells = len(np.unique(hires_masks)) - 1

                # Also transform low-res image for QA
                lowres_warped = apply_tps_2d(
                    src_pts, dst_pts,
                    lowres_2d.astype(float),
                    output_shape=hires_2d.shape
                )
                print(f"  TPS warp: {n_landmarks} landmarks, {n_cells} cells")

            except Exception as e:
                print(f"  TPS transformation failed: {e}")
                continue

        else:
            # NON-REFERENCE PLANES: Use reference TPS + phase correlation refinement
            if reference_landmarks is None:
                print(f"  Error: Reference plane must be processed first")
                continue

            try:
                src_pts, dst_pts = reference_landmarks

                # Apply reference TPS transformation to get initial alignment
                lowres_tps = apply_tps_2d(
                    src_pts, dst_pts,
                    lowres_2d.astype(float),
                    output_shape=hires_2d.shape
                )

                from skimage.registration import phase_cross_correlation
                from scipy.ndimage import shift as ndimage_shift

                # Normalize images for phase correlation
                lowres_norm = (lowres_tps - lowres_tps.min()) / (lowres_tps.max() - lowres_tps.min() + 1e-8)
                hires_norm = (hires_2d - hires_2d.min()) / (hires_2d.max() - hires_2d.min() + 1e-8)

                # Find best shift using phase correlation
                shift, error, diffphase = phase_cross_correlation(
                    hires_norm, lowres_norm,
                    upsample_factor=10,  # Sub-pixel accuracy
                    space='real'
                )

                # Limit shift to ±100 pixels
                shift_y = np.clip(shift[0], -100, 100)
                shift_x = np.clip(shift[1], -100, 100)

                # Apply shift to TPS-transformed lowres image for QA
                lowres_warped = ndimage_shift(lowres_tps, shift=(shift_y, shift_x), order=1, mode='constant', cval=0)

                # First apply TPS to masks with nearest-neighbor interpolation
                masks_tps = apply_tps_2d(
                    src_pts, dst_pts,
                    lowres_masks.astype(float),
                    output_shape=hires_2d.shape,
                    order=0
                )

                # Then apply shift
                hires_masks = ndimage_shift(masks_tps, shift=(shift_y, shift_x), order=0, mode='constant', cval=0)
                hires_masks = hires_masks.astype(lowres_masks.dtype)
                n_cells = len(np.unique(hires_masks)) - 1
                print(f"  TPS + shift ({shift_y:.1f}, {shift_x:.1f})px, {n_cells} cells")

            except Exception as e:
                print(f"  TPS + phase correlation failed: {e}")
                continue

        # Save transformed masks
        output_masks_path = output_dir / f'lowres_plane{plane_idx}_masks_in_hires_space.tiff'
        tif_imwrite(str(output_masks_path), hires_masks.astype(np.uint16))

        # Generate QA overlay
        try:
            from skimage.exposure import rescale_intensity
            hires_norm = rescale_intensity(hires_2d, out_range=(0, 255)).astype(np.uint8)
            lowres_norm = rescale_intensity(lowres_warped, out_range=(0, 255)).astype(np.uint8)
            overlay = np.stack([hires_norm, lowres_norm], axis=0)
            qa_path = qa_dir / f'plane{plane_idx}_lowres_to_hires_overlay.tiff'
            tif_imwrite(str(qa_path), overlay, imagej=True, metadata={'axes': 'CYX'})
        except Exception as e:
            rprint(f"  [yellow]Warning: QA overlay failed: {e}[/yellow]")

    rprint(f"\n[green]Low-res to high-res registration complete.[/green] QA: {qa_dir.name}/")
