import re
import os
import sys
import hjson
import zarr
import shutil
import numpy as np
from pathlib import Path

try:
    from .meta import parse_json, get_hcr_to_hcr_registration_config, get_rotation_config  # Relative import (for running as part of a package)
    from . import automation as auto
except ImportError:
    from meta import parse_json, get_hcr_to_hcr_registration_config, get_rotation_config  # Absolute import (for running in Jupyter Notebook)
    import automation as auto

try:
    from .registrations_utils import load_landmarks, build_z_map, tps_warp_2p_to_hcr, erode_labels, global_alignment, sample_hcr_binary_at_zmap, compute_iou, apply_shift_fields, compute_tile_shifts, interpolate_shift_field, shift_2d, rotate_2d, register_lowres_to_hires_single_plane, apply_lowres_to_hires_transform, refine_lowres_to_hires_with_tiles, prompt_overwrite_per_plane
except ImportError:
    from registrations_utils import load_landmarks, build_z_map, tps_warp_2p_to_hcr, erode_labels, global_alignment, sample_hcr_binary_at_zmap, compute_iou, apply_shift_fields, compute_tile_shifts, interpolate_shift_field, shift_2d, rotate_2d, register_lowres_to_hires_single_plane, apply_lowres_to_hires_transform, refine_lowres_to_hires_with_tiles, prompt_overwrite_per_plane
import pandas as pd

from rich.progress import track
import SimpleITK as sitk
from rich import print as rprint
from tifffile import imwrite as tif_imwrite
from tifffile import imread as tif_imread
from scipy.interpolate import RBFInterpolator, griddata

# Path for bigstream unless you did pip install
sys.path = [fr"\\nasquatch\data\2p\jonna\Code_Python\Notebooks_Jonna\BigStream\bigstream_v2_andermann"] + sys.path 
sys.path = [fr"C:\Users\jonna\Notebooks_Jonna\BigStream\bigstream_v2_andermann"] + sys.path 
sys.path = [fr'{os.getcwd()}/bigstream_v2_andermann'] + sys.path
sys.path = ["/mnt/nasquatch/data/2p/jonna/Code_Python/Notebooks_Jonna/BigStream/bigstream_v2_andermann"] + sys.path 

from bigstream.piecewise_transform import distributed_apply_transform



def HCR_confocal_imaging(manifest, only_paths=False):
    """
    print instructions on how to register the HCR data round to round
    only_paths: if True, return only the paths to the files and not registration instructions
    """

    mov_rounds = []
    reference_round_number = manifest['HCR_confocal_imaging']['reference_round']
    for i in manifest['HCR_confocal_imaging']['rounds']:
        if i['round'] == reference_round_number:
            reference_round = Path(manifest['base_path']) / manifest['mouse_name'] / 'HCR' / f"{manifest['mouse_name']}_HCR{reference_round_number}.tiff"
        else:
            mov_rounds.append(Path(manifest['base_path']) / manifest['mouse_name'] / 'HCR' / f"{manifest['mouse_name']}_HCR{i['round']}_To_HCR{reference_round_number}.tiff")
    
    while True:
        missing_files = []
        if not reference_round.exists():
            missing_files.append(f"Reference: {reference_round.name}")
        for i in mov_rounds:
            if not i.exists():
                missing_files.append(f"Round: {i.name}")

        if not missing_files:
            rprint("[green]All HCR round files found[/green]")
            break

        rprint(f"\n[bold yellow]Missing HCR round files:[/bold yellow]")
        for f in missing_files:
            rprint(f"  [yellow]{f}[/yellow]")
        rprint(f"\nExpected directory: [dim]{reference_round.parent}[/dim]")
        rprint("\nAdd the missing files, then press [green]Enter[/green] to continue...")
        input()
    if only_paths:
        return reference_round, mov_rounds
    # register the rounds

def registration_apply(full_manifest):
    """
    Register the rounds in the manifest that was selected in params.hjson
    """
    manifest = full_manifest['data']
    round_to_rounds, reference_round, register_rounds = verify_rounds(full_manifest, parse_registered = True, print_rounds = True, print_registered = True)

    # Pre-load fix image path (optimization: avoid reloading per round)
    HCR_fix_image_path = reference_round['image_path']
    HCR_fix_round = None  # Lazy load on first use

    for HCR_round_to_register in register_rounds:
        mov_round = round_to_rounds[HCR_round_to_register]
        HCR_mov_image_path = mov_round['image_path']

        round_folder_name = f"HCR{HCR_round_to_register}_to_HCR{reference_round['round']}"

        reg_path =  Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'HCR' / 'registrations'/ round_folder_name / round_to_rounds[HCR_round_to_register]['registrations'][0]
        full_stack_path =  Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'HCR' / 'full_registered_stacks' / f"{round_folder_name}.tiff"
        if full_stack_path.exists():
            print(f"Round {HCR_round_to_register} already registered")
            continue
        full_stack_path.parent.mkdir(exist_ok=True, parents=True)
        rprint(f"[bold]Applying Registration to round - {HCR_round_to_register}[/bold]")


        # resolution of the images
        fix_image_spacing = np.array(reference_round['resolution']) # Y,X,Z
        mov_image_spacing = np.array(mov_round['resolution']) # Y,X,Z


        # spatial down-sampling, probably no need to test. (Changed x and y from 3 to 2 for CIM round 5)
        hcr_reg_config = get_hcr_to_hcr_registration_config(full_manifest.get('params', {}))
        downsampling = hcr_reg_config['downsampling']  # [x, y, z]
        red_mut_x, red_mut_y, red_mut_z = downsampling[0], downsampling[1], downsampling[2]

        fix_lowres_spacing = fix_image_spacing * np.array([red_mut_y, red_mut_x, red_mut_z])
        mov_lowres_spacing = mov_image_spacing * np.array([red_mut_y, red_mut_x, red_mut_z])


        # get block size from the registration file
        blocksize_match = re.findall(r'bs(\d+)_(\d+)_(\d+)', Path(round_to_rounds[HCR_round_to_register]['registrations'][0]).name)
        blocksize = [int(num) for num in blocksize_match[0]]

        # Load fix image only once (lazy load on first round that needs processing)
        if HCR_fix_round is None:
            print("Loading fix image (once)")
            HCR_fix_round = tif_imread(HCR_fix_image_path)[:, 0]
        print("Loading moving image")
        HCR_mov_round = tif_imread(HCR_mov_image_path)

        # load the registration files
        affine = np.loadtxt(fr"{reg_path}/_affine.mat")
        deform = zarr.load(fr"{reg_path}/deform.zarr")
        fix_highres = HCR_fix_round.transpose(2, 1, 0) # from Z,X,Y to Y,X,Z

        # Build full_stack incrementally (optimization: avoid double zarr loads)
        full_stack = []

        # Loop through channels starting with 1, which ignores the first channel which has already been registered
        for channel in track(range(HCR_mov_round.shape[1]), description="Registering channels"):
            output_channel_path = Path(fr"{reg_path}/out_c{channel}.zarr")
            output_channel_tiff_path = output_channel_path.parent / output_channel_path.name.replace('.zarr','.tiff')
            if os.path.exists(output_channel_path) and os.path.exists(output_channel_tiff_path):
                print(f"Channel {channel} already registered")
                # Load existing data for full_stack (unavoidable - wasn't processed this run)
                full_stack.append(zarr.load(output_channel_path))
                continue

            HCR_mov_round_C = HCR_mov_round[:,channel]

            # mov Image
            mov_highres = HCR_mov_round_C.transpose(2,1,0)

            # Interpolation:
            # - Channel 0 (DAPI): use linear ('1') for smooth cell boundaries in segmentation
            # - Other channels (HCR probes): use nearest neighbor ('0') to preserve signal intensity
            if channel == 0:
                interpolator = '1'  # Linear for DAPI
            else:
                interpolator = '0'  # Nearest neighbor for HCR probes

            # register the images
            local_aligned = distributed_apply_transform(
                fix_highres, mov_highres,
                fix_image_spacing, mov_image_spacing,
                transform_list=[affine, deform],
                blocksize=blocksize,
                write_path=output_channel_path,
                interpolator=interpolator)
            print(fr'saved output {output_channel_path} (interpolator={interpolator})')

            # Load once and reuse for both TIFF write and full_stack
            data = zarr.load(output_channel_path)
            full_stack.append(data)

            tif_imwrite(output_channel_tiff_path
                        ,data.transpose(2,1,0))

        print(f"Saving full stack -{full_stack_path}")
        full_stack = np.stack(full_stack)

        tif_imwrite(full_stack_path, full_stack.transpose(3, 0, 2, 1), imagej=True, metadata={'axes': 'ZCYX'})

    # Now let's also copy the reference_round to the full_registered_stacks folder
    reference_round_full_stack_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'HCR' / 'full_registered_stacks' / f"HCR{reference_round['round']}.tiff"
    if not reference_round_full_stack_path.exists():
        print(f"Copying reference round {reference_round['round']} to full_registered_stacks")
        shutil.copy(HCR_fix_image_path, reference_round_full_stack_path)
            

def verify_rounds(full_manifest, parse_registered = False, print_rounds = False, print_registered = False, func='registering-apply'):
    '''
    if parse_registered is True, return the rounds that have been registered
    
    '''
    manifest = full_manifest['data']

    # verify that all rounds exists.
    reference_round_path, mov_rounds_path = HCR_confocal_imaging(manifest, only_paths=True)
    reference_round_number = manifest['HCR_confocal_imaging']['reference_round']
    if print_rounds: print("\nRounds available:")

    round_to_rounds = {}
    j=0
    for i in manifest['HCR_confocal_imaging']['rounds']:
        if i['round'] != reference_round_number:
            if print_rounds: print(i['round'],i['channels'])
            round_to_rounds[i['round']] = i
            round_to_rounds[i['round']]['image_path'] = mov_rounds_path[j]
            j+=1
        else:
            reference_round = i
            reference_round['image_path'] = reference_round_path
    
    ready_to_apply = []
    if parse_registered:
        selected_registrations = parse_json(full_manifest['manifest_path'])['params']
        txt_to_rich = f"[green]Rounds available for {func} [/green]:"
        for i in selected_registrations['HCR_selected_registrations']['rounds']:
            assert i['round'] in round_to_rounds, f"Round {i['round']} not defined in manifest!"
            selected_registration_path =  Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'HCR' / 'registrations'/ f"HCR{i['round']}_to_HCR{reference_round_number}" / i['selected_registrations'][0]
            assert os.path.exists(selected_registration_path), f"Registration {selected_registration_path} not found although it exists in the manifest params"

            round_to_rounds[i['round']]['registrations'] = i['selected_registrations']
            txt_to_rich+= f" {i['round']}"
            ready_to_apply.append(i['round'])
        if print_registered: rprint(txt_to_rich)
    return round_to_rounds, reference_round, ready_to_apply


def register_rounds(full_manifest):
    
    """
    Register the rounds in the manifest
    """
    manifest = full_manifest['data']
    round_to_rounds, reference_round, ready_to_apply = verify_rounds(full_manifest)
    
    # Clean header
    rprint("\n" + "="*80)
    rprint("[bold green] HCR Rounds Registrations[bold green]")
    rprint("="*80)
    rprint(f"Found {len(manifest['HCR_confocal_imaging']['rounds'])} HCR rounds in manifest")
    rprint("Registration process uses step-by-step jupyter notebooks\n")

    # Ensure reference round is copied in case user has only Rounds = 1
    reference_round_full_stack_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'HCR' / 'full_registered_stacks' / f"HCR{reference_round['round']}.tiff"
    if not reference_round_full_stack_path.exists():
        reference_round_full_stack_path.parent.mkdir(parents=True, exist_ok=True)
        rprint(f"[yellow]Copying reference round {reference_round['round']} to full_registered_stacks[/yellow]")
        shutil.copy(reference_round['image_path'], reference_round_full_stack_path)
        rprint(f"[green]✓ Reference round HCR{reference_round['round']} ready in full_registered_stacks[/green]")


    # Step A
    rprint("[bold cyan] Step 1): Configure Low-Resolution Parameters[/bold cyan]")
    rprint("   Notebook: [blue]ezfish_pipeline/src/processing_notebooks/HCR_rounds/1_scan_lowres_parameters.ipynb[/blue]")
    rprint(f"   Manifest: [yellow]{full_manifest['manifest_path']}[/yellow]\n")

    # Step B  
    rprint("[bold cyan] Step 2): Configure High-Resolution Parameters[/bold cyan]")
    rprint("   Notebook: [blue]ezfish_pipeline/src/processing_notebooks/HCR_rounds/2_scan_highres_parameters.ipynb[/blue]")
    rprint(f"   Manifest: [yellow]{full_manifest['manifest_path']}[/yellow]\n")
    
    # Step C
    rprint("[bold cyan] Step 3): Select Registration Rounds[/bold cyan]")
    rprint(f"   • Add 'HCR_selected_registrations' to: [yellow]{full_manifest['manifest_path']}[/yellow]")
    rprint("   • This will specify which rounds you want to register")
    rprint("\n[bold]Press [green]Enter[/green] when configuration is complete...[/bold]")
    input()

    # Reload and verify
    round_to_rounds, reference_round, ready_to_apply = verify_rounds(full_manifest, parse_registered = True)
    rprint("[green]✅ Configuration loaded successfully[/green]\n")

    # Step D
    rprint("[bold cyan] Step 4): Apply Registration[/bold cyan]")
    if ready_to_apply:
        rprint(f"    Rounds ready for registration: [green]{', '.join(ready_to_apply)}[/green]")
        rprint("\n[bold]Press [green]Enter[/green] to apply registration matrix...[/bold]")
        input()
        registration_apply(full_manifest)
        rprint("\n[green]✅ Registration applied successfully[/green]")
    else:
        rprint("   [yellow]⚠️  No rounds ready for registration[/yellow]")
    
    # Clean footer
    rprint("\n" + "="*80)
    rprint("[bold green] HCR Rounds Registrations COMPLETE[/bold green]")
    rprint("="*80 + "\n")


def register_lowres_to_hires(full_manifest, session):
    """
    Register low-res 2P images and masks to high-res stitched space using SIFT.

    This function:
    1. Loads low-res mean images (from Suite2p) and high-res stitched images
    2. Performs SIFT feature matching + RANSAC affine registration
    3. Applies transformation to low-res masks
    4. Generates QA overlays (before/after registration)
    5. Saves transformed masks in high-res space

    Parameters
    ----------
    full_manifest : dict
        Full manifest including data and params
    session : dict
        Session information from manifest

    Configuration (in manifest params.lowres_to_hires_registration):
        n_features : int (default 5000)
            Max SIFT keypoints to detect
        ratio_threshold : float (default 0.75)
            Lowe's ratio test threshold for feature matching
        ransac_reproj_threshold : float (default 5.0)
            RANSAC reprojection threshold in pixels
        min_matches : int (default 10)
            Minimum feature matches required before failing
    """

    print("\nLow-Res to High-Res Registration (SIFT)")

    manifest = full_manifest['data']
    params = full_manifest.get('params', {})
    base_path = Path(manifest['base_path'])
    mouse = manifest['mouse_name']

    # Get SIFT registration config (with defaults)
    sift_config = params.get('lowres_to_hires_registration', {})

    # Get planes to process
    if 'functional_planes' in session:
        TARGET_PLANES = session['functional_planes']
    else:
        REFERENCE_PLANE = session['functional_plane'][0]
        ADDITIONAL_PLANES = session.get('additional_functional_planes', [])
        TARGET_PLANES = [REFERENCE_PLANE] + ADDITIONAL_PLANES

    print(f"Planes: {TARGET_PLANES}")

    # Output directories
    output_dir = base_path / mouse / 'OUTPUT' / '2P' / 'registered'
    qa_dir = output_dir / 'QualityCheck' / 'lowres_to_hires'
    qa_dir.mkdir(parents=True, exist_ok=True)

    # Track overwrite state across planes (allows 'all'/'none' to apply to remaining)
    overwrite_state = [None]

    # Process each plane
    for plane_idx in TARGET_PLANES:
        # Check if output already exists and prompt user
        output_masks_path = output_dir / f'lowres_plane{plane_idx}_masks_in_hires_space.tiff'
        if not prompt_overwrite_per_plane(plane_idx, output_masks_path, overwrite_state):
            rprint(f"[dim]Plane {plane_idx}: skipped (output exists)[/dim]")
            continue

        print(f"\nPlane {plane_idx}:")

        # --- LOAD IMAGES ---
        # Low-res: rotated mean image from Suite2p
        lowres_img_path = base_path / mouse / 'OUTPUT' / '2P' / 'registered' / f'lowres_meanImg_C0_plane{plane_idx}_rotated.tiff'

        # High-res: rotated stitched image (use mean projection of channel 0)
        hires_img_path = base_path / mouse / 'OUTPUT' / '2P' / 'registered' / f'hires_stitched_plane{plane_idx}_rotated.tiff'

        # Low-res masks: cellpose output (before rotation)
        lowres_masks_path = base_path / mouse / 'OUTPUT' / '2P' / 'cellpose' / f'lowres_meanImg_C0_plane{plane_idx}_seg.npy'

        # Check files exist
        missing = []
        if not lowres_img_path.exists():
            missing.append("lowres_img")
        if not hires_img_path.exists():
            missing.append("hires_img")
        if not lowres_masks_path.exists():
            missing.append("lowres_masks")
        if missing:
            print(f"  Missing: {', '.join(missing)}, skipping")
            continue

        # Load images
        lowres_img = tif_imread(str(lowres_img_path))
        hires_img = tif_imread(str(hires_img_path))
        lowres_masks = np.load(str(lowres_masks_path), allow_pickle=True).item()['masks']

        # Apply rotation to masks to match rotated images (same logic as registrations_landmarks.py:193-207)
        from scipy.ndimage import rotate as ndimage_rotate
        rotation_params = get_rotation_config(full_manifest.get('params', {}))
        rotation_angle = rotation_params.get('rotation', 0)
        flip_lr = rotation_params.get('fliplr', False)
        flip_ud = rotation_params.get('flipud', False)

        if rotation_angle != 0:
            lowres_masks = ndimage_rotate(lowres_masks, rotation_angle, reshape=False, order=0, mode='constant', cval=0).astype(lowres_masks.dtype)
        if flip_lr:
            lowres_masks = np.fliplr(lowres_masks)
        if flip_ud:
            lowres_masks = np.flipud(lowres_masks)

        # Save rotated masks as TIFF (same location as landmark path uses)
        rotated_masks_path = base_path / mouse / 'OUTPUT' / '2P' / 'cellpose' / f'lowres_meanImg_C0_plane{plane_idx}_seg_rotated.tiff'
        tif_imwrite(str(rotated_masks_path), lowres_masks.astype(np.uint16))

        # Extract 2D from high-res (could be ZCYX or CYX)
        if hires_img.ndim == 4:
            hires_2d = hires_img[:, 0].mean(axis=0)
        elif hires_img.ndim == 3:
            hires_2d = hires_img[0]
        else:
            hires_2d = hires_img

        # Extract 2D from low-res
        if lowres_img.ndim == 2:
            lowres_2d = lowres_img
        else:
            lowres_2d = lowres_img[0] if lowres_img.ndim == 3 else lowres_img

        # Register using SIFT feature matching + RANSAC affine
        transform_params, lowres_aligned = register_lowres_to_hires_single_plane(
            lowres_2d, hires_2d,
            config=sift_config
        )

        # Apply transform to masks
        hires_masks = apply_lowres_to_hires_transform(
            lowres_masks,
            transform_params,
            hires_2d.shape
        )
        n_cells = len(np.unique(hires_masks)) - 1
        n_matches = transform_params.get('n_matches', 0)
        n_inliers = transform_params.get('n_inliers', 0)
        sim = transform_params.get('similarity', 0)
        rot = transform_params.get('rotation', 0)
        print(f"  SIFT: {n_matches} matches, {n_inliers} inliers, rot={rot:.2f}°, NCC={sim:.3f}, {n_cells} cells")

        # Save transformed masks
        output_masks_path = output_dir / f'lowres_plane{plane_idx}_masks_in_hires_space.tiff'
        tif_imwrite(str(output_masks_path), hires_masks.astype(np.uint16))

        # --- TILE-BASED REFINEMENT ---
        # Optimized parameters from notebook testing (tests/lowres_hires_registration_test.ipynb)
        # Key finding: direct per-tile corrections outperform griddata interpolation (+7% NCC)
        tile_refinement_enabled = sift_config.get('tile_refinement', True)
        lowres_final = lowres_aligned  # Default to global-only if tile refinement disabled/fails

        if tile_refinement_enabled:
            scale_factor = hires_2d.shape[0] / lowres_2d.shape[0]
            tile_size_hires = int(300 * scale_factor)  # 300px in lowres space

            # Best parameters from notebook testing (tests/lowres_hires_registration_test.ipynb)
            # Best by Quality Rate: ratio=0.8, ransac=5.0, min_matches=3, feat_size=8, spatial=0
            tile_config = {
                'tile_size': tile_size_hires, 'tile_overlap': 0.3, 'blend_width': 10, 'max_shift': 40,
                'n_features': 100, 'min_matches': 3, 'ratio_threshold': 0.8, 'ransac_reproj_threshold': 5.0,
                'min_feature_size': 8, 'max_spatial_distance': 0,  # 0 = no spatial limit (best results)
                'percentile_norm': (2, 98),  # Percentile-based normalization for feature detection
                'require_ncc_improvement': True, 'min_overlap_fraction': 0.5,
            }

            hires_masks_refined, lowres_refined, tile_info = refine_lowres_to_hires_with_tiles(
                lowres_aligned, hires_2d, hires_masks, config=tile_config
            )

            actual_tiles = [t for t in tile_info if 'cy' in t]
            n_success = sum(1 for t in actual_tiles if t.get('success'))
            print(f"  Tile refinement: {n_success}/{len(actual_tiles)} tiles successful")

            if n_success >= 1:
                hires_masks = hires_masks_refined
                lowres_final = lowres_refined
                tif_imwrite(str(output_masks_path), hires_masks.astype(np.uint16))

        # Generate separate QA overlay files for easy composite comparison
        from skimage.transform import resize
        lowres_upsampled = resize(lowres_2d, hires_2d.shape, order=1, preserve_range=True).astype(np.float32)

        # Before: hires vs upsampled lowres (no alignment)
        overlay_before = np.stack([hires_2d.astype(np.float32), lowres_upsampled])
        tif_imwrite(str(qa_dir / f'plane{plane_idx}_BEFORE_overlay.tiff'), overlay_before, imagej=True, metadata={'axes': 'CYX'})

        # After global: hires vs globally-aligned lowres
        overlay_global = np.stack([hires_2d.astype(np.float32), lowres_aligned.astype(np.float32)])
        tif_imwrite(str(qa_dir / f'plane{plane_idx}_AFTER_global_overlay.tiff'), overlay_global, imagej=True, metadata={'axes': 'CYX'})

        # After tiling: hires vs tile-refined lowres
        overlay_tiled = np.stack([hires_2d.astype(np.float32), lowres_final.astype(np.float32)])
        tif_imwrite(str(qa_dir / f'plane{plane_idx}_AFTER_tiling_overlay.tiff'), overlay_tiled, imagej=True, metadata={'axes': 'CYX'})

    print(f"\nLow-res to high-res registration complete. QA: {qa_dir.name}/")


def twop_to_hcr_registration(full_manifest, session, has_hires=False, automation_enabled=False):
    """
    Register 2P masks to HCR reference space.

    Parameters
    ----------
    full_manifest : dict
        Pipeline manifest
    session : dict
        Session metadata
    has_hires : bool
        Whether using high-res stitched workflow
    automation_enabled : bool
        If True, generate corrected landmarks and prompt for acceptance.
        If False, use manual landmarks (required).
    """
    workflow_type = "high-res" if has_hires else "low-res"
    mode_str = " + Automation" if automation_enabled else ""
    print(f"\n2P-to-HCR Registration ({workflow_type}{mode_str})")

    manifest = full_manifest['data']
    params = full_manifest['params']
    base_path = Path(manifest['base_path'])
    mouse = manifest['mouse_name']

    # --- REGISTRATION PARAMETERS ---
    # All parameters are configurable via manifest under params.twop_to_hcr_registration
    reg_params = params.get('twop_to_hcr_registration', {})

    # Erosion: shrink masks before alignment to improve precision, 2p tends to overestimate mask size.
    EROSION = reg_params.get('erosion', 1)

    # Global alignment: coarse rotation + XY/Z shifts
    ROTATION_RANGE = tuple(reg_params.get('rotation_range', [-1, 1]))  # degrees
    ROTATION_STEP = reg_params.get('rotation_step', 0.5)               # degrees
    Z_RANGE_GLOBAL = tuple(reg_params.get('z_range_global', [-20, 20]))  # planes
    XY_MAX_GLOBAL = reg_params.get('xy_max_global', 200)               # pixels

    # Local tile alignment: coarse-to-fine pyramid
    TILE_SIZES = reg_params.get('tile_sizes', [150, 75])               # pyramid levels (px)
    TILE_OVERLAP = reg_params.get('tile_overlap', 0.30)                # overlap fraction (0-1)
    TILE_XY_MAX = reg_params.get('tile_xy_max', 15)                    # max XY shift per tile (px)
    TILE_Z_RANGE = tuple(reg_params.get('tile_z_range', [-2, 2]))     # Z search range for tiles
    MIN_TILE_PIXELS = reg_params.get('min_tile_pixels', 50)            # min pixels to process tile

    # Print configuration (compact)
    print(f"\n2P→HCR Registration: erosion={EROSION}, rot={ROTATION_RANGE}°/{ROTATION_STEP}°, "
          f"Z={Z_RANGE_GLOBAL}, XY={XY_MAX_GLOBAL}px, tiles={TILE_SIZES}px\n")

    # Get ORIGINAL reference plane from manifest (not modified session)
    # master_pipeline modifies session['functional_plane'] per-plane, but landmarks are only for reference
    original_session = full_manifest['data']['two_photons_imaging']['sessions'][0]
    if 'functional_planes' in original_session:
        REFERENCE_PLANE = int(original_session['functional_planes'][0])
    else:
        REFERENCE_PLANE = int(original_session['functional_plane'][0])

    # Current plane being processed (from modified session)
    CURRENT_PLANE = int(session['functional_plane'][0])
    TARGET_PLANES = [CURRENT_PLANE]  # Process only the current plane per call

    # Check existing outputs and prompt user for each plane
    output_dir = base_path / mouse / 'OUTPUT' / '2P' / 'registered'
    overwrite_state = [None]
    planes_to_process = []

    for plane in TARGET_PLANES:
        output_path = output_dir / f'twop_plane{plane}_aligned_3d.tiff'
        if prompt_overwrite_per_plane(plane, output_path, overwrite_state):
            planes_to_process.append(plane)
        else:
            rprint(f"[dim]Plane {plane}: skipped (output exists)[/dim]")

    if not planes_to_process:
        rprint("[dim]All planes already processed, nothing to do[/dim]")
        return

    TARGET_PLANES = planes_to_process

    round_to_rounds, reference_round, register_rounds = verify_rounds(full_manifest, parse_registered = True,
                                                                    print_rounds = False, print_registered = False)

    # HCR paths
    hcr_ref_round_path = base_path / mouse / 'OUTPUT' / 'HCR' / 'full_registered_stacks' / f"HCR{reference_round['round']}.tiff"
    hcr_ref_masks_path = base_path / mouse / 'OUTPUT' / 'HCR' / 'cellpose' / f"HCR{reference_round['round']}_masks.tiff"

    # WORKFLOW DETECTION: Choose reference image and landmarks based on has_hires
    landmarks_dir = base_path / mouse / 'OUTPUT' / '2P' / 'registered'
    if has_hires:
        twop_ref_image_path = base_path / mouse / 'OUTPUT' / '2P' / 'registered' / f'hires_stitched_plane{REFERENCE_PLANE}_rotated.tiff'
        landmarks_path, landmarks_source = auto.find_landmark_file(
            landmarks_dir, REFERENCE_PLANE, prefix="hires_stitched_"
        )
        rprint(f"[dim]Mode: high-res stitched[/dim]")
    else:
        twop_ref_image_path = base_path / mouse / 'OUTPUT' / '2P' / 'cellpose' / f'lowres_meanImg_C0_plane{REFERENCE_PLANE}_seg_rotated.tiff'
        landmarks_path, landmarks_source = auto.find_landmark_file(
            landmarks_dir, REFERENCE_PLANE, prefix=""
        )
        rprint(f"[dim]Mode: standard low-res[/dim]")

    # Check if landmarks exist
    if landmarks_path is None:
        expected_name = f"{'hires_stitched_' if has_hires else ''}plane{REFERENCE_PLANE}_TO_HCR1_landmarks.csv"
        expected_path = landmarks_dir / expected_name

        if automation_enabled:
            # In automation mode, skip if no landmarks exist
            # User must provide initial landmarks (manual or from previous run) for TPS warp
            rprint(f"\n[yellow]Skipping 2P-to-HCR registration: No landmarks found[/yellow]")
            rprint(f"  Expected: {expected_path}")
            rprint(f"  [dim]Initial landmarks are required for TPS warp.[/dim]")
            rprint(f"  [dim]Run once with automation.twop_to_hcr: false to create them,[/dim]")
            rprint(f"  [dim]or create landmarks manually in BigWarp.[/dim]")
            return
        else:
            # Manual mode: wait for user to create landmarks
            instructions = f"Map 2P Plane {REFERENCE_PLANE} to HCR reference round"

            auto.prompt_for_missing_file(
                expected_path,
                f"2P-to-HCR landmarks for Plane {REFERENCE_PLANE}",
                instructions=instructions
            )

            # Re-check after user creates file
            landmarks_path, landmarks_source = auto.find_landmark_file(
                landmarks_dir, REFERENCE_PLANE, prefix="hires_stitched_" if has_hires else ""
            )

    print(f"Using {landmarks_source} landmarks: {landmarks_path.name}")

    # Create output folder
    output_folder = base_path / mouse / 'OUTPUT' / 'MERGED' / 'aligned_masks'
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load and prepare reference landmarks
    landmarks_df = load_landmarks(landmarks_path, hcr_ref_round_path, twop_ref_image_path)
    print(f"Loaded {len(landmarks_df)} landmarks from {landmarks_path.name}")

    # Load HCR reference masks
    hcr_ref_masks = tif_imread(str(hcr_ref_masks_path))
    y1, x1 = hcr_ref_masks.shape[1], hcr_ref_masks.shape[2]

    # Build z-map from reference landmarks
    z_map_base = build_z_map(landmarks_df, (y1, x1))

    # Load 2P masks for each plane
    print(f"\nLoading 2P masks for planes {TARGET_PLANES}...")
    twop_2d_planes = {}
    for plane in TARGET_PLANES:
        if has_hires:
            twop_path = base_path / mouse / 'OUTPUT' / '2P' / 'registered' / f'lowres_plane{plane}_masks_in_hires_space.tiff'
            if not twop_path.exists():
                raise FileNotFoundError(
                    f"High-res transformed masks not found: {twop_path}\n"
                    f"Please ensure register_lowres_to_hires() has completed successfully."
                )
        else:
            twop_path = base_path / mouse / 'OUTPUT' / '2P' / 'cellpose' / f'lowres_meanImg_C0_plane{plane}_seg_rotated.tiff'
        twop_2d_planes[plane] = tif_imread(str(twop_path))

    # Store results for each plane
    print(f"\nAligning {len(TARGET_PLANES)} planes...")
    plane_results = {}
    for plane in TARGET_PLANES:
        ref_marker = "*" if plane == REFERENCE_PLANE else ""
        n_cells = len(np.unique(twop_2d_planes[plane])) - 1
        print(f"\nPlane {plane}{ref_marker} ({n_cells} cells):")

        # TPS warp using reference landmarks
        twop_warped = tps_warp_2p_to_hcr(twop_2d_planes[plane], landmarks_df, hcr_ref_masks.shape)

        # Erode
        twop_eroded = erode_labels(twop_warped, EROSION)
        twop_binary = twop_eroded > 0
        n_cells = len(np.unique(twop_eroded)) - 1

        # Baseline IoU
        hcr_baseline = sample_hcr_binary_at_zmap(hcr_ref_masks, z_map_base, z_offset=0)
        iou_baseline = compute_iou(twop_binary, hcr_baseline)

        # Global alignment
        global_params = global_alignment(
            twop_binary, hcr_ref_masks, z_map_base,
            ROTATION_RANGE, ROTATION_STEP,
            Z_RANGE_GLOBAL, XY_MAX_GLOBAL,
            desc=f"Plane {plane}"
        )

        g_theta, g_dz, g_dy, g_dx = global_params['theta'], global_params['dz'], global_params['dy'], global_params['dx']
        print(f"  Global: θ={g_theta}°, dz={g_dz}, dy={g_dy}, dx={g_dx}, mask IoU: {iou_baseline:.3f}→", end="")

        # Apply global to BINARY for IoU calculation
        twop_global_binary = rotate_2d(twop_binary, g_theta)
        twop_global_binary = shift_2d(twop_global_binary, g_dy, g_dx)
        hcr_global = sample_hcr_binary_at_zmap(hcr_ref_masks, z_map_base, z_offset=g_dz)
        iou_global = compute_iou(twop_global_binary, hcr_global)

        # Apply global to ORIGINAL (non-eroded) LABELED MASKS for final output
        # Note: Eroded masks are used only for IOU alignment optimization (twop_binary),
        # but the saved output should contain original masks for accurate downstream IOU matching
        twop_global_labels = rotate_2d(twop_warped.astype(float), g_theta)
        twop_global_labels = shift_2d(twop_global_labels, g_dy, g_dx)
        twop_global_labels = twop_global_labels.astype(twop_warped.dtype)

        # Local tile alignment (coarse-to-fine pyramid)
        cumulative_dz = np.zeros((y1, x1))
        cumulative_dy = np.zeros((y1, x1))
        cumulative_dx = np.zeros((y1, x1))
        twop_current_binary = twop_global_binary.copy()
        twop_current_labels = twop_global_labels.copy()

        all_tile_results = []
        if len(TILE_SIZES) > 0:
            for i, tile_size in enumerate(TILE_SIZES):
                tile_results = compute_tile_shifts(
                    twop_current_binary, hcr_ref_masks, z_map_base, g_dz,
                    tile_size, TILE_OVERLAP, TILE_XY_MAX, TILE_Z_RANGE, MIN_TILE_PIXELS
                )
                all_tile_results.append({'tile_size': tile_size, 'tiles': tile_results})

                dz_field, dy_field, dx_field = interpolate_shift_field(tile_results, (y1, x1))
                cumulative_dz += dz_field
                cumulative_dy += dy_field
                cumulative_dx += dx_field

                twop_current_binary = apply_shift_fields(twop_global_binary, cumulative_dy, cumulative_dx)
                twop_current_labels = apply_shift_fields(twop_global_labels, cumulative_dy, cumulative_dx, order=0, return_labels=True)

        # Final local result
        twop_local_binary = twop_current_binary
        twop_local_labels = twop_current_labels
        z_map_local = z_map_base + g_dz + cumulative_dz
        hcr_local = sample_hcr_binary_at_zmap(hcr_ref_masks, z_map_local, z_offset=0)
        iou_local = compute_iou(twop_local_binary, hcr_local)
        print(f"{iou_global:.3f}→{iou_local:.3f}")

        # Store results
        plane_results[plane] = {
            'twop_original': twop_2d_planes[plane],
            'twop_warped': twop_warped,
            'twop_eroded': twop_eroded,
            'twop_binary': twop_binary,
            'twop_global': twop_global_binary,
            'twop_local': twop_local_binary,
            'hcr_baseline': hcr_baseline,
            'hcr_global': hcr_global,
            'hcr_local': hcr_local,
            'iou_baseline': iou_baseline,
            'iou_global': iou_global,
            'iou_local': iou_local,
            'global_params': global_params,
            'z_map_local': z_map_local,
            'cumulative_dz': cumulative_dz,
            'cumulative_dy': cumulative_dy,
            'cumulative_dx': cumulative_dx,
            'n_cells': n_cells,
        }

        # Create 3D volume matching HCR dimensions using LABELED MASKS
        nz, ny, nx = hcr_ref_masks.shape
        twop_3d = np.zeros((nz, ny, nx), dtype=twop_local_labels.dtype)

        # Get z-coordinates for each pixel (rounded to nearest integer)
        z_coords = np.round(z_map_local).astype(int)

        # Clip z-coordinates to valid range
        z_coords = np.clip(z_coords, 0, nz - 1)

        # Create coordinate arrays for indexing
        yy, xx = np.mgrid[0:ny, 0:nx]

        # Place 2D LABELED mask into 3D volume at z-positions specified by z_map
        twop_3d[z_coords, yy, xx] = twop_local_labels

        # Save the final aligned 3D volume
        output_path = base_path / mouse / 'OUTPUT' / '2P' / 'registered' / f'twop_plane{plane}_aligned_3d.tiff'
        tif_imwrite(str(output_path), twop_3d.astype(np.uint16))

        # Save QualityCheck overlay TIFFs
        qa_folder = base_path / mouse / 'OUTPUT' / '2P' / 'registered' / 'QualityCheck'
        qa_folder.mkdir(parents=True, exist_ok=True)

        overlay_before = np.stack([hcr_baseline.astype(np.uint16), twop_binary.astype(np.uint16)], axis=0)
        qa_before_path = qa_folder / f'plane{plane}_BEFORE_registration_overlay.tiff'
        tif_imwrite(str(qa_before_path), overlay_before, imagej=True, metadata={'axes': 'CYX'})

        overlay_after = np.stack([hcr_local.astype(np.uint16), twop_local_binary.astype(np.uint16)], axis=0)
        qa_after_path = qa_folder / f'plane{plane}_AFTER_registration_overlay.tiff'
        tif_imwrite(str(qa_after_path), overlay_after, imagej=True, metadata={'axes': 'CYX'})

    # --- SAVE PER-PLANE AUTO LANDMARKS ---
    # Generate corrected landmarks for EACH plane based on mask-mask alignment
    # This allows users to refine per-plane alignment if needed

    # Load original landmarks to get coordinates
    orig_landmarks = pd.read_csv(landmarks_path, header=None)
    orig_landmarks = orig_landmarks.replace([np.inf, -np.inf], np.nan).dropna()

    # Parse BigWarp format (6 or 8+ columns)
    if orig_landmarks.shape[1] == 6:
        # 2D format: name, enabled, src_x, src_y, dst_x, dst_y
        names = orig_landmarks[0].values
        enabled = orig_landmarks[1].values
        src_x = orig_landmarks[2].values
        src_y = orig_landmarks[3].values
        src_z = np.zeros(len(orig_landmarks))
        dst_x = orig_landmarks[4].values
        dst_y = orig_landmarks[5].values
        dst_z = np.zeros(len(orig_landmarks))
    else:
        # 3D format: name, enabled, src_x, src_y, src_z, dst_x, dst_y, dst_z
        names = orig_landmarks[0].values
        enabled = orig_landmarks[1].values
        src_x = orig_landmarks[2].values
        src_y = orig_landmarks[3].values
        src_z = orig_landmarks[4].values if orig_landmarks.shape[1] > 4 else np.zeros(len(orig_landmarks))
        dst_x = orig_landmarks[5].values if orig_landmarks.shape[1] > 5 else orig_landmarks[4].values
        dst_y = orig_landmarks[6].values if orig_landmarks.shape[1] > 6 else orig_landmarks[5].values
        dst_z = orig_landmarks[7].values if orig_landmarks.shape[1] > 7 else np.zeros(len(orig_landmarks))

    # Save auto landmarks for each plane
    prefix = "hires_stitched_" if has_hires else ""
    cy, cx = y1 / 2, x1 / 2  # Center for rotation

    for plane in TARGET_PLANES:
        g_params = plane_results[plane]['global_params']

        # Apply global shifts to HCR (destination) coordinates
        theta_rad = np.deg2rad(g_params['theta'])
        cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)

        # Apply rotation around center then shift
        dst_x_centered = dst_x - cx
        dst_y_centered = dst_y - cy
        dst_x_rot = dst_x_centered * cos_t - dst_y_centered * sin_t + cx
        dst_y_rot = dst_x_centered * sin_t + dst_y_centered * cos_t + cy

        # Apply XY shifts
        dst_x_corrected = dst_x_rot + g_params['dx']
        dst_y_corrected = dst_y_rot + g_params['dy']
        dst_z_corrected = dst_z + g_params['dz']

        # Create corrected landmarks DataFrame
        corrected_df = pd.DataFrame({
            'name': names,
            'enabled': enabled,
            'src_x': src_x,
            'src_y': src_y,
            'src_z': src_z,
            'dst_x': dst_x_corrected,
            'dst_y': dst_y_corrected,
            'dst_z': dst_z_corrected,
        })

        # Save as _auto file for this plane
        auto_landmarks_path = landmarks_dir / f'{prefix}plane{plane}_TO_HCR1_auto.csv'
        corrected_df.to_csv(auto_landmarks_path, index=False, header=False)

    print(f"\nSaved auto landmarks for {len(TARGET_PLANES)} planes")

    # --- AUTOMATION CHECKPOINT ---
    # Only show checkpoint when processing the reference plane (where landmarks were created)
    # Other planes use the same landmarks with per-plane refinement
    if automation_enabled and CURRENT_PLANE == REFERENCE_PLANE:
        ref_auto_path = landmarks_dir / f'{prefix}plane{REFERENCE_PLANE}_TO_HCR1_auto.csv'
        qa_paths = [
            qa_folder / f'plane{REFERENCE_PLANE}_BEFORE_registration_overlay.tiff',
            qa_folder / f'plane{REFERENCE_PLANE}_AFTER_registration_overlay.tiff'
        ]
        choice = auto.prompt_registration_checkpoint(
            qa_paths, ref_auto_path, "2P-to-HCR", REFERENCE_PLANE
        )

        if choice == "skip":
            rprint(f"[yellow]Registration skipped by user. Outputs remain but may need re-running.[/yellow]")
            return
        elif choice == "refine":
            rprint(f"[yellow]Refinement requested. Re-run pipeline with updated landmarks.[/yellow]")
            rprint(f"[yellow]Per-plane _auto.csv files saved. Edit and re-run to apply changes.[/yellow]")
            return
        # accept: continue normally
    elif not automation_enabled and CURRENT_PLANE == REFERENCE_PLANE:
        # Only prompt for review on reference plane in manual mode
        rprint(f"\n[green]Alignment complete.[/green]")
        rprint(f"[bold]Review QA overlays:[/bold]")
        rprint(f"  Directory: [yellow]{qa_folder}[/yellow]")
        rprint(f"  [dim]Ch1=HCR (magenta), Ch2=2P (green)[/dim]")
        rprint("\nPress [green]Enter[/green] to continue with mask matching...")
        input()




