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
    from .registrations_utils import (
        # Existing functions (still used by other workflows)
        load_landmarks, build_z_map, tps_warp_2p_to_hcr, erode_labels,
        sample_hcr_binary_at_zmap, compute_iou, apply_shift_fields,
        shift_2d, rotate_2d,
        register_lowres_to_hires_single_plane, apply_lowres_to_hires_transform,
        refine_lowres_to_hires_with_tiles, prompt_overwrite_per_plane,
        # v8 registration functions
        global_search_moving_iou, create_landmark_hull_mask, create_fov_mask_convex_hull,
        build_z_quad_blend, extract_centroids, find_cell_displacements,
        fit_affine_ransac, run_local_tile_ransac,
    )
except ImportError:
    from registrations_utils import (
        load_landmarks, build_z_map, tps_warp_2p_to_hcr, erode_labels,
        sample_hcr_binary_at_zmap, compute_iou, apply_shift_fields,
        shift_2d, rotate_2d,
        register_lowres_to_hires_single_plane, apply_lowres_to_hires_transform,
        refine_lowres_to_hires_with_tiles, prompt_overwrite_per_plane,
        global_search_moving_iou, create_landmark_hull_mask, create_fov_mask_convex_hull,
        build_z_quad_blend, extract_centroids, find_cell_displacements,
        fit_affine_ransac, run_local_tile_ransac,
    )
import pandas as pd

from rich.progress import track
import SimpleITK as sitk
from rich import print as rprint
from tifffile import imwrite as tif_imwrite
from tifffile import imread as tif_imread
# RBFInterpolator/griddata now handled in registrations_utils.py

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
            mov_rounds.append(Path(manifest['base_path']) / manifest['mouse_name'] / 'HCR' / f"{manifest['mouse_name']}_HCR{i['round']}_to_HCR{reference_round_number}.tiff")
    
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
            continue
        full_stack_path.parent.mkdir(exist_ok=True, parents=True)
        rprint(f"[bold]Applying Registration to round {HCR_round_to_register}[/bold]")


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
            HCR_fix_round = tif_imread(HCR_fix_image_path)[:, 0]
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

            # register the images - returns zarr array reference (already written to disk)
            output_zarr = distributed_apply_transform(
                fix_highres, mov_highres,
                fix_image_spacing, mov_image_spacing,
                transform_list=[affine, deform],
                blocksize=blocksize,
                write_path=output_channel_path,
                interpolator=interpolator)

            # Use returned zarr array directly instead of re-reading from disk
            # output_zarr[:] reads the data that was just written, avoiding redundant I/O
            data = np.asarray(output_zarr)
            full_stack.append(data)

            tif_imwrite(output_channel_tiff_path
                        ,data.transpose(2,1,0))

        full_stack = np.stack(full_stack)
        tif_imwrite(full_stack_path, full_stack.transpose(3, 0, 2, 1), imagej=True, metadata={'axes': 'ZCYX'})

    # Now let's also copy the reference_round to the full_registered_stacks folder
    reference_round_full_stack_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'HCR' / 'full_registered_stacks' / f"HCR{reference_round['round']}.tiff"
    if not reference_round_full_stack_path.exists():
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


def create_rotated_masks_for_standard_mode(full_manifest, session):
    """Create rotated mask TIFFs for standard (non-hires) mode.

    In hi-res mode, register_lowres_to_hires() creates these files.
    In standard mode, we need to create them separately before 2P-to-HCR registration.
    """
    from scipy.ndimage import rotate as ndimage_rotate

    data = full_manifest['data']
    params = full_manifest.get('params', {})
    base_path = Path(data['base_path'])
    mouse = data['mouse_name']

    # Get all planes
    if 'functional_planes' in session:
        all_planes = list(session['functional_planes'])
    else:
        all_planes = list(session['functional_plane'])
        if 'additional_functional_planes' in session:
            all_planes.extend(session['additional_functional_planes'])

    # Get rotation config
    rotation_params = get_rotation_config(params)
    rotation_angle = rotation_params.get('rotation', 0)
    flip_lr = rotation_params.get('fliplr', False)
    flip_ud = rotation_params.get('flipud', False)

    rprint(f"[cyan]Creating rotated masks for standard mode (rot={rotation_angle}°, fliplr={flip_lr}, flipud={flip_ud})[/cyan]")

    for plane_idx in all_planes:
        plane_idx = int(plane_idx)
        masks_path = base_path / mouse / 'OUTPUT' / '2P' / 'cellpose' / f'lowres_meanImg_C0_plane{plane_idx}_seg.npy'
        if not masks_path.exists():
            rprint(f"  [yellow]Plane {plane_idx}: masks not found, skipping[/yellow]")
            continue

        masks = np.load(str(masks_path), allow_pickle=True).item()['masks']

        if rotation_angle != 0:
            masks = ndimage_rotate(masks, rotation_angle, reshape=False, order=0, mode='constant', cval=0).astype(masks.dtype)
        if flip_lr:
            masks = np.fliplr(masks)
        if flip_ud:
            masks = np.flipud(masks)

        output_path = base_path / mouse / 'OUTPUT' / '2P' / 'cellpose' / f'lowres_meanImg_C0_plane{plane_idx}_seg_rotated.tiff'
        tif_imwrite(str(output_path), masks.astype(np.uint16))
        rprint(f"  Plane {plane_idx}: saved rotated masks")


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

    rprint("[bold]Low-Res to High-Res Registration (SIFT)[/bold]")

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
            continue

        rprint(f"Plane {plane_idx}:")

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
            rprint(f"  [yellow]Missing: {', '.join(missing)}, skipping[/yellow]")
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
        rprint(f"  SIFT: {n_matches} matches, {n_inliers} inliers, rot={rot:.2f}°, NCC={sim:.3f}, {n_cells} cells")

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
            rprint(f"  [dim]Tile refinement: {n_success}/{len(actual_tiles)} tiles[/dim]")

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

    rprint(f"[green]Low-res to high-res registration complete.[/green] QA: {qa_dir.name}/")


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
    rprint(f"[bold]2P-to-HCR Registration ({workflow_type}{mode_str})[/bold]")

    manifest = full_manifest['data']
    params = full_manifest['params']
    base_path = Path(manifest['base_path'])
    mouse = manifest['mouse_name']

    # --- REGISTRATION PARAMETERS (v8: 5-tier progressive refinement) ---
    # All parameters configurable via manifest under params.twop_to_hcr_registration
    reg_params = params.get('twop_to_hcr_registration', {})

    # Erosion: shrink masks before alignment to improve precision
    EROSION = reg_params.get('erosion', 2)
    EROSION_HCR = reg_params.get('erosion_hcr', 0)

    # Global search: moving-mask IoU with inward hull (no rotation search)
    HULL_MARGIN = reg_params.get('hull_margin', -100)
    Z_RANGE_GLOBAL = tuple(reg_params.get('z_range_global', [-15, 15]))
    XY_MAX_GLOBAL = reg_params.get('xy_max_global', 500)
    QUAD_BLEND_DIST = reg_params.get('quad_blend_dist', 300)
    FOV_CROP_MARGIN = reg_params.get('fov_crop_margin', 150)

    # Tier 1: Coarse matching (pre-affine, residuals ~50-80px)
    PATCH_RADIUS_COARSE = reg_params.get('patch_radius_coarse', 120)
    SEARCH_XY_COARSE = reg_params.get('search_xy_coarse', 150)
    SEARCH_Z_COARSE = reg_params.get('search_z_coarse', 9)

    # Tier 2: Tight matching (post-affine-pass-1, residuals <40px)
    PATCH_RADIUS_TIGHT = reg_params.get('patch_radius_tight', 60)
    SEARCH_XY_TIGHT = reg_params.get('search_xy_tight', 60)
    SEARCH_Z_TIGHT = reg_params.get('search_z_tight', 4)

    # Tier 3: Fine matching (post-composed-affine, residuals <5px)
    PATCH_RADIUS_FINE = reg_params.get('patch_radius_fine', 30)
    SEARCH_XY_FINE = reg_params.get('search_xy_fine', 30)
    SEARCH_Z_FINE = reg_params.get('search_z_fine', 2)

    # Tier 4: Ultra-fine matching (post-300px-tiles, residuals <3px)
    PATCH_RADIUS_ULTRAFINE = reg_params.get('patch_radius_ultrafine', 25)
    SEARCH_XY_ULTRAFINE = reg_params.get('search_xy_ultrafine', 8)
    SEARCH_Z_ULTRAFINE = reg_params.get('search_z_ultrafine', 1)

    # Per-cell FFT-IoU settings
    FOV_DILATION = reg_params.get('fov_dilation', 10)
    MIN_PEAK_RATIO = reg_params.get('min_peak_ratio', 1.02)

    # Match quality thresholds
    MIN_IOU_GLOBAL = reg_params.get('min_iou_global', 0.06)
    MIN_GAIN_GLOBAL = reg_params.get('min_gain_global', 0.005)
    MIN_IOU_LOCAL = reg_params.get('min_iou_local', 0.10)
    MIN_GAIN_LOCAL = reg_params.get('min_gain_local', 0.01)

    # RANSAC parameters
    RANSAC_RESIDUAL = reg_params.get('ransac_residual', 10.0)
    RANSAC_MIN_SAMPLES = reg_params.get('ransac_min_samples', 6)
    RANSAC_MAX_TRIALS = reg_params.get('ransac_max_trials', 2000)

    # Local tiles (300px)
    TILE_SIZE = reg_params.get('tile_size_coarse', 300)
    TILE_OVERLAP = reg_params.get('tile_overlap', 0.5)
    MIN_CELLS_PER_TILE = reg_params.get('min_cells_per_tile', 8)
    RBF_SMOOTHING = reg_params.get('rbf_smoothing_coarse', 50)

    # Fine tiles (100px)
    FINE_TILE_SIZE = reg_params.get('tile_size_fine', 100)
    FINE_TILE_OVERLAP = reg_params.get('fine_tile_overlap', 0.5)
    FINE_MIN_CELLS_PER_TILE = reg_params.get('min_cells_per_tile_fine', 4)
    FINE_RBF_SMOOTHING = reg_params.get('rbf_smoothing_fine', 100)

    # Print configuration (compact)
    rprint(f"[dim]Parameters: erosion={EROSION}, Z={Z_RANGE_GLOBAL}, "
          f"XY={XY_MAX_GLOBAL}px, hull_margin={HULL_MARGIN}, "
          f"tiles={TILE_SIZE}/{FINE_TILE_SIZE}px[/dim]")

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
    # Reference round number for file naming (strip leading zeros: "01" -> "1")
    hcr_ref = str(int(reference_round['round']))

    if has_hires:
        twop_ref_image_path = base_path / mouse / 'OUTPUT' / '2P' / 'registered' / f'hires_stitched_plane{REFERENCE_PLANE}_rotated.tiff'
        landmarks_path, landmarks_source = auto.find_landmark_file(
            landmarks_dir, REFERENCE_PLANE, prefix="hires_stitched_", hcr_ref=hcr_ref
        )
        rprint(f"[dim]Mode: high-res stitched[/dim]")
    else:
        twop_ref_image_path = base_path / mouse / 'OUTPUT' / '2P' / 'cellpose' / f'lowres_meanImg_C0_plane{REFERENCE_PLANE}_seg_rotated.tiff'
        landmarks_path, landmarks_source = auto.find_landmark_file(
            landmarks_dir, REFERENCE_PLANE, prefix="", hcr_ref=hcr_ref
        )
        rprint(f"[dim]Mode: standard low-res[/dim]")

    # Check if landmarks exist
    if landmarks_path is None:
        expected_name = f"{'hires_stitched_' if has_hires else ''}plane{REFERENCE_PLANE}_to_HCR{hcr_ref}_landmarks.csv"
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
                landmarks_dir, REFERENCE_PLANE, prefix="hires_stitched_" if has_hires else "", hcr_ref=hcr_ref
            )

    rprint(f"[dim]Using {landmarks_source} landmarks: {landmarks_path.name}[/dim]")

    # Create output folders
    output_folder = base_path / mouse / 'OUTPUT' / 'MERGED' / 'aligned_masks'
    output_folder.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare reference landmarks (HCR resolution from manifest)
    hcr_resolution = full_manifest['data']['HCR_confocal_imaging']['rounds'][0]['resolution']
    landmarks_df = load_landmarks(landmarks_path, hcr_resolution)
    rprint(f"[dim]Loaded {len(landmarks_df)} landmarks[/dim]")

    # Load HCR reference masks
    hcr_ref_masks = tif_imread(str(hcr_ref_masks_path))
    y1, x1 = hcr_ref_masks.shape[1], hcr_ref_masks.shape[2]

    # Load 2P masks for each plane
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
    plane_results = {}
    for plane in TARGET_PLANES:
        ref_marker = "*" if plane == REFERENCE_PLANE else ""
        n_cells = len(np.unique(twop_2d_planes[plane])) - 1
        rprint(f"Plane {plane}{ref_marker} ({n_cells} cells):")

        # TPS warp using reference landmarks
        twop_warped = tps_warp_2p_to_hcr(twop_2d_planes[plane], landmarks_df, hcr_ref_masks.shape)

        # Erode
        twop_eroded = erode_labels(twop_warped, EROSION)
        twop_binary = twop_eroded > 0
        n_cells = len(np.unique(twop_eroded)) - 1

        # ---- CROP STAGE 1: Bounding box + margin for global search ----
        ys, xs = np.where(twop_warped > 0)
        CROP_MARGIN_OUTER = 500
        bb_y0 = max(0, ys.min() - CROP_MARGIN_OUTER)
        bb_y1 = min(y1, ys.max() + CROP_MARGIN_OUTER)
        bb_x0 = max(0, xs.min() - CROP_MARGIN_OUTER)
        bb_x1 = min(x1, xs.max() + CROP_MARGIN_OUTER)

        hcr_3d_crop = hcr_ref_masks[:, bb_y0:bb_y1, bb_x0:bb_x1]
        if EROSION_HCR > 0:
            hcr_3d_crop = hcr_3d_crop.copy()
            for z_idx in range(hcr_3d_crop.shape[0]):
                hcr_3d_crop[z_idx] = erode_labels(hcr_3d_crop[z_idx], EROSION_HCR)
        hcr_3d_bin_crop = hcr_3d_crop > 0
        twop_labels_crop = twop_eroded[bb_y0:bb_y1, bb_x0:bb_x1]
        twop_binary_crop = twop_labels_crop > 0
        twop_warped_crop = twop_warped[bb_y0:bb_y1, bb_x0:bb_x1]  # non-eroded for output

        # Adjust landmark coordinates for crop space
        lm_crop = landmarks_df.copy()
        lm_crop['hcr_y_px'] = lm_crop['hcr_y_px'] - bb_y0
        lm_crop['hcr_x_px'] = lm_crop['hcr_x_px'] - bb_x0

        # Z-map with quad-blend extrapolation
        lm_coords = lm_crop[['hcr_y_px', 'hcr_x_px']].values
        lm_z_vals = lm_crop['hcr_z_px'].values
        z_map_baseline = build_z_quad_blend(lm_coords, lm_z_vals, twop_binary_crop.shape,
                                             max_dist=QUAD_BLEND_DIST)

        # FOV mask and baseline IoU
        fov_mask = create_fov_mask_convex_hull(twop_binary_crop, margin=30)
        hcr_baseline = sample_hcr_binary_at_zmap(hcr_3d_bin_crop, z_map_baseline)
        iou_baseline = compute_iou(twop_binary_crop, hcr_baseline, fov_mask=fov_mask)

        # Inward hull mask for global search
        inward_hull_mask = create_landmark_hull_mask(lm_crop, twop_binary_crop.shape, margin=HULL_MARGIN)

        # ---- GLOBAL SEARCH: Moving-mask IoU ----
        global_params = global_search_moving_iou(
            twop_binary_crop, hcr_3d_bin_crop, z_map_baseline,
            mask=inward_hull_mask,
            z_range=Z_RANGE_GLOBAL,
            xy_max=XY_MAX_GLOBAL,
        )

        g_dy, g_dx, g_dz = global_params['dy'], global_params['dx'], global_params['dz']
        twop_global_labels = shift_2d(twop_labels_crop, g_dy, g_dx)
        twop_global_binary = twop_global_labels > 0
        z_map_global = z_map_baseline + g_dz

        fov_global = create_fov_mask_convex_hull(twop_global_binary, margin=30)
        hcr_global = sample_hcr_binary_at_zmap(hcr_3d_bin_crop, z_map_global)
        iou_global = compute_iou(twop_global_binary, hcr_global, fov_mask=fov_global)
        rprint(f"  [dim]Global: IoU {iou_baseline:.3f}→{iou_global:.3f} "
               f"(dy={g_dy:+d}, dx={g_dx:+d}, dz={g_dz:+d})[/dim]")

        # Also shift the non-eroded warped labels by global offset
        twop_warped_global = shift_2d(twop_warped_crop, g_dy, g_dx)

        # ---- CROP STAGE 2: Tight FOV crop for affine/tile work ----
        _fov_for_crop = create_fov_mask_convex_hull(twop_global_binary, margin=0)
        _ys_fov, _xs_fov = np.where(_fov_for_crop)
        cy0 = int(max(0, _ys_fov.min() - FOV_CROP_MARGIN))
        cy1 = int(min(twop_global_binary.shape[0], _ys_fov.max() + FOV_CROP_MARGIN))
        cx0 = int(max(0, _xs_fov.min() - FOV_CROP_MARGIN))
        cx1 = int(min(twop_global_binary.shape[1], _xs_fov.max() + FOV_CROP_MARGIN))

        twop_crop_labels = twop_global_labels[cy0:cy1, cx0:cx1]
        twop_crop_binary = twop_crop_labels > 0
        hcr_crop_3d = hcr_3d_bin_crop[:, cy0:cy1, cx0:cx1]
        z_map_crop = z_map_global[cy0:cy1, cx0:cx1]
        fov_crop = create_fov_mask_convex_hull(twop_crop_binary, margin=20)
        centroids_crop = extract_centroids(twop_crop_labels)

        # Safety check: patch must be smaller than crop
        crop_h, crop_w = cy1 - cy0, cx1 - cx0
        min_dim = min(crop_h, crop_w)
        _prc = PATCH_RADIUS_COARSE
        if 2 * _prc >= min_dim:
            _prc = min_dim // 4
            rprint(f"  [yellow]Coarse patch reduced to {2 * _prc}px (crop={min_dim}px)[/yellow]")

        # ---- AFFINE PASS 1: Coarse matching + RANSAC ----
        rprint(f"  [dim]Affine Pass 1: {2 * _prc}px patches, +/-{SEARCH_XY_COARSE}px search[/dim]")
        cell_results_coarse = find_cell_displacements(
            twop_crop_binary, twop_crop_labels, hcr_crop_3d, z_map_crop, centroids_crop,
            patch_radius=_prc, search_xy=SEARCH_XY_COARSE,
            search_z=SEARCH_Z_COARSE, fov_dilation=FOV_DILATION,
            min_peak_ratio=MIN_PEAK_RATIO)

        df_coarse = pd.DataFrame(cell_results_coarse)
        ny_c, nx_c = twop_crop_binary.shape

        # Filter and fit RANSAC
        df_A = df_coarse[df_coarse.peak_ratio >= MIN_PEAK_RATIO].copy()
        df_A = df_A[df_A.cell_iou > 0.0]
        if len(df_A) < RANSAC_MIN_SAMPLES:
            df_A = df_coarse

        affine_A, z_model_A, df_ann_A = fit_affine_ransac(
            df_A, MIN_IOU_GLOBAL, MIN_GAIN_GLOBAL, RANSAC_RESIDUAL,
            min_samples=RANSAC_MIN_SAMPLES, max_trials=RANSAC_MAX_TRIALS)

        iou_pass1 = iou_global  # default if RANSAC fails
        if affine_A is not None:
            yy_c, xx_c = np.mgrid[:ny_c, :nx_c]
            M = np.stack([yy_c.ravel(), xx_c.ravel(), np.ones(ny_c * nx_c)], axis=1)
            dy_field_A = (M @ affine_A['A_dy']).reshape(ny_c, nx_c)
            dx_field_A = (M @ affine_A['A_dx']).reshape(ny_c, nx_c)
            dz_field_A = (M @ z_model_A).reshape(ny_c, nx_c)

            twop_A_labels = apply_shift_fields(twop_crop_labels, dy_field_A, dx_field_A, return_labels=True)
            twop_A_binary = twop_A_labels > 0
            z_map_A = z_map_crop + dz_field_A
            fov_A = create_fov_mask_convex_hull(twop_A_binary, margin=20)
            hcr_A = sample_hcr_binary_at_zmap(hcr_crop_3d, z_map_A)
            iou_pass1 = compute_iou(twop_A_binary, hcr_A, fov_mask=fov_A)
            rprint(f"  [dim]Affine Pass 1: IoU {iou_global:.3f}→{iou_pass1:.3f}[/dim]")
        else:
            rprint(f"  [yellow]Affine Pass 1: RANSAC failed, using zero fields[/yellow]")
            dy_field_A = np.zeros((ny_c, nx_c))
            dx_field_A = np.zeros((ny_c, nx_c))
            dz_field_A = np.zeros((ny_c, nx_c))
            twop_A_labels = twop_crop_labels
            twop_A_binary = twop_crop_binary
            z_map_A = z_map_crop

        # ---- AFFINE PASS 2: Tight re-matching on pass-1-corrected ----
        rprint(f"  [dim]Affine Pass 2: {2 * PATCH_RADIUS_TIGHT}px patches, +/-{SEARCH_XY_TIGHT}px search[/dim]")
        centroids_A = extract_centroids(twop_A_labels)
        iou_affine = iou_pass1  # default

        if len(centroids_A) > 0:
            cell_results_tight = find_cell_displacements(
                twop_A_binary, twop_A_labels, hcr_crop_3d, z_map_A, centroids_A,
                patch_radius=PATCH_RADIUS_TIGHT, search_xy=SEARCH_XY_TIGHT,
                search_z=SEARCH_Z_TIGHT, fov_dilation=FOV_DILATION,
                min_peak_ratio=MIN_PEAK_RATIO)

            df_tight = pd.DataFrame(cell_results_tight)
            df_filt = df_tight[df_tight.peak_ratio >= MIN_PEAK_RATIO].copy()
            df_filt = df_filt[df_filt.cell_iou > 0.0]
            if len(df_filt) < RANSAC_MIN_SAMPLES:
                df_filt = df_tight

            affine_B, z_model_B, df_ann_B = fit_affine_ransac(
                df_filt, MIN_IOU_GLOBAL, MIN_GAIN_GLOBAL, RANSAC_RESIDUAL,
                min_samples=RANSAC_MIN_SAMPLES, max_trials=RANSAC_MAX_TRIALS)

            if affine_B is not None:
                dy2 = (M @ affine_B['A_dy']).reshape(ny_c, nx_c)
                dx2 = (M @ affine_B['A_dx']).reshape(ny_c, nx_c)
                dz2 = (M @ z_model_B).reshape(ny_c, nx_c)

                # Compose: Pass 1 + residual
                dy_field = dy_field_A + dy2
                dx_field = dx_field_A + dx2
                dz_field = dz_field_A + dz2

                twop_affine_labels = apply_shift_fields(twop_crop_labels, dy_field, dx_field, return_labels=True)
                twop_affine_binary = twop_affine_labels > 0
                z_map_affine = z_map_crop + dz_field
                fov_affine = create_fov_mask_convex_hull(twop_affine_binary, margin=20)
                hcr_affine = sample_hcr_binary_at_zmap(hcr_crop_3d, z_map_affine)
                iou_affine = compute_iou(twop_affine_binary, hcr_affine, fov_mask=fov_affine)
                rprint(f"  [dim]Composed Affine: IoU {iou_pass1:.3f}→{iou_affine:.3f}[/dim]")
            else:
                rprint(f"  [yellow]Affine Pass 2: RANSAC failed, using Pass 1[/yellow]")
                dy_field = dy_field_A
                dx_field = dx_field_A
                dz_field = dz_field_A
                twop_affine_labels = twop_A_labels
                twop_affine_binary = twop_A_binary
                z_map_affine = z_map_A
        else:
            dy_field = dy_field_A
            dx_field = dx_field_A
            dz_field = dz_field_A
            twop_affine_labels = twop_A_labels
            twop_affine_binary = twop_A_binary
            z_map_affine = z_map_A

        # ---- LOCAL TILES 300px ----
        iou_local_300 = iou_affine  # default
        dy_local = np.zeros((ny_c, nx_c))
        dx_local = np.zeros((ny_c, nx_c))
        dz_local = np.zeros((ny_c, nx_c))
        twop_local_labels = twop_affine_labels
        twop_local_binary = twop_affine_binary
        z_map_local_crop = z_map_affine

        if iou_affine > iou_global:
            rprint(f"  [dim]Local tiles 300px: {2 * PATCH_RADIUS_FINE}px patches, +/-{SEARCH_XY_FINE}px search[/dim]")
            centroids_affine = extract_centroids(twop_affine_labels)

            if len(centroids_affine) > 0:
                cell_results_local = find_cell_displacements(
                    twop_affine_binary, twop_affine_labels, hcr_crop_3d, z_map_affine,
                    centroids_affine,
                    patch_radius=PATCH_RADIUS_FINE, search_xy=SEARCH_XY_FINE,
                    search_z=SEARCH_Z_FINE, fov_dilation=FOV_DILATION,
                    min_peak_ratio=MIN_PEAK_RATIO)

                df_local = pd.DataFrame(cell_results_local)
                df_local_filt = df_local[df_local.peak_ratio >= MIN_PEAK_RATIO].copy()
                if len(df_local_filt) < MIN_CELLS_PER_TILE:
                    df_local_filt = df_local

                if len(df_local_filt) > 0:
                    dy_local, dx_local, dz_local, tile_res = run_local_tile_ransac(
                        twop_affine_binary, hcr_crop_3d, z_map_affine, centroids_affine,
                        df_local_filt,
                        tile_size=TILE_SIZE, overlap=TILE_OVERLAP,
                        min_cells=MIN_CELLS_PER_TILE,
                        min_iou=MIN_IOU_LOCAL, min_gain=MIN_GAIN_LOCAL,
                        smoothing=RBF_SMOOTHING)

                    # Displacement clamping (95th percentile * 2.0)
                    _accepted = [t for t in tile_res if t['accepted']]
                    if len(_accepted) > 0:
                        _tile_mags = np.sqrt(np.array([t['dy'] for t in _accepted])**2 +
                                             np.array([t['dx'] for t in _accepted])**2)
                        _max_xy = np.percentile(_tile_mags, 95) * 2.0
                        _tile_dz = np.abs([t['dz'] for t in _accepted])
                        _max_dz = np.percentile(_tile_dz, 95) * 2.0
                        _mag = np.sqrt(dy_local**2 + dx_local**2)
                        _scale = np.minimum(1.0, _max_xy / (_mag + 1e-8))
                        dy_local = dy_local * _scale
                        dx_local = dx_local * _scale
                        dz_local = np.clip(dz_local, -_max_dz, _max_dz)

                    twop_local_labels = apply_shift_fields(twop_affine_labels, dy_local, dx_local, return_labels=True)
                    twop_local_binary = twop_local_labels > 0
                    z_map_local_crop = z_map_affine + dz_local
                    fov_local = create_fov_mask_convex_hull(twop_local_binary, margin=20)
                    hcr_local_300 = sample_hcr_binary_at_zmap(hcr_crop_3d, z_map_local_crop)
                    iou_local_300 = compute_iou(twop_local_binary, hcr_local_300, fov_mask=fov_local)
                    rprint(f"  [dim]Local 300px: IoU {iou_affine:.3f}→{iou_local_300:.3f}[/dim]")

        # ---- FINE TILES 100px ----
        iou_fine_100 = iou_local_300  # default
        dy_fine = np.zeros((ny_c, nx_c))
        dx_fine = np.zeros((ny_c, nx_c))
        dz_fine = np.zeros((ny_c, nx_c))
        twop_fine_labels = twop_local_labels
        twop_fine_binary = twop_local_binary
        z_map_fine_crop = z_map_local_crop

        if iou_local_300 > iou_affine:
            rprint(f"  [dim]Fine tiles 100px: {2 * PATCH_RADIUS_ULTRAFINE}px patches, +/-{SEARCH_XY_ULTRAFINE}px search[/dim]")
            centroids_local = extract_centroids(twop_local_labels)

            if len(centroids_local) > 0:
                cell_results_fine = find_cell_displacements(
                    twop_local_binary, twop_local_labels, hcr_crop_3d, z_map_local_crop,
                    centroids_local,
                    patch_radius=PATCH_RADIUS_ULTRAFINE, search_xy=SEARCH_XY_ULTRAFINE,
                    search_z=SEARCH_Z_ULTRAFINE, fov_dilation=FOV_DILATION,
                    min_peak_ratio=MIN_PEAK_RATIO)

                df_fine = pd.DataFrame(cell_results_fine)
                df_fine_filt = df_fine[df_fine.peak_ratio >= MIN_PEAK_RATIO].copy()
                if len(df_fine_filt) < FINE_MIN_CELLS_PER_TILE:
                    df_fine_filt = df_fine

                if len(df_fine_filt) > 0:
                    dy_fine, dx_fine, dz_fine, tile_res_fine = run_local_tile_ransac(
                        twop_local_binary, hcr_crop_3d, z_map_local_crop, centroids_local,
                        df_fine_filt,
                        tile_size=FINE_TILE_SIZE, overlap=FINE_TILE_OVERLAP,
                        min_cells=FINE_MIN_CELLS_PER_TILE,
                        min_iou=MIN_IOU_LOCAL, min_gain=MIN_GAIN_LOCAL,
                        smoothing=FINE_RBF_SMOOTHING)

                    # Displacement clamping (tighter 1.5x multiplier)
                    _accepted_f = [t for t in tile_res_fine if t['accepted']]
                    if len(_accepted_f) > 0:
                        _tile_mags_f = np.sqrt(np.array([t['dy'] for t in _accepted_f])**2 +
                                               np.array([t['dx'] for t in _accepted_f])**2)
                        _max_xy_f = np.percentile(_tile_mags_f, 95) * 1.5
                        _tile_dz_f = np.abs([t['dz'] for t in _accepted_f])
                        _max_dz_f = np.percentile(_tile_dz_f, 95) * 1.5
                        _mag_f = np.sqrt(dy_fine**2 + dx_fine**2)
                        _scale_f = np.minimum(1.0, _max_xy_f / (_mag_f + 1e-8))
                        dy_fine = dy_fine * _scale_f
                        dx_fine = dx_fine * _scale_f
                        dz_fine = np.clip(dz_fine, -_max_dz_f, _max_dz_f)

                    twop_fine_labels = apply_shift_fields(twop_local_labels, dy_fine, dx_fine, return_labels=True)
                    twop_fine_binary = twop_fine_labels > 0
                    z_map_fine_crop = z_map_local_crop + dz_fine
                    fov_fine = create_fov_mask_convex_hull(twop_fine_binary, margin=20)
                    hcr_fine = sample_hcr_binary_at_zmap(hcr_crop_3d, z_map_fine_crop)
                    iou_fine_100 = compute_iou(twop_fine_binary, hcr_fine, fov_mask=fov_fine)

                    # Safety revert if fine tiles made things worse
                    if iou_fine_100 < iou_local_300:
                        rprint(f"  [yellow]Fine tiles decreased IoU ({iou_local_300:.4f}→{iou_fine_100:.4f}), reverting[/yellow]")
                        twop_fine_labels = twop_local_labels
                        twop_fine_binary = twop_local_binary
                        z_map_fine_crop = z_map_local_crop
                        dy_fine = np.zeros((ny_c, nx_c))
                        dx_fine = np.zeros((ny_c, nx_c))
                        dz_fine = np.zeros((ny_c, nx_c))
                        iou_fine_100 = iou_local_300
                    else:
                        rprint(f"  [dim]Fine 100px: IoU {iou_local_300:.3f}→{iou_fine_100:.3f}[/dim]")

        iou_final = iou_fine_100
        rprint(f"  [bold]Final: IoU {iou_baseline:.3f}→{iou_final:.3f} ({iou_final - iou_baseline:+.3f})[/bold]")

        # ---- UNCROP: Compose displacement fields back to full HCR space ----
        # Total displacement in inner-crop space
        total_dy_inner = dy_field + dy_local + dy_fine
        total_dx_inner = dx_field + dx_local + dx_fine
        total_dz_inner = dz_field + dz_local + dz_fine

        # Pad inner → outer crop
        outer_h, outer_w = bb_y1 - bb_y0, bb_x1 - bb_x0
        total_dy_outer = np.zeros((outer_h, outer_w))
        total_dx_outer = np.zeros((outer_h, outer_w))
        total_dz_outer = np.zeros((outer_h, outer_w))
        total_dy_outer[cy0:cy1, cx0:cx1] = total_dy_inner
        total_dx_outer[cy0:cy1, cx0:cx1] = total_dx_inner
        total_dz_outer[cy0:cy1, cx0:cx1] = total_dz_inner

        # Pad outer → full HCR space
        cumulative_dy = np.zeros((y1, x1))
        cumulative_dx = np.zeros((y1, x1))
        cumulative_dz = np.zeros((y1, x1))
        cumulative_dy[bb_y0:bb_y1, bb_x0:bb_x1] = total_dy_outer
        cumulative_dx[bb_y0:bb_y1, bb_x0:bb_x1] = total_dx_outer
        cumulative_dz[bb_y0:bb_y1, bb_x0:bb_x1] = total_dz_outer

        # ---- Apply final composed displacement to NON-ERODED labels ----
        # Eroded masks were used for alignment; non-eroded for output
        twop_global_noneroded = shift_2d(twop_warped, g_dy, g_dx)
        twop_final_labels = apply_shift_fields(twop_global_noneroded, cumulative_dy, cumulative_dx,
                                                return_labels=True)

        # Final z_map in full HCR space
        z_map_base_full = build_z_quad_blend(
            landmarks_df[['hcr_y_px', 'hcr_x_px']].values,
            landmarks_df['hcr_z_px'].values,
            (y1, x1), max_dist=QUAD_BLEND_DIST)
        z_map_local = z_map_base_full + g_dz + cumulative_dz

        # Compute final IoU in full space for QA
        hcr_final_full = sample_hcr_binary_at_zmap(hcr_ref_masks > 0, z_map_local)
        twop_final_binary = twop_final_labels > 0
        fov_final = create_fov_mask_convex_hull(twop_final_binary, margin=30)
        hcr_local = hcr_final_full  # for QA overlay

        # Store results
        plane_results[plane] = {
            'iou_baseline': iou_baseline,
            'iou_global': iou_global,
            'iou_affine': iou_affine,
            'iou_local_300': iou_local_300,
            'iou_fine_100': iou_fine_100,
            'iou_final': iou_final,
            'global_params': global_params,
            'z_map_local': z_map_local,
            'cumulative_dz': cumulative_dz,
            'cumulative_dy': cumulative_dy,
            'cumulative_dx': cumulative_dx,
            'n_cells': n_cells,
        }

        # Create 3D volume matching HCR dimensions using NON-ERODED LABELED MASKS
        nz, ny, nx = hcr_ref_masks.shape
        twop_3d = np.zeros((nz, ny, nx), dtype=twop_final_labels.dtype)
        z_coords = np.round(z_map_local).astype(int)
        z_coords = np.clip(z_coords, 0, nz - 1)
        yy, xx = np.mgrid[0:ny, 0:nx]
        twop_3d[z_coords, yy, xx] = twop_final_labels

        # Save the final aligned 3D volume
        output_path = base_path / mouse / 'OUTPUT' / '2P' / 'registered' / f'twop_plane{plane}_aligned_3d.tiff'
        tif_imwrite(str(output_path), twop_3d.astype(np.uint16))

        # Save QualityCheck overlay TIFFs
        qa_folder = base_path / mouse / 'OUTPUT' / '2P' / 'registered' / 'QualityCheck'
        qa_folder.mkdir(parents=True, exist_ok=True)

        hcr_baseline_full = sample_hcr_binary_at_zmap(hcr_ref_masks > 0, z_map_base_full)
        twop_binary_full = (twop_warped > 0).astype(np.uint16)
        overlay_before = np.stack([hcr_baseline_full.astype(np.uint16), twop_binary_full], axis=0)
        qa_before_path = qa_folder / f'plane{plane}_BEFORE_registration_overlay.tiff'
        tif_imwrite(str(qa_before_path), overlay_before, imagej=True, metadata={'axes': 'CYX'})

        overlay_after = np.stack([hcr_local.astype(np.uint16), twop_final_binary.astype(np.uint16)], axis=0)
        qa_after_path = qa_folder / f'plane{plane}_AFTER_registration_overlay.tiff'
        tif_imwrite(str(qa_after_path), overlay_after, imagej=True, metadata={'axes': 'CYX'})

        # --- SAVE REGISTRATION PARAMS ---
        params_path = base_path / mouse / 'OUTPUT' / '2P' / 'registered' / f'twop_plane{plane}_registration_params.npz'
        params_path.parent.mkdir(parents=True, exist_ok=True)
        import io as _io
        buf = _io.BytesIO()
        np.savez_compressed(
            buf,
            # Global transform (backward compatible)
            global_theta=0.0,
            global_dy=global_params['dy'],
            global_dx=global_params['dx'],
            global_dz=global_params['dz'],
            # Composed shift fields (backward compatible: affine+local+fine)
            cumulative_dy=cumulative_dy,
            cumulative_dx=cumulative_dx,
            cumulative_dz=cumulative_dz,
            # Z-map
            z_map_local=z_map_local,
            # IoU scores (backward compatible)
            iou_baseline=iou_baseline,
            iou_global=iou_global,
            iou_local=iou_final,
            # Metadata
            erosion=EROSION,
            erosion_hcr=EROSION_HCR,
            hcr_shape=np.array(hcr_ref_masks.shape),
            # v9: erode_labels with cell-cell boundaries, wider coarse search
            algorithm_version=np.array(9),
            iou_affine_pass1=iou_pass1,
            iou_affine_composed=iou_affine,
            iou_local_300=iou_local_300,
            iou_fine_100=iou_fine_100,
            crop_offsets=np.array([bb_y0, bb_x0, bb_y1, bb_x1]),
        )
        with open(str(params_path), 'wb') as f:
            f.write(buf.getvalue())

    # --- SAVE PER-PLANE AUTO LANDMARKS ---
    orig_landmarks = pd.read_csv(landmarks_path, header=None)
    orig_landmarks = orig_landmarks.replace([np.inf, -np.inf], np.nan).dropna()

    if orig_landmarks.shape[1] == 6:
        names = orig_landmarks[0].values
        enabled = orig_landmarks[1].values
        src_x = orig_landmarks[2].values
        src_y = orig_landmarks[3].values
        src_z = np.zeros(len(orig_landmarks))
        dst_x = orig_landmarks[4].values
        dst_y = orig_landmarks[5].values
        dst_z = np.zeros(len(orig_landmarks))
    else:
        names = orig_landmarks[0].values
        enabled = orig_landmarks[1].values
        src_x = orig_landmarks[2].values
        src_y = orig_landmarks[3].values
        src_z = orig_landmarks[4].values if orig_landmarks.shape[1] > 4 else np.zeros(len(orig_landmarks))
        dst_x = orig_landmarks[5].values if orig_landmarks.shape[1] > 5 else orig_landmarks[4].values
        dst_y = orig_landmarks[6].values if orig_landmarks.shape[1] > 6 else orig_landmarks[5].values
        dst_z = orig_landmarks[7].values if orig_landmarks.shape[1] > 7 else np.zeros(len(orig_landmarks))

    prefix = "hires_stitched_" if has_hires else ""
    hcr_res_x, hcr_res_y, hcr_res_z = hcr_resolution[0], hcr_resolution[1], hcr_resolution[2]

    for plane in TARGET_PLANES:
        g_params = plane_results[plane]['global_params']

        # v8: No rotation (theta=0), just translation
        dx_um = g_params['dx'] * hcr_res_x
        dy_um = g_params['dy'] * hcr_res_y
        dz_um = g_params['dz'] * hcr_res_z

        dst_x_corrected = dst_x + dx_um
        dst_y_corrected = dst_y + dy_um
        dst_z_corrected = dst_z + dz_um

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

        auto_landmarks_path = landmarks_dir / f'{prefix}plane{plane}_to_HCR{hcr_ref}_auto.csv'
        corrected_df.to_csv(auto_landmarks_path, index=False, header=False)

    # --- AUTOMATION CHECKPOINT ---
    # Only show checkpoint when processing the reference plane (where landmarks were created)
    # Other planes use the same landmarks with per-plane refinement
    if automation_enabled and CURRENT_PLANE == REFERENCE_PLANE:
        ref_auto_path = landmarks_dir / f'{prefix}plane{REFERENCE_PLANE}_to_HCR{hcr_ref}_auto.csv'
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




