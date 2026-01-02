import re
import os
import sys
import hjson
import zarr
import shutil
import numpy as np
from pathlib import Path

try:
    from .meta import parse_json  # Relative import (for running as part of a package)
except ImportError:
    from meta import parse_json  # Absolute import (for running in Jupyter Notebook)
    
from rich.progress import track
import SimpleITK as sitk
from rich import print as rprint
from tifffile import imwrite as tif_imwrite
from tifffile import imread as tif_imread
from scipy.interpolate import RBFInterpolator, griddata
from .registrations_utils import load_landmarks, build_z_map, tps_warp_2p_to_hcr, erode_labels, global_alignment, sample_hcr_binary_at_zmap, compute_iou, apply_shift_fields, compute_tile_shifts, interpolate_shift_field, shift_2d, rotate_2d

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
    
    missing = False
    while 1:
        if not reference_round.exists():
            print(f"Reference round {reference_round} not found\n")
            missing = True
        for i in mov_rounds:
            if not i.exists():
                print(f"Round {i} not found\n")
                missing = True
        if not missing:
            print("All files found")
            break
        print("Please add the missing files and press enter to continue")
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

    for HCR_round_to_register in register_rounds:
        mov_round = round_to_rounds[HCR_round_to_register]
        # File names
        HCR_fix_image_path = reference_round['image_path'] # The fix image that all other rounds will be registerd to (include all channels!)
        HCR_mov_image_path = mov_round['image_path'] # The image that will be registered to the fix image (include all channels!)

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
        red_mut_x = manifest['HCR_to_HCR_params']['red_mut_x']
        red_mut_y = manifest['HCR_to_HCR_params']['red_mut_y']
        red_mut_z = manifest['HCR_to_HCR_params']['red_mut_z']

        fix_lowres_spacing = fix_image_spacing * np.array([red_mut_y, red_mut_x, red_mut_z])
        mov_lowres_spacing = mov_image_spacing * np.array([red_mut_y, red_mut_x, red_mut_z])


        # get block size from the registration file
        blocksize_match = re.findall(r'bs(\d+)_(\d+)_(\d+)', Path(round_to_rounds[HCR_round_to_register]['registrations'][0]).name)
        blocksize = [int(num) for num in blocksize_match[0]]

        print("Loading images")
        HCR_fix_round = tif_imread(HCR_fix_image_path)[:, 0]
        HCR_mov_round = tif_imread(HCR_mov_image_path)

        # load the registration files
        affine = np.loadtxt(fr"{reg_path}/_affine.mat")
        deform = zarr.load(fr"{reg_path}/deform.zarr")
        data_paths = []
        fix_highres = HCR_fix_round.transpose(2, 1, 0) # from Z,X,Y to Y,X,Z

        # Loop through channels starting with 1, which ignores the first channel which has already been registered
        for channel in track(range(HCR_mov_round.shape[1]), description="Registering channels"):
            output_channel_path = Path(fr"{reg_path}/out_c{channel}.zarr")
            output_channel_tiff_path = output_channel_path.parent / output_channel_path.name.replace('.zarr','.tiff')
            if os.path.exists(output_channel_path) and os.path.exists(output_channel_tiff_path):
                print(f"Channel {channel} already registered")
                data_paths.append(output_channel_path)
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

            data = zarr.load(output_channel_path)
            data_paths.append(output_channel_path)

            tif_imwrite(output_channel_tiff_path
                        ,data.transpose(2,1,0))
        
        print(f"Saving full stack -{full_stack_path}")
        full_stack = [] 
        for path in data_paths:
            full_stack.append(zarr.load(path))
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



def twop_to_hcr_registration(full_manifest, session):

    rprint("\n" + "="*80)
    # if reference_plane is not None:
    #     rprint(f"[bold green] Align Masks for Additional Plane (Reference: Plane {reference_plane})[bold green]")
    # else:
    #     rprint("[bold green] Align Rounds Masks[bold green]")
    rprint("="*80)

    manifest = full_manifest['data']
    params = full_manifest['params']
    base_path = Path(manifest['base_path'])
    mouse = manifest['mouse_name']

    # def extract registration params
    # TODO

    # --- EROSION ---
    EROSION = 2  # Erode 2P masks

    # --- GLOBAL ALIGNMENT ---
    ROTATION_RANGE = (-1, 1)      # degrees
    ROTATION_STEP = 0.5           # degrees
    Z_RANGE_GLOBAL = (-2, 2)    # planes (wider for cross-plane)
    XY_MAX_GLOBAL = 80            # pixels

    # --- LOCAL TILES ---
    TILE_SIZES = [150, 75]        # pyramid
    TILE_OVERLAP = 0.30           # 30% overlap between tiles
    TILE_XY_MAX = 15              # Max shift allowed for the tiles !pixels!
    TILE_Z_RANGE = (-2, 2)        # slightly wider for cross-plane -> TILE_Z_MAX = 2 = (-2,2) !pixels!
    MIN_TILE_PIXELS = 50          # Min non-zero pixles in the tile, if there is less we just keep the tile as is.

    REFERENCE_PLANE = session['functional_plane'][0]
    TARGET_PLANES = session['additional_functional_planes'] + [REFERENCE_PLANE]
    round_to_rounds, reference_round, register_rounds = verify_rounds(full_manifest, parse_registered = True, 
                                                                    print_rounds = False, print_registered = False)

    # HCR paths
    hcr_ref_round_path = base_path / mouse / 'OUTPUT' / 'HCR' / 'full_registered_stacks' / f"HCR{reference_round['round']}.tiff"
    hcr_ref_masks_path = base_path / mouse / 'OUTPUT' / 'HCR' / 'cellpose' / f"HCR{reference_round['round']}_masks.tiff"
    # 2P reference path
    twop_ref_masks_path =  base_path / mouse / 'OUTPUT' / '2P' / 'cellpose' / f'lowres_meanImg_C0_plane{REFERENCE_PLANE}_seg_rotated.tiff'

    # Create output folder
    output_folder = base_path / mouse / 'OUTPUT' / 'MERGED' / 'aligned_masks'
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load and prepare reference landmarks (filter and modify from um to pixels)
    landmarks_path = base_path / mouse / 'OUTPUT' / '2P' / 'registered' / f'stitched_C01_plane{REFERENCE_PLANE}_rotated_TO_HCR1_landmarks.csv'
    landmarks_df = load_landmarks(landmarks_path, hcr_ref_round_path, twop_ref_masks_path)

    # Load HCR reference masks
    hcr_ref_masks = tif_imread(str(hcr_ref_masks_path))
    y1, x1 = hcr_ref_masks.shape[1], hcr_ref_masks.shape[2]

    # Build z-map from reference landmarks
    z_map_base = build_z_map(landmarks_df, (y1, x1))

    print("\n" + "="*70)
    print("ALIGNING ALL PLANES")
    print("="*70)

    # Loading data 
    # Load 2P masks for each plane
    twop_2d_planes = {}
    for plane in TARGET_PLANES:
        twop_path = base_path / mouse / 'OUTPUT' / '2P' / 'cellpose' / f'lowres_meanImg_C0_plane{plane}_seg_rotated.tiff'
        twop_2d_planes[plane] = tif_imread(str(twop_path))
        print(f"Plane {plane}: {twop_2d_planes[plane].shape}, {len(np.unique(twop_2d_planes[plane]))-1} cells")

    # Store results for each plane
    plane_results = {}
    for plane in TARGET_PLANES:
        # All planes to align (including reference), the refernce just gives a guess but it goes through the full alignment anyway
        print(f"\n{'='*70}")
        print(f"PLANE {plane}" + (" (REFERENCE)" if plane == REFERENCE_PLANE else ""))
        print("="*70)

        # TPS warp using reference landmarks
        print("  TPS warp...")
        twop_warped = tps_warp_2p_to_hcr(twop_2d_planes[plane], landmarks_df, hcr_ref_masks.shape)
        assert twop_warped.shape == (y1, x1), f"Warped shape {twop_warped.shape} does not match HCR shape {(y1, x1)}" #<<-- delete check!

        # Erode
        twop_eroded = erode_labels(twop_warped, EROSION)
        twop_binary = twop_eroded > 0

        n_cells = len(np.unique(twop_eroded)) - 1
        print(f"  Cells after erosion: {n_cells}")

        # Baseline IoU
        hcr_baseline = sample_hcr_binary_at_zmap(hcr_ref_masks, z_map_base, z_offset=0)
        iou_baseline = compute_iou(twop_binary, hcr_baseline)
        print(f"  Baseline IoU: {iou_baseline:.4f}")

        # Global alignment
        print("  Global search...")
        global_params = global_alignment(
            twop_binary, hcr_ref_masks, z_map_base,
            ROTATION_RANGE, ROTATION_STEP,
            Z_RANGE_GLOBAL, XY_MAX_GLOBAL,
            desc=f"Plane {plane}"
        )

        g_theta, g_dz, g_dy, g_dx = global_params['theta'], global_params['dz'], global_params['dy'], global_params['dx']
        print(f"  Global: θ={g_theta}°, dz={g_dz}, dy={g_dy}, dx={g_dx}")

        # Apply global
        twop_global = rotate_2d(twop_binary, g_theta)
        twop_global = shift_2d(twop_global, g_dy, g_dx)
        hcr_global = sample_hcr_binary_at_zmap(hcr_ref_masks, z_map_base, z_offset=g_dz)
        iou_global = compute_iou(twop_global, hcr_global)
        print(f"  Global IoU: {iou_global:.4f}")

        # Local tile alignment
        print("  Local tiles...")
        cumulative_dz = np.zeros((y1, x1))
        cumulative_dy = np.zeros((y1, x1))
        cumulative_dx = np.zeros((y1, x1))
        twop_current = twop_global.copy()

        all_tile_results = []
        for tile_size in TILE_SIZES:
            tile_results = compute_tile_shifts(
                twop_current, hcr_ref_masks, z_map_base, g_dz,
                tile_size, TILE_OVERLAP, TILE_XY_MAX, TILE_Z_RANGE, MIN_TILE_PIXELS
            )
            all_tile_results.append({'tile_size': tile_size, 'tiles': tile_results})
            
            dz_field, dy_field, dx_field = interpolate_shift_field(tile_results, (y1, x1))
            cumulative_dz += dz_field
            cumulative_dy += dy_field
            cumulative_dx += dx_field
            
            twop_current = apply_shift_fields(twop_global, cumulative_dy, cumulative_dx)
            print(f"    Tiles {tile_size}px: {len(tile_results)} tiles")
        
        # Final local result
        twop_local = twop_current
        z_map_local = z_map_base + g_dz + cumulative_dz
        hcr_local = sample_hcr_binary_at_zmap(hcr_ref_masks, z_map_local, z_offset=0)
        iou_local = compute_iou(twop_local, hcr_local)
        print(f"  Final IoU after gloabl + local transform: {iou_local:.4f}")
        
        # Store results
        plane_results[plane] = {
            'twop_original': twop_2d_planes[plane],
            'twop_warped': twop_warped,
            'twop_eroded': twop_eroded,
            'twop_binary': twop_binary,
            'twop_global': twop_global,
            'twop_local': twop_local,
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
        
        # Create 3D volume matching HCR dimensions
        nz, ny, nx = hcr_ref_masks.shape
        twop_3d = np.zeros((nz, ny, nx), dtype=twop_local.dtype)

        # Get z-coordinates for each pixel (rounded to nearest integer)
        z_coords = np.round(z_map_local).astype(int)

        # Clip z-coordinates to valid range
        z_coords = np.clip(z_coords, 0, nz - 1)

        # Create coordinate arrays for indexing
        yy, xx = np.mgrid[0:ny, 0:nx]

        # Place 2D mask into 3D volume at z-positions specified by z_map
        twop_3d[z_coords, yy, xx] = twop_local

        # Save as TIFF
        output_path = base_path / mouse / 'OUTPUT' / '2P' / 'registered' / f'twop_plane{plane}_aligned_3d.tiff'
        tif_imwrite(str(output_path), twop_3d)
        print(f" Saved twop_local stack to {output_path}")

    print("\n" + "="*70)
    print("ALIGNMENT COMPLETE")
    print("="*70)




