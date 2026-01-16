import os
import time
from pathlib import Path

import cv2
import hjson
import numpy as np
from rich import print as rprint
from sbxreader import sbx_get_metadata, sbx_memmap
from tifffile import imread as tif_imread
from tifffile import imwrite as tif_imwrite
from .registrations import verify_rounds
from .meta import check_rotation, get_automation_config, get_rotation_config, get_stitching_config
from .auto_stitching import auto_stitch_tiles, StitchingError
from skimage.transform import rotate

def update_map(map_x, map_y, poly_vals, src):
    for i in range(map_x.shape[0]):
        #map_x[i,:] = [poly_vals[x] for x in range(map_x.shape[1])]
        map_x[i,:] = (np.cumsum(poly_vals)/max(np.cumsum(poly_vals)))*src.shape[3]# for x in range(map_x.shape[1])]
        #map_x[i,:] = np.cumsum(poly_vals)
    for j in range(map_y.shape[1]):
        map_y[:,j] = [y for y in range(map_y.shape[0])]


def unwarp_tile(image_path: Path, unwarp_config: Path, steps: int, image_output_path: Path):
    #load the image
    images = tif_imread(image_path)
    src = images.copy()

    # load warpping parameters
    vals = np.load(unwarp_config)
    # Ensure steps is an integer (in case it's loaded as string from config)
    steps = int(steps)
    poly_vals = np.poly1d(vals)(np.arange(src.shape[3])/src.shape[3]*steps)

    # create remap map
    map_x = np.zeros((src.shape[2], src.shape[3]), dtype=np.float32)
    map_y = np.zeros((src.shape[2], src.shape[3]), dtype=np.float32)
    update_map(map_x, map_y, poly_vals, src)
    
    # apply unwarping
    output = np.zeros_like(src)
    for z in range(src.shape[0]):
        for c in range(src.shape[1]):
            src_im = src[z,c]
            dst = cv2.remap(src_im, map_x, map_y, cv2.INTER_LINEAR)
            output[z,c]=dst
        

    # save the output
    tif_imwrite(image_output_path, output, imagej=True,metadata={'axes': 'ZCYX'})

    for plane in range(output.shape[0]):
        save_file_name = Path(image_output_path).parent / f'plane{plane}' / image_output_path.name
        save_file_name.parent.mkdir(exist_ok=True)
        tif_imwrite(save_file_name,output[plane],imagej=True,metadata={'axes': 'CYX'})


def unwarp_tiles(full_manifest: dict, session: dict):
    manifest = full_manifest['data']
    if len(session['anatomical_hires_green_runs']) == 0:
        return
    print(f'Unwarping tiles for session {session["date"]}')
    for i in range(1,1+len(session['anatomical_hires_green_runs'])):
        warp_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / '2P' / 'tile' / 'warped' / f'stack_warped_C12_{i:03}.tiff'
        unwarp_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / '2P' / 'tile' / 'unwarped' / f'stack_unwarped_C12_{i:03}.tiff'   
        if os.path.exists(unwarp_path):
            continue
        unwarp_path.parent.mkdir(exist_ok=True, parents=True)
        unwarp_tile(warp_path, 
                       session['unwarp_config'], 
                       session['unwarp_steps'], 
                       unwarp_path)
        

def process_session_sbx(full_manifest: dict , session:dict):
    '''
    extract the mean of anatomical hires green and red runs to tiff file in the correct pathways.
    manifest: json dict
    session: the session to process, extracted from the manifest 
    '''
    manifest = full_manifest['data']
    if len(session['anatomical_hires_green_runs']) == 0:
        return
    print(f'processing session {session["date"]}')
    
    base_2P = Path(manifest['base_path']) / manifest['mouse_name'] / '2P' 
    save_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / '2P' / 'tile' / 'warped' 
    mouse_name = manifest['mouse_name']
    date = session['date']

    # Support variable number of tiles (not just 3)
    num_tiles = len(session['anatomical_hires_green_runs'])

    for j in range(num_tiles):
        # Sequential numbering: 001, 002, 003, 004, etc.
        stack_name_new = f'{j+1:03d}'
        save_path.mkdir(exist_ok=True, parents=True)
        save_filename = save_path / f'stack_warped_C12_{stack_name_new}.tiff'
        if save_filename.exists():
            continue

        green_run = session['anatomical_hires_green_runs'][j]
        green_sbx = base_2P / f'{mouse_name}_{date}_{green_run}' / f'{mouse_name}_{date}_{green_run}.sbx'

        red_run = session['anatomical_hires_red_runs'][j]
        red_sbx = base_2P / f'{mouse_name}_{date}_{red_run}' / f'{mouse_name}_{date}_{red_run}.sbx'
        
        print(f'processing {green_sbx} and {red_sbx}, mean across time')
        green_stack = np.expand_dims(np.array(sbx_memmap(green_sbx)[:,:,0]).mean(0),1)
        red_stack = np.expand_dims(np.array(sbx_memmap(red_sbx)[:,:,-1]).mean(0),1)
        combined_stack = np.concatenate([green_stack,red_stack],1)


        tif_imwrite(save_filename, combined_stack.astype(np.float32),
                    imagej=True, metadata={'axes': 'ZCYX'})
        for plane in range(combined_stack.shape[0]):
            save_path_plane = save_path / f'plane{plane}'
            save_path_plane.mkdir(exist_ok=True)
            tif_imwrite(save_path_plane/ f'stack_warped_C12_{stack_name_new}.tiff',combined_stack.astype(np.float32)[plane],
                    imagej=True,metadata={'axes': 'CYX'})



def stitch_tiles_and_rotate(full_manifest: dict, session: dict):
    '''
    stitch the tiles of the session
    '''
    manifest = full_manifest['data']
    if len(session['anatomical_hires_green_runs']) == 0:
        return
    tile_to_num = {'left':'001', 'center':'002', 'right':'003'}
    plane = session['functional_plane'][0]

    # NEW NAMING: Use hires_stitched_plane{X}.tiff instead of stack_stitched_C01_plane{X}.tiff
    stitched_file  = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / '2P' / 'tile' / 'stitched' / f'hires_stitched_plane{plane}.tiff'
    tile_base_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / '2P' / 'tile'
    unwarped_path = tile_base_path / 'unwarped'
    save_path_registered = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / '2P' / 'registered'
    save_path_registered.mkdir(exist_ok=True, parents=True)
    stitched_file.parent.mkdir(exist_ok=True, parents=True)

    rotation_file = stitched_file.parent / 'rotation.txt'

    # NEW NAMING: Output will be hires_stitched_plane{X}_rotated.tiff
    save_path_registered_rotated = save_path_registered / f'hires_stitched_plane{plane}_rotated.tiff'
    if save_path_registered_rotated.exists():
        rprint(f"[dim]Plane {plane}: hires stitched file already rotated[/dim]")
        return

    # Check if automated stitching is enabled
    params = full_manifest.get('params', {})
    automation_config = get_automation_config(params)
    use_auto_stitch = automation_config['stitching'] == 'auto'

    if use_auto_stitch and not stitched_file.exists():
        # Automated stitching workflow
        rprint("[bold cyan]Starting automated stitching...[/bold cyan]")

        num_tiles = len(session['anatomical_hires_green_runs'])

        # Determine whether to use unwarped or warped tiles
        # Use unwarped if unwarp_config is specified (not empty string) and files exist
        use_unwarped = False
        unwarp_config = session.get('unwarp_config', '')

        # Check if unwarp_config is specified and not empty
        has_unwarp_config = unwarp_config and (
            (isinstance(unwarp_config, str) and unwarp_config.strip()) or
            (not isinstance(unwarp_config, str))
        )

        if has_unwarp_config:
            # Check if unwarped files exist
            test_unwarp = unwarped_path / f'stack_unwarped_C12_001.tiff'
            if test_unwarp.exists():
                use_unwarped = True
                rprint(f"[green]Using unwarped tiles from: {unwarped_path}[/green]")
            else:
                rprint(f"[yellow]Unwarp config specified but unwarped files not found.[/yellow]")
                rprint(f"[yellow]Falling back to warped tiles.[/yellow]")
        else:
            rprint(f"[cyan]No unwarp config specified, using warped tiles.[/cyan]")

        # Get stitching parameters from config or use defaults
        stitch_params = get_stitching_config(params)
        overlap_fraction = stitch_params.get('overlap_fraction', 0.50)  # ~400px for 796px tiles (auto-detected ~420-430px)
        noise_floor = stitch_params.get('noise_floor', 15.0)
        min_signal_frac = stitch_params.get('min_signal_frac', 0.01)
        upsample_factor = stitch_params.get('upsample_factor', 10)

        try:
            # Run automated stitching
            auto_stitch_tiles(
                tile_dir=tile_base_path,
                num_tiles=num_tiles,
                output_path=stitched_file,
                use_unwarped=use_unwarped,
                overlap_fraction=overlap_fraction,
                noise_floor=noise_floor,
                min_signal_frac=min_signal_frac,
                upsample_factor=upsample_factor
            )
            rprint(f"[bold green]✓ Automated stitching successful![/bold green]")

        except StitchingError as e:
            rprint(f"[bold red]✗ Automated stitching failed: {e}[/bold red]")
            rprint(f"[yellow]Falling back to manual stitching workflow...[/yellow]")
            use_auto_stitch = False  # Fall through to manual workflow
        except Exception as e:
            rprint(f"[bold red]✗ Unexpected error during stitching: {e}[/bold red]")
            rprint(f"[yellow]Falling back to manual stitching workflow...[/yellow]")
            use_auto_stitch = False  # Fall through to manual workflow

    # Manual stitching workflow (original behavior)
    if not use_auto_stitch:
        # Wait for stitched file
        while not stitched_file.exists():
            output_string = f'''
            Please stitch the tiles using BigStitcher or other software:
            Input files:  [red]{unwarped_path if unwarped_path.exists() else tile_base_path / 'warped'}[/red]
            Output file:  [green]{stitched_file}[/green]

            Once you've created the stitched file, press Enter to continue...
            '''
            rprint(output_string)
            input()
        rprint(f"[green]✓ Found stitched file: {stitched_file}[/green]")

    rotation_config = get_rotation_config(full_manifest['params'])
    data = tif_imread(stitched_file)

    # Handle both ZCYX (auto-stitched) and CYX (manual) formats
    if data.ndim == 4:
        # Auto-stitched output: (Z, C, Y, X) - extract functional plane
        # plane was extracted earlier: plane = session['functional_plane'][0]
        data = data[int(plane)]  # Extract the functional plane, now (C, Y, X)

    for k in rotation_config:
        if k == 'rotation' and rotation_config[k]:
            data = np.stack([rotate(data[0], rotation_config['rotation'],resize=True,preserve_range=True),
                             rotate(data[1], rotation_config['rotation'],resize=True,preserve_range=True)])
        if k == 'fliplr' and rotation_config[k]:
            data = data[:, :, ::-1]
        if k == 'flipud' and rotation_config[k]:
            data =data[:, ::-1, :]

    # Save rotated stitched image in both locations
    tif_imwrite(stitched_file.parent / f'hires_stitched_plane{plane}_rotated.tiff', data.astype(np.float32), imagej=True, metadata={'axes': 'CYX'})
    tif_imwrite(save_path_registered_rotated, data.astype(np.float32), imagej=True, metadata={'axes': 'CYX'})
