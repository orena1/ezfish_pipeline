import os
import time
from pathlib import Path

import cv2
import hjson
import numpy as np
from sbxreader import sbx_get_metadata, sbx_memmap
from tifffile import imread as tif_imread
from tifffile import imwrite as tif_imwrite
try:
    from .registrations import verify_rounds  # Relative import (running as part of a package)
    from .meta import check_rotation, get_automation_config, get_rotation_config, get_stitching_config, parse_json, output_root, rprint
    from .auto_stitching import auto_stitch_tiles, StitchingError
except ImportError:
    from registrations import verify_rounds  # Absolute import (running in Jupyter notebook)
    from meta import check_rotation, get_automation_config, get_rotation_config, get_stitching_config, parse_json, output_root, rprint
    from auto_stitching import auto_stitch_tiles, StitchingError
from skimage.transform import rotate

def update_map(map_x, map_y, poly_vals, src):
    for i in range(map_x.shape[0]):
        map_x[i,:] = (np.cumsum(poly_vals)/max(np.cumsum(poly_vals)))*src.shape[3]
    for j in range(map_y.shape[1]):
        map_y[:,j] = [y for y in range(map_y.shape[0])]


def unwarp_2d(img: np.ndarray, unwarp_config: Path, steps: int) -> np.ndarray:
    """Apply resonant-scanner distortion correction to a 2D image.

    Mirrors the inline lowres unwarp used in the JS078 test notebook
    (cv2.remap with a per-column polynomial map). Returns a float32 copy.
    """
    img2d = img.astype(np.float32)
    vals = np.load(str(unwarp_config))
    steps = int(steps)
    poly_vals = np.poly1d(vals)(np.arange(img2d.shape[1]) / img2d.shape[1] * steps)
    map_x = np.zeros(img2d.shape, dtype=np.float32)
    map_y = np.zeros(img2d.shape, dtype=np.float32)
    col_map = (np.cumsum(poly_vals) / max(np.cumsum(poly_vals))) * img2d.shape[1]
    for i in range(map_x.shape[0]):
        map_x[i, :] = col_map
    for j in range(map_y.shape[1]):
        map_y[:, j] = np.arange(map_y.shape[0])
    return cv2.remap(img2d, map_x, map_y, cv2.INTER_LINEAR)


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

    # Check how many tiles need unwarping
    num_tiles = len(session['anatomical_hires_green_runs'])
    tiles_to_unwarp = []
    for i in range(1, 1 + num_tiles):
        unwarp_path = output_root(full_manifest) / '2P' / 'tile' / 'unwarped' / f'stack_unwarped_C12_{i:03}.tiff'
        if not os.path.exists(unwarp_path):
            tiles_to_unwarp.append(i)

    if not tiles_to_unwarp:
        # All tiles already unwarped - no message needed (avoids per-plane spam)
        return

    print(f'Unwarping {len(tiles_to_unwarp)}/{num_tiles} tiles for session {session["date"]}')
    for i in tiles_to_unwarp:
        warp_path = output_root(full_manifest) / '2P' / 'tile' / 'warped' / f'stack_warped_C12_{i:03}.tiff'
        unwarp_path = output_root(full_manifest) / '2P' / 'tile' / 'unwarped' / f'stack_unwarped_C12_{i:03}.tiff'
        unwarp_path.parent.mkdir(exist_ok=True, parents=True)
        unwarp_tile(warp_path,
                    session['unwarp_config'],
                    session['unwarp_steps'],
                    unwarp_path)
        

def _hires_sbx_channel_mean(sbx_path: Path, channel: int, register: bool,
                            subsample: int = 1, tmats_per_z=None,
                            return_tmats: bool = False):
    '''
    Return the time-mean of one channel of an SBX file as a (Z, Y, X) array.

    When register=True, each Z-plane's frames are motion-corrected with
    StackReg(RIGID_BODY) before averaging (tmats computed on the center crop,
    ported from register_sbx_general.py). When register=False, a plain
    time-mean is returned (original pipeline behavior).

    Optimizations:
      - The (T, Z, Y, X) chunk for the chosen channel is read in a single
        contiguous pass instead of slicing per-Z (1 network read instead of n_planes).
      - subsample > 1 keeps every Nth frame (after the leading skip) for both
        tmat estimation and averaging — ~Nx fewer StackReg fits at the cost of
        a slightly noisier mean.
      - If tmats_per_z is provided (list of (T', 3, 3) arrays from a same-file
        green pass), tmats are reused instead of re-estimated.
      - return_tmats returns ((Z, Y, X), tmats_per_z) for downstream reuse.

    Handles both single-plane and optotune multi-plane SBX (shape (T, Z, C, Y, X)).
    '''
    dat = sbx_memmap(sbx_path)  # memmap (T, Z, C, Y, X)
    n_planes = dat.shape[1]
    ch = channel if channel >= 0 else dat.shape[2] + channel

    if not register:
        out = np.array(dat[:, :, ch]).mean(0)  # (Z, Y, X)
        return (out, None) if return_tmats else out

    from pystackreg import StackReg

    # Match register_sbx_general.py: drop 3 leading frames for single-plane,
    # 2 leading cycles for multiplane (each memmap T index is one full cycle).
    skip = 3 if n_planes == 1 else 2

    # Single contiguous read of (T', Z, Y, X) for this channel.
    stack = np.array(dat[skip::max(int(subsample), 1), :, ch, :, :])

    sr = StackReg(StackReg.RIGID_BODY)
    plane_means = []
    out_tmats = [] if (return_tmats or tmats_per_z is not None) else None
    for z in range(n_planes):
        plane = stack[:, z]  # (T', Y, X)
        if tmats_per_z is not None:
            tmats = tmats_per_z[z]
        else:
            tmats = sr.register_stack(plane[:, 50:-50, 50:-50], reference='mean', verbose=False)
        registered = sr.transform_stack(plane, tmats=tmats)
        plane_means.append(registered.mean(0))
        if out_tmats is not None and tmats_per_z is None:
            out_tmats.append(tmats)
    out = np.stack(plane_means, axis=0)  # (Z, Y, X)
    if return_tmats:
        return out, (out_tmats if tmats_per_z is None else tmats_per_z)
    return out


def process_session_sbx(full_manifest: dict , session:dict):
    '''
    extract the mean of anatomical hires green and red runs to tiff file in the correct pathways.
    manifest: json dict
    session: the session to process, extracted from the manifest
    '''
    manifest = full_manifest['data']
    if len(session['anatomical_hires_green_runs']) == 0:
        return

    base_2P = Path(manifest['base_path']) / manifest['mouse_name'] / '2P'
    save_path = output_root(full_manifest) / '2P' / 'tile' / 'warped'
    mouse_name = manifest['mouse_name']
    date = session['date']

    # Motion-correct SBX frames before averaging into the hi-res tile.
    # Default on: sbx/suite2p users haven't pre-registered these stacks.
    # Tiff-mode users provide their own post-registration hires tiff and never
    # reach this function, but gate defensively on input_format anyway.
    input_format = session.get('input_format', 'sbx')
    params = full_manifest.get('params', {})
    register_hires = params.get('hires_sbx_registration', True) and input_format != 'tiff'
    # Frame subsampling for StackReg estimation+averaging. Default 4 gives ~4x
    # speedup with negligible quality loss on anatomical hi-res tiles. Set to 1
    # in the manifest to recover the original use-every-frame behavior.
    hires_subsample = max(int(params.get('hires_sbx_subsample', 4)), 1)

    # Support variable number of tiles (not just 3)
    num_tiles = len(session['anatomical_hires_green_runs'])

    # Check which tiles need processing
    tiles_to_process = []
    for j in range(num_tiles):
        stack_name_new = f'{j+1:03d}'
        save_filename = save_path / f'stack_warped_C12_{stack_name_new}.tiff'
        if not save_filename.exists():
            tiles_to_process.append(j)

    if not tiles_to_process:
        # All tiles already processed - no message needed (avoids per-plane spam)
        return

    reg_suffix = ' (with per-plane StackReg motion correction)' if register_hires else ''
    print(f'Processing {len(tiles_to_process)}/{num_tiles} tiles for session {date}{reg_suffix}')
    save_path.mkdir(exist_ok=True, parents=True)

    for j in tiles_to_process:
        stack_name_new = f'{j+1:03d}'
        save_filename = save_path / f'stack_warped_C12_{stack_name_new}.tiff'

        green_run = session['anatomical_hires_green_runs'][j]
        green_sbx = base_2P / f'{mouse_name}_{date}_{green_run}' / f'{mouse_name}_{date}_{green_run}.sbx'

        red_run = session['anatomical_hires_red_runs'][j]
        red_sbx = base_2P / f'{mouse_name}_{date}_{red_run}' / f'{mouse_name}_{date}_{red_run}.sbx'

        print(f'  Tile {j+1}: {green_sbx.name} + {red_sbx.name}')
        # If green and red come from the same SBX file (typical: both channels
        # acquired simultaneously), motion is identical — estimate tmats on green
        # and reuse for red. Halves the StackReg work per tile.
        same_file = register_hires and green_sbx.resolve() == red_sbx.resolve()
        if same_file:
            green_mean, tmats_per_z = _hires_sbx_channel_mean(
                green_sbx, 0, True, subsample=hires_subsample, return_tmats=True
            )
            red_mean = _hires_sbx_channel_mean(
                red_sbx, -1, True, subsample=hires_subsample, tmats_per_z=tmats_per_z
            )
        else:
            green_mean = _hires_sbx_channel_mean(
                green_sbx, 0, register_hires, subsample=hires_subsample
            )
            red_mean = _hires_sbx_channel_mean(
                red_sbx, -1, register_hires, subsample=hires_subsample
            )
        green_stack = np.expand_dims(green_mean, 1)
        red_stack = np.expand_dims(red_mean, 1)
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

    # Full stitched volume (Z,C,Y,X) — plane-neutral name so subsequent
    # functional planes short-circuit auto_stitch_tiles instead of re-writing
    # the identical stack under a different filename.
    stitched_file  = output_root(full_manifest) / '2P' / 'tile' / 'stitched' / 'hires_stitched.tiff'
    tile_base_path = output_root(full_manifest) / '2P' / 'tile'
    unwarped_path = tile_base_path / 'unwarped'
    save_path_registered = output_root(full_manifest) / '2P' / 'registered'
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
        stitch_channel = stitch_params.get('stitch_channel', 0)

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
                upsample_factor=upsample_factor,
                stitch_channel=stitch_channel,
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

    # First-run detection: prompt only if NO rotated files exist yet for any plane.
    mouse_name = manifest['mouse_name']
    any_rotated_exists = any(save_path_registered.glob('*_rotated.tiff'))
    manifest_path = full_manifest['manifest_path']

    if not any_rotated_exists:
        reference_HCR_round = verify_rounds(full_manifest)[1]['image_path']

        print('')
        print('=' * 70)
        print(f'  ROTATION SETUP — {mouse_name} (hi-res stitched)')
        print('=' * 70)
        print(f'  No rotated image found for plane {plane}.')
        print(f'  Before applying rotation, check your unrotated hi-res stitched image:')
        print(f'')
        print(f'    {stitched_file}')
        print(f'')
        print(f'  Compare it to the HCR reference round:')
        print(f'')
        print(f'    {reference_HCR_round}')
        print(f'')
        print(f'  Then update rotation_2p_to_HCR in your manifest:')
        print(f'    {manifest_path}')
        print(f'')
        print(f'  Set "rotation" (degrees), "fliplr" (true/false), "flipud" (true/false)')
        print('=' * 70)
        input('  Press Enter after updating the manifest...\n')

    # Always re-read manifest from disk before applying rotation so subsequent
    # planes pick up edits the user made during plane 0's prompt.
    rotation_config = get_rotation_config(parse_json(manifest_path)['params'])

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
