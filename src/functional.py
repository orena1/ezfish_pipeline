from pathlib import Path

import numpy as np
from skimage.transform import rotate
try:
    from .registrations import verify_rounds  # Relative import (running as part of a package)
    from .meta import check_rotation, get_rotation_config, parse_json, output_root, rprint, track
except ImportError:
    from registrations import verify_rounds  # Absolute import (running in Jupyter notebook)
    from meta import check_rotation, get_rotation_config, parse_json, output_root, rprint, track
from tifffile import imread as tif_imread
from tifffile import imwrite as tif_imwrite
from sbxreader import sbx_get_metadata, sbx_memmap

def _maybe_apply_lowres_unwarp(img, full_manifest, session):
    """Optionally apply resonant-scanner distortion correction to a 2D lowres image.

    Enabled when ``session['lowres_unwarp_config']`` is set. The config path is
    resolved directly or, if missing, under
    ``{base_path}/Calibration_files_for_unwarping/``. Number of polynomial steps
    comes from ``session['unwarp_steps']`` (default 28).
    """
    cfg = session.get('lowres_unwarp_config')
    if not cfg:
        return img
    from .tiling import unwarp_2d
    cfg_path = Path(cfg)
    if not cfg_path.exists():
        base_path = Path(full_manifest['data']['base_path'])
        candidate = base_path / 'Calibration_files_for_unwarping' / cfg
        if candidate.exists():
            cfg_path = candidate
        else:
            rprint(f"[yellow]  lowres_unwarp_config not found ({cfg}); skipping lowres unwarp[/yellow]")
            return img
    steps = int(session.get('unwarp_steps', 28))
    rprint(f"  Applying lowres scanner unwarp: {cfg_path.name} (steps={steps})")
    return unwarp_2d(img, cfg_path, steps)


def get_number_of_suite2p_planes(suite2p_path: Path):
    """
    Get the number of planes in a suite2p run.
    The ops file is very big (300MBs) so it will try to write a cache file
    with the number of planes.
    """
    cache_file = suite2p_path / 'plane0' / 'nplanes_cache.txt'
    if not cache_file.exists() or \
        (suite2p_path / 'plane0' / 'ops.npy').stat().st_mtime > cache_file.stat().st_mtime:
        # if the cache file does not exist or is older than the ops file,
        ops = np.load(suite2p_path / 'plane0/ops.npy', allow_pickle=True).item()
        nplanes = ops['nplanes']
        with open(cache_file, 'w') as f:
            f.write(str(nplanes))

    with open(cache_file, 'r') as f:
        nplanes = int(f.read())
    return nplanes


def _get_input_format(session):
    return session.get('input_format', 'sbx')


def prepare_tiff_input(full_manifest: dict, session: dict):
    '''
    Copy user-provided mean image tiff from 2P/ into OUTPUT/2P/cellpose/ with
    standard naming, then rotate. Autodetects 1 vs 2 channel from tiff shape.
    If plane_{N}_hires.tiff exists, also copies and rotates it.
    '''
    manifest = full_manifest['data']
    base_path = Path(manifest['base_path'])
    mouse_name = manifest['mouse_name']
    functional_plane = int(session['functional_plane'][0])

    src_path = base_path / mouse_name / '2P' / f'plane_{functional_plane}.tiff'
    hires_src = base_path / mouse_name / '2P' / f'plane_{functional_plane}_hires.tiff'
    save_path = output_root(full_manifest) / '2P' / 'cellpose'
    save_path_registered = output_root(full_manifest) / '2P' / 'registered'
    save_path.mkdir(exist_ok=True, parents=True)
    save_path_registered.mkdir(exist_ok=True, parents=True)

    img = tif_imread(str(src_path))

    # Optional resonant-scanner distortion correction on the lowres input
    # (applied before channel split so the downstream pipeline sees corrected pixels).
    if img.ndim == 2:
        img = _maybe_apply_lowres_unwarp(img, full_manifest, session)
    elif img.ndim == 3 and img.shape[0] == 2:
        img = np.stack([
            _maybe_apply_lowres_unwarp(img[0], full_manifest, session),
            _maybe_apply_lowres_unwarp(img[1], full_manifest, session),
        ])

    # Autodetect channels: 2D = single channel, 3D with shape (2,Y,X) = two channel
    if img.ndim == 3 and img.shape[0] == 2:
        channels_needed = 'C01'
        # Save green channel separately (used by cellpose and registration)
        green_path = save_path / f'lowres_meanImg_C0_plane{functional_plane}.tiff'
        if not green_path.exists():
            tif_imwrite(str(green_path), img[0])
        # Save combined
        combined_path = save_path / f'lowres_meanImg_C01_plane{functional_plane}.tiff'
        if not combined_path.exists():
            tif_imwrite(str(combined_path), img.astype(np.float32),
                        imagej=True, metadata={'axes': 'CYX'})
    else:
        channels_needed = 'C0'
        dest_path = save_path / f'lowres_meanImg_C0_plane{functional_plane}.tiff'
        if not dest_path.exists():
            tif_imwrite(str(dest_path), img)

    # Rotate lowres
    save_filename_C = save_path / f'lowres_meanImg_{channels_needed}_plane{functional_plane}.tiff'
    _rotate_plane(full_manifest, save_filename_C, save_path_registered, functional_plane)

    # If hires tiff provided, copy and rotate it too
    if hires_src.exists():
        hires_rotated = save_path_registered / f'hires_stitched_plane{functional_plane}_rotated.tiff'
        if not hires_rotated.exists():
            print(f"Rotating hires image for plane {functional_plane}")
            _rotate_plane(full_manifest, hires_src, save_path_registered, functional_plane,
                          output_name=f'hires_stitched_plane{functional_plane}_rotated.tiff')


def prepare_suite2p_input(full_manifest: dict, session: dict):
    '''
    Extract meanImg from user-provided suite2p folder at 2P/suite2p/plane{N}/ops.npy,
    save to OUTPUT/2P/cellpose/, then rotate.
    '''
    manifest = full_manifest['data']
    base_path = Path(manifest['base_path'])
    mouse_name = manifest['mouse_name']
    functional_plane = int(session['functional_plane'][0])

    suite2p_path = base_path / mouse_name / '2P' / 'suite2p'
    save_path = output_root(full_manifest) / '2P' / 'cellpose'
    save_path_registered = output_root(full_manifest) / '2P' / 'registered'
    save_path.mkdir(exist_ok=True, parents=True)
    save_path_registered.mkdir(exist_ok=True, parents=True)

    save_filename = save_path / f'lowres_meanImg_C0_plane{functional_plane}.tiff'
    if not save_filename.exists():
        ops = np.load(suite2p_path / f'plane{functional_plane}' / 'ops.npy', allow_pickle=True).item()
        img = ops['meanImg']
        assert img.ndim == 2, f"meanImg should be 2D, got shape {img.shape}"
        img = _maybe_apply_lowres_unwarp(img, full_manifest, session)
        tif_imwrite(str(save_filename), img)
        print(f"Extracted meanImg for plane {functional_plane}")

    _rotate_plane(full_manifest, save_filename, save_path_registered, functional_plane)


def _rotate_plane(full_manifest, save_filename_C, save_path_registered, functional_plane, output_name=None):
    '''Shared rotation logic for all input formats.'''
    if output_name:
        save_filename_rotated = save_path_registered / output_name
    else:
        save_filename_rotated = save_path_registered / f'{save_filename_C.stem}_rotated.tiff'

    if save_filename_rotated.exists():
        print(f"Plane {functional_plane}: rotated file already exists")
        return

    any_rotated_exists = any(save_path_registered.glob('*_rotated.tiff'))
    manifest_path = full_manifest['manifest_path']
    # Presence of rotation block = user-configured; banner only fires for true first-run.
    rotation_explicitly_provided = (
        'rotation_2p_to_HCR' in full_manifest['params']
        or 'rotation_2p_to_HCRspec' in full_manifest['params']
    )

    if not any_rotated_exists and rotation_explicitly_provided:
        cfg = get_rotation_config(full_manifest['params'])
        rprint(f"  Applying rotation_2p_to_HCR: rotation={cfg.get('rotation', 0)}, "
               f"fliplr={cfg.get('fliplr', False)}, flipud={cfg.get('flipud', False)}")
    elif not any_rotated_exists:
        reference_HCR_round = verify_rounds(full_manifest)[1]['image_path']

        print('')
        print('=' * 70)
        print(f'  ROTATION SETUP — {full_manifest["data"]["mouse_name"]}')
        print('=' * 70)
        print(f'  No rotated image found for plane {functional_plane}.')
        print(f'  Before applying rotation, check your unrotated 2P image:')
        print(f'')
        print(f'    {save_filename_C}')
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

    # Always re-read manifest from disk before applying rotation so that any
    # edits the user made during an earlier prompt (e.g. during hires stitching
    # on plane 0) are picked up for this plane's rotation too.
    rotation_config = get_rotation_config(parse_json(manifest_path)['params'])

    # Apply rotation
    data = tif_imread(str(save_filename_C))
    print(f"  Loaded {save_filename_C.name if hasattr(save_filename_C, 'name') else save_filename_C}: shape={data.shape}, dtype={data.dtype}")
    # Hires tiff inputs may be multi-dimensional (ZCYX, CYX, etc.) — reduce to 2D
    if output_name:
        while data.ndim > 2:
            print(f"  Reducing {data.ndim}D array (shape {data.shape}) → taking [0]")
            data = data[0]
    if data.ndim not in (2, 3):
        raise ValueError(f"Unexpected image dimensions {data.ndim} (shape {data.shape}) for {save_filename_C}")
    for k in rotation_config:
        if k == 'rotation' and rotation_config[k]:
            if data.ndim == 2:
                data = rotate(data, rotation_config['rotation'], resize=True, preserve_range=True)
            else:
                data = np.stack([rotate(ch, rotation_config['rotation'], resize=True, preserve_range=True)
                                 for ch in data])
        if k == 'fliplr' and rotation_config[k]:
            data = np.flip(data, axis=-1)
        if k == 'flipud' and rotation_config[k]:
            data = np.flip(data, axis=-2)
    file_specs = {'axes': 'YX' if data.ndim == 2 else 'CYX'}

    tif_imwrite(str(save_filename_rotated),
                data.astype(np.float32),
                imagej=True,
                metadata=file_specs)
    print(f"Rotated plane {functional_plane}")


def extract_suite2p_registered_planes(full_manifest: dict , session: dict, combine_with_red = False):
    '''
    extract the mean of registered plane from the functional run of suite2p
    manifest: json dict
    session: the session to process, extracted from the manifest
    combine_with_red: if True, combine the green channel with the red channel from a different run sbx run.
    '''
    manifest = full_manifest['data']
    mouse_name = manifest['mouse_name']
    date = session['date']
    suite2p_run = session['functional_run'][0]
    base_path = Path(manifest['base_path'])

    suite2p_path = base_path / mouse_name / '2P' /  f'{mouse_name}_{date}_{suite2p_run}' / 'suite2p'
    save_path = output_root(full_manifest) / '2P' / 'cellpose'
    save_path_registered = output_root(full_manifest) / '2P' / 'registered'

    save_path.mkdir(exist_ok=True, parents=True)
    save_path_registered.mkdir(exist_ok=True, parents=True)

    functional_plane = int(session['functional_plane'][0])
    planes = get_number_of_suite2p_planes(suite2p_path)
    channels_needed = 'C0'

    # Check which planes need extraction
    planes_to_extract = []
    for plane in range(planes):
        save_filename = save_path / f'lowres_meanImg_C0_plane{plane}.tiff'
        if not save_filename.exists():
            planes_to_extract.append(plane)

    if not planes_to_extract:
        # All planes already extracted - skip silently (this runs once per session)
        pass
    else:
        # Only show progress bar if there's work to do
        for plane in track(planes_to_extract, description=f'Extracting {len(planes_to_extract)}/{planes} suite2p planes'):
            save_filename = save_path / f'lowres_meanImg_C0_plane{plane}.tiff'
            ops = np.load(suite2p_path / f'plane{plane}/ops.npy',allow_pickle=True).item()
            img = ops['meanImg']
            assert len(img.shape)==2, f"meanImg - should be 2D, not 3D!"
            tif_imwrite(save_filename, img)

    if combine_with_red:
        save_filename_C01 = save_path / f'lowres_meanImg_C01_plane{functional_plane}.tiff'
        if not save_filename_C01.exists():
            save_filename_green = save_path / f'lowres_meanImg_C0_plane{functional_plane}.tiff'
            img = tif_imread(save_filename_green)
            red_run = session['anatomical_lowres_red_runs'][0]
            red_sbx = base_path / mouse_name / '2P' / f'{mouse_name}_{date}_{red_run}' / f'{mouse_name}_{date}_{red_run}.sbx'

            print(f'processing {red_sbx}, mean across time')
            red_stack = np.array(sbx_memmap(red_sbx)[:,functional_plane,-1]).mean(0)

            combined_stack = np.stack([img, red_stack])
            tif_imwrite(save_filename_C01, combined_stack.astype(np.float32),
                        imagej=True, metadata={'axes': 'CYX'})
        channels_needed = 'C01'

    # Rotate the current functional plane
    save_filename_C = save_path / f'lowres_meanImg_{channels_needed}_plane{functional_plane}.tiff'
    _rotate_plane(full_manifest, save_filename_C, save_path_registered, functional_plane)
