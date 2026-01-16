from pathlib import Path

import hjson
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from rich import print as rprint
from rich.progress import track
from skimage.transform import rotate
from .registrations import verify_rounds
from .meta import check_rotation, get_rotation_config
from tifffile import imread as tif_imread
from tifffile import imwrite as tif_imwrite
from sbxreader import sbx_get_metadata, sbx_memmap

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
    save_path = base_path / mouse_name / 'OUTPUT' / '2P' / 'cellpose'
    save_path_registered = base_path / mouse_name / 'OUTPUT' / '2P' / 'registered'

    save_path.mkdir(exist_ok=True, parents=True)
    save_path_registered.mkdir(exist_ok=True, parents=True)

    functional_plane = int(session['functional_plane'][0])
    planes = get_number_of_suite2p_planes(suite2p_path)
    channels_needed = 'C0'
    skipped_planes = []
    for plane in track(range(planes), description='Extracting suite2p registered planes'):
        ## Extract the mean of the registered plane

        save_filename = save_path / f'lowres_meanImg_C0_plane{plane}.tiff'
        if save_filename.exists():
            skipped_planes.append(plane)
            continue
        ops = np.load(suite2p_path / f'plane{plane}/ops.npy',allow_pickle=True).item()
        img = ops['meanImg']
        assert len(img.shape)==2, f"meanImg - should be 2D, not 3D!"
        tif_imwrite(save_filename, img)

    if skipped_planes:
        rprint(f"[dim]Planes {skipped_planes}: already exist[/dim]")

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

    # Determine which planes to rotate
    # Support both old and new manifest formats
    if 'functional_planes' in session:
        # New format: all planes in one list
        planes_to_rotate = session['functional_planes']
    else:
        # Old format: functional_plane + optional additional_functional_planes
        planes_to_rotate = [functional_plane]
        if 'additional_functional_planes' in session:
            planes_to_rotate.extend(session['additional_functional_planes'])

    # Rotate and flip all planes that will be used
    reference_HCR_round = verify_rounds(full_manifest)[1]['image_path']
    while not check_rotation(full_manifest):
        output_string = f'''
        Missing rotation specs in {full_manifest['manifest_path']}
        for rotating low-res images to {reference_HCR_round}
        Once you create these files press enter
        '''
        rprint(output_string)
        input()

    rotation_config = get_rotation_config(full_manifest['params'])

    skipped_rotations = []
    for plane_to_rotate in planes_to_rotate:
        # For additional planes, always use C0 (no red channel combination)
        if plane_to_rotate == functional_plane:
            channels = channels_needed
        else:
            channels = 'C0'

        save_filename_C = save_path / f'lowres_meanImg_{channels}_plane{plane_to_rotate}.tiff'
        save_filename_rotated = save_path_registered / f'{save_filename_C.stem}_rotated.tiff'

        if save_filename_rotated.exists():
            skipped_rotations.append(plane_to_rotate)
            continue

        data = tif_imread(save_filename_C)
        if data.ndim == 2:
            for k in rotation_config:
                if k == 'rotation' and rotation_config[k]:
                    data = rotate(data, rotation_config['rotation'], resize=True, preserve_range=True)
                if k == 'fliplr' and rotation_config[k]:
                    data = data[:,::-1]
                if k == 'flipud' and rotation_config[k]:
                    data =data[::-1,:]
            file_specs = {'axes': 'YX'}
        if data.ndim == 3:
            for k in rotation_config:
                if k == 'rotation' and rotation_config[k]:
                    data = np.stack([rotate(data[0], rotation_config['rotation'], resize=True, preserve_range=True),
                                     rotate(data[1], rotation_config['rotation'], resize=True, preserve_range=True)])
                if k == 'fliplr' and rotation_config[k]:
                    data = data[:,:,::-1]
                if k == 'flipud' and rotation_config[k]:
                    data =data[:,::-1,:]
            file_specs = {'axes': 'CYX'}

        tif_imwrite(save_filename_rotated,
                    data.astype(np.float32),
                    imagej=True,
                    metadata=file_specs)
        rprint(f"[green]Rotated plane {plane_to_rotate}[/green]")

    if skipped_rotations:
        rprint(f"[dim]Planes {skipped_rotations}: rotated files already exist[/dim]")
