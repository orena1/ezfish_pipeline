from pathlib import Path
import shutil
import hjson
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from rich import print as rprint
from rich.progress import track
from skimage.transform import rotate
from .registrations import verify_rounds
from .meta import check_rotation
from tifffile import imread as tif_imread
from tifffile import imwrite as tif_imwrite
from tifffile import TiffFile
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


def hires_rotate_and_save(full_manifest: dict , session: dict):
    '''
    stitch the tiles of the session
    '''
    manifest = full_manifest['data']
    if len(session['anatomical_hires_green_runs']) == 0:
        return

    plane = session['functional_plane'][0]
    original_hires_run = session['anatomical_hires_run']
    hires_file_C01  = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / '2P' / 'hires' / f'hires_C01.tiff' 
    registered_folder = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / '2P' / 'registered'

    registered_folder.mkdir(exist_ok=True, parents=True)
    hires_file_C01.parent.mkdir(exist_ok=True, parents=True)

    save_path_registered_rotated = registered_folder / f'{hires_file_C01.stem}_rotated.tiff'
    if save_path_registered_rotated.exists():
        print(f'already rotated {save_path_registered_rotated}')
        return
    
    # Copy the original hires file to the expected location
    print(f'Copying {original_hires_run} to {hires_file_C01}')
    shutil.copyfile(original_hires_run, hires_file_C01)

    rotation_config = full_manifest['params']['rotation_2p_to_HCRspec']
    data = tif_imread(hires_file_C01)
    for k in rotation_config:
        if k == 'rotation' and rotation_config[k]:
            data = np.stack([rotate(data[0], rotation_config['rotation'],resize=True,preserve_range=True),
                             rotate(data[1], rotation_config['rotation'],resize=True,preserve_range=True)])
        if k == 'fliplr' and rotation_config[k]:
            data = data[:, :, ::-1]
        if k == 'flipud' and rotation_config[k]:
            data =data[:, ::-1, :]
    tif_imwrite(hires_file_C01.parent / f'{hires_file_C01.stem}_rotated.tiff', data.astype(np.float32), imagej=True, metadata={'axes': 'CYX'})
    tif_imwrite(save_path_registered_rotated, data.astype(np.float32), imagej=True, metadata={'axes': 'CYX'})


def load_stack(stack_path: Path):
    '''
    load a stack from the given path, path can be tiff or suite2p bin file
    '''
    frames_to_load_for_mean = 50 # read every 50th frame to speed up loading <-- Need to take out!!
    if stack_path.suffix == '.binary':
        suite2p.io
    elif stack_path.suffix in ['.tiff', '.tif']:
        number_of_frames = len(TiffFile(stack_path).pages)
        frames_to_load = range(0, number_of_frames, number_of_frames//frames_to_load_for_mean) if number_of_frames > frames_to_load_for_mean else None
        data = tif_imread(stack_path, key=frames_to_load)
    return data

def extract_functional_planes(full_manifest: dict , session: dict):
    '''
    extract the mean of registered plane from the functional run of suite2p
    manifest: json dict
    session: the session to process, extracted from the manifest
    combine_with_red: if True, combine the green channel with the red channel from a different run sbx run.
    '''
    manifest = full_manifest['data']
    mouse_name = manifest['mouse_name']
    date = session['date']
    base_path = Path(manifest['base_path'])

    #save_path = base_path / mouse_name / 'OUTPUT' / '2P' / 'cellpose'
    lowres_mean_path = base_path / mouse_name / 'OUTPUT' / '2P' / 'lowres' / 'lowres_C01_meanImg.tiff'
    registered_folder = base_path / mouse_name / 'OUTPUT' / '2P' / 'registered'

    lowres_mean_path.parent.mkdir(exist_ok=True, parents=True)
    registered_folder.mkdir(exist_ok=True, parents=True)

    lowres_mean_rotated_path = registered_folder / f'{lowres_mean_path.stem}_rotated.tiff'

    if lowres_mean_rotated_path.exists():
        return

    # load stack 
    subsampled_stack = load_stack(lowres_mean_path)
    assert subsampled_stack.ndim == 4, f'Expected 4D stack, got {subsampled_stack.ndim}D'
    assert subsampled_stack.shape[1] == 2, f'Expected 2 channels, got {subsampled_stack.shape[1]} channels'
    mean_stack = subsampled_stack.mean(axis=0)  # mean across time
    tif_imwrite(lowres_mean_path, mean_stack.astype(np.float32), imagej=True, metadata={'axes': 'CYX'})

    # If we do not have highres we need to verify that rotation file is created,
    # If we have highres the rotation file should already exist otherwise the pipeline would have stopped before
    reference_HCR_round = verify_rounds(full_manifest)[1]['image_path']
    while not check_rotation(full_manifest):
        output_string = f'''
        Missing rotation specs in {full_manifest['manifest_path']} 
        for rotating [red]{save_filename_C}[/red] to {reference_HCR_round}
        Once you create these files press enter
        '''
        rprint(output_string)
        input()


    rotation_config = full_manifest['params']['rotation_2p_to_HCRspec']
    data = tif_imread(save_filename_C)

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
