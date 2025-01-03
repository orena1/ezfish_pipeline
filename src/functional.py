from pathlib import Path

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
from sbxreader import sbx_get_metadata, sbx_memmap

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
    functional_plane = int(session['functional_plane'][0])
    print("Loading ops!")
    ops = np.load(suite2p_path / 'plane0/ops.npy',allow_pickle=True).item()
    print("Ops loaded!")
    planes = ops['nplanes']
    channels_needed = 'C0'
    for plane in track(range(planes), description='Extracting suite2p registered planes'):
        ## Extract the mean of the registered plane

        save_filename = save_path / f'meanImg_C0_plane{plane}.tiff'
        if save_filename.exists():
            print(f'{save_filename} already exists')
            continue
        ops = np.load(suite2p_path / f'plane{plane}/ops.npy',allow_pickle=True).item()
        img = ops['meanImg']
        assert len(img.shape)==2, f"meanImg - should be 2D, not 3D!"
        tif_imwrite(save_filename, img)

    if combine_with_red:
        save_filename_C01 = save_path / f'meanImg_C01_plane{functional_plane}.tiff'
        if not save_filename_C01.exists():
            img = tif_imread(save_filename)
            
            red_run = session['anatomical_lowres_red_runs'][0]
            red_sbx = base_path / mouse_name / '2P' / f'{mouse_name}_{date}_{red_run}' / f'{mouse_name}_{date}_{red_run}.sbx'

            print(f'processing {red_sbx}, mean across time')
            red_stack = np.array(sbx_memmap(red_sbx)[:,functional_plane,-1]).mean(0)

            combined_stack = np.stack([img, red_stack])
            tif_imwrite(save_filename_C01, combined_stack.astype(np.float32),
                        imagej=True, metadata={'axes': 'CYX'})
        channels_needed = 'C01'

    # rotate and flip the selected functional plane
    save_filename_C = save_path / f'meanImg_{channels_needed}_plane{functional_plane}.tiff'
    save_filename_rotated = save_path_registered / f'{save_filename_C.stem}_rotated.tiff'
    
    if save_filename_rotated.exists():
        return

    # this is for the case of no-hires
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
