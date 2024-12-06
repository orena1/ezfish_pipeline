from pathlib import Path

import hjson
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from rich import print as rprint
from rich.progress import track
from scipy.sparse import csr_matrix
from skimage.transform import rotate
from suite2p.io import binary, tiff
from .registrations import verify_rounds
from tifffile import imread as tif_imread
from tifffile import imwrite as tif_imwrite


def extract_suite2p_registered_planes(manifest: dict , session: dict):
    '''
    extract the mean of registered plane from the functional run of suite2p
    manifest: json dict
    session: the session to process, extracted from the manifest
    '''
    mouse_name = manifest['mouse_name']
    date = session['date']
    suite2p_run = session['functional_run'][0]

    suite2p_path = Path(manifest['base_path']) / manifest['mouse_name'] / '2P' /  f'{mouse_name}_{date}_{suite2p_run}' / 'suite2p'
    save_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / '2P' / 'cellpose'
    save_path_registered = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / '2P' / 'registered'
    save_path.mkdir(exist_ok=True, parents=True)
    functional_plane = session['functional_plane'][0]
    ops = np.load(suite2p_path / 'plane0/ops.npy',allow_pickle=True).item()
    planes = ops['nplanes']

    for plane in track(range(planes), description='Extracting suite2p registered planes'):
        
        ## Extract the mean of the registered plane
        for img_key in ['meanImg', 'meanImgE', 'max_proj']:
            save_filename = save_path / f'{img_key}_C01_plane{plane}.tiff'
            if save_filename.exists():
                print(f'{save_filename} already exists')
                continue
            ops = np.load(suite2p_path / f'plane{plane}/ops.npy',allow_pickle=True).item()
            img = ops[img_key]
            tif_imwrite(save_filename, img)


    # rotate and flip the selected functional plane
    save_filename_C01  = save_path / f'meanImg_C01_plane{functional_plane}.tiff'
    save_filename_rotated = save_path_registered / f'{save_filename_C01.stem}_rotated.tiff'
    
    if save_filename_rotated.exists():
        return

    # this is for the case of no-hires
    rotation_file =  Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / '2P' / 'tile' / 'stitched' / 'rotation.txt'
    reference_HCR_round = verify_rounds(manifest)[1]['image_path']
    while not rotation_file.exists():
        output_string = f'''
        Missing rotation file for rotating [red]{save_filename_C01}[/red] to {reference_HCR_round}
        Once you create these files press enter
        '''
        rprint(output_string)
        input()


    rotation_config = hjson.load(open(rotation_file,'r'))
    data = tif_imread(save_filename_C01)
    assert data.ndim == 2, 'suite2p binary files should be 2D, not 3D!, can fix but not implemented yet'
    for k in rotation_config:
        if k == 'rotation' and rotation_config[k]:
            data = rotate(data, rotation_config['rotation'])
        if k == 'fliplr' and rotation_config[k]:
            data = data[:,::-1]
        if k == 'flipud' and rotation_config[k]:
            data =data[::-1,:]
    

    tif_imwrite(save_filename_rotated, 
                data.astype(np.float32), 
                imagej=True, 
                metadata={'axes': 'YX'})
