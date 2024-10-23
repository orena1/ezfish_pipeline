import numpy as np
from scipy.sparse import csr_matrix
from tifffile import imread as tif_imread
from suite2p.io import binary
from suite2p.io import tiff
from pathlib import Path
from tifffile import imread as tif_imread
from tifffile import imwrite as tif_imwrite
import scipy.io as sio
import matplotlib.pyplot as plt
from rich.progress import track
from skimage.transform import rotate
import hjson


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
        ## Jonna check if needed.

        save_filename = save_path / f'mean_over_time_C01_plane{plane}.tiff'
        print(save_filename)
        if save_filename.exists():
            print(f'{save_filename} already exists')
            continue
        # set up binary file
        bin_file = binary.BinaryFile(filename=suite2p_path / f'plane{plane}/data.bin', Lx=ops['Lx'],Ly=ops['Ly'])

        # move data to a numpy
        all_data = bin_file.data

        #Save as mean tiff (from suite2p)
        mean_mat = np.mean(all_data, axis = 0)

        tif_imwrite(save_filename, mean_mat)


    # rotate and flip the selected functional plane
    save_filename_C01  = save_path / f'mean_over_time_C01_plane{functional_plane}.tiff'
    save_filename_rotated = save_path_registered / f'{save_filename_C01.stem}_rotated.tiff'

    if save_filename_rotated.exists():
        return
    
    rotation_file =  Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / '2P' / 'tile' / 'stitched' / 'rotation.txt'
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
