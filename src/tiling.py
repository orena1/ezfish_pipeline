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
from .meta import check_rotation
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
    tile_to_num = {'left':'001', 'center':'002', 'right':'003'}
    
    base_2P = Path(manifest['base_path']) / manifest['mouse_name'] / '2P' 
    save_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / '2P' / 'tile' / 'warped' 
    mouse_name = manifest['mouse_name']
    date = session['date']

    scans_names = ['left','center','right']
    for j, tile_loc in enumerate(scans_names):
        # get output stack name from tile location
        stack_name_new = tile_to_num[tile_loc]
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
    stitched_file_C01  = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / '2P' / 'tile' / 'stitched' / f'stack_stitched_C01_plane{plane}.tiff' 
    unwarped_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / '2P' / 'tile' / 'unwarped'
    save_path_registered = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / '2P' / 'registered'
    save_path_registered.mkdir(exist_ok=True, parents=True)
    stitched_file_C01.parent.mkdir(exist_ok=True, parents=True)

    rotation_file = stitched_file_C01.parent / 'rotation.txt'
    
    save_path_registered_rotated = save_path_registered / f'{stitched_file_C01.stem}_rotated.tiff'
    if save_path_registered_rotated.exists():
        print(f'already rotated {save_path_registered_rotated}')
        return
    

    reference_HCR_round = verify_rounds(full_manifest)[1]['image_path']
    while not stitched_file_C01.exists() or not check_rotation(full_manifest):
        output_string = f'''
        need to stitch the files using big-stitcher or other software, 
        input unwarped files are - [red]{unwarped_path}[/red]
        output file is - [green]{stitched_file_C01}[/green]
        we also need a rotation specs in {full_manifest['manifest_path']} that contains the rotation to fit {reference_HCR_round}
        we did not detect the file yet, once you create these files press enter
        '''
        rprint(output_string)
        input()
    rprint(f"Found stitched file {stitched_file_C01} and rotation file {rotation_file}")

    rotation_config = full_manifest['params']['rotation_2p_to_HCRspec']
    data = tif_imread(stitched_file_C01)
    for k in rotation_config:
        if k == 'rotation' and rotation_config[k]:
            data = np.stack([rotate(data[0], rotation_config['rotation'],resize=True,preserve_range=True),
                             rotate(data[1], rotation_config['rotation'],resize=True,preserve_range=True)])
        if k == 'fliplr' and rotation_config[k]:
            data = data[:, :, ::-1]
        if k == 'flipud' and rotation_config[k]:
            data =data[:, ::-1, :]
    tif_imwrite(stitched_file_C01.parent / f'{stitched_file_C01.stem}_rotated.tiff', data.astype(np.float32), imagej=True, metadata={'axes': 'CYX'})
    tif_imwrite(save_path_registered_rotated, data.astype(np.float32), imagej=True, metadata={'axes': 'CYX'})
