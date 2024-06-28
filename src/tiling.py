import cv2
import numpy as np
from tifffile import imwrite as tif_imwrite
from tifffile import imread as tif_imread
from sbxreader import sbx_get_metadata, sbx_memmap
import numpy as np
from pathlib import Path


def update_map(map_x, map_y, poly_vals, src):
    for i in range(map_x.shape[0]):
        #map_x[i,:] = [poly_vals[x] for x in range(map_x.shape[1])]
        map_x[i,:] = (np.cumsum(poly_vals)/max(np.cumsum(poly_vals)))*src.shape[3]# for x in range(map_x.shape[1])]
        #map_x[i,:] = np.cumsum(poly_vals)
    for j in range(map_y.shape[1]):
        map_y[:,j] = [y for y in range(map_y.shape[0])]


def unwarp_tile(image_path: Path, unwarp_config: Path, steps: int, output_path: Path):
    #load the image
    images = tif_imread(image_path)
    src = images.copy()

    # load warpping parameters
    vals = np.load(unwarp_config)
    steps = 28
    poly_vals = np.poly1d(vals)(np.arange(src.shape[3])/src.shape[3]*steps)

    # create remap map
    map_x = np.zeros((src.shape[2], src.shape[3]), dtype=np.float32)
    map_y = np.zeros((src.shape[2], src.shape[3]), dtype=np.float32)
    update_map(map_x, map_y, poly_vals)
    
    # apply unwarping
    output = np.zeros_like(src)
    for z in range(src.shape[0]):
        for c in range(src.shape[1]):
            src_im = src[z,c]
            dst = cv2.remap(src_im, map_x, map_y, cv2.INTER_LINEAR)
            output[z,c]=dst
        

    # save the output
    tif_imwrite(output_path, output, imagej=True,metadata={'axes': 'ZCYX'})

    for plane in [0,1,2]:
        save_file_name = Path(output_path).parent / f'plane{plane}' / output_path.name
        save_file_name.parent.mkdir(exist_ok=True)
        tif_imwrite(save_file_name,output[plane],imagej=True,metadata={'axes': 'CYX'})


def process_session_sbx(manifest: dict , session:dict):
    '''
    extract the mean of analomical hires green and red runs to tiff file in the correct pathways.
    manifest: json dict
    session: the session to process, extracted from the manifest 
    '''
    tile_to_num = {'left':'001', 'center':'002', 'right':'003'}
    
    base_2P = Path(manifest['base_path']) / manifest['mouse_name'] / '2P' 
    save_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / '2P' / 'tile' / 'warped' 
    mouse_name = manifest['mouse_name']
    date = session['date']

    for j, tile_loc in enumerate(['left','center','right']):

        green_run = session['anatomical_hires_green_runs'][j]
        green_sbx = base_2P / f'{mouse_name}_{date}_{green_run}' / f'{mouse_name}_{date}_{green_run}.sbx'

        red_run = session['anatomical_hires_red_runs'][j]
        red_sbx = base_2P / f'{mouse_name}_{date}_{red_run}' / f'{mouse_name}_{date}_{red_run}.sbx'
        

        green_stack = np.expand_dims(np.array(sbx_memmap(green_sbx))[:,:,0].mean(0),1)
        red_stack = np.expand_dims(np.array(sbx_memmap(red_sbx))[:,:,-1].mean(0),1)
        combined_stack = np.concatenate([green_stack,red_stack],1)

        # get output stack name from tile location
        stack_name_new = tile_to_num[tile_loc]
        
        save_path.mkdir(exist_ok=True, parents=True)
        tif_imwrite(save_path / f'stack_warped_C12_{stack_name_new}.tiff',combined_stack.astype(np.float32),
                    imagej=True,metadata={'axes': 'ZCYX'})
        for plane in range(combined_stack.shape[0]):
            save_path_plane = save_path / f'plane{plane}'
            save_path_plane.mkdir(exist_ok=True)
            tif_imwrite(save_path_plane/ f'stack_warped_C12_{stack_name_new}.tiff',combined_stack.astype(np.float32)[plane],
                    imagej=True,metadata={'axes': 'CYX'})



def unwarp_exp(unwarp_config,steps):
    ## unwarp all 2P files in an experiment
    for run_path in anatomical_lowres_green_paths:

        unwarp_tile(run_path, unwarp_config, steps, f_run['output'])