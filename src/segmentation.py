import sys
from collections import defaultdict
from pathlib import Path
import hjson
from cellpose import models
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy
import skimage
import pickle as pkl
from suite2p.io import binary
from IPython.display import HTML, display
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix
from tifffile import imread as tif_imread
from tifffile import imwrite as tif_imsave
from tqdm.auto import tqdm
from skimage.transform import rotate
from rich import print as rprint
from .registrations import verify_rounds


# CellposeModelWrapper class
# This class encapsulates the Cellpose model and its configuration.
# It's needed to:
# 1. Provide a consistent interface for Cellpose model initialization and evaluation
# 2. Allow lazy loading of the model, which can be computationally expensive
# 3. Centralize the configuration management for Cellpose parameters
class CellposeModelWrapper:
    def __init__(self, manifest):
        self.manifest = manifest
        self.model = None

    def eval(self, raw_image):
        if self.model is None:
            self.model = models.CellposeModel(
                pretrained_model=self.manifest['HCR_cellpose']['model_path'],
                gpu=self.manifest['HCR_cellpose']['gpu']
            )
        
        return self.model.eval(
            raw_image,
            channels=[0,0],
            diameter=self.manifest['HCR_cellpose']['diameter'],
            flow_threshold=self.manifest['HCR_cellpose']['flow_threshold'],
            cellprob_threshold=self.manifest['HCR_cellpose']['cellprob_threshold'],
            do_3D=True,
        )

def run_cellpose(manifest):
    round_to_rounds, reference_round, register_rounds = verify_rounds(manifest, parse_registered=True, 
                                                                      print_rounds=True, print_registered=True, func='cellpose')
    
    model_wrapper = CellposeModelWrapper(manifest)
    print(f"Running cellpose for {register_rounds}")
    for HCR_round_to_register in register_rounds + [reference_round['round']]:
        if HCR_round_to_register == reference_round['round']:
            round_folder_name = f"HCR{HCR_round_to_register}"
        else:
            round_folder_name = f"HCR{HCR_round_to_register}_to_HCR{reference_round['round']}"
        
        full_stack_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'HCR' / 'full_registered_stacks' / f"{round_folder_name}.tiff"
        output_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'HCR' / 'cellpose' / f"{round_folder_name}_masks.tiff"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            print(f"Cellpose segmentation already exists for {round_folder_name} - skipping")
            continue
        
        raw_image = tif_imread(full_stack_path)
        print(f"Running cellpose for {round_folder_name}, please wait")
        masks, _, _ = model_wrapper.eval(raw_image)
        
        tif_imsave(output_path, masks)
        print(f"Cellpose segmentation saved for {round_folder_name} - {output_path}")

def compute_M(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)),
                      shape=(data.max() + 1, data.size))

def get_indices_sparse(data):
    M = compute_M(data)
    inds = []
    for row in tqdm(M):
        inds.append(np.unravel_index(row.data, data.shape))
    return inds



def get_neuropil_mask_square(volume, radius, bound, inds):
    '''
    Get the neuropil mask square for each mask in the volume
    '''
    all_masks_locs = {}
    x_max = volume.shape[1]
    y_max = volume.shape[2]
    
    for i,plane in enumerate(tqdm(volume)):
        masks_expanded_b = skimage.segmentation.expand_labels(plane, distance=bound)
        all_masks_ids = set(list(zip(*np.where(plane>0))))

        print(" mask_expanded_b not used! fix, go to the notebooks and see how you fixed it!!"*3)
        for mask_id in np.unique(plane):
            if mask_id == 0:
                continue
            #mask_ids = np.where(plane==mask_id)
            #mx,my = mask_ids[0].mean().astype(int),mask_ids[1].mean().astype(int)
            
            mx,my = inds[mask_id][1][inds[mask_id][0]==i].mean().astype(int),inds[mask_id][2][inds[mask_id][0]==i].mean().astype(int)
            
            tm = np.meshgrid(range(max(mx-radius,0),min(mx+radius,x_max)),range(max(my-radius,0),min(my+radius,y_max)))
            
            mask_square =set(list(zip(tm[0].flatten(),tm[1].flatten())))

            mask_locs = list(zip(*list(mask_square - all_masks_ids)))
            if len(mask_locs):
                mask_locs = [np.ones(len(mask_locs[0]))*i,np.array(mask_locs[0]),np.array(mask_locs[1])]

                if mask_id not in all_masks_locs: all_masks_locs[mask_id] =[]
                all_masks_locs[mask_id].append(mask_locs)
    for mask_id in all_masks_locs:
        all_masks_locs[mask_id] = np.hstack(all_masks_locs[mask_id]).astype(int)
    return all_masks_locs



def extract_probs_intensities(manifest):
    round_to_rounds, reference_round, register_rounds = verify_rounds(manifest, parse_registered = True, 
                                                                      print_rounds = True, print_registered = True)
    
    HCR_fix_image_path = reference_round['image_path'] # The fix image that all other rounds will be registerd to (include all channels!)

    # neuropil parameters
    neuropil_radius = manifest['HCR_prob_intenisty_extraction']['neuropil_radius']
    neuropil_boundary = manifest['HCR_prob_intenisty_extraction']['neuropil_boundary']
    neuropil_pooling = manifest['HCR_prob_intenisty_extraction']['neuropil_pooling']
    
    for HCR_round_to_register in register_rounds + [reference_round['round']]:
        if HCR_round_to_register == reference_round['round']:
            round_folder_name = f"HCR{HCR_round_to_register}"
        else:
            round_folder_name = f"HCR{HCR_round_to_register}_to_HCR{reference_round['round']}"

        full_stack_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'HCR' / 'full_registered_stacks' / f"{round_folder_name}.tiff"
        full_stack_masks_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'HCR' / 'cellpose' / f"{round_folder_name}_masks.tiff"
        output_folder = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'HCR' / 'extract_intensities'
        output_folder.mkdir(parents=True, exist_ok=True)
        csv_output_path = output_folder / f"{round_folder_name}_probs_intensities.csv"
        pkl_output_path = output_folder / f"{round_folder_name}_probs_intensities.pkl"
        if pkl_output_path.exists():
            print(f"HCR - Intensities already extracted for {round_folder_name} - skipping")
            continue

        
        # Load images and verify sizes
        raw_image = tif_imread(full_stack_path)
        masks = tif_imread(full_stack_masks_path)
        assert raw_image[:,0,:,:].shape == masks.shape

        # acceleration step
        inds = get_indices_sparse(masks)
        neuropil_masks_inds = get_neuropil_mask_square(masks, neuropil_radius, neuropil_boundary, inds)

        number_of_channels = int(raw_image.shape[1])
        to_pnd = defaultdict(list)
        for mask_id, mask_inds in enumerate(tqdm(inds)):
            if mask_id == 0:
                continue
            if len(mask_inds[0]) == 0:
                raise Exception("Mask with zero pixels")
            Z, Y, X = mask_inds
            
            vals_per_mask_per_channel = raw_image[Z, :, Y, X]
            
            to_pnd['mask_id'].extend([mask_id]*number_of_channels)
            to_pnd['channel'].extend(list(range(number_of_channels)))
            
            to_pnd['mean'].extend(list(vals_per_mask_per_channel.mean(axis=0)))
            
            # add X,Y,Z 
            to_pnd['Z'].extend([Z.mean()]*number_of_channels)
            to_pnd['X'].extend([X.mean()]*number_of_channels)
            to_pnd['Y'].extend([Y.mean()]*number_of_channels)
            
            # Extract neuropil values
            neuropil_Z, neuropil_Y, neuropil_X = neuropil_masks_inds[mask_id]
            neuropil_vals_per_channel = raw_image[neuropil_Z, :, neuropil_Y, neuropil_X]

            # Use the neuropil_pooling list to extract the correct values
            for pooling_method in neuropil_pooling:
                if pooling_method == 'mean':
                    to_pnd[f'neuropil_mean_nr{neuropil_radius}_nb_{neuropil_boundary}'].extend(neuropil_vals_per_channel.mean(0))
                elif pooling_method == 'median':
                    to_pnd[f'neuropil_median_nr{neuropil_radius}_nb_{neuropil_boundary}'].extend(np.median(neuropil_vals_per_channel, axis=0))
                elif pooling_method.startswith('percentile-'):
                    percentile = int(pooling_method.split('-')[1])
                    to_pnd[f'neuropil_{percentile}pct_nr{neuropil_radius}_nb_{neuropil_boundary}'].extend(np.percentile(neuropil_vals_per_channel, percentile, axis=0))
                else:
                    raise ValueError(f"Unsupported pooling method: {pooling_method}")

        #Save to `masks_path` folder
        df = pd.DataFrame(to_pnd)
        df.attrs['raw_image_path'] = str(full_stack_path)
        df.attrs['masks_path'] = str(full_stack_masks_path)
        df.attrs['HCR_round_number'] = HCR_round_to_register
        df.to_csv(csv_output_path)
        df.to_pickle(pkl_output_path)
        print(f"Intensities extracted and saved for {round_folder_name} - {pkl_output_path}_probs_intensities.csv")
        



def extract_electrophysiology_intensities(manifest: dict , session: dict):
    #Edited dictionary version

    mouse_name = manifest['mouse_name']
    date = session['date']
    suite2p_run = session['functional_run'][0]

    suite2p_path = Path(manifest['base_path']) / manifest['mouse_name'] / '2P' /  f'{mouse_name}_{date}_{suite2p_run}' / 'suite2p'
    save_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / '2P' / 'suite2p'
    cellpose_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / '2P' / 'cellpose'
    save_path.mkdir(exist_ok=True, parents=True)
    functional_plane = session['functional_plane'][0]

    ops = np.load(suite2p_path / 'plane0/ops.npy',allow_pickle=True).item()

    planes = ops['nplanes']
    # Assuming `planes`, `suite2p_path`, `savepath`, `mouse`, `run`, and `ops` are defined
    for plane in [functional_plane]:
        pkl_save_path = save_path / f'lowres_meanImg_C0_plane{plane}.pkl'
        if pkl_save_path.exists():
            print(f"2p activity already extracted for plane {plane} - skipping")
            continue

        # Set up binary file
        bin_file = binary.BinaryFile(filename=suite2p_path / f'plane{plane}' / 'data.bin', Lx=ops['Lx'], Ly=ops['Ly'])
        # Move data to a numpy array
        all_data = bin_file.data
        # Load masks
        stats = np.load(cellpose_path / f'lowres_meanImg_C0_plane{plane}_seg.npy', allow_pickle=True).item()
        masks_locs = get_indices_sparse(stats['masks'])  # Get (x, y) indices per mask
    
        # Process each mask to get mean values
        mean_frames = []
        for mask_loc in tqdm(masks_locs[1:]):
            mean_frames.append(all_data[:, mask_loc[0], mask_loc[1]].mean(axis=1))
        
        mean_frames = np.array(mean_frames)
        # Save masks_locs as a dictionary
        masks_locs_dict = {f'cell_{i}': masks_locs[i] for i in range(len(masks_locs))}

        # Save data to .mat files
        sio.savemat(save_path / f'lowres_meanImg_C0_plane{plane}_locs.mat', {'masks_locs': masks_locs_dict})
        sio.savemat(save_path / f'lowres_meanImg_C0_plane{plane}_traces.mat', {'mean_frames': mean_frames})

        # Save data to .pkl files
        pkl.dump({'masks_locs': masks_locs_dict,
                  'mean_frames': mean_frames}, 
                  open(pkl_save_path, 'wb'))



def match_masks(stack1_masks_path: np.ndarray, stack2_masks_path: np.ndarray) -> dict:
    '''
    Input should be 2 stacks of masks where stack1 is registered to stack2.
    '''

    stack1_masks = tif_imread(stack1_masks_path)
    stack2_masks= tif_imread(stack2_masks_path)
    
    stack1_masks_inds = get_indices_sparse(stack1_masks.astype(np.uint16))

    #collect mask to mask values with overlap and put in pandas dataframe

    to_pandas = {'mask1':[], 'mask2':[], 'overlap':[]}
    for mask1_inds in stack1_masks_inds[1:]: #skip the first one because it is background
        if len(mask1_inds[0]) == 0: #skip if there are no pixels in the mask
            continue

        mask1 = stack1_masks[mask1_inds]
        assert len(set(mask1)) == 1, 'mask1_inds should only have 1 value, it means that you have floats instead of ints for masks'

        mask2_at_mask1 = stack2_masks[mask1_inds]

        # get the mode of the mask2_at_mask1
        most_overlapped_mask_in_s1 = scipy.stats.mode(mask2_at_mask1[mask2_at_mask1>0],keepdims=False).mode

        # How many pixel in the overlap have the same value out of all overlap pixels 
        overlap = sum(mask2_at_mask1 == most_overlapped_mask_in_s1)/len(mask2_at_mask1)
        to_pandas['mask1'].append(mask1[0])
        to_pandas['mask2'].append(most_overlapped_mask_in_s1)
        to_pandas['overlap'].append(overlap)

    df = pd.DataFrame(to_pandas)
    # keep the mask with the higher overlap
    df_removed_dups = df.sort_values('overlap', ascending=False).drop_duplicates('mask2', keep='first')
    
    ##3<-- Add a print on the number of masks that where removed

    
    return df_removed_dups

## All dimensions must match, all must be 1 channel, all mask files must have 1 value per mask
## MAKE SURE TO ALWAYS LEAVE ONE CONSTANT AS HCR 1, DON'T SWITCH THE IDENTITIES
#Output dataframes (dfs) saved in the HCR round folder in NASQUATCH for now

def align_masks(manifest: dict, session: dict):

    print("### HCR masks alignment - start")
    round_to_rounds, reference_round, register_rounds = verify_rounds(manifest, parse_registered = True, 
                                                                    print_rounds = False, print_registered = False)
    
    reference_round_tiff = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'HCR' / 'full_registered_stacks' / f"HCR{reference_round['round']}.tiff"
    reference_round_masks = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'HCR' / 'cellpose' / f"HCR{reference_round['round']}_masks.tiff"

    output_folder = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'MERGED' / 'aligned_masks'
    output_folder.mkdir(parents=True, exist_ok=True)
    
    for HCR_round_to_register in register_rounds:
        round_folder_name = f"HCR{HCR_round_to_register}_to_HCR{reference_round['round']}"

        mov_stack_masks = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'HCR' / 'cellpose' / f"{round_folder_name}_masks.tiff"

        save_path = output_folder / f"{round_folder_name}.csv"
        if save_path.exists():
            print(f"Mask alignment already exists for {round_folder_name} - skipping")
            continue
        # calculate the matching masks and the overlap
        mask1_to_mask2_df = match_masks(mov_stack_masks, reference_round_masks)
        mask1_to_mask2_df.to_csv(save_path)

    print("### HCR masks alignment - finish")

    # 2p masks alignment

    # Rotate the masks


    plane = session['functional_plane'][0]
    stitched_file_C01  = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / '2P' / 'tile' / 'stitched' / f'stack_stitched_C01_plane{plane}.tiff' 
    rotation_file = stitched_file_C01.parent / 'rotation.txt'
    rotation_config = hjson.load(open(rotation_file,'r'))

    cellpose_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / '2P' / 'cellpose'
    stats = np.load(cellpose_path / f'lowres_meanImg_C0_plane{plane}_seg.npy', allow_pickle=True).item()
    
    from IPython import embed; embed()
    stats = np.load(cellpose_path / f'lowres_meanImg_C0_plane{plane}_seg.npy', allow_pickle=True).item()

    masks_2p = np.array(stats['masks'])  # Get (x, y) indices per mask
    # Check the shape and type of masks_2p before rotation
    print(f"'masks_2p' type: {type(masks_2p)}, shape: {masks_2p.shape}")
    
    # Check the shape of individual masks
    print(f"'masks_2p[0]' shape: {masks_2p[0].shape}")
    print(f"'masks_2p[1]' shape: {masks_2p[1].shape}")

    for k in rotation_config:
        if k == 'rotation' and rotation_config[k]:
            masks_2p = rotate(masks_2p, rotation_config['rotation'])
        if k == 'fliplr' and rotation_config[k]:
            masks_2p = masks_2p[:,::-1]
        if k == 'flipud' and rotation_config[k]:
            masks_2p = masks_2p[::-1,:]


    masks_2p_rotated = masks_2p
    masks_2p_rotated_path = cellpose_path / f'lowres_meanImg_C0_plane{plane}_seg_rotated_11.tiff'
    tif_imsave(masks_2p_rotated_path,  masks_2p_rotated)

    masks_2p_rotated_to_HCR1 = cellpose_path / f'lowres_meanImg_C0_plane{plane}_seg_rotated_to_HCR.tiff'
    # saved rotated masks_2p
    while not masks_2p_rotated_to_HCR1.exists():
        output_string = f'''
        Please bigwarp {masks_2p_rotated_path} to HCR1, this requires two bigwarp steps,
        step 1 - ???
        setp 2 - ???
        once you are done save the file here {masks_2p_rotated_to_HCR1}
        '''
        rprint(output_string)
        input()

    save_path = output_folder / f"twop_to_HCR{reference_round['round']}.csv"

    mask1_to_mask2_df = match_masks(masks_2p_rotated_to_HCR1, reference_round_masks)
    mask1_to_mask2_df.to_csv(save_path)







        # TODO: Add visualization code here
        # # filter only overlaps above min_overlap
        # mask1_to_mask2 = mask1_to_mask2_df.query(f'overlap>@min_overlap')[['mask1','mask2']].astype('int').set_index('mask1')['mask2'].to_dict()

        # # creating visualization files
        # visualize_match(stack1_image_path, stack1_masks_path, stack2_image_path, stack2_masks_path,
        #                 mask1_to_mask2, output_base_filename)

    
    

