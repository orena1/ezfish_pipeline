import sys
from collections import defaultdict
from pathlib import Path

from cellpose import models
import numpy as np
import pandas as pd
import scipy.io as sio
import skimage
import pickle as pkl
from suite2p.io import binary
from IPython.display import HTML, display
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix
from tifffile import imread as tif_imread
from tifffile import imwrite as tif_imsave
from tqdm.auto import tqdm

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
                                                                      print_rounds=True, print_registered=True)
    
    model_wrapper = CellposeModelWrapper(manifest)
    print(f"Running cellpose for {register_rounds}")
    for HCR_round_to_register in register_rounds:
        round_folder_name = f"HCR{HCR_round_to_register}_to_HCR{reference_round['round']}"
        
        full_stack_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'HCR' / 'full_registered_stacks' / f"{round_folder_name}.tiff"
        output_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'HCR' / 'cellpose' / f"{round_folder_name}_masks.tiff"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            print(f"Cellpose segmentation already exists for {round_folder_name} - skipping")
            continue
        
        raw_image = tif_imread(full_stack_path)
        
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

        raise Exception(" mask_expanded_b not used! fix, go to the notebooks and see how you fixed it!")
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
    neuropil_pooling = manifest['HCR_prob_intenisty_extraction']['neuroupil_pooling']
    
    for HCR_round_to_register in register_rounds:

        round_folder_name = f"HCR{HCR_round_to_register}_to_HCR{reference_round['round']}"
        
        full_stack_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'HCR' / 'full_registered_stacks' / f"{round_folder_name}.tiff"
        full_stack_masks_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'HCR' / 'cellpose' / f"{round_folder_name}_masks.tiff"
        output_folder = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'HCR' / 'extract_intensities'
        output_folder.mkdir(parents=True, exist_ok=True)

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
        df.to_csv(output_folder / f"{round_folder_name}_probs_intensities.csv")
        df.to_pickle(output_folder / f"{round_folder_name}_probs_intensities.pkl")
        print(f"Intensities extracted and saved for {round_folder_name} - {output_folder}/{round_folder_name}_probs_intensities.csv")
        



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
    for plane in range(planes):
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
                  open(save_path / f'lowres_meanImg_C0_plane{plane}.pkl', 'wb'))
