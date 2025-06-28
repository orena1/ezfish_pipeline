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
from skimage import filters
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
from scipy.spatial import Delaunay, ConvexHull
from scipy.interpolate import LinearNDInterpolator
from .registrations import verify_rounds
from .functional import get_number_of_suite2p_planes



# CellposeModelWrapper class
# This class encapsulates the Cellpose model and its configuration.
# It's needed to:
# 1. Provide a consistent interface for Cellpose model initialization and evaluation
# 2. Allow lazy loading of the model, which can be computationally expensive
# 3. Centralize the configuration management for Cellpose parameters
class CellposeModelWrapper:
    def __init__(self, params):
        self.params = params
        self.model = None

    def eval(self, raw_image):
        if self.model is None:
            self.model = models.CellposeModel(
                pretrained_model=self.params['HCR_cellpose']['model_path'],
                gpu=self.params['HCR_cellpose']['gpu']
            )
        
        return self.model.eval(
            raw_image,
            channels=[0,0],
            diameter=self.params['HCR_cellpose']['diameter'],
            flow_threshold=self.params['HCR_cellpose']['flow_threshold'],
            cellprob_threshold=self.params['HCR_cellpose']['cellprob_threshold'],
            do_3D=True,
        )

def run_cellpose(full_manifest):
    manifest = full_manifest['data']
    params = full_manifest['params']
    rprint("\n" + "="*80)
    rprint("[bold green] Cellpose on Rounds [/bold green]")
    rprint("="*80)

    round_to_rounds, reference_round, register_rounds = verify_rounds(full_manifest, parse_registered=True, 
                                                                      print_rounds=True, print_registered=True, func='cellpose')
    
    model_wrapper = CellposeModelWrapper(params)
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
    
    rprint("\n" + "="*80)
    rprint("[bold green]✨ Cellpose on Rounds COMPLETE[/bold green]")
    rprint("="*80 + "\n")

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



def get_neuropil_mask_square_old(volume, radius, bound, inds):
    '''
    Get the neuropil mask square for each mask in the volume
    '''
    all_masks_locs = {}
    x_max = volume.shape[1]
    y_max = volume.shape[2]
    
    for i,plane in enumerate(tqdm(volume)):
        masks_expanded_b = skimage.segmentation.expand_labels(plane, distance=bound)
        all_masks_ids = set(list(zip(*np.where(plane>0))))
        if i<3:
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


def get_neuropil_mask_square(volume, radius, bound, inds):
    '''
    Get the neuropil mask square for each mask in the volume using vectorized operations
    '''
    all_masks_locs = {}
    x_max, y_max = volume.shape[1], volume.shape[2]
    
    # Pre-compute unique mask IDs excluding background (0)
    unique_masks = np.unique(volume)
    unique_masks = unique_masks[unique_masks != 0]
    
    # Create meshgrid once
    x_grid, y_grid = np.meshgrid(np.arange(x_max), np.arange(y_max), indexing='ij')
    
    for mask_id in tqdm(unique_masks):
        mask_coords = []
        
        # Get all z-planes where this mask appears
        z_planes = np.unique(inds[mask_id][0])
        if mask_id<3:
            print(" mask_expanded_b not used! fix, go to the notebooks and see how you fixed it!!"*3)
        for z in z_planes:
            # Get mask center for this z-plane
            mask_points = (inds[mask_id][0] == z)
            if not np.any(mask_points):
                continue
                
            mx = int(inds[mask_id][1][mask_points].mean())
            my = int(inds[mask_id][2][mask_points].mean())
            
            # Create square mask using broadcasting
            x_min, x_max_local = max(mx-radius, 0), min(mx+radius, x_max)
            y_min, y_max_local = max(my-radius, 0), min(my+radius, y_max)
            
            # Get all points in square that aren't part of any cell
            square_mask = (x_grid[x_min:x_max_local, y_min:y_max_local] >= x_min) & \
                         (x_grid[x_min:x_max_local, y_min:y_max_local] < x_max_local) & \
                         (y_grid[x_min:x_max_local, y_min:y_max_local] >= y_min) & \
                         (y_grid[x_min:x_max_local, y_min:y_max_local] < y_max_local)
            
            neuropil_points = (volume[z, x_min:x_max_local, y_min:y_max_local] == 0) & square_mask
            
            if np.any(neuropil_points):
                x_coords, y_coords = np.where(neuropil_points)
                z_coords = np.full_like(x_coords, z)
                mask_coords.append(np.stack([z_coords, x_coords + x_min, y_coords + y_min]))
        
        if mask_coords:
            all_masks_locs[mask_id] = np.hstack(mask_coords).astype(np.int32)
    
    return all_masks_locs


def extract_probs_intensities(full_manifest):
    rprint("\n" + "="*80)
    rprint("[bold green] Extract Rounds Intensities[/bold green]")
    rprint("="*80)

    manifest = full_manifest['data']
    params = full_manifest['params']
    round_to_rounds, reference_round, register_rounds = verify_rounds(full_manifest, parse_registered = True, 
                                                                      print_rounds = True, print_registered = True, func='intensities-extraction')
    
    HCR_fix_image_path = reference_round['image_path'] # The fix image that all other rounds will be registerd to (include all channels!)

    # neuropil parameters
    neuropil_radius = params['HCR_probe_intensity_extraction']['neuropil_radius']
    neuropil_boundary = params['HCR_probe_intensity_extraction']['neuropil_boundary']
    neuropil_pooling = params['HCR_probe_intensity_extraction']['neuropil_pooling']

    median_filter = None
    # Create custom 3x5x5 structuring element
    if params['HCR_probe_intensity_extraction'].get('stack_median_filter'):
        median_filter = params['HCR_probe_intensity_extraction'].get('stack_median_filter')
        assert isinstance(median_filter, (list, tuple)) and len(median_filter) == 3
        median_filter = np.ones((median_filter[0], 1, median_filter[1], median_filter[2]))
        median_filter_label = f'medflt_{median_filter[0]}x{median_filter[1]}x{median_filter[2]}'

    for HCR_round_to_register in register_rounds + [reference_round['round']]:
        if HCR_round_to_register == reference_round['round']:
            round_folder_name = f"HCR{HCR_round_to_register}"
            channels_names =  reference_round['channels']
        else:
            round_folder_name = f"HCR{HCR_round_to_register}_to_HCR{reference_round['round']}"
            channels_names = round_to_rounds[HCR_round_to_register]['channels']

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
        if median_filter is not None:
            print(f"Applying median filter with shape {median_filter.shape} to raw image, might take some time...")
            med_filter_stack = filters.median(raw_image, median_filter)
        # acceleration step
        inds = get_indices_sparse(masks)
        neuropil_masks_inds = get_neuropil_mask_square(masks, neuropil_radius, neuropil_boundary, inds) #Confirm whether the neuropil_boundary is being used currently JSA

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
            to_pnd['channel_name'].extend(channels_names)
            to_pnd['mean'].extend(list(vals_per_mask_per_channel.mean(axis=0)))

            # add X,Y,Z 
            to_pnd['Z'].extend([Z.mean()]*number_of_channels)
            to_pnd['X'].extend([X.mean()]*number_of_channels)
            to_pnd['Y'].extend([Y.mean()]*number_of_channels)
            
            # Extract neuropil values
            neuropil_Z, neuropil_Y, neuropil_X = neuropil_masks_inds[mask_id]
            neuropil_vals_per_channel = raw_image[neuropil_Z, :, neuropil_Y, neuropil_X]

            if median_filter is not None:
                to_pnd['mean_' + median_filter_label].extend(list(med_filter_stack[Z, :, Y, X].mean(axis=0)))
                neuropil_vals_per_channel_median = med_filter_stack[neuropil_Z, :, neuropil_Y, neuropil_X]
            

            # UGLY CODE, create function.
            # Use the neuropil_pooling list to extract the correct values
            for pooling_method in neuropil_pooling:
                if pooling_method == 'mean':
                    to_pnd[f'neuropil_mean_nr{neuropil_radius}_nb_{neuropil_boundary}'].extend(neuropil_vals_per_channel.mean(0))
                    if median_filter is not None:
                        to_pnd[f'{median_filter_label}_neuropil_mean_nr{neuropil_radius}_nb_{neuropil_boundary}'].extend(neuropil_vals_per_channel_median.mean(0))
                
                elif pooling_method == 'median':
                    to_pnd[f'neuropil_median_nr{neuropil_radius}_nb_{neuropil_boundary}'].extend(np.median(neuropil_vals_per_channel, axis=0))
                    if median_filter is not None:
                        to_pnd[f'{median_filter_label}_neuropil_median_nr{neuropil_radius}_nb_{neuropil_boundary}'].extend(np.median(neuropil_vals_per_channel_median, axis=0))
                
                elif pooling_method.startswith('percentile-'):
                    percentile = int(pooling_method.split('-')[1])
                    to_pnd[f'neuropil_{percentile}pct_nr{neuropil_radius}_nb_{neuropil_boundary}'].extend(np.percentile(neuropil_vals_per_channel, percentile, axis=0))
                    if median_filter is not None:
                        to_pnd[f'{median_filter_label}_neuropil_{percentile}pct_nr{neuropil_radius}_nb_{neuropil_boundary}'].extend(np.percentile(neuropil_vals_per_channel_median, percentile, axis=0))

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
    rprint("\n" + "="*80)
    rprint("[bold green]✨ Extract Rounds Intensities COMPLETE[/bold green]")
    rprint("="*80 + "\n")



def extract_electrophysiology_intensities(full_manifest: dict , session: dict):
    #Edited dictionary version
    rprint("\n" + "="*80)
    rprint("[bold green] Extract 2P Intensities[/bold green]")
    rprint("="*80)

    manifest = full_manifest['data'] 
    mouse_name = manifest['mouse_name']
    date = session['date']
    suite2p_run = session['functional_run'][0]

    suite2p_path = Path(manifest['base_path']) / manifest['mouse_name'] / '2P' /  f'{mouse_name}_{date}_{suite2p_run}' / 'suite2p'
    save_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / '2P' / 'suite2p'
    cellpose_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / '2P' / 'cellpose'
    save_path.mkdir(exist_ok=True, parents=True)
    functional_plane = session['functional_plane'][0]

    planes = get_number_of_suite2p_planes(suite2p_path)
    # Assuming `planes`, `suite2p_path`, `savepath`, `mouse`, `run`, and `ops` are defined
    for plane in [functional_plane]:
        pkl_save_path = save_path / f'lowres_meanImg_C0_plane{plane}.pkl'
        if pkl_save_path.exists():
            print(f"2p activity already extracted for plane {plane} - skipping")
            continue
        ops = np.load(suite2p_path / f'plane0/ops.npy', allow_pickle=True).item()
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

    rprint("\n" + "="*80)
    rprint("[bold green] Extract 2P Intensities COMPLETE[/bold green]")
    rprint("="*80 + "\n")

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

def convex_mask(landmarks_path: str, stack_path: str, Ydist: int, full_manifest: dict):
    '''
    Use landmarks to create two boundary surfaces and mask out everything outside them.

    Args:
        landmarks_path: Path to CSV file containing landmarks used for High res -> HCR Round 1 registration
        stack_path: Path to masks that have been fully bigwarped (2x) to align with HCR Round 1
        Ydist: Distance in microns beyond which everything will be masked out
        
    Returns:
        numpy.ndarray: Masked image stack with regions outside boundary surfaces set to 0
    '''

    # Load landmark coordinates (X,Y,Z in microns) from CSV
    df = pd.read_csv(landmarks_path, header=None)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    x_values = df[5]  # X coordinates in microns
    y_values = df[6]  # Y coordinates in microns 
    z_values = df[7]  # Z coordinates in microns
    points = np.column_stack((x_values, y_values, z_values))

    # Create upper and lower boundary surfaces by offsetting landmark points
    top_points = points.copy()
    top_points[:, 2] += Ydist  # Shift up by Ydist microns
    bottom_points = points.copy()
    bottom_points[:, 2] -= Ydist  # Shift down by Ydist microns

    # Image resolution factors to convert microns to voxels
    resolution = full_manifest['data']['HCR_confocal_imaging']['rounds'][0]['resolution']

    # Convert point coordinates from microns to voxels
    top_points[:, 0] /= resolution[0]  # Scale X 
    top_points[:, 1] /= resolution[1]  # Scale Y
    top_points[:, 2] /= resolution[2]  # Scale Z
    bottom_points[:, 0] /= resolution[0]
    bottom_points[:, 1] /= resolution[1]
    bottom_points[:, 2] /= resolution[2]

    tiff_stack = tif_imread(stack_path)
    # Handle both single channel (3D) and multichannel (4D) images
    if tiff_stack.ndim < 4:
        tiff_stack_first_channel = tiff_stack
    elif tiff_stack.ndim == 4:
        tiff_stack_first_channel = tiff_stack[:, 0, :, :]
    else:
        raise ValueError(f"Unsupported number of dimensions: {tiff_stack.ndim}")

    z_slices, height, width = tiff_stack_first_channel.shape

    def extrapolate_surface_to_image_edges(points, height, width):
        """
        Extrapolate Z values across full image using Delaunay triangulation.
        
        Args:
            points: Landmark points
            height: Image height in pixels
            width: Image width in pixels
            
        Returns:
            numpy.ndarray: Extrapolated Z values for each X,Y position
        """
        print("Extrapolating surface Z-values to image edges...")
        
        X, Y = np.meshgrid(np.arange(width), np.arange(height))
        xy_grid = np.column_stack([X.ravel(), Y.ravel()])

        interpolator = LinearNDInterpolator(points[:, :2], points[:, 2])
        z_values = interpolator(xy_grid).reshape(height, width)

        # Fill NaN values using nearest convex hull points
        nan_mask = np.isnan(z_values)
        if np.any(nan_mask):
            convex_hull = ConvexHull(points[:, :2])
            hull_points = points[convex_hull.vertices]

            for i, j in zip(*np.where(nan_mask)):
                x, y = X[i, j], Y[i, j]
                nearest_point = hull_points[np.argmin(np.linalg.norm(hull_points[:, :2] - np.array([x, y]), axis=1))]
                z_values[i, j] = nearest_point[2]

        print("Extrapolation complete.")
        return z_values
    
    # Generate boundary surfaces
    top_z_values = extrapolate_surface_to_image_edges(top_points, height, width)
    bottom_z_values = extrapolate_surface_to_image_edges(bottom_points, height, width)

    def blackout_above_and_below(tiff_stack, top_z_values, bottom_z_values):
        """
        Mask out regions above top surface and below bottom surface.

        Args:
            tiff_stack: Input image stack
            top_z_values: Z coordinates of upper boundary surface  
            bottom_z_values: Z coordinates of lower boundary surface

        Returns:
            numpy.ndarray: Masked image stack
        """
        volume = np.copy(tiff_stack)

        total_slices = tiff_stack.shape[0]
        for z in range(total_slices):
            if z % 5 == 0:
                progress = (z / total_slices) * 100
                print(f"Processing slice {z + 1}/{total_slices} - {progress:.2f}% complete")

            mask = (z > top_z_values) | (z < bottom_z_values)
            volume[z, mask] = 0

        print("Masking complete.")
        return volume

    # Apply masking to image stack
    blacked_out_stack_first_channel = blackout_above_and_below(tiff_stack_first_channel, top_z_values, bottom_z_values)
    return blacked_out_stack_first_channel


def align_masks(full_manifest: dict, session: dict, only_hcr: bool = False):

    rprint("\n" + "="*80)
    rprint("[bold green] Align Rounds Masks[bold green]")
    rprint("="*80)

    manifest = full_manifest['data']
    params = full_manifest['params']
    round_to_rounds, reference_round, register_rounds = verify_rounds(full_manifest, parse_registered = True, 
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
    
    if only_hcr:
        print("Skipping 2p masks alignment")
        return
    rprint("\n" + "="*80)
    rprint("[bold green] HCR Rounds Registrations COMPLETE[/bold green]")
    rprint("="*80 + "\n")

    rprint("\n" + "="*80)
    rprint("[bold green] Align 2P Masks[/bold green]")
    rprint("="*80)

    plane = session['functional_plane'][0]
    rotation_config = params['rotation_2p_to_HCRspec']

    cellpose_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / '2P' / 'cellpose'
    twop_cellpose = cellpose_path / f'lowres_meanImg_C0_plane{plane}_seg.npy'
    while not twop_cellpose.exists():
        tiff_path = cellpose_path / f'lowres_meanImg_C0_plane{plane}.tiff'
        output_string = f'''
        Please run cellpose on plane {tiff_path}
        once you are done save the file in the cellpose directory as {twop_cellpose}
        '''
        rprint(output_string)
        input()
    stats = np.load(twop_cellpose, allow_pickle=True).item()
    masks_2p = np.array(stats['masks'])  # Get (x, y) indices per mask
    save_path = output_folder / f"twop_plane{plane}_to_HCR{reference_round['round']}.csv"
    if save_path.exists():
        print(f"2p masks alignment already exists for plane {plane} - skipping")
        rprint("\n" + "="*80)
        rprint("[bold green] Align 2P Masks COMPLETE[/bold green]")
        rprint("="*80 + "\n")
        
        return
    # Check the shape and type of masks_2p before rotation
    print(f"'masks_2p' type: {type(masks_2p)}, shape: {masks_2p.shape}")

    for k in rotation_config:
        if k == 'rotation' and rotation_config[k]:
            masks_2p = rotate(masks_2p, rotation_config['rotation'], preserve_range=True, resize=True)
        if k == 'fliplr' and rotation_config[k]:
            masks_2p = masks_2p[:,::-1]
        if k == 'flipud' and rotation_config[k]:
            masks_2p = masks_2p[::-1,:]


    masks_2p_rotated = masks_2p
    masks_2p_rotated_path = cellpose_path / f'lowres_meanImg_C0_plane{plane}_seg_rotated.tiff'
    tif_imsave(masks_2p_rotated_path,  masks_2p_rotated.astype(np.uint16))

    reg_save_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / '2P' / 'registered'
    masks_2p_rotated_to_HCR1 = reg_save_path / f'lowres_meanImg_C0_plane{plane}_seg_rotated_bigwarped_to_HCR1.tiff'
    masks_2p_rotated_to_HCR1_blacked_save_path = reg_save_path / f'lowres_meanImg_C0_plane{plane}_seg_rotated_bigwarped_to_HCR1_blacked.tiff'
    bigwarp_landmarks_path =  reg_save_path /  f'stitched_C01_plane{plane}_rotated_TO_HCR1_landmarks.csv'

    # saved rotated masks_2p
    while not masks_2p_rotated_to_HCR1.exists() or not bigwarp_landmarks_path.exists():
        output_string = f'''
        Please apply bigwarp on masks in {masks_2p_rotated_path}, two steps required
        step 1 - low res to high res transform
        setp 2 - high res to HCR Round 1 transform
        once you are done save the file in the registered directory as {masks_2p_rotated_to_HCR1}
        and also save step 2 landmarks in {bigwarp_landmarks_path}
        '''
        rprint(output_string)
        input()
    
    masks_2p_rotated_to_HCR1_blacked = convex_mask(bigwarp_landmarks_path, masks_2p_rotated_to_HCR1, params['2p_to_HCR_params']['convex_masking_distance'], full_manifest)
    tif_imsave(masks_2p_rotated_to_HCR1_blacked_save_path, masks_2p_rotated_to_HCR1_blacked)

    mask1_to_mask2_df = match_masks(masks_2p_rotated_to_HCR1, reference_round_masks)
    mask1_to_mask2_df.to_csv(save_path)
    rprint("\n" + "="*80)
    rprint("[bold green] Align 2P Masks COMPLETE[/bold green]")
    rprint("="*80 + "\n")






    # TODO: Add visualization code here
    # # filter only overlaps above min_overlap
    # mask1_to_mask2 = mask1_to_mask2_df.query(f'overlap>@min_overlap')[['mask1','mask2']].astype('int').set_index('mask1')['mask2'].to_dict()

    # # creating visualization files
    # visualize_match(stack1_image_path, stack1_masks_path, stack2_image_path, stack2_masks_path,
    #                 mask1_to_mask2, output_base_filename)

def merge_masks(full_manifest: dict, session: dict, only_hcr: bool = False):
    rprint("\n" + "="*80)
    rprint("[bold green] Match Aligned Masks[bold green]")
    rprint("="*80)

    manifest = full_manifest['data']
    params = full_manifest['params']
    
    plane = session['functional_plane'][0]

    round_to_rounds, reference_round, register_rounds = verify_rounds(full_manifest, parse_registered = True, 
                                                                    print_rounds = False, print_registered = False)
    HCR_intensities_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'HCR' / 'extract_intensities'
    HCR_mapping_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'MERGED' / 'aligned_masks'
    merged_table_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'MERGED' / 'aligned_extracted_features'
    merged_table_path.mkdir(parents=True, exist_ok=True)
    
    
    # get available features to create merged table for
    neuropil_radius = params['HCR_probe_intensity_extraction']['neuropil_radius']
    neuropil_boundary = params['HCR_probe_intensity_extraction']['neuropil_boundary']
    neuropil_pooling = params['HCR_probe_intensity_extraction']['neuropil_pooling']
    # Use the neuropil_pooling list to extract the correct values
    features_to_extract = ['mean']
    for pooling_method in neuropil_pooling:
        if pooling_method == 'mean':
            features_to_extract.append(f'neuropil_mean_nr{neuropil_radius}_nb_{neuropil_boundary}')
        elif pooling_method == 'median':
            features_to_extract.append(f'neuropil_median_nr{neuropil_radius}_nb_{neuropil_boundary}')
        elif pooling_method.startswith('percentile-'):
            percentile = int(pooling_method.split('-')[1])
            features_to_extract.append(f'neuropil_{percentile}pct_nr{neuropil_radius}_nb_{neuropil_boundary}')
        else:
            raise ValueError(f"Unsupported pooling method: {pooling_method}")
    
    
    for feature in features_to_extract:
        merged_table_file_path = merged_table_path / f'full_table_{feature}_twop_plane{plane}.pkl'
        if merged_table_file_path.exists():
            print(f"Feature extraction already merged for {feature} - skipping")
            continue
        print(f"Extracting feature: {feature}")
        if only_hcr:
            twoP_mapping_dict = {0:0}
        else:
            # load 2p mapping
            towP_to_reference_mapping = pd.read_csv(HCR_mapping_path / f"twop_plane{plane}_to_HCR{reference_round['round']}.csv")
            twoP_mapping_dict = {mask_2:mask_1 for mask_1,mask_2 in towP_to_reference_mapping[['mask1','mask2']].values}

        # load reference round intensities
        reference_round_intensities = pd.read_pickle(HCR_intensities_path / f"HCR{reference_round['round']}_probs_intensities.pkl")
        reference_round_intensities_pivot = pd.pivot(reference_round_intensities, index='mask_id', columns=['channel_name'], values=[feature]).reset_index()
        reference_round_intensities_pivot.rename(columns={'mask_id':'mask_id_main'},inplace=True)
        # load HCR rounds intensities and matching files
        HCR_rounds_mapping_tables = []
        HCR_round_mapping_dict = []
        HCR_rounds_intensities_pivot = []
        HCR_rounds_names = register_rounds
        for HCR_round_to_register in register_rounds:
            # load mapping data
            round_file_name = f"HCR{HCR_round_to_register}_to_HCR{reference_round['round']}"
            round_mapping_file_path = HCR_mapping_path / f"{round_file_name}.csv"
            HCR_rounds_mapping_tables.append(pd.read_csv(round_mapping_file_path))
            HCR_round_mapping_dict.append({mask_2:mask_1 for mask_1,mask_2 in HCR_rounds_mapping_tables[-1][['mask1','mask2']].values})

            # load intensities
            HCR_round_intensities = pd.read_pickle(HCR_intensities_path / f"{round_file_name}_probs_intensities.pkl")
            HCR_rounds_intensities_pivot.append(pd.pivot(HCR_round_intensities, index='mask_id', columns=['channel_name'], values=[feature]).reset_index())


        ####       ####
        ###  MATCH  ###
        ####       ####
        # first let's match the 2p
        mask_tp_matched = []
        for i in reference_round_intensities_pivot.mask_id_main:
            if i in twoP_mapping_dict:
                mask_tp_matched.append(twoP_mapping_dict[i])
            else:
                mask_tp_matched.append(None)    
        reference_round_intensities_pivot['twoP_mask']  = mask_tp_matched


        for j in range(len(HCR_round_mapping_dict)):
            HCR_main_2_HCR_round = HCR_round_mapping_dict[j]
            mask_matched = []
            for i in reference_round_intensities_pivot.mask_id_main:
                if i in HCR_main_2_HCR_round:
                    mask_matched.append(HCR_main_2_HCR_round[i])
                else:
                    mask_matched.append(None)
            reference_round_intensities_pivot[f'mask_round_{HCR_rounds_names[j]}'] = mask_matched


        ####       ####
        ###  MERGE  ###
        ####       ####
        HCR_main_pivot_merged = reference_round_intensities_pivot.copy().reset_index()

        for j in range(len(HCR_round_mapping_dict)):
            round_mask_name = f'mask_round_{HCR_rounds_names[j]}'
            print(round_mask_name)
            HCR_main_pivot_merged = pd.merge(HCR_main_pivot_merged, 
                                        HCR_rounds_intensities_pivot[j],
                                        left_on=round_mask_name,
                                        right_on='mask_id',
                                        suffixes=['',f'_round_{HCR_rounds_names[j]}'],
                                        how='left').drop(columns=['mask_id'])#,how='outer')
        pd.set_option('display.max_columns', None)
        print(f'final table saved to {merged_table_file_path}')
        HCR_main_pivot_merged.to_pickle(merged_table_file_path)

    rprint("\n" + "="*80)
    rprint("[bold green] Match Aligned Masks COMPLETE[/bold green]")
    rprint("="*80 + "\n")
