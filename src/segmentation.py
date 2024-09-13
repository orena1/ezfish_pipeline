import sys
from collections import defaultdict
from pathlib import Path
import skimage
import numpy as np
import pandas as pd
from IPython.display import HTML, display
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix
from tifffile import imread as tif_imread
from tifffile import imwrite as tif_imsave
from tqdm.auto import tqdm
from registrations import verify_rounds

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

# extraction_functions
def max_mean_per_channel(raw_image, Z,Y,X):
    '''
    Get the max for each channel per Z plane:
    calc the mean value per Z plane and get the max of those
    '''
    mean_per_z = []
    for Z_u in np.unique(Z):
        Z_t = Z[Z==Z_u]
        Y_t = Y[Z==Z_u]
        X_t = X[Z==Z_u]
        mean_per_z.append(raw_image[Z_t,:,Y_t,X_t].mean(0))
    return np.max(mean_per_z,0)




def get_neuropil_mask_square(volume, radius, bound, inds):
    '''
    Get the neuropil mask square for each mask in the volume
    '''
    new_volumes = []
    all_masks_locs = {}
    x_max = volume.shape[1]
    y_max = volume.shape[2]
    
    
    
    for i,plane in enumerate(tqdm(volume)):
        masks_expanded_b = skimage.segmentation.expand_labels(plane, distance=bound)
        all_masks_ids = set(list(zip(*np.where(plane>0))))
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



def process_volume(volume, num_tiles=3, block_size=(7, 7), median_filter_size=3):
    processed_volume = []
    for slice_index in tqdm(range(volume.shape[0]), desc='Processing slices'):
        slice_image = volume[slice_index].astype(float)
        processed_slice = process_image(slice_image, num_tiles, block_size, median_filter_size)
        processed_volume.append(processed_slice)
    return np.array(processed_volume)

def process_image(image, num_tiles=3, block_size=(7, 7), median_filter_size=3):
    tile_height = image.shape[0] // num_tiles
    tile_width = image.shape[1] // num_tiles
    
    # Add a dimension to block size to stay within the same z-plane
    block_size_with_z = (block_size[0], block_size[1], 1)
    
    # Block size for block_reduce
    num_pixels_per_block = block_size[0] * block_size[1]
    
    # Initialize lists to store thresholds
    thresholds = []

    # Process each tile to find the maximum Otsu threshold
    max_otsu_threshold = None

    for i in range(num_tiles):
        for j in range(num_tiles):
            # Extract the tile
            tile = image[i * tile_height:(i + 1) * tile_height, j * tile_width:(j + 1) * tile_width]
            
            # Compute the block sums using block_reduce
            block_sums = skimage.measure.block_reduce(tile, block_size_with_z, func=np.sum)
            
            # Flatten the block sums to get the histogram values
            histogram_values = block_sums.flatten()
            
            # Apply Otsu's thresholding
            otsu_threshold = skimage.filters.threshold_otsu(histogram_values)
            
            # Check if the maximum Otsu threshold exceeds other tiles by a certain factor
            if max_otsu_threshold is not None and otsu_threshold > (max_otsu_threshold + 1) * 2:
                continue  # Skip this threshold and look for the next largest
            
            # Update maximum Otsu threshold
            if max_otsu_threshold is None or otsu_threshold > max_otsu_threshold:
                max_otsu_threshold = otsu_threshold
            
            # Append the threshold to the list
            thresholds.append(otsu_threshold)

    # Calculate the pixel intensity threshold per pixel
    if max_otsu_threshold is not None:
        pixel_intensity_threshold = np.round(max_otsu_threshold / num_pixels_per_block)

        # Subtract the threshold from the entire image
        background_subtracted = image - pixel_intensity_threshold

        # Set negative pixel values to 0
        background_subtracted[background_subtracted < 0] = 0

        # Apply median filtering to the background-subtracted image
        median_filtered = scipy.ndimage.median_filter(background_subtracted, size=median_filter_size)

        return median_filtered

    else:
        return None
    


def extract_intensities(manifest):
    round_to_rounds, reference_round, register_rounds = verify_rounds(manifest, parse_registered = True, 
                                                                      print_rounds = True, print_registered = True)
    
    HCR_fix_image_path = reference_round['image_path'] # The fix image that all other rounds will be registerd to (include all channels!)

    # get parameters
    gaus_blur_params = (0,0,2,2) #not over Z, not over C, 2x2 on XxY

    #Generate background subtracted, median filtered image
    block_size = (7,7)
    num_tiles = 3 #your tile side length
    median_filter_size = 3

    # neuropil parameters
    radius = 25 # radius on neuropil square
    bound = 1 # how much to extand each mask DOES NOT WORK FOR NOW, NEED TO DEBUG
    

    
    for HCR_round_to_register in register_rounds:
        mov_round = round_to_rounds[HCR_round_to_register]
        # File names
        HCR_mov_image_path = mov_round['image_path'] # The image that will be registered to the fix image (include all channels!)
        round_folder_name = f"HCR{HCR_round_to_register}_to_HCR{reference_round['round']}"
        

        full_stack_path =  Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'HCR' / 'full_registered_stacks' / f"{round_folder_name}.tiff"
        full_stack_masks_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'HCR' / 'cellpose' / f"{round_folder_name}_masks.tiff"
        
        ## load images and verify that sizes are correct
        raw_image = tif_imread(full_stack_path)
        masks = tif_imread(full_stack_masks_path)
        assert raw_image[:,0,:,:].shape == masks.shape
        # acceleration step
        inds = get_indices_sparse(masks)

        # get neuropil masks
        neuropil_masks_inds = get_neuropil_mask_square(masks, radius, bound, inds)

        # filter stack
        gaus_blured = gaussian_filter(raw_image,gaus_blur_params)

        # run otsu filter on mask
        otsu_tile_img = process_volume(raw_image, num_tiles, block_size, median_filter_size) #per-slice otsu tile, pick max, med filt

        number_of_channels = int(raw_image.shape[1])
        to_pnd = defaultdict(list)
        for mask_id, mask_inds in enumerate(tqdm(inds)):
            if mask_id == 0: # first mask is just
                continue
            if len(mask_inds[0])==0:
                raise Exception("mask with zero pixels")
            Z = mask_inds[0]
            Y = mask_inds[1]
            X = mask_inds[2]
            
            vals_per_mask_per_channel = raw_image[mask_inds[0],:,mask_inds[1],mask_inds[2]]
            
            to_pnd['mask_id'].extend([mask_id]*number_of_channels)
            to_pnd['channel'].extend(list(range(number_of_channels)))
            
            to_pnd['mean'].extend(list(vals_per_mask_per_channel.mean(0)))
            
            # calc mean for each plan, and get the value of the plan with the highest mean value
            to_pnd['mean_of_max_Z'].extend(list(max_mean_per_channel(raw_image,Z,Y,X)))
            
            # Mean of otsu-filtered image
            vals_per_mask_per_channel_otsu_median = otsu_tile_img[mask_inds[0],:,mask_inds[1],mask_inds[2]]
            to_pnd[f'otsu_medfilt{"_".join(map(str, block_size))}_{num_tiles}_{median_filter_size}_mean'].extend(list(vals_per_mask_per_channel_otsu_median.mean(0)))
            
            # Mean of gaussian-filtered image
            vals_per_mask_per_channel_gaus = gaus_blured[mask_inds[0],:,mask_inds[1],mask_inds[2]]
            to_pnd[f'gaus_blured{"_".join(map(str,gaus_blur_params))}_mean'].extend(list(vals_per_mask_per_channel_gaus.mean(0)))
            
            # add X,Y,Z 
            to_pnd['Z'].extend([Z.mean()]*number_of_channels)
            to_pnd['X'].extend([X.mean()]*number_of_channels)
            to_pnd['Y'].extend([Y.mean()]*number_of_channels)
            
            # add neuropil mean
            Z = neuropil_masks_inds[mask_id][0]
            Y = neuropil_masks_inds[mask_id][1]
            X = neuropil_masks_inds[mask_id][2]
            
            vals_per_mask_per_channel = raw_image[Z,:,Y,X]
            to_pnd['neuropil_mean'].extend(list(vals_per_mask_per_channel.mean(0)))
            to_pnd['neuropil_median'].extend(list(np.median(vals_per_mask_per_channel, axis=0)))
            to_pnd['neuropil_30pct'].extend(list(np.percentile(vals_per_mask_per_channel, 30, axis=0)))
            to_pnd['neuropil_20pct'].extend(list(np.percentile(vals_per_mask_per_channel, 20, axis=0)))
            to_pnd['neuropil_10pct'].extend(list(np.percentile(vals_per_mask_per_channel, 10, axis=0)))


        #Save to `masks_path` folder
        df = pd.DataFrame(to_pnd)
        df.attrs['raw_image_path'] = str(full_stack_path)
        df.attrs['masks_path'] = str(full_stack_masks_path)
        df.attrs['HCR_round_number'] = HCR_round_to_register
        df.to_csv(masks_path.parent / 'extract_values.csv')
        df.to_pickle(masks_path.parent / 'extract_values.pkl')
        print(masks_path.parent / 'extract_values.pkl')
