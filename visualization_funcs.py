'''
Helper functions for visualizing data from EZfish

'''

from tifffile import imwrite as tif_imwrite
from tifffile import imread as tif_imread
import numpy as np
from scipy.sparse import csr_matrix
import scipy
import pandas as pd

def compute_M(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)),
                      shape=(data.max() + 1, data.size))

def get_indices_sparse(data):
    M = compute_M(data)
    inds = []
    for row in M:
        inds.append(np.unravel_index(row.data, data.shape))
    return inds

def match_masks(stack1_masks_path: np.ndarray, stack2_masks_path: np.ndarray) -> dict:
    '''
    Input should be 2 stacks of masks where stack1 is registered to stack2.
    '''
    mask1_to_mask2 = {}

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

def visualize_match(stack1_image_path: str, stack1_masks_path: str, 
                    stack2_image_path: str, stack2_masks_path: str, mask_s1_2_mask_s2: dict,
                    output_base_filename: str) -> bool:
	
    '''
    Visualize the matching of masks between stack1 and stack2.
    stack1_image_path: path to stack1 image
    stack1_masks_path: path to stack1 masks
    stack2_image_path: path to stack2 image
    stack2_masks_path: path to stack2 masks
    mask_s1_2_mask_s2: dictionary of mask1 to mask2
    output_base_filename: base filename for output images
    '''

    # Load images and masks
    stack1_image = tif_imread(stack1_image_path)
    stack1_masks = tif_imread(stack1_masks_path)
    stack2_image = tif_imread(stack2_image_path)
    stack2_masks = tif_imread(stack2_masks_path)

    stack1_masks_inds = get_indices_sparse(stack1_masks.astype(np.uint16))
    stack2_masks_inds = get_indices_sparse(stack2_masks.astype(np.uint16))

    stack1_masks_out = np.zeros_like(stack1_masks)
    stack2_masks_out = np.zeros_like(stack2_masks)
    for mask_s1 in mask_s1_2_mask_s2:
        mask_s2_corresponding = mask_s1_2_mask_s2[mask_s1]
        stack1_masks_out[stack1_masks_inds[mask_s1]] = mask_s1 # don't need to do this
        stack2_masks_out[stack2_masks_inds[mask_s2_corresponding]] = mask_s1

    # now lets get all masks that where not matched:
    stack1_masks -= stack1_masks_out
    stack2_masks -= stack2_masks_out
    

    # Create a new stack with stack1 image and stack2 image
    image_concat = np.swapaxes(np.array([stack1_image, stack2_image]),0,1) # swap axes to make it ZCYX
    # Create a new stack with stack1 mask and stack2 mask of matched
    mask_concat_match = np.swapaxes(np.array([stack1_masks_out, stack2_masks_out]),0,1) # swap axes to make it ZCYX
    
    # Create a new stack with stack1 mask and stack2 mask of non_matched
    mask_concat_nonmatch = np.swapaxes(np.array([stack1_masks, stack2_masks]),0,1) # swap axes to make it ZCYX
    

    tif_imwrite(fr'{output_base_filename}_msk.tif', mask_concat_match.astype(np.uint16), imagej=True, 
                metadata={'axes': 'ZCYX'})
    
    tif_imwrite(fr'{output_base_filename}_msk_nonmatch.tif', mask_concat_nonmatch.astype(np.uint16), imagej=True, 
                metadata={'axes': 'ZCYX'})
    
    tif_imwrite(fr'{output_base_filename}_img.tif', image_concat.astype(np.float32), imagej=True, 
                metadata={'axes': 'ZCYX'})
    
    return False



# def examples(example_name):
#     if example_name == 'visualize_match_0':
#         stack1_image_path = fr'\\nasquatch\data\EASI_FISH\tests_small_samples\230107_RG002_Region_2_Merged_RAW_ch00_small.tif'
#         stack1_masks_path = fr'\\nasquatch\data\EASI_FISH\tests_small_samples\230107_RG002_Region_2_Merged_RAW_ch00_masks_small.tif'
        
#         stack2_image_path = fr'\\nasquatch\data\EASI_FISH\tests_small_samples\RG002_221209_007_c0_registered_flipZ_rot30p6_regigstered_to_EZfish_small.tif'
#         stack2_masks_path = fr'\\nasquatch\data\EASI_FISH\tests_small_samples\RG002_221209_007_c0_registered_flipZ_rot30p6_masks_registered_to_EZfish_small.tif'

#         min_overlap = 0.5
#         output_base_filename = fr'\\nasquatch\data\EASI_FISH\tests_small_samples\vis_min_overlap_{min_overlap}'


#         mask1_to_mask2_df = match_masks(stack1_masks_path, stack2_masks_path)
#          # to mask1_to_mask2 dict
#         mask1_to_mask2 = mask1_to_mask2_df.query(f'overlap>@min_overlap')[['mask1','mask2']].astype('int').set_index('mask1')['mask2'].to_dict()
#         visualize_match(stack1_image_path, stack1_masks_path, stack2_image_path, stack2_masks_path,
#                         mask1_to_mask2, output_base_filename)
        
# examples(example_name='visualize_match_0')