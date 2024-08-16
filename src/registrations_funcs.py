import os
import sys
import hjson
import numpy as np
from pathlib import Path
import SimpleITK as sitk
from tifffile import imwrite as tif_imwrite
from tifffile import imread as tif_imread
from functools import lru_cache
# Path for bigstream unless you did pip install
sys.path = [fr"\\nasquatch\data\2p\jonna\Code_Python\Notebooks_Jonna\BigStream\bigstream_github"] + sys.path 
sys.path = [fr"C:\Users\jonna\Notebooks_Jonna\BigStream\bigstream_github"] + sys.path 
sys.path = [fr'{os.getcwd()}/bigstream_github'] + sys.path
sys.path = ["/mnt/nasquatch/data/2p/jonna/Code_Python/Notebooks_Jonna/BigStream/bigstream_github"] + sys.path 

from bigstream.align import feature_point_ransac_affine_align
from bigstream.application_pipelines import easifish_registration_pipeline
from bigstream.transform import apply_transform

def get_registration_score(fixed, mov):
    fixed_image = sitk.GetImageFromArray(fixed.astype(np.float32))
    registered_image = sitk.GetImageFromArray(mov.astype(np.float32))
    # Initialize the registration method
    irm = sitk.ImageRegistrationMethod()

    # Set the metric to ANTS Neighborhood Correlation
    irm.SetMetricAsMattesMutualInformation()  # 4 is the radius of the neighborhood

    # Set the fixed and moving images for the metric evaluation
    out = irm.MetricEvaluate(fixed_image,registered_image)
    return out

# @lru_cache(maxsize=128)
def register_lowres(
    fix_lowres,
    mov_lowres,
    fix_lowres_spacing,
    mov_lowres_spacing,
    write_directory,
    global_ransac_kwargs={},
    fname='',
    write_only_aligned=True,
):
    """

    """

    # ensure lowres datasets are in memory
    fix_lowres = fix_lowres[...]
    mov_lowres = mov_lowres[...]

    # configure global affine alignment at lowres
    alignment_spacing = np.min(fix_lowres_spacing)*4
    blob_min = int(round(np.min(fix_lowres_spacing)*4))
    blob_max = int(round(np.min(fix_lowres_spacing)*16))
    print(f'1, {blob_min=} , {blob_max=}')
    a = {'alignment_spacing':alignment_spacing,
         'blob_sizes':[blob_min, blob_max]}
    
    #numberOfIterations = 10 instead of 100
    global_ransac_kwargs_full = {**a, **global_ransac_kwargs}

    affine = feature_point_ransac_affine_align(fix_lowres, mov_lowres, 
                                                fix_lowres_spacing, mov_lowres_spacing, safeguard_exceptions=False,
                                                **global_ransac_kwargs_full)
    
    if write_only_aligned:
        # if affine is not a solution, return None
        if (np.eye(fix_lowres.ndim + 1) == affine).all():
            return None
        
    # apply global affine and save result
    aligned = apply_transform(
        fix_lowres, mov_lowres,
        fix_lowres_spacing, mov_lowres_spacing,
        transform_list=[affine],
    )
#     from IPython import embed; embed()
    reg_score = get_registration_score(aligned,fix_lowres)
    reg_score_text = str(np.round(reg_score,3)).replace('-','m')
    print(f'{write_directory}/{reg_score_text}_{fname}_both.tiff',flush=True)
    tif_imwrite(f'{write_directory}/{reg_score_text}_{fname}_both.tiff', np.swapaxes(np.array([ aligned.transpose(2,1,0), 
                                                                        fix_lowres.transpose(2,1,0)]),0,1),
                                                                        imagej=True)
    return aligned




def HCR_confocal_imaging(manifest, only_paths=False):
    """
    print instructions on how to register the HCR data round to round
    only_paths: if True, return only the paths to the files and not registration instructions
    """

    mov_rounds = []
    reference_round_number = manifest['HCR_confocal_imaging']['reference_round']
    for i in manifest['HCR_confocal_imaging']['rounds']:
        if i['round'] == reference_round_number:
            reference_round = Path(manifest['base_path']) / manifest['mouse_name'] / 'HCR' / f"{manifest['mouse_name']}_HCR{reference_round_number}.tif"
        else:
            mov_rounds.append(Path(manifest['base_path']) / manifest['mouse_name'] / 'HCR' / f"{manifest['mouse_name']}_HCR{i['round']}_To_HCR{reference_round_number}.tif")
    
    missing = False
    while 1:
        if not reference_round.exists():
            print(f"Reference round {reference_round} not found\n")
            missing = True
        for i in mov_rounds:
            if not i.exists():
                print(f"Round {i} not found\n")
                missing = True
        if not missing:
            print("All files found")
            break
        print("Please add the missing files and press enter to continue")
        input()
    if only_paths:
        return reference_round, mov_rounds
    # register the rounds


def verify_rounds(manifest, registered_paths = None):

    # verify that all rounds exists.
    reference_round_path, mov_rounds_path = HCR_confocal_imaging(manifest, only_paths=True)
    reference_round_number = manifest['HCR_confocal_imaging']['reference_round']
    if registered_paths is None: print("Rounds available for register:")
    round_to_rounds = {}
    j=0
    for i in manifest['HCR_confocal_imaging']['rounds']:
        if i['round'] != reference_round_number:
            if registered_paths is None: print(i['round'],i['channels'])
            round_to_rounds[i['round']] = i
            round_to_rounds[i['round']]['image_path'] = mov_rounds_path[j]
            j+=1
        else:
            reference_round = i
            reference_round['image_path'] = reference_round_path

    if registered_paths:
        print("Rounds available for registerion apply:", end=' ')
        params = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'params.hjson'
        selected_registrations = hjson.load(open(params, 'r'))
        for i in selected_registrations['HCR_selected_registrations']['rounds']:
            assert i['round'] in round_to_rounds, f"Round {i['round']} not defined in manifest!"
            round_to_rounds[i['round']]['registrations'] = i['selected_registrations']
            print(i['round'], end=' ')
    return round_to_rounds, reference_round