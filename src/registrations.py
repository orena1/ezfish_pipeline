import re
import os
import sys
import hjson
import zarr
import numpy as np
from pathlib import Path
from rich.progress import track
import SimpleITK as sitk
from rich import print as rprint
from tifffile import imwrite as tif_imwrite
from tifffile import imread as tif_imread

# Path for bigstream unless you did pip install
sys.path = [fr"\\nasquatch\data\2p\jonna\Code_Python\Notebooks_Jonna\BigStream\bigstream_github"] + sys.path 
sys.path = [fr"C:\Users\jonna\Notebooks_Jonna\BigStream\bigstream_github"] + sys.path 
sys.path = [fr'{os.getcwd()}/bigstream_github'] + sys.path
sys.path = ["/mnt/nasquatch/data/2p/jonna/Code_Python/Notebooks_Jonna/BigStream/bigstream_github"] + sys.path 

from bigstream.align import feature_point_ransac_affine_align
from bigstream.application_pipelines import easifish_registration_pipeline
from bigstream.transform import apply_transform
from bigstream.piecewise_transform import distributed_apply_transform

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
    Function to register lowres images in paramters scan, this function is used by 1_scan_lowres_parameters.ipynb
    """

    # ensure lowres datasets are in memory
    fix_lowres = fix_lowres[...]
    mov_lowres = mov_lowres[...]

    # configure global affine alignment at lowres
    alignment_spacing = np.min(fix_lowres_spacing)*4
    blob_min = int(round(np.min(fix_lowres_spacing)*4))
    blob_max = int(round(np.min(fix_lowres_spacing)*16))
    #print(f'1, {blob_min=} , {blob_max=}')
    a = {'alignment_spacing':alignment_spacing,'blob_sizes':[blob_min, blob_max]}
    
    #numberOfIterations = 10 instead of 100
    global_ransac_kwargs_full = {**a, **global_ransac_kwargs}

    affine = feature_point_ransac_affine_align(fix_lowres, mov_lowres, 
                                                fix_lowres_spacing, mov_lowres_spacing, 
                                                safeguard_exceptions=False,
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

    reg_score = get_registration_score(aligned, fix_lowres)
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
            reference_round = Path(manifest['base_path']) / manifest['mouse_name'] / 'HCR' / f"{manifest['mouse_name']}_HCR{reference_round_number}.tiff"
        else:
            mov_rounds.append(Path(manifest['base_path']) / manifest['mouse_name'] / 'HCR' / f"{manifest['mouse_name']}_HCR{i['round']}_To_HCR{reference_round_number}.tiff")
    
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

def registarion_apply(manifest):
    """
    Register the rounds in the manifest that was selected in params.hjson
    """

    round_to_rounds, reference_round, register_rounds = verify_rounds(manifest, parse_registered = True, print_rounds = True, print_registered = True)

    for HCR_round_to_register in register_rounds:
        mov_round = round_to_rounds[HCR_round_to_register]
        # File names
        HCR_fix_image_path = reference_round['image_path'] # The fix image that all other rounds will be registerd to (include all channels!)
        HCR_mov_image_path = mov_round['image_path'] # The image that will be registered to the fix image (include all channels!)

        round_folder_name = f"HCR{HCR_round_to_register}_to_HCR{reference_round['round']}"
        
        reg_path =  Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'HCR' / 'registrations'/ round_folder_name / round_to_rounds[HCR_round_to_register]['registrations'][0]
        full_stack_path =  Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'HCR' / 'full_registered_stacks' / f"{round_folder_name}.tiff"
        if full_stack_path.exists():
            print(f"Round {HCR_round_to_register} already registered")
            continue
        full_stack_path.parent.mkdir(exist_ok=True, parents=True)
        rprint(f"[bold]Applying Registering to round - {HCR_round_to_register}[/bold]")


        # resolution of the images
        fix_image_spacing = np.array(reference_round['resolution']) # Y,X,Z
        mov_image_spacing = np.array(mov_round['resolution']) # Y,X,Z


        # spatial downsampling, probably no need to test. (Changed x and y from 3 to 2 for CIM round 5)
        red_mut_x = manifest['HCR_to_HCR_params']['red_mut_x']
        red_mut_y = manifest['HCR_to_HCR_params']['red_mut_y']
        red_mut_z = manifest['HCR_to_HCR_params']['red_mut_z']

        fix_lowres_spacing = fix_image_spacing * np.array([red_mut_y,red_mut_x,red_mut_z])
        mov_lowres_spacing = mov_image_spacing * np.array([red_mut_y,red_mut_x,red_mut_z])


        # get block size from the registration file
        blocksize_match = re.findall(r'bs(\d+)_(\d+)_(\d+)', Path(round_to_rounds[HCR_round_to_register]['registrations'][0]).name)
        blocksize = [int(num) for num in blocksize_match[0]]

        print("Loding images")
        HCR_fix_round = tif_imread(HCR_fix_image_path)[:,0]
        HCR_mov_round = tif_imread(HCR_mov_image_path)

        # load the registration files
        affine = np.loadtxt(fr"{reg_path}/_affine.mat")
        deform = zarr.load(fr"{reg_path}/deform.zarr")
        data_paths = []
        fix_highres = HCR_fix_round.transpose(2,1,0) # from Z,X,Y to Y,X,Z

        #Loop through channels starting with 1, which ignores the first channel which has already been registered
        for channel in track(range(HCR_mov_round.shape[1]), description="Registering channels"):
            output_channel_path = Path(fr"{reg_path}/out_c{channel}.zarr")
            if os.path.exists(output_channel_path):
                print(f"Channel {channel} already registered")
                data_paths.append(output_channel_path)
                continue

            HCR_mov_round_C = HCR_mov_round[:,channel]
            
            # mov Image
            mov_highres = HCR_mov_round_C.transpose(2,1,0)
            
            # register the images
            local_aligned = distributed_apply_transform(
                fix_highres, mov_highres,
                fix_image_spacing, mov_image_spacing,
                transform_list=[affine, deform],
                blocksize=blocksize,
                write_path=output_channel_path)
            print(fr'saved output {output_channel_path}')

            data = zarr.load(output_channel_path)
            data_paths.append(output_channel_path)

            tif_imwrite(output_channel_path.parent / output_channel_path.name.replace('.zarr','.tiff')
                        ,data.transpose(2,1,0))
        
        print(f"Saving full stack -{full_stack_path}")
        full_stack = [] 
        for path in data_paths:
            full_stack.append(zarr.load(path))
        full_stack = np.stack(full_stack)

        tif_imwrite(full_stack_path, full_stack.transpose(2,3,0,1), imagej=True, metadata={'axes': 'ZCYX'})

def verify_rounds(manifest, parse_registered = False, print_rounds = False, print_registered = False):
    '''
    if parse_registered is True, return the rounds that have been registered
    
    '''

    # verify that all rounds exists.
    reference_round_path, mov_rounds_path = HCR_confocal_imaging(manifest, only_paths=True)
    reference_round_number = manifest['HCR_confocal_imaging']['reference_round']
    if print_rounds: print("Rounds available for register:")

    round_to_rounds = {}
    j=0
    for i in manifest['HCR_confocal_imaging']['rounds']:
        if i['round'] != reference_round_number:
            if print_rounds: print(i['round'],i['channels'])
            round_to_rounds[i['round']] = i
            round_to_rounds[i['round']]['image_path'] = mov_rounds_path[j]
            j+=1
        else:
            reference_round = i
            reference_round['image_path'] = reference_round_path
    
    ready_to_apply = []
    if parse_registered:
        txt_to_rich = "[green]Rounds available for registerion-apply [/green]:"
        params = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'params.hjson'
        selected_registrations = hjson.load(open(params, 'r'))
        for i in selected_registrations['HCR_selected_registrations']['rounds']:
            assert i['round'] in round_to_rounds, f"Round {i['round']} not defined in manifest!"
            selected_registration_path =  Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'HCR' / 'registrations'/ f"HCR{i['round']}_to_HCR{reference_round_number}" / i['selected_registrations'][0]
            assert os.path.exists(selected_registration_path), f"Registration {selected_registration_path} not found although params.hjson says it should be there"

            round_to_rounds[i['round']]['registrations'] = i['selected_registrations']
            txt_to_rich+= f" {i['round']}"
            ready_to_apply.append(i['round'])
        if print_registered: rprint(txt_to_rich)
    return round_to_rounds, reference_round, ready_to_apply



def register_rounds(manifest, manifest_path):
    """
    Register the rounds in the manifest
    """
    round_to_rounds, reference_round, ready_to_apply = verify_rounds(manifest)
    rprint("[green]Registering rounds: [/green]")
    rprint(f"There are {len(manifest['HCR_confocal_imaging']['rounds'])} HCR rounds in the manifest, registartion is done round to round using juptyer notebooks")

    rprint("\n[green]Step A:[/green]")
    rprint("Open the notebook ezfish_pipeline/src/processing_notebooks/HCR_rounds/1_scan_lowres_parameters.ipynb")
    rprint(f"Change the manifest path to this = {manifest_path}")

    rprint("\n[green]Step B:[/green]")
    rprint("Open the notebook ezfish_pipeline/src/processing_notebooks/HCR_rounds/2_scan_highres_parameters.ipynb")
    rprint(f"Change the manifest path to this = {manifest_path}")
    
    rprint("\n[green]Step C:[/green]")
    rprint("Create the OUTPUT/params.hjson file to select the rounds you want to register (see example in examples/param_example.hjson)")
    rprint(f"Once files is created Press [bold]Enter[/bold] to load the params.hjson file")
    input()

    round_to_rounds, reference_round, ready_to_apply = verify_rounds(manifest, parse_registered = True)
    rprint("\n[green]Step D:[/green]")
    rprint(f"Currenlty registartion that are ready to apply are {ready_to_apply}")
    rprint(f"Press Enter to apply registerion matrix to these rounds {ready_to_apply}")
    input()
    registarion_apply(manifest)
