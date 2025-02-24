import re
import os
import sys
import hjson
import zarr
import shutil
import numpy as np
from pathlib import Path
try:
    from .meta import parse_json  # Relative import (for running as part of a package)
except ImportError:
    from meta import parse_json  # Absolute import (for running in Jupyter Notebook)
from rich.progress import track
import SimpleITK as sitk
from rich import print as rprint
from tifffile import imwrite as tif_imwrite
from tifffile import imread as tif_imread

# Path for bigstream unless you did pip install
sys.path = [fr"\\nasquatch\data\2p\jonna\Code_Python\Notebooks_Jonna\BigStream\bigstream_v2_andermann"] + sys.path 
sys.path = [fr"C:\Users\jonna\Notebooks_Jonna\BigStream\bigstream_v2_andermann"] + sys.path 
sys.path = [fr'{os.getcwd()}/bigstream_v2_andermann'] + sys.path
sys.path = ["/mnt/nasquatch/data/2p/jonna/Code_Python/Notebooks_Jonna/BigStream/bigstream_v2_andermann"] + sys.path 



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

def registarion_apply(full_manifest):
    """
    Register the rounds in the manifest that was selected in params.hjson
    """
    manifest = full_manifest['data']
    round_to_rounds, reference_round, register_rounds = verify_rounds(full_manifest, parse_registered = True, print_rounds = True, print_registered = True)

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
        rprint(f"[bold]Applying Registration to round - {HCR_round_to_register}[/bold]")


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
            output_channel_tiff_path = output_channel_path.parent / output_channel_path.name.replace('.zarr','.tiff')
            if os.path.exists(output_channel_path) and os.path.exists(output_channel_tiff_path):
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

            tif_imwrite(output_channel_tiff_path
                        ,data.transpose(2,1,0))
        
        print(f"Saving full stack -{full_stack_path}")
        full_stack = [] 
        for path in data_paths:
            full_stack.append(zarr.load(path))
        full_stack = np.stack(full_stack)

        tif_imwrite(full_stack_path, full_stack.transpose(3, 0, 2, 1), imagej=True, metadata={'axes': 'ZCYX'})

    # Now let's also copy the reference_round to the full_registered_stacks folder
    reference_round_full_stack_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'HCR' / 'full_registered_stacks' / f"HCR{reference_round['round']}.tiff"
    if not reference_round_full_stack_path.exists():
        print(f"Copying reference round {reference_round['round']} to full_registered_stacks")
        shutil.copy(HCR_fix_image_path, reference_round_full_stack_path)
        
            

def verify_rounds(full_manifest, parse_registered = False, print_rounds = False, print_registered = False, func='registering-apply'):
    '''
    if parse_registered is True, return the rounds that have been registered
    
    '''
    manifest = full_manifest['data']

    # verify that all rounds exists.
    reference_round_path, mov_rounds_path = HCR_confocal_imaging(manifest, only_paths=True)
    reference_round_number = manifest['HCR_confocal_imaging']['reference_round']
    if print_rounds: print("\nRounds available:")

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
        selected_registrations = parse_json(full_manifest['manifest_path'])['params']
        txt_to_rich = f"[green]Rounds available for {func} [/green]:"
        for i in selected_registrations['HCR_selected_registrations']['rounds']:
            assert i['round'] in round_to_rounds, f"Round {i['round']} not defined in manifest!"
            selected_registration_path =  Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / 'HCR' / 'registrations'/ f"HCR{i['round']}_to_HCR{reference_round_number}" / i['selected_registrations'][0]
            assert os.path.exists(selected_registration_path), f"Registration {selected_registration_path} not found although it exists in the manifest params"

            round_to_rounds[i['round']]['registrations'] = i['selected_registrations']
            txt_to_rich+= f" {i['round']}"
            ready_to_apply.append(i['round'])
        if print_registered: rprint(txt_to_rich)
    return round_to_rounds, reference_round, ready_to_apply



def register_rounds(full_manifest):
    """
    Register the rounds in the manifest
    """
    manifest = full_manifest['data']
    round_to_rounds, reference_round, ready_to_apply = verify_rounds(full_manifest)
    
    rprint("\n [green]---------------------------Registering rounds---------------------------- [/green]")
    rprint(f"There are {len(manifest['HCR_confocal_imaging']['rounds'])} HCR rounds in the manifest, registartion is done round to round using juptyer notebooks")

    rprint("\n[green]Step A:[/green]")
    rprint("Open the notebook ezfish_pipeline/src/processing_notebooks/HCR_rounds/1_scan_lowres_parameters.ipynb")
    rprint(f"Change the manifest path to this = {full_manifest['manifest_path']}")

    rprint("\n[green]Step B:[/green]")
    rprint("Open the notebook ezfish_pipeline/src/processing_notebooks/HCR_rounds/2_scan_highres_parameters.ipynb")
    rprint(f"Change the manifest path to this = {full_manifest['manifest_path']}")
    
    rprint("\n[green]Step C:[/green]")
    rprint(f"add HCR_selected_registrations to {full_manifest['manifest_path']} file to select the rounds you want to register")
    rprint(f"Once files everyting is set Press [bold]Enter[/bold] to load the HCR_selected_registrations")
    input()

    round_to_rounds, reference_round, ready_to_apply = verify_rounds(full_manifest, parse_registered = True)
    rprint("\n[green]Step D:[/green]")
    rprint(f"Currently registration that are ready to apply are {ready_to_apply}")
    rprint(f"Press Enter to apply registeration matrix to these rounds {ready_to_apply}")
    input()
    registarion_apply(full_manifest)
