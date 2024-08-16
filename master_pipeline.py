import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
from rich import print as rprint
from rich.prompt import Prompt

from src import tiling as tl
from src.meta import main_pipeline_manifest

# supported HCR probs
HCR_probs = ['CALCA', 'CCK', 'CHAT', 'CHRIMSON', 'DAPI', 'FOXP2', 'GCAMP', 'GLP1R', 'GRP', 'NPR3', 'PDYN', 'RORB', 'SST', 'SYT10', 'TAC1', 'VGAT']

def parse_input_args(args):
    if args is not None:
        return args
    argparser = argparse.ArgumentParser(description='Master pipeline for processing data')
    argparser.add_argument('manifest', help='Path to the pipeline manifest file')



    args = argparser.parse_args(args)




# def construct_paths(manifest):
#     mn = manifest
#     m_name = mn['mouse_name']

#     assert len(mn['two_photons_imaging']['sessions']) == 1, 'Only one session is currenlty supported'
#     ses_number = 0
#     session = mn['two_photons_imaging']['sessions'][ses_number]
#     ses_date = session['date']

#     base_path = Path(mn['base_path'])
#     base_path / mn['mouse_name'] / '2P' / ses_date / f'{date}_{m_name}_{session['']}


def user_input_missing(check_results, message):
    if  (np.array(check_results)[:,1]==False).any():
        print("Missing 2P runs:")
        for i in check_results[check_results[:,1]==False]:
            print(i[0])
        out = Prompt.ask("\n[italic red]Some 2p runs are missing, do you whish to continue?[/italic red]", choices=["y", "n"])
        if out=='n':
            sys.exit()

def verify_manifest(manifest):
    '''
    Verify that the json file is valid
    '''
    #test that only one 2P session is present
    assert len(manifest['two_photons_imaging']['sessions'])==1, 'only support one 2P sessions'
    base_path = Path(manifest['base_path'])
    mouse_name = manifest['mouse_name']
    date_two_photons = manifest['two_photons_imaging']['sessions'][0]['date']

    #test that reference round exists
    reference_round = manifest['HCR_confocal_imaging']['reference_round']
    for i in manifest['HCR_confocal_imaging']['rounds']:
        if i['round'] == reference_round:
            break
    else:
        raise Exception(f"reference round was not found {reference_round} is not un rounds")
    
    # verify that all probs are supported
    for i in manifest['HCR_confocal_imaging']['rounds']:
        for channel in i['channels']:
            assert channel in HCR_probs, f"Probe {channel} not supported"

    # verify that all 2p runs exists.
    session = manifest['two_photons_imaging']['sessions'][0]
    check_results = []
    for k in session:
        if '_run' in k:
            for run in session[k]:
                run_path_sbx = base_path / mouse_name / '2P' /  f'{mouse_name}_{date_two_photons}_{run}' / f'{mouse_name}_{date_two_photons}_{run}.sbx'
                check_results.append([run_path_sbx,os.path.exists(run_path_sbx)])
    check_results = np.array(check_results)
    user_input_missing(check_results, 'Some 2p runs are missing, do you whish to continue?')

    # verify that unwarp_config exists
    if not os.path.exists(manifest['two_photons_imaging']['sessions'][0]['unwarp_config']):
        new_path  = base_path / 'Calibration_files_for_unwarping' /manifest['two_photons_imaging']['sessions'][0]['unwarp_config']
        if not os.path.exists(new_path):
            raise Exception(f"unwarp config file does not exist {manifest['two_photons_imaging']['sessions'][0]['unwarp_config']} and not in {new_path}")
        manifest['two_photons_imaging']['sessions'][0]['unwarp_config'] = new_path
    


    return {'reference_round':reference_round}

def main(args = None):
    '''
    We can either start main with arguments or from command line 
    The following steps are conducted:
    1. Parse the manifest file
    2. unwarp 2P anatomical_runs
    '''

    args = parse_input_args(args)

    # step 1-A: parse the manifest file
    manifest = main_pipeline_manifest(args['manifest'])
    specs = verify_manifest(manifest)

    session = manifest['two_photons_imaging']['sessions'][0]
    tl.process_session_sbx(manifest, session)

    # step 2-A: unwarp 2P anatomical_runs
    tl.unwarp_tiles(manifest, session)

    # step 3-M: stitch the tiles
    tl.stitch_tiles_and_rotate(manifest, session)

    # step 4-AM: register the HCR data round to round
    tl.register_rounds(manifest)
    









if __name__ == "__main__":
    args = {'manifest': 'examples/CIM132.hjson'}
    main(args)
    