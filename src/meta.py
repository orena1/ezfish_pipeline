import hjson
from pathlib import Path
import os
from rich.prompt import Prompt
import numpy as np
import sys
# supported HCR probs
HCR_probs = ['CALCA', 'CCK', 'CHAT', 'CHRIMSON', 'DAPI', 'FOXP2', 'GCAMP', 'GLP1R', 'GRP', 'NPR3', 'PDYN', 
             'RORB', 'SST', 'SYT10', 'TAC1', 'VGAT', 'SYT10', 'CD24A', 'GPR101','PDE11A','MC4R','TH','RUNX1','RUNX4',
            'BRS3', 'TACR1', 'OPRM1', 'ASB4', 'SAMD3', 'EGR1', 'EPHA3']


def parse_json(json_file):
    """
    Parse a json file and return a dictionary object
    """
    with open(json_file, 'r') as f:
        return hjson.load(f)
    
def user_input_missing(check_results, message, color):
    if  (np.array(check_results)[:,1]==False).any():
        print("Missing 2P runs:")
        while True:
            for i in check_results[check_results[:,1]==False]:
                print(i[0])
            out = Prompt.ask(f"\n[italic {color}]Some 2p runs are missing, do you whish to continue?[/italic {color}]", choices=["y", "n", "check-again"])
            if out=='n':
                sys.exit()
            if out=='y':
                return

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
    user_input_missing(check_results, 'Some 2p runs are missing, do you whish to continue?', color='red')

    # verify that functional run exists.
    suite2p_run = session['functional_run'][0]
    suite2p_path = base_path / mouse_name / '2P' /  f'{mouse_name}_{date_two_photons}_{suite2p_run}' / 'suite2p' /'plane0/ops.npy'
    user_input_missing(np.array([[suite2p_path, os.path.exists(suite2p_path)]]), 'Suite2p path is missing, do you wish to continue?', color='pink')

    # verify that unwarp_config exists
    if not os.path.exists(manifest['two_photons_imaging']['sessions'][0]['unwarp_config']):
        new_path  = base_path / 'Calibration_files_for_unwarping' /manifest['two_photons_imaging']['sessions'][0]['unwarp_config']
        if not os.path.exists(new_path):
            raise Exception(f"unwarp config file does not exist {manifest['two_photons_imaging']['sessions'][0]['unwarp_config']} and not in {new_path}")
        manifest['two_photons_imaging']['sessions'][0]['unwarp_config'] = new_path
    
    return {'reference_round':reference_round}

def main_pipeline_manifest(json_file):
    """
    Parse the pipeline manifest json file and verify that the required fields are present
    """
    manifest = parse_json(json_file)
    required_fields = ['base_path', 'mouse_name']
    for field in required_fields:
        if field not in manifest:
            raise ValueError(f"Required field {field} not found in pipeline manifest")
    
    # ToDo, add more checks here

    return manifest

