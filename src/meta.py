import os
import sys
import hjson
import numpy as np
from pathlib import Path
from rich.prompt import Prompt
# supported HCR probs
HCR_probs = [
    'ADRA1A', 'ADRA1B', 'ADRA2A', 'ADRA2B', 'ASB4', 'BRS3', 'CALCA', 'CCK', 'CD24A', 'CHAT',
    'CHRIMSON', 'CRH', 'DAPI', 'DRD1', 'DSREDV1', 'DSREDV2', 'EBF2', 'EGR1', 'EPHA3', 'FOS', 'FOXP2', 'GCAMP',
    'GLP1R', 'GPR101', 'GRP', 'MC4R', 'NPR3', 'NPY1R', 'OPRM1', 'PDE11A', 'PDYN', 'RORB',
    'RUNX1', 'RUNX4', 'SAMD3', 'SATB2', 'SERPINB1B', 'SST', 'SSTR2', 'SYT10', 'TAC1', 'TACR1', 'TH',
    'TRHR', 'VGAT'
]


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

def verify_manifest(manifest, args):
    '''
    Verify that the json file is valid
    '''

    if 'cellpose_channel' not in manifest['params']['HCR_cellpose']:
        raise ValueError("'cellpose_channel' must be specified in HCR_cellpose params. (e.g., 0 for first channel, 1 for second channel).")
    
    if not args.only_hcr and '2p_cellpose' not in manifest['params']:
        raise ValueError("'2p_cellpose' configuration must be specified in params when processing 2P data.")
            
    
    manifest = manifest['data']

    # backward compat: accept old key name
    if 'two_photons_imaging' in manifest and 'two_photon_imaging' not in manifest:
        manifest['two_photon_imaging'] = manifest.pop('two_photons_imaging')

    base_path = Path(manifest['base_path'])
    mouse_name = manifest['mouse_name']

    # 2P validation only when not in HCR-only mode
    session = None
    has_hires = False
    if not args.only_hcr:
        assert 'two_photon_imaging' in manifest, "'two_photon_imaging' section required for full pipeline mode"
        assert len(manifest['two_photon_imaging']['sessions'])==1, 'only support one 2P sessions'
        date_two_photons = manifest['two_photon_imaging']['sessions'][0]['date']
        session = manifest['two_photon_imaging']['sessions'][0]

    #test that reference round exists
    reference_round = manifest['HCR_confocal_imaging']['reference_round']
    for i in manifest['HCR_confocal_imaging']['rounds']:
        if i['round'] == reference_round:
            break
    else:
        raise Exception(f"reference round was not found {reference_round} is not in rounds")
    
    # verify that all probs are supported
    for i in manifest['HCR_confocal_imaging']['rounds']:
        for channel in i['channels']:
            assert channel in HCR_probs, f"Probe {channel} not supported"

    # verify that all 2p runs exists.
    if not args.only_hcr:
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

    # verify anatomical runs configuration
    if not args.only_hcr:
        has_lowres_green = len(session.get('anatomical_lowres_green_runs', [])) > 0
        has_hires_green = len(session.get('anatomical_hires_green_runs', [])) > 0

        if has_hires_green:
            assert not has_lowres_green, "Cannot have both lowres and hires runs"
            assert len(session['anatomical_hires_green_runs'])==len(session['anatomical_hires_red_runs']), "Number of hires green and red runs do not match"
            has_hires = True
        elif has_lowres_green:
            assert len(session['anatomical_lowres_green_runs'])==len(session['anatomical_lowres_red_runs']), "Number of lowres green and red runs do not match"
        # else: no anatomical runs — lowres mode using Suite2p mean image only

    # verify that unwarp_config exists (only needed for high-res workflow)
    if has_hires and 'unwarp_config' in session:
        if not os.path.exists(session['unwarp_config']):
            new_path = base_path / 'Calibration_files_for_unwarping' / session['unwarp_config']
            if not os.path.exists(new_path):
                raise Exception(f"unwarp config file does not exist {session['unwarp_config']} and not in {new_path}")
            session['unwarp_config'] = new_path

    return {'reference_round':reference_round, 'session':session}, has_hires

def main_pipeline_manifest(json_file):
    """
    Parse the pipeline manifest json file and verify that the required fields are present
    """
    manifest = parse_json(json_file)
    manifest['manifest_path'] = json_file
    required_fields = ['base_path', 'mouse_name']
    for field in required_fields:
        if field not in manifest['data']:
            raise ValueError(f"Required field {field} not found in pipeline manifest")
    
    # ToDo, add more checks here

    return manifest

def get_automation_config(params):
    """
    Get automation config with sensible defaults.

    If 'automation' section is missing, defaults to manual (backward compatible).

    Config options (all use 'auto' or 'manual'):
    - twop_to_hcr: 'auto' or 'manual'
        - 'manual' (default): User-provided BigWarp landmarks only
        - 'auto': Automated registration refinement (still requires manual landmarks as starting point)
    - lowres_to_hires: 'auto' or 'manual'
        - 'manual' (default): User-provided BigWarp landmarks + TPS
        - 'auto': Automated SIFT feature matching + RANSAC affine
    - stitching: 'auto' or 'manual'
        - 'manual' (default): User-provided BigStitcher coordinates
        - 'auto': Automated SIFT + phase correlation stitching
    """
    automation = params.get('automation', {})
    return {
        'twop_to_hcr': automation.get('twop_to_hcr', 'manual'),
        'lowres_to_hires': automation.get('lowres_to_hires', 'manual'),
        'stitching': automation.get('stitching', 'manual'),
    }

def check_rotation(manifest):
    manifest = parse_json(manifest['manifest_path'])
    # Support both old and new name
    if 'rotation_2p_to_HCR' in manifest['params'] or 'rotation_2p_to_HCRspec' in manifest['params']:
        return True
    else:
        return False


# =============================================================================
# Parameter accessor functions (with backward compatibility)
# =============================================================================

def get_rotation_config(params):
    """Get rotation/coordinate transform config. Supports old and new names."""
    # New name first, fall back to old
    return params.get('rotation_2p_to_HCR', params.get('rotation_2p_to_HCRspec', {}))


def get_hcr_to_hcr_registration_config(params):
    """
    Get HCR-to-HCR registration config. Supports old and new names/formats.

    Returns dict with 'downsampling' as [x, y, z] array.
    """
    # Try new name first
    config = params.get('HCR_to_HCR_registration', {})
    if config:
        # New format: downsampling as array
        if 'downsampling' in config:
            return config
        # Old field names in new location
        return {
            'downsampling': [
                config.get('red_mut_x', 3),
                config.get('red_mut_y', 3),
                config.get('red_mut_z', 2)
            ]
        }

    # Fall back to old name (HCR_to_HCR_params)
    old_config = params.get('HCR_to_HCR_params', {})
    return {
        'downsampling': [
            old_config.get('red_mut_x', 3),
            old_config.get('red_mut_y', 3),
            old_config.get('red_mut_z', 2)
        ]
    }


def get_stitching_config(params):
    """Get stitching config. Supports old and new names."""
    # New name first, fall back to old
    return params.get('stitching', params.get('auto_stitch_params', {}))


def get_intensity_extraction_config(params):
    """Get intensity extraction config. Supports old and new names."""
    # New name first, fall back to old
    return params.get('intensity_extraction', params.get('HCR_probe_intensity_extraction', {}))


def get_round_folder_name(round_num: int, reference_round_num: int) -> str:
    """Get the folder name for an HCR round based on whether it's the reference.

    Args:
        round_num: The HCR round number
        reference_round_num: The reference HCR round number

    Returns:
        "HCR{N}" for reference round, "HCR{N}_to_HCR{ref}" for other rounds
    """
    if round_num == reference_round_num:
        return f"HCR{round_num}"
    return f"HCR{round_num}_to_HCR{reference_round_num}"


