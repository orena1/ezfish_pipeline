import os
import sys
import hjson
import numpy as np
from pathlib import Path
from rich.prompt import Prompt


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

def _validate_tiff_inputs(base_path, mouse_name, session, has_hires):
    """Validate that expected TIFF files exist for TIFF input mode."""
    # Get planes to process
    if 'functional_planes' in session:
        all_planes = list(session['functional_planes'])
    elif 'functional_plane' in session:
        all_planes = list(session['functional_plane'])
    else:
        raise ValueError("TIFF mode requires 'functional_planes' (or 'functional_plane') in session config.")

    missing = []
    output_base = base_path / mouse_name / 'OUTPUT' / '2P'

    for plane in all_planes:
        if has_hires:
            expected = output_base / 'tile' / 'stitched' / f'hires_stitched_plane{plane}.tiff'
            if not expected.exists():
                missing.append(str(expected))
        else:
            c01 = output_base / 'cellpose' / f'lowres_meanImg_C01_plane{plane}.tiff'
            c0 = output_base / 'cellpose' / f'lowres_meanImg_C0_plane{plane}.tiff'
            if not c01.exists() and not c0.exists():
                missing.append(f"{c0}  (green only)\n    or {c01}  (green+red)")

    if missing:
        msg = "TIFF input mode: missing required files. Place your images at:\n"
        for m in missing:
            msg += f"    {m}\n"
        raise FileNotFoundError(msg)


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
        session = manifest['two_photon_imaging']['sessions'][0]

        # input_format: defaults to 'sbx' for backward compatibility with existing manifests
        input_format = session.get('input_format', 'sbx')
        if input_format not in ('sbx', 'tiff'):
            raise ValueError(f"input_format must be 'sbx' or 'tiff', got '{input_format}'")

    #test that reference round exists
    reference_round = manifest['HCR_confocal_imaging']['reference_round']
    for i in manifest['HCR_confocal_imaging']['rounds']:
        if i['round'] == reference_round:
            break
    else:
        raise Exception(f"reference round was not found {reference_round} is not in rounds")

    if not args.only_hcr:
        if input_format == 'sbx':
            # --- SBX validation (existing logic) ---
            date_two_photons = session['date']

            # verify that all 2p runs exists.
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

        elif input_format == 'tiff':
            # --- TIFF validation ---
            has_hires = session.get('has_hires', False)
            _validate_tiff_inputs(base_path, mouse_name, session, has_hires)

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


