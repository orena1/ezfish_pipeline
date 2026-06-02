import os
import sys
import hjson
import numpy as np
from pathlib import Path

# Rich-compat shim. All pipeline modules import rprint / track / Prompt from
# here so the fallback lives in one place. If rich isn't installed (e.g. a
# stripped-down notebook env), modules still import cleanly and lose only
# the Rich markup formatting.
try:
    from rich import print as rprint
    from rich.progress import track
    from rich.prompt import Prompt
except ImportError:
    rprint = print

    def track(iterable, *args, **kwargs):
        return iterable

    class Prompt:
        @staticmethod
        def ask(prompt, choices=None, default=None, **kwargs):
            ans = input(str(prompt) + ' ').strip()
            if not ans and default is not None:
                return default
            return ans


def parse_json(json_file):
    """
    Parse a json/hjson manifest. Normalizes base_path slashes so the same
    manifest works whether loaded on Linux (/mnt/...) or Windows (\\...).
    """
    with open(json_file, 'r') as f:
        manifest = hjson.load(f)
    bp = manifest.get('data', {}).get('base_path')
    if isinstance(bp, str):
        manifest['data']['base_path'] = bp.replace('\\', '/')
    return manifest


def output_root(full_manifest) -> Path:
    """Pipeline OUTPUT directory: base_path / mouse_name / OUTPUT."""
    data = full_manifest['data']
    return Path(data['base_path']) / data['mouse_name'] / 'OUTPUT'
    
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
        session = manifest['two_photon_imaging']['sessions'][0]
        if getattr(args, 'tiff_only', False):
            if session.get('input_format') and session['input_format'] != 'tiff':
                rprint(f"[yellow]--tiff_only: ignoring manifest input_format={session['input_format']!r}[/yellow]")
            session['input_format'] = 'tiff'
        input_format = session.get('input_format', 'sbx')

    #test that reference round exists
    reference_round = manifest['HCR_confocal_imaging']['reference_round']
    for i in manifest['HCR_confocal_imaging']['rounds']:
        if i['round'] == reference_round:
            break
    else:
        raise Exception(f"reference round was not found {reference_round} is not in rounds")

    if not args.only_hcr and input_format == 'tiff':
        # Validate tiff inputs exist
        twop_dir = base_path / mouse_name / '2P'
        planes = session.get('functional_planes', session.get('functional_plane', []))
        for plane in planes:
            tiff_path = twop_dir / f'plane_{plane}.tiff'
            if not tiff_path.exists():
                raise FileNotFoundError(f"Expected 2P tiff input not found: {tiff_path}")
            # Detect pre-stitched hires from presence of plane_{N}_hires.tiff
            if (twop_dir / f'plane_{plane}_hires.tiff').exists():
                has_hires = True

    elif not args.only_hcr and input_format == 'suite2p':
        # Validate suite2p folder exists
        suite2p_path = base_path / mouse_name / '2P' / 'suite2p' / 'plane0' / 'ops.npy'
        user_input_missing(np.array([[suite2p_path, os.path.exists(suite2p_path)]]), 'Suite2p path is missing, do you wish to continue?', color='pink')

    elif not args.only_hcr:
        # Default SBX mode — original validation
        date_two_photons = session['date']
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

    # Verify anatomical runs configuration. Hires tile stitching from raw
    # sbx tiles is orthogonal to the functional input source — it runs for
    # both sbx and suite2p modes whenever anatomical_hires_*_runs are
    # declared. Tiff mode skips sbx-based stitching entirely (users supply
    # a pre-stitched plane_{N}_hires.tiff instead) and has_hires is set
    # above based on that file's presence.
    if not args.only_hcr and input_format != 'tiff':
        has_lowres_green = len(session.get('anatomical_lowres_green_runs', [])) > 0
        has_hires_green = len(session.get('anatomical_hires_green_runs', [])) > 0

        if has_hires_green:
            assert not has_lowres_green, "Cannot have both lowres and hires runs"
            assert len(session['anatomical_hires_green_runs'])==len(session['anatomical_hires_red_runs']), "Number of hires green and red runs do not match"
            has_hires = True
            # Validate hires tile .sbx files exist (suite2p mode skips sbx
            # validation above but still needs the tile sbx files for stitching)
            if input_format == 'suite2p':
                date_two_photons = session.get('date')
                if date_two_photons:
                    tile_check = []
                    for run in session['anatomical_hires_green_runs'] + session['anatomical_hires_red_runs']:
                        tile_sbx = base_path / mouse_name / '2P' / f'{mouse_name}_{date_two_photons}_{run}' / f'{mouse_name}_{date_two_photons}_{run}.sbx'
                        tile_check.append([tile_sbx, os.path.exists(tile_sbx)])
                    user_input_missing(np.array(tile_check), 'Some hires tile sbx files are missing, do you wish to continue?', color='red')
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


