import argparse
import os
import sys
import logging
from src.meta import main_pipeline_manifest
from src import tiling as tl
from pathlib import Path

# supported HCR probs
HCR_probs = ['CCK', 'CHAT', 'CHRIMSON', 'DAPI', 'GCAMP', 'GLP1R', 'NPR3', 'PDYN', 'RORB', 'TAC1', 'VGAT']

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


def verify_manifest(manifest):
    '''
    Verify that the json file is valid
    '''
    #test that only one 2P session is present
    assert len(manifest['two_photons_imaging']['sessions'])==1, 'only support one 2P sessions'
    
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
    tl.stitch_tiles(manifest, session)

    # step 4-AM: register the HCR data round to round
    tl.register_rounds(manifest)
    









if __name__ == "__main__":
    args = {'manifest': 'examples/RG026.hjson',
            'unwarp_file': 'examples/unwarp_config.json',
            'unwarp_steps': 10}
    main(args)
    