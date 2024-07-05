import argparse
import os
import sys
import logging
from src.meta import main_pipeline_manifest
from src import tiling as tl
from pathlib import Path


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





def main(args = None):
    '''
    We can either start main with arguments or from command line 
    The following steps are conducted:
    1. Parse the manifest file
    2. unwarp 2P anatomical_runs
    '''

    args = parse_input_args(args)

    # step 1: parse the manifest file
    manifest = main_pipeline_manifest(args['manifest'])
    assert len(manifest['two_photons_imaging']['sessions'])==1, 'only support one 2P sessions'

    session = manifest['two_photons_imaging']['sessions'][0]
    tl.process_session_sbx(manifest, session)

    # step 2: unwarp 2P anatomical_runs
    for i in range(1,1+len(session['anatomical_hires_green_runs'])):
        warp_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / '2P' / 'tile' / 'warped' / f'stack_warped_C12_{i:03}.tiff'
        unwarp_path = Path(manifest['base_path']) / manifest['mouse_name'] / 'OUTPUT' / '2P' / 'tile' / 'unwarped' / f'stack_unwarped_C12_{i:03}.tiff'   
        tl.unwarp_tile(warp_path, 
                       session['unwarp_config'], 
                       session['unwarp_steps'], 
                       unwarp_path)
    
    # step 2: create the base directories
    #?dsdf









if __name__ == "__main__":
    args = {'manifest': 'examples/RG026.json',
            'unwarp_file': 'examples/unwarp_config.json',
            'unwarp_steps': 10}
    main(args)
    