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
    argparser.add_argument('unwarp_file', help='Path to the file containing the unwarping parameters')
    argparser.add_argument('unwarp_steps', type=int, help='how many steps for unwarping')



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

    tl.process_session_sbx(manifest, manifest['two_photons_imaging']['sessions'][0])

    # step 2: unwarp 2P anatomical_runs
    tl.unwarp_tile(manifest['anatomical_runs']['input'], args.unwarp_file, args.unwarp_steps, manifest['anatomical_runs']['output'])
    
    # step 2: create the base directories
    #?dsdf









if __name__ == "__main__":
    args = {'manifest': 'examples/RG026.json',
            'unwarp_file': 'examples/unwarp_config.json',
            'unwarp_steps': 10}
    main(args)
    