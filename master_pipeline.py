import argparse
import logging
from pathlib import Path
import argparse
from rich import print as rprint

from src import functional as fc
from src import tiling as tl
from src import registrations as rf
from src import meta as mt
from src import segmentation as sg


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




# https://drive.google.com/file/d/1HZNh7aqJr-vTsLsSGlFmi11HuEvYlgZ-/view?usp=sharing

def main(args = None):
    '''
    We can either start main with arguments or from command line 
    See README.md for more information
    '''

    args = parse_input_args(args)

    # step 1-A: parse the manifest file
    manifest = mt.main_pipeline_manifest(args.manifest)
    specs = mt.verify_manifest(manifest)

    session = manifest['two_photons_imaging']['sessions'][0]
    tl.process_session_sbx(manifest, session)

    # step 2-A: unwarp 2P anatomical_runs
    tl.unwarp_tiles(manifest, session)

    # step 3-M: stitch the tiles
    tl.stitch_tiles_and_rotate(manifest, session)

    # step 4-AM: register the HCR data round to round
    rf.register_rounds(manifest, manifest_path=args.manifest)


    ### move to the start!!! fix also the README after you do that!
    fc.extract_suite2p_registered_planes(manifest, session)
    
    # Run cellpose on HCR rounds
    sg.run_cellpose(manifest)

    # extract probs values from cellpose segmentation
    sg.extract_probs_intensities(manifest)

    sg.extract_electrophysiology_intensities(manifest, session)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, help='Path to the pipeline manifest file e.g. examples/CIM132.hjson')
    args = parser.parse_args()
    #args = {'manifest': 'examples/CIM132.hjson'}
    main(args)
    
