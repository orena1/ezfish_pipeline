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
    argparser.add_argument('only_hcr', action='store_true', help='Run only HCR part of the pipeline')
    
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
    session =[]

    # step 1-A: parse the manifest file
    manifest = mt.main_pipeline_manifest(args.manifest)
    specs = mt.verify_manifest(manifest, args)
    
    if not args.only_hcr:
        session = manifest['two_photons_imaging']['sessions'][0]
        tl.process_session_sbx(manifest, session)
    
        # # step 2-A: unwarp 2P anatomical_runs
        tl.unwarp_tiles(manifest, session)

        # # step 3-M: stitch the tiles
        tl.stitch_tiles_and_rotate(manifest, session)

        # ### moved temporarily to generate mean image from suite2p
        fc.extract_suite2p_registered_planes(manifest, session)
        
    # # step 4-AM: register the HCR data round to round
    rf.register_rounds(manifest, manifest_path=args.manifest)

    # # Run cellpose on HCR rounds
    sg.run_cellpose(manifest)

    # # extract probs values from cellpose segmentation
    sg.extract_probs_intensities(manifest)

    if not args.only_hcr:
        sg.extract_electrophysiology_intensities(manifest, session)

    sg.align_masks(manifest, session, only_hcr=args.only_hcr)

    # merge aligned masks to single files
    sg.merge_masks(manifest, session, only_hcr=args.only_hcr)

    rprint('Pipeline completed successfully!')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, help='Path to the pipeline manifest file e.g. examples/CIM132.hjson')
    parser.add_argument('--only_hcr', action='store_true', help='Run only HCR part of the pipeline')
    args = parser.parse_args()
    #args = {'manifest': 'examples/CIM132.hjson'}
    main(args)
    
