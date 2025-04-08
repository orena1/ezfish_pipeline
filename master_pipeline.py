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

from bigstream.piecewise_transform import distributed_apply_transform

# This is the main pipeline script that runs the entire pipeline

# https://drive.google.com/file/d/1HZNh7aqJr-vTsLsSGlFmi11HuEvYlgZ-/view?usp=sharing

def main(args = None):
    '''
    We can either start main with arguments or from command line 
    See README.md for more information
    '''
    session =[]

    # Parse the manifest file
    full_manifest = mt.main_pipeline_manifest(args.manifest)
    specs, has_hires = mt.verify_manifest(full_manifest, args)
    
    if not args.only_hcr:
        session = full_manifest['data']['two_photons_imaging']['sessions'][0]
        if has_hires:
            tl.process_session_sbx(full_manifest, session)

            # Unwrap 2P anatomical_runs
            tl.unwarp_tiles(full_manifest, session)

            # Stitch the tiles sdf sd
            tl.stitch_tiles_and_rotate(full_manifest, session)

            # extract registered suite2p planes
            fc.extract_suite2p_registered_planes(full_manifest, session)
        else:
            fc.extract_suite2p_registered_planes(full_manifest, session, combine_with_red=True)
        
    # Register the HCR data round to round
    rf.register_rounds(full_manifest)

    # Run cellpose on HCR rounds
    sg.run_cellpose(full_manifest)

    # Extract probs values from cellpose segmentation
    sg.extract_probs_intensities(full_manifest)

    if not args.only_hcr:
        sg.extract_electrophysiology_intensities(full_manifest, session)

    sg.align_masks(full_manifest, session, only_hcr=args.only_hcr)

    # Merge aligned masks to single files
    sg.merge_masks(full_manifest, session, only_hcr=args.only_hcr)

    rprint('Pipeline completed successfully!')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, help='Path to the pipeline manifest file e.g. examples/CIM132.hjson')
    parser.add_argument('--only_hcr', action='store_true', help='Run only HCR part of the pipeline')
    args = parser.parse_args()
    #args = {'manifest': 'examples/CIM132.hjson'}
    main(args)
    
