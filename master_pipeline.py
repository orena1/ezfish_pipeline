print("Running main pipeline...", flush=True)
import argparse
import logging
from pathlib import Path
import argparse
from rich import print as rprint
from src import functional as fc
# from src import tiling as tl
from src import registrations as rf
from src import meta as mt
from src import segmentation as sg
print("Running main Finish loading", flush=True)
# This is the main pipeline script that runs the entire pipeline

# https://drive.google.com/file/d/1HZNh7aqJr-vTsLsSGlFmi11HuEvYlgZ-/view?usp=sharing

def main(args = None):
    '''
    We can either start main with arguments or from command line 
    See README.md for more information
    '''
    
    # Parse the manifest file
    full_manifest = mt.main_pipeline_manifest(args.manifest)
    specs, has_hires = mt.verify_manifest(full_manifest, args)
    session = full_manifest['data']['two_photons_imaging']['sessions'][0]
    process_plane(args, full_manifest, session, has_hires)

    # if args.only_hcr:
    #     process_plane(args, full_manifest, session, has_hires)

    # # process_plane(args, full_manifest, session, has_hires, reference_plane=reference_plane)


def process_plane(args, full_manifest, session, has_hires, reference_plane=None):

    if not args.only_hcr:

        if has_hires:
            # Rotate hires file and save in correct directory.
            fc.hires_rotate_and_save(full_manifest, session)

            # Extract registered suite2p planes
            fc.extract_functional_planes(full_manifest, session)
        else:
            # Will need to rotate here
            fc.extract_functional_planes(full_manifest, session, combine_with_red=True)

    # Register the HCR data round to round
    rf.register_rounds(full_manifest)

    # Run cellpose on HCR rounds
    sg.run_cellpose(full_manifest)

    # Extract probs values from cellpose segmentation
    sg.extract_probs_intensities(full_manifest)

    if not args.only_hcr:
        # Extract cellpose masks from 2p images
        sg.extract_2p_cellpose_masks(full_manifest, session)

        # Extract electrophysiology intensities from 2p images
        sg.extract_electrophysiology_intensities(full_manifest, session)

    sg.align_masks(full_manifest, session, only_hcr=args.only_hcr, reference_plane=reference_plane)
    # Merge aligned masks to single files
    sg.merge_masks(full_manifest, session, only_hcr=args.only_hcr)

    rprint('Pipeline completed successfully!')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, help='Path to the pipeline manifest file e.g. examples/CIM132.hjson')
    parser.add_argument('--only_hcr', action='store_true', help='Run only HCR part of the pipeline')
    parser.add_argument('--add_planes', action='store_true', help='Process additional functional planes beyond the reference plane')

    args = parser.parse_args()
    #args = {'manifest': 'examples/CIM132.hjson'}
    main(args)
    
