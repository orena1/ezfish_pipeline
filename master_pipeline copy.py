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

# This is the main pipeline script that runs the entire pipeline

# https://drive.google.com/file/d/1HZNh7aqJr-vTsLsSGlFmi11HuEvYlgZ-/view?usp=sharing

def main(args = None):
    '''
    We can either start main with arguments or from command line 
    See README.md for more information
    '''
    session = []

    # Parse the manifest file
    full_manifest = mt.main_pipeline_manifest(args.manifest)
    specs, has_hires = mt.verify_manifest(full_manifest, args)
    if args.only_hcr:
        process_plane(args, full_manifest, session, has_hires)

    else:
        session = full_manifest['data']['two_photons_imaging']['sessions'][0]
        
        if args.add_planes:
            reference_plane = session['functional_plane'][0]
            functional_planes = session['additional_functional_planes']
        else:
            reference_plane = None
            functional_planes = list(session['functional_plane'])
        
        for plane in functional_planes:
            session['functional_plane'] = [plane]
            rprint(f"\n[bold yellow]Processing plane {plane}{f' (ref: {reference_plane})' if reference_plane else ''}[/bold yellow]\n")
            process_plane(args, full_manifest, session, has_hires, reference_plane=reference_plane)


def process_plane(args, full_manifest, session, has_hires, reference_plane=None):

    if not args.only_hcr:

        if has_hires:
            # in case of highres anatomical runs, we need to process them first
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
    
