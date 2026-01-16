import argparse
from pathlib import Path
from rich import print as rprint
from src import functional as fc
from src import tiling as tl
from src import registrations as rf
from src import registrations_landmarks as rf_landmarks
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

    # Get automation config (defaults to 'manual' if not specified)
    automation = mt.get_automation_config(full_manifest.get('params', {}))
    if automation['twop_to_hcr'] == 'auto':
        rprint(f"[bold cyan]Automation enabled:[/bold cyan] twop_to_hcr={automation['twop_to_hcr']}")

    # HCR processing - runs once regardless of number of 2P planes
    rf.register_rounds(full_manifest)
    sg.run_cellpose(full_manifest)
    sg.extract_probs_intensities(full_manifest)

    if args.only_hcr:
        # For HCR-only mode, just do align_masks and merge
        sg.align_masks(full_manifest, session, only_hcr=True, reference_plane=None)
        sg.merge_masks(full_manifest, session, only_hcr=True)

    else:
        session = full_manifest['data']['two_photons_imaging']['sessions'][0]

        # Get all planes to process
        # Support both old and new manifest formats
        if 'functional_planes' in session:
            # New format: all planes in one list
            all_planes = session['functional_planes']
        else:
            # Old format: functional_plane + optional additional_functional_planes
            all_planes = list(session['functional_plane'])
            if 'additional_functional_planes' in session:
                all_planes.extend(session['additional_functional_planes'])

        # First plane is the reference (used for HCR alignment and landmarks)
        reference_plane = all_planes[0]

        # Process 2P data for each plane
        for plane in all_planes:
            session['functional_plane'] = [plane]
            process_plane(full_manifest, session, has_hires)

        # Now do low-res to high-res registration for ALL planes at once
        # Method depends on automation config: 'landmarks' (default) or 'auto' (SIFT-based)
        if has_hires:
            # Temporarily set session to have all planes for registration
            session_with_all_planes = session.copy()
            if 'functional_planes' in full_manifest['data']['two_photons_imaging']['sessions'][0]:
                # Already has functional_planes
                pass
            else:
                # Add additional_functional_planes to session for registration
                session_with_all_planes['functional_plane'] = [reference_plane]
                session_with_all_planes['additional_functional_planes'] = [p for p in all_planes if p != reference_plane]

            lowres_method = automation.get('lowres_to_hires', 'manual')
            if lowres_method == 'auto':
                rprint(f"\n[bold cyan]Running automated (SIFT) low-res to high-res registration[/bold cyan]\n")
                rf.register_lowres_to_hires(full_manifest, session_with_all_planes)
            else:
                rprint(f"\n[bold cyan]Running landmark-based low-res to high-res registration[/bold cyan]\n")
                rf_landmarks.register_lowres_to_hires_landmarks(full_manifest, session_with_all_planes)

        # Now run 2P-to-HCR registration for each plane (needs transformed masks from above)
        for plane in all_planes:
            session['functional_plane'] = [plane]
            rprint(f"\n[bold yellow]Running 2P→HCR registration for plane {plane}[/bold yellow]\n")
            try:
                rf.twop_to_hcr_registration(
                    full_manifest, session, has_hires,
                    automation_enabled=(automation['twop_to_hcr'] == 'auto')
                )
                sg.extract_electrophysiology_intensities(full_manifest, session)
            except FileNotFoundError as e:
                rprint(f"[red]Plane {plane} failed - required file not created: {e}[/red]")
                continue

        # Now align 2P masks to HCR and merge (needs twop_aligned_3d.tiff from above)
        for plane in all_planes:
            session['functional_plane'] = [plane]
            rprint(f"\n[bold yellow]Aligning and merging masks for plane {plane}[/bold yellow]\n")
            # Determine if this is the reference plane
            plane_reference = None if plane == reference_plane else reference_plane
            sg.align_masks(full_manifest, session, only_hcr=args.only_hcr, reference_plane=plane_reference)
            sg.merge_masks(full_manifest, session, only_hcr=args.only_hcr)

        rprint('\n' + '='*80)
        rprint('[bold green]Pipeline completed successfully for all planes![/bold green]')
        rprint('='*80)


def process_plane(full_manifest, session, has_hires):
    """Process 2P data for a single plane (no HCR processing - that's done once in main)."""
    plane = session['functional_plane'][0]
    rprint(f"\n[bold green]Processing 2P plane {plane}[/bold green]")

    if has_hires:
        tl.process_session_sbx(full_manifest, session)
        tl.unwarp_tiles(full_manifest, session)
        tl.stitch_tiles_and_rotate(full_manifest, session)
        fc.extract_suite2p_registered_planes(full_manifest, session)
    else:
        fc.extract_suite2p_registered_planes(full_manifest, session, combine_with_red=True)

    # Extract cellpose masks from 2p images
    sg.extract_2p_cellpose_masks(full_manifest, session)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, help='Path to the pipeline manifest file e.g. examples/CIM132.hjson')
    parser.add_argument('--only_hcr', action='store_true', help='Run only HCR part of the pipeline')

    args = parser.parse_args()
    #args = {'manifest': 'examples/CIM132.hjson'}
    main(args)
    
