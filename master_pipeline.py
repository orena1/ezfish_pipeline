import argparse
from pathlib import Path
from src import functional as fc
from src import tiling as tl
from src import registrations as rf
from src import registrations_landmarks as rf_landmarks
from src import meta as mt
from src import segmentation as sg
from src import importers as im
from src.meta import rprint

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

    if getattr(args, 'tiff_only', False):
        rprint("[bold cyan]in vivo 2P -> ex vivo volume alignment[/bold cyan]")
    elif args.only_hcr:
        rprint("[bold cyan]HCR rounds + segmentation[/bold cyan]")

    # Get automation config (defaults to 'manual' if not specified)
    automation = mt.get_automation_config(full_manifest.get('params', {}))
    if automation['twop_to_hcr'] == 'auto':
        rprint(f"[bold cyan]Automation enabled:[/bold cyan] twop_to_hcr={automation['twop_to_hcr']}")

    # HCR processing - runs once regardless of number of 2P planes.
    # Cellpose runs first (on each acquired round), then HCR-HCR registration, then
    # the labels are aligned into the HCR01 frame for matching, then intensities are
    # measured on the acquired rounds.
    sg.run_cellpose(full_manifest)
    rf.register_rounds(full_manifest)
    rf.align_masks_to_reference(full_manifest)
    sg.extract_probe_intensity(full_manifest)

    if args.only_hcr:
        # For HCR-only mode, just do align_masks and merge
        sg.align_masks(full_manifest, session, only_hcr=True, reference_plane=None)
        # Hybrid HCR↔HCR matcher (best-plane IoU overlap + soma repair); augments
        # the per-round CSVs so merge uses the consensus pick (no-op if disabled).
        sg.align_somaprint_hcr(full_manifest, session, only_hcr=True)
        sg.merge_masks(full_manifest, session, only_hcr=True)

    else:
        # TODO: Refactor plane iteration - currently we copy session and mutate
        # functional_plane each iteration, while functions also read from the original
        # manifest to get reference_plane. This works but is confusing. Consider passing
        # current_plane and reference_plane as explicit parameters instead.
        session = full_manifest['data']['two_photon_imaging']['sessions'][0].copy()

        # Get all planes to process
        # Support both old and new manifest formats
        if 'functional_planes' in session:
            # New format: all planes in one list
            all_planes = list(session['functional_planes'])
        else:
            # Old format: functional_plane + optional additional_functional_planes
            all_planes = list(session['functional_plane'])
            if 'additional_functional_planes' in session:
                all_planes.extend(session['additional_functional_planes'])

        # First plane is the reference (used for HCR alignment and landmarks)
        reference_plane = all_planes[0]

        # Process 2P data for each plane. Cellpose runs eagerly per plane; the
        # verification prompt is consolidated into a single Enter press after
        # the loop so the user doesn't get prompted between every plane.
        cellpose_seg_files = []
        for plane in all_planes:
            session['functional_plane'] = [plane]
            seg_file = process_plane(full_manifest, session, has_hires)
            cellpose_seg_files.append((plane, seg_file))
        sg.verify_2p_cellpose_segmentations(cellpose_seg_files)

        # Now do low-res to high-res registration for ALL planes at once
        # Method depends on automation config: 'landmarks' (default) or 'auto' (SIFT-based)
        if has_hires:
            # Temporarily set session to have all planes for registration
            session_with_all_planes = session.copy()
            if 'functional_planes' in full_manifest['data']['two_photon_imaging']['sessions'][0]:
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
        else:
            # Standard mode: create rotated masks (in hi-res mode, register_lowres_to_hires does this)
            session_with_all_planes = session.copy()
            if 'functional_planes' not in full_manifest['data']['two_photon_imaging']['sessions'][0]:
                session_with_all_planes['functional_plane'] = [reference_plane]
                session_with_all_planes['additional_functional_planes'] = [p for p in all_planes if p != reference_plane]
            rf.create_rotated_masks_for_standard_mode(full_manifest, session_with_all_planes)

        # Now run 2P-to-HCR registration for each plane (needs transformed masks from above)
        for plane in all_planes:
            session['functional_plane'] = [plane]
            rprint(f"\n[bold yellow]Running 2P→HCR registration for plane {plane}[/bold yellow]\n")
            rf.twop_to_hcr_registration(
                full_manifest, session, has_hires,
                automation_enabled=(automation['twop_to_hcr'] == 'auto')
            )
            input_format = fc._get_input_format(session)
            if input_format not in ('tiff', 'suite2p'):
                sg.extract_electrophysiology_intensities(full_manifest, session)

        # Now align 2P masks to HCR and merge (needs twop_aligned_3d.tiff from above)
        for plane in all_planes:
            session['functional_plane'] = [plane]
            rprint(f"\n[bold yellow]Aligning and merging masks for plane {plane}[/bold yellow]\n")
            # Determine if this is the reference plane
            plane_reference = None if plane == reference_plane else reference_plane
            sg.align_masks(full_manifest, session, only_hcr=args.only_hcr, reference_plane=plane_reference)
            # Augment IoU CSV with somaprint columns (geometric matcher,
            # parallel to IoU; no-op if disabled in manifest).
            sg.align_somaprint(full_manifest, session, only_hcr=args.only_hcr)
            # Hybrid HCR↔HCR matcher → consensus round_{R}_mask. The per-round CSVs
            # are (re)generated only on the reference-plane pass (plane_reference is
            # None), which runs first, so populate them once here; idempotent.
            if plane_reference is None:
                sg.align_somaprint_hcr(full_manifest, session, only_hcr=args.only_hcr)
            sg.merge_masks(full_manifest, session, only_hcr=args.only_hcr)

        sg.print_match_summary(full_manifest, all_planes)

        rprint('\n' + '='*80)
        rprint(f"[bold green]Pipeline completed successfully for {full_manifest['data']['mouse_name']}![/bold green]")
        rprint('='*80)


def process_plane(full_manifest, session, has_hires):
    """Process 2P data for a single plane (no HCR processing - that's done once in main).

    Functional lowres mean extraction (driven by input_format) and hires tile
    stitching (driven by has_hires) are orthogonal: any input_format can opt
    into hires stitching by declaring anatomical_hires_*_runs in the manifest.
    """
    plane = session['functional_plane'][0]
    input_format = fc._get_input_format(session)
    rprint(f"\n[bold green]Processing 2P plane {plane}[/bold green]")

    # 1. Hires tile stitching first (sbx + suite2p modes). Running stitching
    # ahead of the lowres rotation prompt means the user gets a 2-channel
    # (green + red) stitched preview to judge the flip/rotation against HCR.
    # Tiff mode skips here because the user supplies a pre-stitched
    # plane_{N}_hires.tiff which prepare_tiff_input handles.
    if has_hires and input_format != 'tiff':
        tl.process_session_sbx(full_manifest, session)
        tl.unwarp_tiles(full_manifest, session)
        tl.stitch_tiles_and_rotate(full_manifest, session)

    # 2. Functional lowres mean image. If stitching prompted and wrote a
    # *_rotated.tiff above, the lowres rotation step below will skip the
    # prompt and pick up the same rotation_config from the updated manifest.
    # Dispatch lives in src/importers.py (behavior-identical across formats).
    im.load_functional_mean(full_manifest, session)

    # 2P segmentation → seg.npy path (collected across planes; verified once after
    # the loop). Source is 'compute' (cellpose, default) or 'accept' (rasterize the
    # user's Suite2p ROIs); dispatch lives in src/importers.py.
    return im.load_functional_masks(full_manifest, session)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, help='Path to the pipeline manifest file e.g. examples/CIM132.hjson')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--only_hcr', action='store_true', help='HCR rounds + segmentation only; no 2P data')
    mode.add_argument('--tiff_only', action='store_true',
                      help='Align a pre-processed 2P mean image (TIFF) to an ex vivo volume. '
                           'See examples/demo_tiff_minimal.hjson.')

    args = parser.parse_args()
    #args = {'manifest': 'examples/CIM132.hjson'}
    main(args)
    
