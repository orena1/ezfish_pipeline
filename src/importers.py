"""Functional-input importers for the 2P arm of the pipeline.

Step 1 of the ingest-mode refactor (see docs/refactor_easipass_ingest.md): a
behavior-preserving seam. Today this is a thin dispatcher over the existing
format-specific loaders in ``functional.py``; later steps grow the shared
``FunctionalInput`` contract (movie accessor, compute/accept segmentation) here
so the rest of the pipeline becomes format-agnostic after ingest.
"""
try:
    from . import functional as fc
except ImportError:  # allow import when src/ is on the path directly (mirrors segmentation.py)
    import functional as fc


def load_functional_mean(full_manifest: dict, session: dict):
    """Produce the functional low-res mean image for the current plane.

    Dispatches on the session's ``input_format``. Behaviour-identical to the
    inline branch that previously lived in ``master_pipeline.process_plane`` --
    each importer writes the standard ``lowres_meanImg_*`` artifact that
    ``extract_2p_cellpose_masks`` and the downstream registration read.
    """
    input_format = fc._get_input_format(session)
    if input_format == 'tiff':
        fc.prepare_tiff_input(full_manifest, session)
    elif input_format == 'suite2p':
        fc.prepare_suite2p_input(full_manifest, session)
    else:
        has_red = bool(session.get('anatomical_lowres_red_runs'))
        fc.extract_suite2p_registered_planes(
            full_manifest, session, combine_with_red=has_red)
