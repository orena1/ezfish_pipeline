"""Functional-input importers for the 2P arm of the pipeline.

Step 1 of the ingest-mode refactor (see docs/refactor_easipass_ingest.md): a
behavior-preserving seam. Today this is a thin dispatcher over the existing
format-specific loaders in ``functional.py``; later steps grow the shared
``FunctionalInput`` contract (movie accessor, compute/accept segmentation) here
so the rest of the pipeline becomes format-agnostic after ingest.
"""
from pathlib import Path
import numpy as np

try:
    from . import functional as fc
    from .meta import output_root, rprint
except ImportError:  # allow import when src/ is on the path directly (mirrors segmentation.py)
    import functional as fc
    from meta import output_root, rprint


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


def _segmentation_source(full_manifest: dict, session: dict) -> str:
    """'compute' (cellpose, default — reproduces the paper) or 'accept' (reuse the
    user's Suite2p ROIs). Read from session then params; anything else -> 'compute'
    so existing runs are unchanged until a manifest opts in with
    ``segmentation_source: accept``."""
    src = session.get('segmentation_source') or full_manifest.get('params', {}).get('segmentation_source')
    return src if src in ('compute', 'accept') else 'compute'


def load_functional_masks(full_manifest: dict, session: dict):
    """Return the 2P segmentation ``_seg.npy`` for the current plane.

    compute (default): cellpose on the mean image.
    accept: rasterize the user's Suite2p ROIs into the SAME ``_seg.npy`` artifact,
    so matching/registration/merge are untouched.
    """
    if _segmentation_source(full_manifest, session) == 'accept':
        return accept_suite2p_masks(full_manifest, session)
    from . import segmentation as sg  # lazy to avoid any import cycle
    return sg.extract_2p_cellpose_masks(full_manifest, session)


def accept_suite2p_masks(full_manifest: dict, session: dict):
    """Rasterize the user's Suite2p ROIs into the cellpose ``_seg.npy`` artifact.

    Reads ``stat.npy``/``iscell.npy``/``ops.npy`` from ``2P/suite2p/plane{N}``,
    paints each iscell ROI into a labeled image with **single ownership** (a shared
    pixel goes to the ROI with the higher ``lam``), and writes it where cellpose
    would. **Mask label = Suite2p ROI index + 1** (0 = background), so the user
    joins their own ``F.npy`` by ``label - 1``. Traces are NOT handled here.

    Assumes ROIs share the Suite2p ``ops['meanImg']`` pixel space (true for a
    plain Suite2p run). If a low-res unwarp is applied to the mean, ROIs would need
    the same transform — that is the Scanbox case, which uses ``compute`` instead.
    """
    manifest = full_manifest['data']
    plane = session['functional_plane'][0]

    cellpose_path = output_root(full_manifest) / '2P' / 'cellpose'
    seg_path = cellpose_path / f'lowres_meanImg_C0_plane{plane}_seg.npy'
    if seg_path.exists():
        rprint(f"[dim]2P Suite2p ROIs plane {plane}: exists[/dim]")
        return seg_path

    s2p_plane = Path(manifest['base_path']) / manifest['mouse_name'] / '2P' / 'suite2p' / f'plane{plane}'
    for f in ('stat.npy', 'iscell.npy', 'ops.npy'):
        if not (s2p_plane / f).exists():
            raise FileNotFoundError(f"segmentation_source=accept needs Suite2p output: missing {s2p_plane / f}")
    stat = np.load(s2p_plane / 'stat.npy', allow_pickle=True)
    iscell = np.load(s2p_plane / 'iscell.npy', allow_pickle=True)
    ops = np.load(s2p_plane / 'ops.npy', allow_pickle=True).item()
    Ly, Lx = int(ops['Ly']), int(ops['Lx'])

    mask_img = np.zeros((Ly, Lx), dtype=np.uint16)
    best_lam = np.zeros((Ly, Lx), dtype=np.float32)
    n_cells = 0
    for i, roi in enumerate(stat):
        if not iscell[i, 0]:
            continue
        ypix = np.asarray(roi['ypix']); xpix = np.asarray(roi['xpix'])
        lam = np.asarray(roi['lam'], dtype=np.float32)
        win = lam > best_lam[ypix, xpix]         # keep only pixels this ROI wins
        yy, xx = ypix[win], xpix[win]
        mask_img[yy, xx] = i + 1                  # label = Suite2p ROI id + 1
        best_lam[yy, xx] = lam[win]
        n_cells += 1

    cellpose_path.mkdir(parents=True, exist_ok=True)
    np.save(str(seg_path), {'masks': mask_img, 'img': ops.get('meanImg')})
    rprint(f"  [green]Accepted {n_cells} Suite2p ROIs[/green] as 2P masks for plane {plane} "
           f"(label = ROI id + 1; join F.npy by label-1)")
    return seg_path
