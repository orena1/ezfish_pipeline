"""
hcr_centroid_registration.py -- in-pipeline HCR round-to-round registration driven by
cellpose nucleus CENTROIDS (no notebook farming).

Two stages, both reusing existing engines (no new registration math):

  GLOBAL  -- a cap-free 2D-context centroid affine, ported verbatim from the validated
             diagnostic `centroid_global_batched` (HCR_global_centroid_diagnostic.ipynb /
             gen_hcr_global_centroid_diagnostic.py). The bigstream no-prior feature-point
             matcher is deliberately NOT used here: the diagnostic showed it falls into the
             identity/local-minimum trap at large inter-round offset. The 2D-um-context
             matcher + cv2 RANSAC is the robust winner across cortex + PBN.

  LOCAL   -- bigstream's distributed_piecewise_alignment_pipeline with the cellpose
             centroids INJECTED as fix_spots_global / mov_spots_global and the global affine
             supplied as the per-block prior (static_transform_list). Ported from
             _profile_local_reg.coverage_sweep, including the adaptive_cc_contexts edge
             monkeypatch and the loosened blob-era spot/match floors.

Outputs land in the EXISTING registrations/ layout so registration_apply(),
verify_rounds() and align_masks_to_reference() need no changes:

    OUTPUT/HCR/registrations/HCR{N}_to_HCR{ref}/<global_tag>/<local_tag>/
        _affine.mat     (4x4 fix->mov physical affine; np.savetxt)
        deform.zarr     (local deformation field)
        overlay_warped.tiff     (QC: 2-channel ImageJ composite, warped mov + fix, at lowres)

`<local_tag>` always contains 'bs{Y}x{X}x{Z}' so _load_round_transform's blocksize regex
parses it. The selected_registrations entry written into the manifest is
'<global_tag>/<local_tag>'.

Public entry: run_hcr_centroid_registration(full_manifest) -> per-round ranked results.
Selection of the winner + manifest write-back is handled by register_rounds() in
registrations.py; this module only computes, scores, and writes candidates.

NOTE: blob-based registration is intentionally NOT handled here -- it stays on the legacy
notebook path (register_rounds), so removing blob later is a single-branch deletion.
"""
import shutil
import hashlib
import contextlib
import numpy as np
from pathlib import Path
from tifffile import imread as tif_imread, imwrite as tif_imwrite

try:
    from .meta import rprint, output_root, get_hcr_to_hcr_registration_config, get_round_folder_name
    from .registrations_utils import resolve_hcr_resolution
    from .bigstream_functions import get_registration_score
except ImportError:  # running in a notebook / as a flat module
    from meta import rprint, output_root, get_hcr_to_hcr_registration_config, get_round_folder_name
    from registrations_utils import resolve_hcr_resolution
    from bigstream_functions import get_registration_score


# --------------------------------------------------------------------------- #
#  CONFIG  (hjson params -> typed config, with defaults)
# --------------------------------------------------------------------------- #
# Defaults chosen from the diagnostic's consistency table (60um global is near-universal)
# and the local threshold-sweep findings (loosened floors 8/4, cc=(12,12,2), mt=0.3).
_GLOBAL_DEFAULTS = dict(
    method="centroid",
    context_radius_um=[60.0],        # a few values may be listed; fast, best is auto-picked
    match_threshold=0.3,
    select_metric="mi",              # "mi" (default) | "median_resid" | "above_chance"
    inlier_gate_um=12.0,             # reciprocal-NN gate for a "matched" cell (~ a cell spacing)
    ransac_threshold_um=15.0,        # cv2.estimateAffine3D inlier threshold
)
_LOCAL_DEFAULTS = dict(
    method="centroid",
    # Trimmed to the two that win in practice (coarse-robust + fine-detail); the dropped
    # [300,300,15] middle almost never beat both. Coarse-first so the cheap config runs first.
    blocksize=[[400, 400, 15], [200, 200, 10]],
    overlap=0.5,                     # A/B (overlap_ab_sweep.py) showed 0.25 costs quality on BOTH
                                     # samples: JS082 medResid 4.60->5.04um (frac<5 53->50%),
                                     # CIM131 11.62->16.49um (frac<5 34->26%). 0.25 is ~5x cheaper
                                     # (efficiency_sweep) but NOT free -- opt-in speed knob only.
    n_workers=1,                     # block-level parallelism; >1 fans blocks across processes
    threads_per_worker=8,
    context_radius=[12, 12, 2],      # voxels (current local style)
    context_radius_um=None,          # OPT-IN: global-style um window for blocks (experiment knob)
    match_threshold=0.3,
    max_spot_match_distance_um=60.0,
    count_floor=8,                   # loosened spot-count floor (blob era was 100)
    match_floor=4,                   # loosened point-matches floor (blob era was 50)
    adaptive_edges=True,
    nspots=5000,
    blob_sizes=[6, 30],              # REQUIRED by bigstream's signature; ignored when spots injected
)


def get_centroid_config(params):
    """Parse the centroid global/local config out of the manifest params, applying defaults.

    Returns (global_cfg, local_cfg, downsampling) or (None, None, ds) when the manifest has
    no global/local block (-> caller falls back to the legacy notebook flow).
    """
    ds = list(get_hcr_to_hcr_registration_config(params).get('downsampling', [3, 3, 2]))
    raw = params.get('HCR_to_HCR_registration', {}) or {}    # read global/local from the raw block
    g_raw, l_raw = raw.get('global'), raw.get('local')
    if g_raw is None and l_raw is None:
        return None, None, ds
    g = {**_GLOBAL_DEFAULTS, **(g_raw or {})}
    l = {**_LOCAL_DEFAULTS, **(l_raw or {})}
    # normalise list-ish fields
    g['context_radius_um'] = _aslist(g['context_radius_um'])
    if l['blocksize'] and np.ndim(l['blocksize'][0]) == 0:   # single [Y,X,Z] -> wrap
        l['blocksize'] = [list(l['blocksize'])]
    else:
        l['blocksize'] = [list(b) for b in l['blocksize']]
    return g, l, ds


def _aslist(x):
    if x is None:
        return []
    return list(x) if isinstance(x, (list, tuple, np.ndarray)) else [x]


# --------------------------------------------------------------------------- #
#  PORTED HELPERS -- geometry + honest metric
#  (verbatim from gen_hcr_global_centroid_diagnostic.py unless noted)
# --------------------------------------------------------------------------- #
def _apply(A, pts):                                        # pts Nx3 physical -> Nx3
    return (A[:3, :3] @ pts.T).T + A[:3, 3]


def _rot_z(theta):                                         # in-plane (y,x) rotation; coords (y,x,z)
    c, s = np.cos(theta), np.sin(theta)
    R = np.eye(4); R[0, 0] = c; R[0, 1] = -s; R[1, 0] = s; R[1, 1] = c
    return R


def mutual_inliers(A, fix_phys, mov_phys, gate_um):
    """Reciprocal NN within gate. A wrong/local-min transform gives few mutual pairs even if
    the one-way NN looks small (dense tissue). Returns (n_mutual, median_resid_um_of_mutual)."""
    from scipy.spatial import cKDTree
    fp = _apply(A, fix_phys)
    mtree = cKDTree(mov_phys); ftree = cKDTree(fp)
    d_fm, j = mtree.query(fp, k=1)
    _, i = ftree.query(mov_phys, k=1)
    mutual = (i[j] == np.arange(len(fp))) & (d_fm < gate_um)
    n = int(mutual.sum())
    return n, (float(np.median(d_fm[mutual])) if n else np.nan)


def null_mutual(A, fix_phys, mov_phys, gate_um, angle=45.0):
    """Chance floor: spin the A-aligned fix cloud by `angle` about its centre, recount mutuals.
    n_mut >> n_null = real signal."""
    from scipy.spatial import cKDTree
    fp = _apply(A, fix_phys); c = fp.mean(0)
    fp = (_rot_z(np.deg2rad(angle))[:3, :3] @ (fp - c).T).T + c
    mtree = cKDTree(mov_phys); ftree = cKDTree(fp)
    d_fm, j = mtree.query(fp, k=1); _, i = ftree.query(mov_phys, k=1)
    return int(((i[j] == np.arange(len(fp))) & (d_fm < gate_um)).sum())


def _contexts(img, spots_vox, cc):
    """Extract + L2-normalize (mean-subtracted) context vectors so a dot product == correlation.
    Returns (NxD unit vectors, keep_mask) for spots far enough from the edge. cc=(ry,rx,rz)."""
    ry, rx, rz = cc; Y, X, Z = img.shape; s = np.asarray(spots_vox, float)
    keep = ((s[:, 0] >= ry) & (s[:, 0] < Y - ry) & (s[:, 1] >= rx) & (s[:, 1] < X - rx) &
            (s[:, 2] >= rz) & (s[:, 2] < Z - rz))
    sp = s[keep].astype(int); D = (2 * ry + 1) * (2 * rx + 1) * (2 * rz + 1)
    out = np.empty((len(sp), D), np.float32)
    for k in range(len(sp)):
        y, x, z = sp[k]; out[k] = img[y - ry:y + ry + 1, x - rx:x + rx + 1, z - rz:z + rz + 1].ravel()
    if len(sp):
        out -= out.mean(1, keepdims=True)
        nrm = np.linalg.norm(out, axis=1, keepdims=True); nrm[nrm == 0] = 1; out /= nrm
    return out, keep


def _keep_sz(cent, area):
    """Drop segmentation artifacts: keep 0.3x-3x median nucleus volume."""
    if not len(area):
        return cent
    m = np.median(area); return cent[(area >= 0.3 * m) & (area <= 3.0 * m)]


def score_mi(A, fix, mov, fsp, msp):
    """Image MI on RAW (non-boosted) DAPI -- an INDEPENDENT sanity verdict (the centroid matcher
    optimises NN, not image MI). More negative = better. NEVER score boosted images."""
    from bigstream.transform import apply_transform
    al = np.asarray(apply_transform(fix, mov, fsp, msp, transform_list=[np.asarray(A)]))
    return float(get_registration_score(al, fix))


# --------------------------------------------------------------------------- #
#  PORTED HELPERS -- centroids + adaptive edges
#  (from _profile_local_reg.py)
# --------------------------------------------------------------------------- #
def _fast_centroids(mask):
    """Per-label centroid + area via bincount (~20x faster than regionprops on big masks).
    Returns (centroids Nx3 in (y,x,z) voxels, areas N)."""
    mask = np.asarray(mask)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return np.empty((0, 3)), np.empty((0,))
    lab = mask.reshape(-1)[idx]
    ys, xs, zs = np.unravel_index(idx, mask.shape)
    n = int(lab.max()) + 1
    cnt = np.bincount(lab, minlength=n).astype(float)
    cy = np.bincount(lab, ys, minlength=n); cx = np.bincount(lab, xs, minlength=n)
    cz = np.bincount(lab, zs, minlength=n)
    keep = cnt > 0; keep[0] = False        # drop background label 0
    cents = np.column_stack([cy[keep] / cnt[keep], cx[keep] / cnt[keep], cz[keep] / cnt[keep]])
    return cents, cnt[keep]


def _score4(pts3, img):
    pts3 = np.asarray(pts3, float)
    if len(pts3) == 0:
        return np.empty((0, 4))
    yy = np.clip(pts3[:, 0].astype(int), 0, img.shape[0] - 1)
    xx = np.clip(pts3[:, 1].astype(int), 0, img.shape[1] - 1)
    zz = np.clip(pts3[:, 2].astype(int), 0, img.shape[2] - 1)
    return np.column_stack([pts3[:, 0], pts3[:, 1], pts3[:, 2], img[yy, xx, zz].astype(float)])


def _adaptive_get_contexts(image, coords, radius):
    """Bounds-safe replacement for bigstream.features.get_contexts: every context window is
    edge-padded to exactly (2r+1) per axis, so injected centroids near a volume edge / in a thin-z
    block degrade gracefully instead of crashing or falling back to identity. Module-level (not a
    closure) so it is picklable -> can be shipped to dask worker processes. Ported from
    _profile_local_reg.py."""
    if isinstance(radius, (int, np.integer)):
        radius = (radius,) * image.ndim
    radius = tuple(int(r) for r in radius)
    shp = image.shape
    out = []
    for coord in coords:
        lo, hi, plo, phi = [], [], [], []
        for x, r, n in zip(coord, radius, shp):
            x = int(round(float(x)))
            a, b = x - r, x + r + 1
            lo.append(max(a, 0)); hi.append(min(b, n))
            plo.append(max(-a, 0)); phi.append(max(b - n, 0))
        crop = image[tuple(slice(l, h) for l, h in zip(lo, hi))]
        if any(plo) or any(phi):
            crop = np.pad(crop, tuple((p, q) for p, q in zip(plo, phi)), mode="edge")
        out.append(crop)
    return out


def _install_adaptive_get_contexts():
    """Permanently install the adaptive get_contexts patch in the CURRENT process. Runs on dask
    worker processes (via client.run / a WorkerPlugin) where the main-process monkeypatch from
    adaptive_cc_contexts() is invisible -- the long-standing reason multiprocess block alignment
    'didn't work': workers silently used bigstream's edge-unsafe get_contexts and edge/thin-z
    blocks failed."""
    import bigstream.features as F
    F.get_contexts = _adaptive_get_contexts
    return True


@contextlib.contextmanager
def adaptive_cc_contexts():
    """Reversibly install the adaptive get_contexts patch in THIS process (main process; also the
    thread-pool workers, which share it). For a MULTIPROCESS cluster this is not enough -- use
    _patch_cluster_workers() to reinstall it inside each worker process too."""
    import bigstream.features as F
    orig = F.get_contexts
    F.get_contexts = _adaptive_get_contexts
    try:
        yield
    finally:
        F.get_contexts = orig


def _patch_cluster_workers(cl):
    """Reinstall the adaptive get_contexts patch on every worker process of a multiprocess cluster.
    Best-effort: warns and proceeds on failure. Covers both current workers (client.run) and any
    nanny-restarted workers (a WorkerPlugin)."""
    client = getattr(cl, 'client', None)
    if client is None:
        rprint("    [yellow]cluster exposes no .client; cannot patch workers for parallelism[/yellow]")
        return
    client.run(_install_adaptive_get_contexts)
    try:
        from distributed.diagnostics.plugin import WorkerPlugin

        class _AdaptiveGetContextsPlugin(WorkerPlugin):
            def setup(self, worker):
                _install_adaptive_get_contexts()

        # register_worker_plugin is deprecated in newer distributed in favour of register_plugin
        reg = getattr(client, 'register_plugin', None) or client.register_worker_plugin
        reg(_AdaptiveGetContextsPlugin())
    except Exception:
        pass    # client.run already covered the live workers; plugin is belt-and-suspenders


# --------------------------------------------------------------------------- #
#  LOAD one round's data (images + centroids, both global-lowres and local-fullres)
# --------------------------------------------------------------------------- #
def _norm_u8(arr, plo_hi=(0, 99.5)):
    from skimage import exposure
    return exposure.rescale_intensity(
        arr, in_range=(0, np.percentile(arr, plo_hi[1])), out_range=(0, 255)).astype(np.uint8)


def _series_shape(path):
    """Cheap tiff shape (header only) -- avoids loading a ~1GB mask just to compare frames."""
    import tifffile
    with tifffile.TiffFile(str(path)) as tf:
        return tuple(tf.series[0].shape)


def _load_round_data(reference_round, mov_round, fix_mask_path, mov_mask_path, ds):
    """Load fix/mov DAPI at full-res (local) + downsampled (global), and size-filtered
    centroids. mov mask MUST be the NATIVE (unregistered) frame; we assert shape != fix.

    Returns a dict with full-res images/centroids and downsampled (global) images/centroids,
    all in (Y,X,Z) order, plus physical spacings.
    """
    ds = np.array(ds)
    fix_sp = np.array(resolve_hcr_resolution(reference_round['image_path'], reference_round.get('resolution')))
    mov_sp = np.array(resolve_hcr_resolution(mov_round['image_path'], mov_round.get('resolution')))

    fix_full = _norm_u8(tif_imread(reference_round['image_path'])[:, 0].transpose(2, 1, 0))  # (Y,X,Z) DAPI
    mov_full = _norm_u8(tif_imread(mov_round['image_path'])[:, 0].transpose(2, 1, 0))

    fmask = tif_imread(str(fix_mask_path)).transpose(2, 1, 0)
    mmask = tif_imread(str(mov_mask_path)).transpose(2, 1, 0)
    # STALE-MASK GUARD: each mask must match its OWN raw image (same shape/orientation). A mismatch
    # means the image was rotated/replaced without re-segmenting -> the centroids are in the wrong
    # frame (garbage global) and intensity extraction crashes later. Fail fast with the fix.
    assert fmask.shape == fix_full.shape, (
        f"fix mask {fmask.shape} != fix image {fix_full.shape} -> stale/mismatched HCR"
        f"{reference_round['round']} mask. Delete its cellpose mask and re-run cellpose.")
    assert mmask.shape == mov_full.shape, (
        f"mov mask {mmask.shape} != mov image {mov_full.shape} -> stale/mismatched HCR"
        f"{mov_round['round']} mask (raw image changed without re-segmenting?). Delete "
        f"cellpose/{get_round_folder_name(mov_round['round'], reference_round['round'])}_masks.tiff "
        f"and re-run cellpose.")
    assert fmask.shape != mmask.shape, (
        f"mov mask shape == fix ({mmask.shape}) -> a WARPED-frame mask, nothing to register. "
        f"Need the native-frame mov mask (HCR{mov_round['round']}_native_masks.tiff).")

    fix_cent, fix_area = _fast_centroids(fmask)
    mov_cent, mov_area = _fast_centroids(mmask)
    fix_cent = _keep_sz(fix_cent, fix_area)
    mov_cent = _keep_sz(mov_cent, mov_area)

    sl = (slice(None, None, ds[0]), slice(None, None, ds[1]), slice(None, None, ds[2]))
    return dict(
        fix_full=fix_full, mov_full=mov_full, fix_sp=fix_sp, mov_sp=mov_sp,
        fix_lo=fix_full[sl], mov_lo=mov_full[sl], lo_sp_f=fix_sp * ds, lo_sp_m=mov_sp * ds, ds=ds,
        fix_cent=fix_cent, mov_cent=mov_cent,                 # full-res voxel coords (y,x,z)
        fix_cent_phys=fix_cent * fix_sp, mov_cent_phys=mov_cent * mov_sp,
    )


# --------------------------------------------------------------------------- #
#  GLOBAL  -- cap-free 2D-context centroid affine (ported centroid_global_batched)
# --------------------------------------------------------------------------- #
def global_centroid(S, gcfg, batch=2000):
    """Sweep gcfg['context_radius_um'] and return (best_A, best_radius, table) where table is a
    list of per-radius dicts {radius_um, above_chance, n_mut, med_resid, mi}. Best is chosen by
    gcfg['select_metric']. Ported from centroid_global_batched (the robust, no-prior, no-rotation
    global) + an MI sanity column."""
    import cv2
    gate = float(gcfg['inlier_gate_um']); mt = float(gcfg['match_threshold'])
    rth = float(gcfg['ransac_threshold_um'])
    fw, mw = S['fix_lo'], S['mov_lo']            # downsampled, nuclei resolved (~3.8um/vox)
    xy = float(S['lo_sp_f'][0]); ds = S['ds']
    fcF, mcF = S['fix_cent'], S['mov_cent']      # already size-filtered, full-res voxels
    fcp, mcp = S['fix_cent_phys'], S['mov_cent_phys']

    table, best = [], (np.eye(4), -np.inf, None)
    for r_um in gcfg['context_radius_um']:
        r = max(1, int(round(r_um / xy))); cc = (r, r, 0)         # 2D window: XY radius, z=0
        fctx, fk = _contexts(fw, fcF / ds, cc); mctx, mk = _contexts(mw, mcF / ds, cc)
        A = np.eye(4)
        if len(fctx) >= 12 and len(mctx) >= 12:
            fphys = fcF[fk] * S['fix_sp']; mphys = mcF[mk] * S['mov_sp']
            src, dst = [], []
            for b in range(0, len(mctx), batch):
                corr = mctx[b:b + batch] @ fctx.T                 # every mov vs ALL fix
                j = corr.argmax(1); cval = corr[np.arange(corr.shape[0]), j]; ok = cval > mt
                if ok.any():
                    dst.append(mphys[b:b + batch][ok]); src.append(fphys[j[ok]])
            if src:
                src = np.vstack(src).astype(np.float32); dst = np.vstack(dst).astype(np.float32)
                if len(src) >= 12:
                    try:
                        _, M, _ = cv2.estimateAffine3D(src, dst, ransacThreshold=rth, confidence=0.999)
                        if M is not None:
                            A = np.eye(4); A[:3, :] = M
                    except Exception as e:
                        rprint(f"    [yellow]global r{r_um}um ransac err {type(e).__name__}: {e}[/yellow]")
        n_mut, med = mutual_inliers(A, fcp, mcp, gate)
        above = n_mut - null_mutual(A, fcp, mcp, gate)
        mi = score_mi(A, S['fix_lo'], S['mov_lo'], S['lo_sp_f'], S['lo_sp_m'])
        row = dict(radius_um=r_um, above_chance=above, n_mut=n_mut,
                   med_resid=med, mi=mi, A=A)
        table.append(row)
    pick = _pick_global(table, 'mi')        # both stages picked by raw-DAPI MI (most negative = best)
    return pick['A'], pick['radius_um'], table


def _pick_global(table, metric):
    """Choose the global radius strictly by `metric`:
      'mi'           -> best (most-negative) raw-DAPI image MI -- an independent verdict, used as-is.
      'above_chance' -> most reciprocal inliers above the 45deg-spin null.
      'median_resid' -> lowest mutual-inlier residual, tie-broken by MORE inliers; GUARDED to radii
                        with above_chance > 0 (a sparse-but-close fit can fake a low residual, so
                        the residual metric -- and only it -- ignores worse-than-chance radii)."""
    if metric == "mi":
        return min(table, key=lambda r: r['mi'])
    if metric == "above_chance":
        return max(table, key=lambda r: r['above_chance'])
    valid = [r for r in table if r['above_chance'] > 0] or table
    return min(valid, key=lambda r: (r['med_resid'] if r['med_resid'] == r['med_resid'] else 1e9, -r['n_mut']))


# --------------------------------------------------------------------------- #
#  LOCAL  -- bigstream piecewise deform with injected centroids (ported)
# --------------------------------------------------------------------------- #
def _local_cfg_dict(lcfg, fix_sp):
    """Build the per-block bigstream ransac config from the manifest local config."""
    if lcfg.get('context_radius_um'):                       # experiment knob: um-window -> voxels
        r = max(1, int(round(float(lcfg['context_radius_um']) / float(fix_sp[0]))))
        cc = (r, r, max(1, int(round(float(lcfg['context_radius_um']) / float(fix_sp[2])))))
    else:
        cc = tuple(int(v) for v in lcfg['context_radius'])
    mt = lcfg['match_threshold']
    return dict(
        blob_sizes=list(lcfg.get('blob_sizes', [6, 30])),   # required positionally; ignored when spots injected
        cc_radius=cc,
        match_threshold=list(mt) if isinstance(mt, (list, tuple)) else [float(mt)],
        max_spot_match_distance=float(lcfg['max_spot_match_distance_um']),
        fix_spots_count_threshold=int(lcfg['count_floor']),
        mov_spots_count_threshold=int(lcfg['count_floor']),
        point_matches_threshold=int(lcfg['match_floor']),
        nspots=int(lcfg['nspots']),
        safeguard_exceptions=False,
    )


def _moved_coverage(deform, blocksize):
    """Fraction of blocks that received a real local deform (95th-pct displacement > 0.5um)."""
    bs = np.array(blocksize); mag = np.linalg.norm(deform, axis=-1)
    grid = np.ceil(np.array(mag.shape) / bs).astype(int); moved = tot = 0
    for i in range(grid[0]):
        for j in range(grid[1]):
            for k in range(grid[2]):
                t = mag[i * bs[0]:(i + 1) * bs[0], j * bs[1]:(j + 1) * bs[1], k * bs[2]:(k + 1) * bs[2]]
                if t.size:
                    tot += 1; moved += int(np.percentile(t, 95) > 0.5)
    return moved, tot


def local_centroid_one(S, A_global, blocksize, lcfg, out_dir, cluster_kwargs=None, overlay_tiff=None,
                       fingerprint=None, resume=True):
    """Run ONE local block config at FULL resolution: write _affine.mat + deform.zarr into
    out_dir (these MUST stay per-candidate -- _load_round_transform reads them), and the QC
    composite to overlay_tiff (a single pooled folder; defaults to out_dir if not given).
    Returns metrics {medResid_um, frac_under5, moved, total, mi, reused}. Ported from
    _profile_local_reg.coverage_sweep._run, but writes PERSISTENT pipeline outputs (no /tmp,
    deform.zarr kept).

    RESUME: when `fingerprint` (mask identity + global affine) matches the one stamped beside an
    existing deform.zarr, the expensive align is SKIPPED and metrics are re-derived from the saved
    field. A changed mask / global / block config flips the fingerprint -> recompute (never reuses
    a stale field)."""
    import zarr
    from scipy.spatial import cKDTree
    from bigstream.piecewise_align import distributed_piecewise_alignment_pipeline
    from bigstream.transform import apply_transform_to_coordinates
    try:
        from ClusterWrap import cluster as cluster_constructor
    except ImportError:
        from ClusterWrap.clusters import cluster as cluster_constructor

    fix, mov = S['fix_full'], S['mov_full']
    fsp, msp = np.asarray(S['fix_sp'], float), np.asarray(S['mov_sp'], float)
    fix_pts = np.asarray(S['fix_cent'], float)                          # full-res voxels (y,x,z)
    fix_spots4 = _score4(fix_pts, fix)
    # mov centroids -> fix frame (inv global affine), injected as fix-frame spots
    mov_fix = apply_transform_to_coordinates(
        S['mov_cent_phys'].astype(float), [np.linalg.inv(A_global)], transform_spacing=fsp) / fsp
    mov_spots4 = _score4(mov_fix, fix)

    loc = _local_cfg_dict(lcfg, fsp)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    deform_path = out_dir / "deform.zarr"
    fp_path = out_dir / "_fingerprint.txt"

    reused = bool(resume and fingerprint and deform_path.exists()
                  and (out_dir / "_affine.mat").exists()
                  and fp_path.exists() and fp_path.read_text().strip() == fingerprint)

    if not reused:
        shutil.rmtree(deform_path, ignore_errors=True)
        nw = int(lcfg.get('n_workers', 1)); tpw = int(lcfg.get('threads_per_worker', 8))
        ck = cluster_kwargs or {"n_workers": nw, "threads_per_worker": tpw, "processes": nw > 1}
        adaptive = lcfg.get('adaptive_edges', True)

        def _run(cluster_kwargs):
            multiproc = bool(cluster_kwargs.get('processes'))
            ctx = adaptive_cc_contexts() if adaptive else contextlib.nullcontext()
            with ctx, cluster_constructor(cluster_kwargs) as cl:
                if adaptive and multiproc:        # propagate the patch into the worker PROCESSES
                    _patch_cluster_workers(cl)
                distributed_piecewise_alignment_pipeline(
                    fix, mov, fsp, msp, steps=[("ransac", loc)],
                    blocksize=list(blocksize), overlap=float(lcfg['overlap']),
                    static_transform_list=[np.asarray(A_global)], write_path=str(deform_path),
                    cluster=cl, fix_spots_global=fix_spots4, mov_spots_global=mov_spots4)

        try:
            _run(ck)
        except Exception as e:
            if ck.get('processes'):               # multiprocess flaked -> fall back to a thread pool
                rprint(f"    [yellow]multiprocess cluster failed ({type(e).__name__}: {e}); "
                       f"retrying single-process thread pool[/yellow]")
                shutil.rmtree(deform_path, ignore_errors=True)
                _run({"n_workers": 1, "threads_per_worker": tpw, "processes": False})
            else:
                raise

        np.savetxt(str(out_dir / "_affine.mat"), np.asarray(A_global))  # required by _load_round_transform
        if fingerprint:
            fp_path.write_text(fingerprint)

    deform = zarr.open(str(deform_path), mode="r")[...]

    # residual: fix centroids warped by [affine, deform], scored with the MUTUAL reciprocal-inlier
    # metric (same as the global) -- robust to volume/coverage mismatch. When one round is much
    # bigger or deeper than the other, its extra cells have no true partner; a one-way all-cells NN
    # median is inflated by them (-> false RED FLAG, e.g. JS079 z-depth, SRC104 1.9x cell count).
    # Mutual inliers exclude the unmatchable cells. all-cells kept for reference only.
    mov_phys = S['mov_cent_phys']
    gate = float(lcfg.get('inlier_gate_um', 12.0))
    fp = apply_transform_to_coordinates(fix_pts * fsp, [np.asarray(A_global), deform], transform_spacing=fsp)
    mtree = cKDTree(mov_phys); ftree = cKDTree(fp)
    d_all, j = mtree.query(fp, k=1)
    _, i = ftree.query(mov_phys, k=1)
    mut = (i[j] == np.arange(len(fp))) & (d_all < gate)
    n_mut = int(mut.sum())
    med_mut = float(np.median(d_all[mut])) if n_mut else float('nan')
    frac5_mut = float(np.mean(d_all[mut] < 5.0)) if n_mut else 0.0
    moved, tot = _moved_coverage(deform, blocksize)

    # MI on RAW lowres DAPI with [affine, deform] -- the boost-proof independent verdict the candidates
    # are RANKED by (more negative = better). Computed from the same lowres warp as the QC composite,
    # so the candidate selection and the overlay you inspect agree.
    from bigstream.transform import apply_transform
    try:
        aligned_lo = np.asarray(apply_transform(
            S['fix_lo'], S['mov_lo'], np.asarray(S['lo_sp_f']), np.asarray(S['lo_sp_m']),
            transform_list=[np.asarray(A_global), deform], transform_spacing=fsp))
        mi = float(get_registration_score(aligned_lo, S['fix_lo']))
    except Exception as e:
        aligned_lo, mi = None, float('nan')
        rprint(f"    [yellow]local MI failed: {type(e).__name__}: {e}[/yellow]")

    # QC composite (warped mov + fix) at lowres -> pooled overlays folder. Reuse the MI warp; affine
    # fallback. On reuse, only (re)write if it's missing -- the warp is the one not-quite-free step here.
    ov = Path(overlay_tiff) if overlay_tiff else (out_dir / "overlay_warped.tiff")
    ov.parent.mkdir(parents=True, exist_ok=True)
    if not (reused and ov.exists()):
        try:
            _write_composite(S['fix_lo'], S['mov_lo'], S['lo_sp_f'], S['lo_sp_m'],
                             [np.asarray(A_global), deform], ov, transform_spacing=fsp, aligned=aligned_lo)
        except Exception as e:
            try:
                _write_composite(S['fix_lo'], S['mov_lo'], S['lo_sp_f'], S['lo_sp_m'],
                                 [np.asarray(A_global)], ov)
            except Exception as e2:
                rprint(f"    [yellow]composite write failed: {type(e2).__name__}: {e2}[/yellow]")

    return dict(medResid_um=med_mut, frac_under5=frac5_mut, n_mut=n_mut,
                medResid_allcells=float(np.median(d_all)), frac_under5_allcells=float(np.mean(d_all < 5.0)),
                moved=int(moved), total=int(tot), mi=mi, reused=reused)


# --------------------------------------------------------------------------- #
#  QC composite: warp the mov IMAGE and save a 2-channel ImageJ TIFF (warped + fix),
#  like the legacy register_lowres _both.tiff. Rendered at the downsampled (lowres) grid
#  so it is cheap. transform_list = [affine] (global) or [affine, deform] (local).
# --------------------------------------------------------------------------- #
def _write_composite(fix, mov, fsp, msp, transform_list, out_tiff, transform_spacing=None, aligned=None):
    from bigstream.transform import apply_transform
    if aligned is None:                          # reuse a precomputed warp (e.g. the MI warp) when given
        aligned = np.asarray(apply_transform(
            fix, mov, np.asarray(fsp), np.asarray(msp),
            transform_list=transform_list, transform_spacing=transform_spacing))
    # (Y,X,Z) -> ImageJ (Z, C=2, Y, X): channel 0 = warped mov, channel 1 = fix
    comp = np.swapaxes(np.array([aligned.transpose(2, 1, 0), fix.transpose(2, 1, 0)]), 0, 1)
    tif_imwrite(str(out_tiff), comp.astype(np.uint8), imagej=True)


# --------------------------------------------------------------------------- #
#  TAGS  (must satisfy _load_round_transform's bs(\d+)[x_](\d+)[x_](\d+) regex)
# --------------------------------------------------------------------------- #
def _global_tag(radius_um, mt):
    return f"global_cent_r{int(round(radius_um))}um_mt{mt}"


def _local_tag(blocksize, lcfg):
    by, bx, bz = (int(v) for v in blocksize)
    ctx = (f"r{int(round(float(lcfg['context_radius_um'])))}um" if lcfg.get('context_radius_um')
           else "r" + "x".join(str(int(v)) for v in lcfg['context_radius']))
    return (f"bs{by}x{bx}x{bz}_ov{lcfg['overlap']}_cent_{ctx}"
            f"_mt{lcfg['match_threshold']}_f{lcfg['count_floor']}")


# --------------------------------------------------------------------------- #
#  OUTPUT  -- structured plan + scored summary + red-flag assessment
# --------------------------------------------------------------------------- #
_SEV = {0: ("✓", "green", "OK"),
        1: ("⚠", "yellow", "WARN"),
        2: ("✗", "red", "RED FLAG")}


def _sev_badge(sev):
    glyph, color, label = _SEV[sev]
    return f"[{color}]{glyph} {label}[/{color}]"


def _print_msgs(sev, msgs, indent="      "):
    glyph, color, _ = _SEV[sev]
    for m in msgs:
        rprint(f"{indent}[{color}]{glyph} {m}[/{color}]")


def _print_ladder(round_to_rounds, ref, gcfg, lcfg):
    """One-time header: what the two stages (coarse affine, fine deform) do for every round, so the
    long run is legible from the first screen."""
    radii = [int(r) for r in gcfg['context_radius_um']]
    blocks = ', '.join('×'.join(str(int(v)) for v in b) for b in lcfg['blocksize'])
    rprint("\n" + "═" * 72)
    rprint(f"[bold] HCR→HCR registration · {len(round_to_rounds)} round(s) → HCR{ref}[/bold]")
    rprint("  [dim]Mode: cellpose centroids (nucleus landmarks)[/dim]")
    rprint("═" * 72)
    rprint(f"  [cyan]1 COARSE[/cyan]  whole-volume affine · context window {radii}µm (best by [b]mi[/b])")
    rprint(f"  [cyan]2 FINE  [/cyan]  local deform · tile sizes {blocks} (best by [b]mi[/b]; "
           f"cached deform reused when unchanged)")
    rprint("═" * 72)


def _assess_global(picked, n_fix, n_mov):
    """Red-flag the chosen global. Returns (severity 0/1/2, [messages])."""
    sev, msgs = 0, []
    ab, nm, mr = picked['above_chance'], picked['n_mut'], picked['med_resid']
    denom = max(1, min(n_fix, n_mov))
    if ab <= 0:
        sev = 2; msgs.append(f"above-chance {ab} ≤ 0 — global is NO BETTER THAN CHANCE (did not lock)")
    elif nm < 0.10 * denom:
        sev = max(sev, 1); msgs.append(f"only {nm} mutual inliers ({100*nm/denom:.0f}% of cells) — sparse lock")
    if mr == mr and mr > 25:
        sev = 2; msgs.append(f"mutual-inlier residual {mr:.1f}µm very high (>25µm)")
    elif mr == mr and mr > 15:
        sev = max(sev, 1); msgs.append(f"mutual-inlier residual {mr:.1f}µm high (>15µm)")
    return sev, msgs


def _assess_local(best, global_med):
    """Red-flag the best local candidate vs absolute gates and vs the global it refined.
    Returns (severity 0/1/2, [messages])."""
    if best is None:
        return 2, ["ALL local blocksizes failed — no deform produced"]
    sev, msgs = 0, []
    mr, fr = best['medResid_um'], best['frac_under5']   # mutual-inlier residual + frac<5 (size-robust)
    cov = best['moved'] / max(1, best['total'])
    if mr > 15:
        sev = 2; msgs.append(f"mutual-inlier residual {mr:.1f}µm > 15µm — local did NOT register")
    elif mr > 10:
        sev = max(sev, 1); msgs.append(f"mutual-inlier residual {mr:.1f}µm > 10µm — marginal")
    if fr < 0.30:                       # fraction of MATCHED (mutual-inlier) cells within 5µm; a
        sev = max(sev, 1); msgs.append(f"only {fr*100:.0f}% of matched cells within 5µm (<30%)")  # genuinely
        # good register (JS082) reaches ~50%, so 30% is the realistic warn floor.
    if cov < 0.50:
        sev = max(sev, 1); msgs.append(f"only {cov*100:.0f}% of blocks deformed (<50%) — field mostly identity")
    if global_med == global_med and mr >= global_med:
        sev = max(sev, 1); msgs.append(f"local {mr:.1f}µm ≥ global {global_med:.1f}µm — deform added nothing")
    return sev, msgs


# --------------------------------------------------------------------------- #
#  DRIVER
# --------------------------------------------------------------------------- #
def run_hcr_centroid_registration(full_manifest, round_to_rounds, reference_round, gcfg, lcfg, ds):
    """Compute global+local centroid registration for every mov round, write candidates into
    the existing registrations/ layout, and return a per-round ranked summary:

        { round: {
            'global': {'tag','radius_um','table'(per-radius),'A'},
            'candidates': [ {'tag','local_tag','blocksize','medResid_um','frac_under5',
                             'moved','total','dir'} ... ranked best-first ],
        } }

    Selection of the winner + manifest write-back is done by the caller (register_rounds).
    """
    out_root = output_root(full_manifest) / 'HCR'
    cellpose_dir = out_root / 'cellpose'
    # Two clearly-labeled group folders for the QC composites; ALL generation info is in the
    # filename (round, global tag, local tag) -- no per-candidate scatter, no vague "summary".
    comp_global = out_root / 'registrations' / 'composites' / 'global'
    comp_local = out_root / 'registrations' / 'composites' / 'local'
    comp_global.mkdir(parents=True, exist_ok=True)
    comp_local.mkdir(parents=True, exist_ok=True)
    ref = reference_round['round']
    results = {}
    _print_ladder(round_to_rounds, ref, gcfg, lcfg)

    for ri, (rnd, mov_round) in enumerate(round_to_rounds.items(), 1):
        rprint(f"\n[bold cyan]── HCR{rnd} → HCR{ref}  ({ri}/{len(round_to_rounds)}) "
               f"{'─' * max(0, 40 - len(str(rnd)) - len(str(ref)))}[/bold cyan]")
        fix_mask = cellpose_dir / f"{get_round_folder_name(ref, ref)}_masks.tiff"
        # mov mask MUST be the NATIVE (unregistered) frame. The canonical
        # cellpose/{round_folder}_masks.tiff is native on most mice, but on some cp4 mice
        # (JS082, CIM131) that name holds the REF-FRAME (warped) copy and the native one is
        # HCR{N}_native_masks.tiff. Pick the first candidate whose shape DIFFERS from fix
        # (== native frame); read tiff HEADERS only, so this stays cheap on ~1GB masks.
        mov_candidates = [cellpose_dir / f"{get_round_folder_name(rnd, ref)}_masks.tiff",
                          cellpose_dir / f"HCR{rnd}_native_masks.tiff"]
        fix_shape = _series_shape(fix_mask) if fix_mask.exists() else None
        mov_mask = next((c for c in mov_candidates
                         if c.exists() and _series_shape(c) != fix_shape), None)
        if mov_mask is None:                     # none look native -> first existing (assert explains)
            mov_mask = next((c for c in mov_candidates if c.exists()), mov_candidates[0])
        missing = [p for p in (fix_mask, mov_mask) if not Path(p).exists()]
        if missing:
            rprint(f"  [red]MISSING {missing} -- skipping HCR{rnd}[/red]")
        else:
            S = _load_round_data(reference_round, mov_round, fix_mask, mov_mask, ds)
            nfix, nmov = len(S['fix_cent']), len(S['mov_cent'])
            fy, fx, fz = S['fix_full'].shape; my, mx, mz = S['mov_full'].shape
            rprint(f"  data: fix {fy}×{fx}×{fz} ([b]{nfix}[/b] nuclei) · "
                   f"mov {my}×{mx}×{mz} ([b]{nmov}[/b] nuclei)")

            # ---- GLOBAL ----
            rfolder = get_round_folder_name(rnd, ref)
            A_g, r_best, gtable = global_centroid(S, gcfg)
            gtag = _global_tag(r_best, gcfg['match_threshold'])
            rprint(f"  [cyan][1/2] COARSE[/cyan]  picked context window [b]{r_best}µm[/b] by mi")
            rprint(f"      {'radius':>7}{'above':>8}{'n_mut':>8}{'medResid':>10}{'MI':>9}")
            for row in gtable:
                mark = "  [b]◄ pick[/b]" if row['radius_um'] == r_best else ""
                mr = row['med_resid'] if row['med_resid'] == row['med_resid'] else float('nan')
                rprint(f"      {str(int(row['radius_um']))+'µm':>7}{row['above_chance']:>8}{row['n_mut']:>8}"
                       f"{mr:>10.2f}{row['mi']:>+9.3f}{mark}")
            picked = next(r for r in gtable if r['radius_um'] == r_best)
            g_sev, g_msgs = _assess_global(picked, nfix, nmov)
            if g_sev == 0:
                rprint(f"      {_sev_badge(0)} locked: {picked['n_mut']} mutual inliers "
                       f"({100*picked['n_mut']/max(1,min(nfix,nmov)):.0f}% of cells), residual {picked['med_resid']:.1f}µm")
            _print_msgs(g_sev, g_msgs)
            gdir = out_root / 'registrations' / rfolder / gtag
            gdir.mkdir(parents=True, exist_ok=True)
            np.savetxt(str(gdir / "_affine.mat"), A_g)
            gcomp = comp_global / f"{rfolder}__{gtag}.tiff"
            try:                                  # coarse composite saved NOW (right after the coarse fit)
                _write_composite(S['fix_lo'], S['mov_lo'], S['lo_sp_f'], S['lo_sp_m'], [A_g], gcomp)
                rprint(f"      [dim]↳ check composite: {gcomp}[/dim]")
            except Exception as e:
                rprint(f"  [yellow]coarse composite write failed: {type(e).__name__}: {e}[/yellow]")

            # RESUME fingerprint: mask identity (size+mtime, changes on re-seg) + the global affine.
            # A match -> reuse the cached deform.zarr and skip the expensive align; any change -> recompute.
            def _mfp(p):
                st = Path(p).stat(); return f"{st.st_size}:{int(st.st_mtime)}"
            round_fp = (f"{_mfp(fix_mask)}|{_mfp(mov_mask)}|"
                        f"aff:{hashlib.md5(np.asarray(A_g).round(6).tobytes()).hexdigest()[:12]}")

            # ---- LOCAL (sweep blocksizes; best by MI) ----
            nbs = len(lcfg['blocksize'])
            rprint(f"  [cyan][2/2] FINE[/cyan]  {nbs} tile size(s) "
                   f"(large→small; deform reused when mask+coarse unchanged)")
            candidates = []
            # No progress bar over the tile sizes: each is ONE long opaque bigstream align (the bar
            # would just sit at 0% then jump). Print a 'computing' line so it's clearly still running.
            for bi, bs in enumerate(lcfg['blocksize'], 1):
                ltag = _local_tag(bs, lcfg)
                ldir = gdir / ltag
                bstr = '×'.join(str(int(v)) for v in bs)
                lcomp = comp_local / f"{rfolder}__{gtag}__{ltag}.tiff"
                rprint(f"      computing deform bs{bstr} ({bi}/{nbs})…")
                try:
                    m = local_centroid_one(S, A_g, bs, lcfg, ldir, fingerprint=round_fp,
                                           overlay_tiff=lcomp)
                except Exception as e:
                    rprint(f"      [red]bs{bstr} ({bi}/{nbs}) failed: {type(e).__name__}: {e}[/red]")
                    continue
                # composite (overlay_warped.tiff) is written inside local_centroid_one
                candidates.append(dict(tag=f"{gtag}/{ltag}", local_tag=ltag, blocksize=bs,
                                       dir=str(ldir), composite=str(lcomp), **m))
                tag = "  [dim](reused)[/dim]" if m.get('reused') else ""
                rprint(f"      bs{bstr:<11} MI [b]{m['mi']:+.3f}[/b]  resid(mutual) {m['medResid_um']:.2f}µm  "
                       f"frac<5 {m['frac_under5']*100:.0f}%  n_mut {m.get('n_mut','?')}  "
                       f"coverage {100*m['moved']/max(1,m['total']):.0f}%  [dim](all-cells {m['medResid_allcells']:.1f}µm)[/dim]{tag}")
            candidates.sort(key=lambda c: c['mi'] if c['mi'] == c['mi'] else float('inf'))   # best MI first

            # ---- per-round VERDICT: best score + combined red-flag assessment ----
            best = candidates[0] if candidates else None
            l_sev, l_msgs = _assess_local(best, picked['med_resid'])
            if best is not None:
                bstr = '×'.join(str(int(v)) for v in best['blocksize'])
                rprint(f"      best: [b]bs{bstr}[/b] → MI {best['mi']:+.3f}, mutual-inlier residual {best['medResid_um']:.2f}µm, "
                       f"{best['frac_under5']*100:.0f}% within 5µm, {100*best['moved']/max(1,best['total']):.0f}% coverage "
                       f"[dim](all-cells {best['medResid_allcells']:.1f}µm)[/dim]")
                if best['mi'] == best['mi'] and picked['mi'] == picked['mi']:   # both non-NaN
                    d_mi = best['mi'] - picked['mi']                            # more negative = better
                    verb = "improved" if d_mi < 0 else "worsened"
                    rprint(f"      MI coarse→fine: {picked['mi']:+.3f} → {best['mi']:+.3f} "
                           f"(Δ {d_mi:+.3f}, deform {verb} the fit)")
                rprint(f"      [dim]↳ check composite: {best['composite']}[/dim]")
            _print_msgs(l_sev, l_msgs)
            sev = max(g_sev, l_sev)
            rprint(f"  VERDICT HCR{rnd} → HCR{ref}: {_sev_badge(sev)}"
                   + ("" if sev == 0 else f"  ([dim]coarse {_SEV[g_sev][2]} · fine {_SEV[l_sev][2]}[/dim])"))

            results[rnd] = dict(global_tag=gtag, radius_um=r_best, table=gtable, A=A_g,
                                candidates=candidates, severity=sev,
                                flags=[f"global: {m}" for m in g_msgs] + [f"local: {m}" for m in l_msgs])
    return results
