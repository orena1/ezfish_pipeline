"""2P-to-HCR cell matching via local geometric soma fingerprint.

Each 2P cell is fingerprinted by the vectors to its m nearest neighbors.
Candidate HCR partners (within a distance radius) are scored by greedy
pairwise vector alignment plus a distance penalty. A GMM separates the
per-cell best-score distribution into 'correct match' vs 'mismatch'
components; the mismatch null is anchored by the second-best score
distribution and a likelihood ratio decides which matches are confident.
A small number of refinement rounds re-scores using only confirmed
identity-paired neighbors instead of geometric neighbors.

Validated against the IoU baseline on JS078 (98-99% concordance at
IoU>=0, zero FAR-disagreements; see
figure_notebooks/figure_4_cortex/somaprint_tests/somaprint_cross_mouse.ipynb).
Inputs come from twop_to_hcr_registration outputs
(twop_plane{P}_aligned_3d.tiff + twop_plane{P}_registration_params.npz)
and HCR cellpose masks (HCR{R}_masks.tiff); the matcher itself adds no
new prerequisites to the pipeline.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numba
import numpy as np
import tifffile
from scipy import stats
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from sklearn.mixture import GaussianMixture
from tqdm.auto import tqdm

try:
    from .meta import output_root, rprint  # Relative import (running as part of a package)
except ImportError:
    from meta import output_root, rprint  # Absolute import (running in Jupyter notebook)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


DEFAULTS = dict(
    enabled=True,
    # TRU-FACT 3D Soma-print (Schnitzer et al., bioRxiv 2026.04.28).
    # 'curved' (default): sample HCR along z_map(y,x)+d for d in [-z_band,+z_band],
    #   run full somaprint on all 2P cells per d, argmax score per cell. Adapts the
    #   paper's flat-slice z-search to tilted-2P geometry — the warp surface IS
    #   the paper's "flat plane" in HCR coordinates.
    # '3d': legacy flat-slice TRU-FACT (fragments local HCR neighborhoods under
    #   tilted 2P; kept for backwards comparison).
    # '2d': single pass at z_map, no z-search.
    mode='curved',
    z_band=1,                       # +/- z-range for 'curved'/'3d' search.
                                    # Production standard is curved +/-1 (~7.5 um,
                                    # ~one soma; least neighbor-aliasing) per the
                                    # cohort band investigation. Was +/-2 through
                                    # 2026-06; switched to +/-1 and all PBN/cortex
                                    # merged tables re-run (see rerun_somaprint_pm1.py).
    m_2p=15,
    m_hcr=30,
    n_pairs=10,
    lambda_scale=10.0,
    radius_factor=3.0,
    lr_threshold=0.05,
    max_rounds=4,
    terminate_pct=5.0,
    n_pairs_r2=10,
    min_confirmed_for_round2=5,
    min_cells_for_gmm=30,
)


def get_params(full_manifest: dict) -> dict:
    """Merge DEFAULTS with any params.somaprint overrides from the manifest."""
    p = dict(DEFAULTS)
    p.update(full_manifest.get('params', {}).get('somaprint', {}) or {})
    return p


def compute_plane_projected_hcr_sizes(full_manifest: dict,
                                      session: dict,
                                      hcr_round: str) -> dict:
    """2D pixel count of each HCR mask at the warped 2P plane.

    Loads the HCR cellpose mask volume, warps it through the 2P plane's
    `z_map_local` (one z per (y,x) pixel — same warp somaprint uses
    internally), then counts pixels per HCR label. The returned size is
    the HCR cell's footprint at exactly where the 2P sheet sits — the
    right denominator for plane-restricted IoU / containment with 2P.

    Returns {hcr_label: int_pixel_count}. HCR cells that don't intersect
    the warped 2P plane get an explicit 0 (rather than being omitted) so
    downstream `df.HCR_mask_size < threshold` filters behave intuitively
    on out-of-plane cells. Background (label 0) is excluded. Returns an
    empty dict if any input file is missing.
    """
    plane = session['functional_plane'][0]
    reg_dir = output_root(full_manifest) / '2P' / 'registered'
    params_path = reg_dir / f'twop_plane{plane}_registration_params.npz'
    hcr_masks_path = output_root(full_manifest) / 'HCR' / 'cellpose' / f'HCR{hcr_round}_masks.tiff'

    for p in (params_path, hcr_masks_path):
        if not Path(p).exists():
            rprint(f"  [yellow][HCR_mask_size] missing input: {p}[/yellow]")
            return {}

    npz = np.load(str(params_path))
    if 'z_map_local' not in npz.files:
        rprint(f"  [yellow][HCR_mask_size] z_map_local missing from {params_path}[/yellow]")
        return {}
    z_map = npz['z_map_local']

    hcr_3d = tifffile.imread(str(hcr_masks_path))
    if hcr_3d.ndim != 3:
        rprint(f"  [yellow][HCR_mask_size] unexpected HCR shape {hcr_3d.shape}[/yellow]")
        return {}

    ny, nx = z_map.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    coords = np.stack([z_map, yy, xx], axis=0)
    hcr_2d = map_coordinates(hcr_3d, coords, order=0, mode='constant', cval=0)

    # All HCR labels present anywhere in the 3D volume; default to 0 at-plane.
    all_ids = np.unique(hcr_3d)
    sizes = {int(i): 0 for i in all_ids if i > 0}
    plane_ids, plane_counts = np.unique(hcr_2d, return_counts=True)
    for i, c in zip(plane_ids, plane_counts):
        if i > 0:
            sizes[int(i)] = int(c)
    return sizes


def compute_full_hcr_sizes(full_manifest: dict, hcr_round: str) -> dict:
    """Full 3D voxel count of each HCR mask in the reference volume.

    Used as the fallback denominator in only_hcr mode, where there's no
    2P plane to project against. Background (label 0) excluded.
    """
    hcr_masks_path = output_root(full_manifest) / 'HCR' / 'cellpose' / f'HCR{hcr_round}_masks.tiff'
    if not hcr_masks_path.exists():
        rprint(f"  [yellow][HCR_mask_size] missing input: {hcr_masks_path}[/yellow]")
        return {}
    hcr_3d = tifffile.imread(str(hcr_masks_path))
    ids, counts = np.unique(hcr_3d, return_counts=True)
    return {int(i): int(c) for i, c in zip(ids, counts) if i > 0}


@dataclass
class _SomaprintInputs:
    twop_labels: np.ndarray
    hcr_labels: np.ndarray
    twop_centroids: np.ndarray
    hcr_centroids: np.ndarray
    twop_label_ids: np.ndarray
    hcr_label_ids: np.ndarray
    fov_span: int


@dataclass
class _RoundResult:
    best_score: np.ndarray
    second_score: np.ndarray
    best_partner: np.ndarray
    confident: dict


def _extract_centroids(labels_2d: np.ndarray):
    """Per-label (y, x) centroid + label id via bincount — same result as regionprops
    (labels ascending) but ~20x faster on full-FOV label images, which matters because
    curved/3d modes re-extract HCR centroids on every z-offset/slice."""
    lab = np.asarray(labels_2d)
    flat = lab.ravel()
    idx = np.flatnonzero(flat)
    if idx.size == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=np.int64)
    lbl = flat[idx]
    ys, xs = np.unravel_index(idx, lab.shape)
    n = int(lbl.max()) + 1
    cnt = np.bincount(lbl, minlength=n)
    sy = np.bincount(lbl, ys, minlength=n)
    sx = np.bincount(lbl, xs, minlength=n)
    keep = np.flatnonzero(cnt)
    keep = keep[keep > 0]                       # drop background label 0
    cents = np.column_stack([sy[keep] / cnt[keep], sx[keep] / cnt[keep]])
    return cents, keep.astype(np.int64)


def _load_inputs(twop_3d_path: Path,
                 params_path: Path,
                 hcr_masks_path: Path) -> Optional[_SomaprintInputs]:
    for p in (twop_3d_path, params_path, hcr_masks_path):
        if not Path(p).exists():
            rprint(f"  [yellow][somaprint] missing input: {p}[/yellow]")
            return None

    twop_3d = tifffile.imread(str(twop_3d_path))
    if twop_3d.ndim != 3:
        rprint(f"  [yellow][somaprint] unexpected 2P shape {twop_3d.shape}[/yellow]")
        return None
    # Each (y, x) holds at most one nonzero z by construction; max over z
    # reproduces the 2D fully-warped labels in the HCR frame.
    twop_labels = twop_3d.max(axis=0).astype(np.uint16)

    npz = np.load(str(params_path))
    if 'z_map_local' not in npz.files:
        rprint(f"  [yellow][somaprint] z_map_local missing from {params_path}[/yellow]")
        return None
    z_map = npz['z_map_local']

    hcr_3d = tifffile.imread(str(hcr_masks_path))
    if hcr_3d.ndim != 3:
        rprint(f"  [yellow][somaprint] unexpected HCR shape {hcr_3d.shape}[/yellow]")
        return None

    ny, nx = twop_labels.shape
    if z_map.shape != (ny, nx):
        rprint(f"  [yellow][somaprint] z_map shape {z_map.shape} != 2P labels {twop_labels.shape}[/yellow]")
        return None

    yy, xx = np.mgrid[0:ny, 0:nx]
    coords = np.stack([z_map, yy, xx], axis=0)
    hcr_labels = map_coordinates(hcr_3d, coords, order=0, mode='constant', cval=0).astype(np.uint16)

    twop_cents, twop_ids = _extract_centroids(twop_labels)
    hcr_cents, hcr_ids = _extract_centroids(hcr_labels)

    return _SomaprintInputs(
        twop_labels=twop_labels,
        hcr_labels=hcr_labels,
        twop_centroids=twop_cents,
        hcr_centroids=hcr_cents,
        twop_label_ids=twop_ids,
        hcr_label_ids=hcr_ids,
        fov_span=int(max(twop_labels.shape)),
    )


@dataclass
class _SomaprintInputs3D:
    """Inputs for 3D modes ('curved' + flat-slice '3d'): full HCR volume + warp."""
    twop_labels: np.ndarray
    twop_centroids: np.ndarray
    twop_label_ids: np.ndarray
    fov_span: int
    z_warp_per_cell: np.ndarray   # z_map sampled at each 2P centroid (n_2p,)
    z_map: np.ndarray             # full 2D z_map(y,x); needed for 'curved' mode
    hcr_3d: np.ndarray            # full HCR label volume (Z, Y, X)


def _load_inputs_3d(twop_3d_path: Path,
                    params_path: Path,
                    hcr_masks_path: Path) -> Optional[_SomaprintInputs3D]:
    """Like _load_inputs but keeps full HCR 3D + z_map for 'curved'/'3d' modes."""
    for p in (twop_3d_path, params_path, hcr_masks_path):
        if not Path(p).exists():
            rprint(f"  [yellow][somaprint] missing input: {p}[/yellow]")
            return None

    twop_3d = tifffile.imread(str(twop_3d_path))
    if twop_3d.ndim != 3:
        rprint(f"  [yellow][somaprint] unexpected 2P shape {twop_3d.shape}[/yellow]")
        return None
    twop_labels = twop_3d.max(axis=0).astype(np.uint16)

    npz = np.load(str(params_path))
    if 'z_map_local' not in npz.files:
        rprint(f"  [yellow][somaprint] z_map_local missing from {params_path}[/yellow]")
        return None
    z_map = npz['z_map_local']

    hcr_3d = tifffile.imread(str(hcr_masks_path))
    if hcr_3d.ndim != 3:
        rprint(f"  [yellow][somaprint] unexpected HCR shape {hcr_3d.shape}[/yellow]")
        return None

    ny, nx = twop_labels.shape
    if z_map.shape != (ny, nx):
        rprint(f"  [yellow][somaprint] z_map shape {z_map.shape} != 2P labels {twop_labels.shape}[/yellow]")
        return None

    twop_cents, twop_ids = _extract_centroids(twop_labels)
    if len(twop_cents) > 0:
        cy = np.clip(np.round(twop_cents[:, 0]).astype(int), 0, ny - 1)
        cx = np.clip(np.round(twop_cents[:, 1]).astype(int), 0, nx - 1)
        z_warp = z_map[cy, cx].astype(np.float32)
    else:
        z_warp = np.zeros(0, dtype=np.float32)

    return _SomaprintInputs3D(
        twop_labels=twop_labels,
        twop_centroids=twop_cents,
        twop_label_ids=twop_ids,
        fov_span=int(max(twop_labels.shape)),
        z_warp_per_cell=z_warp,
        z_map=z_map.astype(np.float32),
        hcr_3d=hcr_3d,
    )


def _build_neighbor_vectors(centroids: np.ndarray, m: int):
    n = len(centroids)
    if n == 0:
        return np.zeros((0, m, 2)), np.zeros((0, m), dtype=np.int64)
    m_eff = max(1, min(m, n - 1))
    tree = cKDTree(centroids)
    _, nbr_idx = tree.query(centroids, k=m_eff + 1)
    if nbr_idx.ndim == 1:
        nbr_idx = nbr_idx[:, None]
    nbr_idx = nbr_idx[:, 1:]
    vecs = centroids[nbr_idx] - centroids[:, None, :]
    if m_eff < m:
        pad_v = np.full((n, m - m_eff, 2), np.nan)
        pad_i = np.full((n, m - m_eff), -1, dtype=np.int64)
        vecs = np.concatenate([vecs, pad_v], axis=1)
        nbr_idx = np.concatenate([nbr_idx, pad_i], axis=1)
    return vecs, nbr_idx


def _radius_for(fov_span: int, params: dict) -> float:
    return params['radius_factor'] * (fov_span / params['lambda_scale'])


def _build_candidates(twop_cents: np.ndarray, hcr_cents: np.ndarray, radius: float):
    """Per-2P-cell HCR candidate indices within `radius`, flattened CSR-style as
    (cand_starts, cand_indices). Geometric — depends only on cell positions, so it is
    IDENTICAL across the identity-refinement rounds within one sheet; build once and reuse
    instead of rebuilding the KD-tree + query every round. query_ball_point is called once
    on the whole (n,2) array (vectorized) rather than per cell."""
    n_2p = len(twop_cents)
    cand_starts = np.zeros(n_2p + 1, dtype=np.int64)
    if n_2p == 0 or len(hcr_cents) == 0:
        return cand_starts, np.empty(0, dtype=np.int64)
    hcr_tree = cKDTree(hcr_cents)
    cand_lists = hcr_tree.query_ball_point(twop_cents, r=radius)
    for i, cl in enumerate(cand_lists):
        cand_starts[i + 1] = cand_starts[i] + len(cl)
    if cand_starts[-1] > 0:
        cand_indices = np.concatenate([np.asarray(cl, dtype=np.int64) for cl in cand_lists])
    else:
        cand_indices = np.empty(0, dtype=np.int64)
    return cand_starts, cand_indices


def _greedy_match_fast(cost: np.ndarray, n: int) -> float:
    c = cost.copy()
    n_actual = min(n, c.shape[0], c.shape[1])
    if n_actual == 0:
        return np.inf
    delta_sum = 0.0
    for _ in range(n_actual):
        ia, ib = np.unravel_index(np.argmin(c), c.shape)
        v = c[ia, ib]
        if not np.isfinite(v):
            break
        delta_sum += v
        c[ia, :] = np.inf
        c[:, ib] = np.inf
    return delta_sum / n_actual


# No cache=True: module is dual-imported as `src.somaprint` and `somaprint`; disk cache poisons across contexts.
@numba.njit
def _score_one_cell_jit(vecs_a, hcr_vecs_cand, dx_arr, lam, n_pairs):
    m_2p = vecs_a.shape[0]
    n_cand = hcr_vecs_cand.shape[0]
    m_hcr = hcr_vecs_cand.shape[1]
    if n_cand < 2:
        return np.nan, np.nan, -1
    n_actual = min(n_pairs, m_2p, m_hcr)
    cost = np.empty((m_2p, m_hcr), dtype=np.float64)
    best_score = -np.inf
    second_score = -np.inf
    best_idx = -1
    for k in range(n_cand):
        for i in range(m_2p):
            for j in range(m_hcr):
                d0 = vecs_a[i, 0] - hcr_vecs_cand[k, j, 0]
                d1 = vecs_a[i, 1] - hcr_vecs_cand[k, j, 1]
                d = np.sqrt(d0 * d0 + d1 * d1)
                if not np.isfinite(d):
                    d = np.inf
                cost[i, j] = d
        n_matched = 0
        delta_sum = 0.0
        for _ in range(n_actual):
            min_val = np.inf
            ia = -1
            ib = -1
            for i in range(m_2p):
                for j in range(m_hcr):
                    if cost[i, j] < min_val:
                        min_val = cost[i, j]
                        ia = i
                        ib = j
            if not np.isfinite(min_val):
                break
            delta_sum += min_val
            n_matched += 1
            for j in range(m_hcr):
                cost[ia, j] = np.inf
            for i in range(m_2p):
                cost[i, ib] = np.inf
        if n_matched == 0:
            continue
        delta_bar = delta_sum / n_matched
        pen = np.exp(-(dx_arr[k] / lam) ** 2)
        score = (100.0 / (1.0 + delta_bar)) * pen
        if score > best_score:
            second_score = best_score
            best_score = score
            best_idx = k
        elif score > second_score:
            second_score = score
    if best_idx < 0:
        return np.nan, np.nan, -1
    return best_score, second_score, best_idx


@numba.njit(parallel=True)
def _score_identity_prange(twop_cents, hcr_cents,
                            conf_starts, conf_indices_2p, conf_indices_hcr,
                            cand_starts, cand_indices, lam):
    n_2p = twop_cents.shape[0]
    best_score = np.full(n_2p, np.nan, dtype=np.float64)
    second_score = np.full(n_2p, np.nan, dtype=np.float64)
    best_partner = np.full(n_2p, -1, dtype=np.int64)
    valid = np.zeros(n_2p, dtype=np.bool_)
    for i_a in numba.prange(n_2p):
        c_start = conf_starts[i_a]
        c_end = conf_starts[i_a + 1]
        n_neigh = c_end - c_start
        if n_neigh < 5:
            continue
        k_start = cand_starts[i_a]
        k_end = cand_starts[i_a + 1]
        n_cand = k_end - k_start
        if n_cand < 2:
            continue
        best = -np.inf
        second = -np.inf
        best_cand = -1
        for k in range(n_cand):
            cand_idx = cand_indices[k_start + k]
            sum_norm = 0.0
            for j in range(n_neigh):
                conf_j_2p = conf_indices_2p[c_start + j]
                conf_j_hcr = conf_indices_hcr[c_start + j]
                va0 = twop_cents[conf_j_2p, 0] - twop_cents[i_a, 0]
                va1 = twop_cents[conf_j_2p, 1] - twop_cents[i_a, 1]
                vb0 = hcr_cents[conf_j_hcr, 0] - hcr_cents[cand_idx, 0]
                vb1 = hcr_cents[conf_j_hcr, 1] - hcr_cents[cand_idx, 1]
                d0 = va0 - vb0
                d1 = va1 - vb1
                sum_norm += np.sqrt(d0 * d0 + d1 * d1)
            dbar = sum_norm / n_neigh
            dx0 = hcr_cents[cand_idx, 0] - twop_cents[i_a, 0]
            dx1 = hcr_cents[cand_idx, 1] - twop_cents[i_a, 1]
            dx = np.sqrt(dx0 * dx0 + dx1 * dx1)
            pen = np.exp(-(dx / lam) ** 2)
            score = (100.0 / (1.0 + dbar)) * pen
            if score >= best:
                second = best
                best = score
                best_cand = cand_idx
            elif score >= second:
                second = score
        if best_cand >= 0:
            best_score[i_a] = best
            second_score[i_a] = second
            best_partner[i_a] = best_cand
            valid[i_a] = True
    return best_score, second_score, best_partner, valid


@numba.njit(parallel=True)
def _score_all_cells_prange(twop_vecs, hcr_vecs, twop_cents, hcr_cents,
                             cand_starts, cand_indices, lam, n_pairs):
    n_2p = twop_vecs.shape[0]
    m_hcr = hcr_vecs.shape[1]
    best_score = np.full(n_2p, np.nan, dtype=np.float64)
    second_score = np.full(n_2p, np.nan, dtype=np.float64)
    best_partner = np.full(n_2p, -1, dtype=np.int64)
    for i_a in numba.prange(n_2p):
        start = cand_starts[i_a]
        end = cand_starts[i_a + 1]
        n_cand = end - start
        if n_cand < 2:
            continue
        hcr_vecs_cand = np.empty((n_cand, m_hcr, 2), dtype=np.float64)
        dx_arr = np.empty(n_cand, dtype=np.float64)
        for k in range(n_cand):
            idx = cand_indices[start + k]
            for j in range(m_hcr):
                hcr_vecs_cand[k, j, 0] = hcr_vecs[idx, j, 0]
                hcr_vecs_cand[k, j, 1] = hcr_vecs[idx, j, 1]
            d0 = hcr_cents[idx, 0] - twop_cents[i_a, 0]
            d1 = hcr_cents[idx, 1] - twop_cents[i_a, 1]
            dx_arr[k] = np.sqrt(d0 * d0 + d1 * d1)
        bs, ss, bi = _score_one_cell_jit(twop_vecs[i_a], hcr_vecs_cand,
                                          dx_arr, lam, n_pairs)
        best_score[i_a] = bs
        second_score[i_a] = ss
        if bi >= 0:
            best_partner[i_a] = cand_indices[start + bi]
    return best_score, second_score, best_partner


def _gmm_density(scores: np.ndarray, gmm: GaussianMixture) -> np.ndarray:
    means = gmm.means_.ravel()
    stds = np.sqrt(gmm.covariances_.ravel())
    ws = gmm.weights_
    return (ws[0] * stats.norm.pdf(scores, means[0], stds[0])
            + ws[1] * stats.norm.pdf(scores, means[1], stds[1]))


def _curate_with_gmm(best_score: np.ndarray,
                     second_score: np.ndarray,
                     best_partner: np.ndarray,
                     valid_mask: np.ndarray,
                     lr_threshold: float,
                     min_cells_for_gmm: int) -> dict:
    n_valid = int(valid_mask.sum())
    if n_valid < min_cells_for_gmm:
        return {}

    best_v = best_score[valid_mask]
    second_v = second_score[valid_mask]
    mu_2nd = float(np.mean(second_v))
    sd_2nd = float(np.std(second_v))
    if sd_2nd <= 0:
        return {}

    try:
        gmm = GaussianMixture(n_components=2, random_state=0).fit(best_v.reshape(-1, 1))
    except Exception:  # noqa: BLE001
        threshold = float(np.percentile(best_v, 75))
        return {int(vi): int(best_partner[vi])
                for vi in np.where(valid_mask)[0]
                if best_score[vi] >= threshold and best_partner[vi] >= 0}

    p_inc = stats.norm.pdf(best_score, mu_2nd, sd_2nd)
    p_gmm = _gmm_density(best_score, gmm)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = p_inc / p_gmm
        lr = np.where(np.isfinite(ratio), ratio, np.inf)

    return {int(vi): int(best_partner[vi])
            for vi in np.where(valid_mask)[0]
            if lr[vi] < lr_threshold and best_partner[vi] >= 0}


def _score_round1(inputs: _SomaprintInputs, params: dict, plane: int, cand=None) -> _RoundResult:
    fov_span = inputs.fov_span
    lam = fov_span / params['lambda_scale']
    n_pairs = params['n_pairs']
    twop_cents = inputs.twop_centroids
    hcr_cents = inputs.hcr_centroids
    n_2p = len(twop_cents)
    if n_2p == 0 or len(hcr_cents) == 0:
        return _RoundResult(
            best_score=np.full(n_2p, np.nan),
            second_score=np.full(n_2p, np.nan),
            best_partner=np.full(n_2p, -1, dtype=np.int64),
            confident={},
        )
    twop_vecs, _ = _build_neighbor_vectors(twop_cents, params['m_2p'])
    hcr_vecs, _ = _build_neighbor_vectors(hcr_cents, params['m_hcr'])
    twop_vecs = np.ascontiguousarray(twop_vecs.astype(np.float64))
    hcr_vecs = np.ascontiguousarray(hcr_vecs.astype(np.float64))
    twop_cents_f = np.ascontiguousarray(twop_cents.astype(np.float64))
    hcr_cents_f = np.ascontiguousarray(hcr_cents.astype(np.float64))
    cand_starts, cand_indices = (cand if cand is not None
                                 else _build_candidates(twop_cents, hcr_cents, _radius_for(fov_span, params)))
    best_score, second_score, best_partner = _score_all_cells_prange(
        twop_vecs, hcr_vecs, twop_cents_f, hcr_cents_f,
        cand_starts, cand_indices, float(lam), int(n_pairs))
    valid_mask = ~np.isnan(best_score) & ~np.isnan(second_score)
    confident = _curate_with_gmm(
        best_score, second_score, best_partner, valid_mask,
        lr_threshold=params['lr_threshold'],
        min_cells_for_gmm=params['min_cells_for_gmm'],
    )
    return _RoundResult(best_score=best_score, second_score=second_score,
                       best_partner=best_partner, confident=confident)


def _score_round_identity(inputs: _SomaprintInputs,
                          params: dict,
                          current_matches: dict,
                          cand=None):
    fov_span = inputs.fov_span
    lam = fov_span / params['lambda_scale']
    n_pairs_r2 = params['n_pairs_r2']
    min_conf = params['min_confirmed_for_round2']

    twop_cents = inputs.twop_centroids
    hcr_cents = inputs.hcr_centroids
    n_2p = len(twop_cents)

    confirmed_2p_idxs = list(current_matches.keys())
    confirmed_2p_array = np.array(confirmed_2p_idxs, dtype=np.int64)
    confirmed_hcr_array = np.array([current_matches[i] for i in confirmed_2p_idxs], dtype=np.int64)

    blank = (np.full(n_2p, np.nan),
             np.full(n_2p, np.nan),
             np.full(n_2p, -1, dtype=np.int64),
             np.zeros(n_2p, dtype=bool))
    if len(confirmed_2p_array) < min_conf:
        return blank

    confirmed_tree = cKDTree(twop_cents[confirmed_2p_array])
    n_query = n_pairs_r2 + 2
    k_query = min(n_query, len(confirmed_2p_array))

    conf_neigh_2p_lists = []
    conf_neigh_hcr_lists = []
    for i_a in range(n_2p):
        d_to_conf, idxs_in_conf = confirmed_tree.query(twop_cents[i_a], k=k_query)
        d_to_conf = np.atleast_1d(d_to_conf)
        idxs_in_conf = np.atleast_1d(idxs_in_conf)
        idxs_in_conf = idxs_in_conf[d_to_conf > 1e-6][:n_pairs_r2]
        conf_neigh_2p_lists.append(confirmed_2p_array[idxs_in_conf])
        conf_neigh_hcr_lists.append(confirmed_hcr_array[idxs_in_conf])
    conf_starts = np.zeros(n_2p + 1, dtype=np.int64)
    for i, cl in enumerate(conf_neigh_2p_lists):
        conf_starts[i + 1] = conf_starts[i] + len(cl)
    conf_indices_2p = (np.concatenate(conf_neigh_2p_lists).astype(np.int64)
                       if any(len(cl) for cl in conf_neigh_2p_lists)
                       else np.empty(0, dtype=np.int64))
    conf_indices_hcr = (np.concatenate(conf_neigh_hcr_lists).astype(np.int64)
                        if any(len(cl) for cl in conf_neigh_hcr_lists)
                        else np.empty(0, dtype=np.int64))

    cand_starts, cand_indices = (cand if cand is not None
                                 else _build_candidates(twop_cents, hcr_cents, _radius_for(fov_span, params)))

    twop_cents_f = np.ascontiguousarray(twop_cents.astype(np.float64))
    hcr_cents_f = np.ascontiguousarray(hcr_cents.astype(np.float64))

    return _score_identity_prange(
        twop_cents_f, hcr_cents_f,
        conf_starts, conf_indices_2p, conf_indices_hcr,
        cand_starts, cand_indices, float(lam))


def _iterate_rounds(inputs: _SomaprintInputs,
                    params: dict,
                    round1: _RoundResult,
                    cand=None) -> _RoundResult:
    """Run identity-paired refinement; return the final-round result. `cand` is the prebuilt
    (cand_starts, cand_indices) for this sheet — geometric and identical every round, so we
    pass it through instead of rebuilding the HCR KD-tree + candidate query each round."""
    current = round1
    for _ in range(params['max_rounds']):
        if len(current.confident) < params['min_confirmed_for_round2']:
            break
        best, second, partner, valid = _score_round_identity(inputs, params, current.confident, cand=cand)
        n_valid = int(valid.sum())
        if n_valid < params['min_cells_for_gmm']:
            break
        new_matches = _curate_with_gmm(
            best, second, partner, valid,
            lr_threshold=params['lr_threshold'],
            min_cells_for_gmm=params['min_cells_for_gmm'],
        )
        prev_n = len(current.confident)
        new_n = len(new_matches)
        delta_pct = 100.0 * abs(new_n - prev_n) / max(prev_n, 1)
        current = _RoundResult(best_score=best, second_score=second,
                              best_partner=partner, confident=new_matches)
        if delta_pct < params['terminate_pct']:
            break
    return current


def _run_curved_match(inputs: _SomaprintInputs3D, params: dict, plane: int) -> dict:
    """Curved-sheet TRU-FACT (paper-accurate for tilted-2P geometry).

    For d in [-z_band, +z_band]: sample HCR along z_map(y,x)+d, run full
    somaprint (round1 + identity rounds + GMM curation) on all 2P cells.
    Per cell: argmax score across offsets; confidence inherits from the
    winning offset's GMM. Returns arrays {score, second, hcr_label, offset, confident}.
    """
    z_band = int(params['z_band'])
    n_2p = len(inputs.twop_centroids)

    top1_score     = np.full(n_2p, -np.inf, dtype=np.float64)
    top1_second    = np.full(n_2p, np.nan,  dtype=np.float64)
    top1_hcr_label = np.zeros(n_2p,         dtype=np.int64)
    top1_offset    = np.full(n_2p, np.iinfo(np.int32).min, dtype=np.int32)
    top1_confident = np.zeros(n_2p,         dtype=bool)

    if n_2p == 0:
        return dict(score=top1_score, second=top1_second,
                    hcr_label=top1_hcr_label, offset=top1_offset,
                    confident=top1_confident)

    z_map = inputs.z_map
    ny, nx = z_map.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    offsets = list(range(-z_band, z_band + 1))

    desc = f"  [somaprint] plane {plane} curved scan (d=-{z_band}..+{z_band})"
    for d in tqdm(offsets, desc=desc, leave=False):
        coords = np.stack([z_map + float(d), yy, xx], axis=0)
        hcr_2d = map_coordinates(inputs.hcr_3d, coords, order=0,
                                 mode='constant', cval=0).astype(np.uint16)
        if hcr_2d.max() == 0:
            continue
        hcr_cents, hcr_ids = _extract_centroids(hcr_2d)
        if len(hcr_cents) < 2:
            continue

        sheet_inputs = _SomaprintInputs(
            twop_labels=inputs.twop_labels,
            hcr_labels=hcr_2d,
            twop_centroids=inputs.twop_centroids,
            hcr_centroids=hcr_cents,
            twop_label_ids=inputs.twop_label_ids,
            hcr_label_ids=hcr_ids,
            fov_span=inputs.fov_span,
        )
        cand = _build_candidates(sheet_inputs.twop_centroids, sheet_inputs.hcr_centroids,
                                 _radius_for(sheet_inputs.fov_span, params))
        round1 = _score_round1(sheet_inputs, params, plane, cand=cand)
        final = _iterate_rounds(sheet_inputs, params, round1, cand=cand)

        # Per-cell argmax aggregation; confidence inherits from winning offset.
        for i_a in range(n_2p):
            s = final.best_score[i_a]
            bp = int(final.best_partner[i_a])
            if not np.isfinite(s) or bp < 0:
                continue
            if s > top1_score[i_a]:
                top1_score[i_a]     = s
                top1_second[i_a]    = final.second_score[i_a]
                top1_hcr_label[i_a] = int(hcr_ids[bp])
                top1_offset[i_a]    = d
                top1_confident[i_a] = i_a in final.confident

    return dict(score=top1_score, second=top1_second,
                hcr_label=top1_hcr_label, offset=top1_offset,
                confident=top1_confident)


def _run_3d_match(inputs: _SomaprintInputs3D, params: dict, plane: int) -> dict:
    """Flat-slice TRU-FACT (legacy; 'curved' mode is the default for tilted 2P).

    For each absolute HCR z in [floor(zwarp.min())-z_band, ceil(zwarp.max())+z_band]:
    run full somaprint on the z-local subset of 2P cells (|z - z_warp| <= z_band).
    Per cell: argmax score across slices. Returns {score, second, hcr_label, z, confident}.
    """
    z_band = int(params['z_band'])
    n_2p = len(inputs.twop_centroids)
    n_z_total = inputs.hcr_3d.shape[0]

    top1_score     = np.full(n_2p, -np.inf, dtype=np.float64)
    top1_second    = np.full(n_2p, np.nan,  dtype=np.float64)
    top1_hcr_label = np.zeros(n_2p,         dtype=np.int64)
    top1_z         = np.full(n_2p, -1,      dtype=np.int32)
    top1_confident = np.zeros(n_2p,         dtype=bool)

    if n_2p == 0:
        return dict(score=top1_score, second=top1_second,
                    hcr_label=top1_hcr_label, z=top1_z, confident=top1_confident)

    z_warp = inputs.z_warp_per_cell
    z_lo = max(0, int(np.floor(z_warp.min())) - z_band)
    z_hi = min(n_z_total, int(np.ceil(z_warp.max())) + z_band + 1)

    desc = f"  [somaprint] plane {plane} 3D scan (z={z_lo}..{z_hi - 1})"
    for z in tqdm(range(z_lo, z_hi), desc=desc, leave=False):
        in_band = np.abs(z - z_warp) <= z_band
        n_in = int(in_band.sum())
        if n_in < params['min_cells_for_gmm']:
            continue   # not enough cells for the GMM curation to be stable
        hcr_2d = inputs.hcr_3d[z].astype(np.uint16)
        if hcr_2d.max() == 0:
            continue
        hcr_cents, hcr_ids = _extract_centroids(hcr_2d)
        if len(hcr_cents) < 2:
            continue

        in_band_idx = np.where(in_band)[0]
        slice_inputs = _SomaprintInputs(
            twop_labels=inputs.twop_labels,
            hcr_labels=hcr_2d,
            twop_centroids=inputs.twop_centroids[in_band],
            hcr_centroids=hcr_cents,
            twop_label_ids=inputs.twop_label_ids[in_band],
            hcr_label_ids=hcr_ids,
            fov_span=inputs.fov_span,
        )
        cand = _build_candidates(slice_inputs.twop_centroids, slice_inputs.hcr_centroids,
                                 _radius_for(slice_inputs.fov_span, params))
        round1 = _score_round1(slice_inputs, params, plane, cand=cand)
        final = _iterate_rounds(slice_inputs, params, round1, cand=cand)

        # Aggregate winners. final.* arrays are indexed against the in-band
        # subset; map back to full-array indices via in_band_idx.
        for j, i_a in enumerate(in_band_idx):
            s = final.best_score[j]
            bp = int(final.best_partner[j])
            if not np.isfinite(s) or bp < 0:
                continue
            if s > top1_score[i_a]:
                top1_score[i_a]     = s
                top1_second[i_a]    = final.second_score[j]
                top1_hcr_label[i_a] = int(hcr_ids[bp])
                top1_z[i_a]         = z
                top1_confident[i_a] = j in final.confident

    return dict(score=top1_score, second=top1_second,
                hcr_label=top1_hcr_label, z=top1_z, confident=top1_confident)


def run_for_plane(full_manifest: dict, session: dict, hcr_round: str) -> dict:
    """Match 2P cells to HCR cells via somaprint for a single plane.

    Inputs are sourced from the pipeline's existing per-plane registration
    outputs and HCR cellpose masks; this function adds no new prerequisites.

    Returns a per-2P-label dict
        {twop_label: (somaprint_hcr_label, best_score, second_score, confident)}
    covering every 2P cell with a candidate; cells with no candidate are
    omitted. `confident` is True for picks that passed the GMM/likelihood
    curation, False for unconfident best-picks (still useful diagnostically
    via the score columns).

    Returns an empty dict if inputs are unavailable, no cells were detected,
    or somaprint is disabled in the manifest.
    """
    params = get_params(full_manifest)
    if not params.get('enabled', True):
        return {}

    plane = session['functional_plane'][0]
    reg_dir = output_root(full_manifest) / '2P' / 'registered'
    twop_3d_path = reg_dir / f'twop_plane{plane}_aligned_3d.tiff'
    params_path = reg_dir / f'twop_plane{plane}_registration_params.npz'
    hcr_masks_path = output_root(full_manifest) / 'HCR' / 'cellpose' / f'HCR{hcr_round}_masks.tiff'

    t0 = time.time()
    mode = str(params.get('mode', 'curved')).lower()

    if mode in ('curved', '3d'):
        inputs3d = _load_inputs_3d(twop_3d_path, params_path, hcr_masks_path)
        if inputs3d is None:
            return {}
        n_2p = len(inputs3d.twop_centroids)
        n_z  = inputs3d.hcr_3d.shape[0]
        if n_2p == 0:
            rprint(f"  [yellow][somaprint] plane {plane}: no 2P cells; skipping[/yellow]")
            return {}
        rprint(f"  [somaprint] plane {plane}: {n_2p} 2P cells, '{mode}' mode "
               f"(z_band=+/-{params['z_band']}, HCR z={n_z}), scoring...")
        if mode == 'curved':
            result = _run_curved_match(inputs3d, params, plane)
        else:
            result = _run_3d_match(inputs3d, params, plane)

        out = {}
        for i_a in range(n_2p):
            hcr_lbl = int(result['hcr_label'][i_a])
            if hcr_lbl == 0:
                continue
            twop_lbl = int(inputs3d.twop_label_ids[i_a])
            bs = float(result['score'][i_a]) if np.isfinite(result['score'][i_a]) else float('nan')
            ss = float(result['second'][i_a]) if np.isfinite(result['second'][i_a]) else float('nan')
            confident = bool(result['confident'][i_a])
            out[twop_lbl] = (hcr_lbl, bs, ss, confident)
        n_conf = sum(1 for v in out.values() if v[3])
        rprint(f"  [somaprint] plane {plane}: {n_conf}/{n_2p} cells confidently matched "
               f"('{mode}' mode, {time.time() - t0:.1f}s)")
        return out

    # mode == '2d' (legacy single-plane sample at z_map)
    inputs = _load_inputs(twop_3d_path, params_path, hcr_masks_path)
    if inputs is None:
        return {}
    n_2p, n_hcr = len(inputs.twop_centroids), len(inputs.hcr_centroids)
    if n_2p == 0 or n_hcr == 0:
        rprint(f"  [yellow][somaprint] plane {plane}: empty masks ({n_2p} 2P / {n_hcr} HCR); skipping[/yellow]")
        return {}

    rprint(f"  [somaprint] plane {plane}: {n_2p} 2P + {n_hcr} HCR cells, 2D mode, scoring...")
    cand = _build_candidates(inputs.twop_centroids, inputs.hcr_centroids,
                             _radius_for(inputs.fov_span, params))
    round1 = _score_round1(inputs, params, plane, cand=cand)
    final = _iterate_rounds(inputs, params, round1, cand=cand)

    # Build per-2P-label output. Index space → label space happens here so
    # downstream code only sees cellpose label IDs.
    out = {}
    for i_a in range(len(inputs.twop_label_ids)):
        bp = int(final.best_partner[i_a])
        if bp < 0:
            continue
        twop_lbl = int(inputs.twop_label_ids[i_a])
        hcr_lbl = int(inputs.hcr_label_ids[bp])
        bs = float(final.best_score[i_a]) if np.isfinite(final.best_score[i_a]) else float('nan')
        ss = float(final.second_score[i_a]) if np.isfinite(final.second_score[i_a]) else float('nan')
        confident = i_a in final.confident
        out[twop_lbl] = (hcr_lbl, bs, ss, confident)

    n_conf = sum(1 for v in out.values() if v[3])
    rprint(f"  [somaprint] plane {plane}: {n_conf}/{n_2p} cells confidently matched "
           f"(2D mode, {time.time() - t0:.1f}s)")
    return out
