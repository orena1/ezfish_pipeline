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

import numpy as np
import tifffile
from scipy import stats
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from skimage.measure import regionprops
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
    # Algorithm mode:
    #   '3d' = TRU-FACT 3D Soma-print (Schnitzer et al., bioRxiv 2026.04.28).
    #          Runs the full 2D somaprint at each HCR z-slice within a per-cell
    #          curved band (z_map(centroid) +/- z_band), then picks the
    #          (slice, HCR cell) with the highest score per 2P cell. The paper
    #          explicitly verifies precision/recall of this "best match in 3D"
    #          is equivalent to the full pipeline with TPS manifold refit +
    #          slab projection (those are for downstream visualisation).
    #   '2d' = legacy single-plane mode: sample HCR at z_map then run 2D
    #          somaprint once. Cheaper, but underestimates matches when the
    #          cascade's z_map has local error > ~1 plane (see
    #          figure_4_cortex/somaprint_tests/ analyses).
    mode='3d',
    z_band=2,                       # per-cell curved z-window for '3d' mode
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
    props = regionprops(labels_2d.astype(np.int32))
    if not props:
        return np.zeros((0, 2)), np.zeros(0, dtype=np.int64)
    cents = np.array([[p.centroid[0], p.centroid[1]] for p in props])
    ids = np.array([p.label for p in props], dtype=np.int64)
    return cents, ids


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
    """Inputs for 3D mode: full HCR volume + per-cell z_warp from z_map."""
    twop_labels: np.ndarray
    twop_centroids: np.ndarray
    twop_label_ids: np.ndarray
    fov_span: int
    z_warp_per_cell: np.ndarray   # z_map sampled at each 2P centroid (n_2p,)
    hcr_3d: np.ndarray            # full HCR label volume (Z, Y, X)


def _load_inputs_3d(twop_3d_path: Path,
                    params_path: Path,
                    hcr_masks_path: Path) -> Optional[_SomaprintInputs3D]:
    """Like _load_inputs but keeps HCR 3D (no per-pixel z_map sampling).

    The 2D loader pre-samples HCR at z_map(y, x) and returns a single 2D
    HCR slice. 3D mode needs the full HCR volume so the per-z somaprint
    scan can look at slices on either side of z_map. We also pre-compute
    each 2P cell's z_warp = z_map(centroid) so the per-z scan can quickly
    select which cells are in-band at a given HCR slice.
    """
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


def _score_round1(inputs: _SomaprintInputs, params: dict, plane: int) -> _RoundResult:
    fov_span = inputs.fov_span
    lam = fov_span / params['lambda_scale']
    radius = params['radius_factor'] * lam
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
    hcr_tree = cKDTree(hcr_cents)

    best_score = np.full(n_2p, np.nan)
    second_score = np.full(n_2p, np.nan)
    best_partner = np.full(n_2p, -1, dtype=np.int64)

    desc = f"  [somaprint] plane {plane} scoring"
    for i_a in tqdm(range(n_2p), desc=desc, leave=False):
        cand = hcr_tree.query_ball_point(twop_cents[i_a], r=radius)
        if len(cand) < 2:
            continue
        vecs_a = twop_vecs[i_a]
        vecs_b_batch = hcr_vecs[cand]
        diff = vecs_a[None, :, None, :] - vecs_b_batch[:, None, :, :]
        cost_batch = np.linalg.norm(diff, axis=-1)
        cost_batch = np.where(np.isnan(cost_batch), np.inf, cost_batch)
        dx_arr = np.linalg.norm(hcr_cents[cand] - twop_cents[i_a], axis=1)
        pen_arr = np.exp(-(dx_arr / lam) ** 2)

        scores = np.empty(len(cand))
        for k, c in enumerate(cost_batch):
            delta_bar = _greedy_match_fast(c, n_pairs)
            scores[k] = (100.0 / (1.0 + delta_bar)) * pen_arr[k]

        order = np.argsort(scores)[::-1]
        best_score[i_a] = scores[order[0]]
        second_score[i_a] = scores[order[1]] if len(order) > 1 else np.nan
        best_partner[i_a] = cand[order[0]]

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
                          current_matches: dict):
    fov_span = inputs.fov_span
    lam = fov_span / params['lambda_scale']
    radius = params['radius_factor'] * lam
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
    hcr_tree = cKDTree(hcr_cents)

    best_score = np.full(n_2p, np.nan)
    second_score = np.full(n_2p, np.nan)
    best_partner = np.full(n_2p, -1, dtype=np.int64)
    valid = np.zeros(n_2p, dtype=bool)
    n_query = n_pairs_r2 + 2

    for i_a in range(n_2p):
        k_query = min(n_query, len(confirmed_2p_array))
        d_to_conf, idxs_in_conf = confirmed_tree.query(twop_cents[i_a], k=k_query)
        d_to_conf = np.atleast_1d(d_to_conf)
        idxs_in_conf = np.atleast_1d(idxs_in_conf)
        idxs_in_conf = idxs_in_conf[d_to_conf > 1e-6][:n_pairs_r2]
        if len(idxs_in_conf) < 5:
            continue

        confirmed_2p_neigh = confirmed_2p_array[idxs_in_conf]
        confirmed_hcr_neigh = confirmed_hcr_array[idxs_in_conf]
        vecs_a = twop_cents[confirmed_2p_neigh] - twop_cents[i_a]

        cands = hcr_tree.query_ball_point(twop_cents[i_a], r=radius)
        if len(cands) < 2:
            continue
        cands_arr = np.array(cands)

        b_centroids = hcr_cents[cands_arr][:, None, :]
        b_neigh_centroids = hcr_cents[confirmed_hcr_neigh][None, :, :]
        vecs_b_batch = b_neigh_centroids - b_centroids

        diff = vecs_a[None, :, :] - vecs_b_batch
        dbar = np.linalg.norm(diff, axis=2).mean(axis=1)
        dx_arr = np.linalg.norm(hcr_cents[cands_arr] - twop_cents[i_a], axis=1)
        pen_arr = np.exp(-(dx_arr / lam) ** 2)
        scores = (100.0 / (1.0 + dbar)) * pen_arr

        order = np.argsort(scores)[::-1]
        best_score[i_a] = scores[order[0]]
        second_score[i_a] = scores[order[1]] if len(order) > 1 else np.nan
        best_partner[i_a] = cands_arr[order[0]]
        valid[i_a] = True

    return best_score, second_score, best_partner, valid


def _iterate_rounds(inputs: _SomaprintInputs,
                    params: dict,
                    round1: _RoundResult) -> _RoundResult:
    """Run identity-paired refinement; return the final-round result."""
    current = round1
    for _ in range(params['max_rounds']):
        if len(current.confident) < params['min_confirmed_for_round2']:
            break
        best, second, partner, valid = _score_round_identity(inputs, params, current.confident)
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


def _run_3d_match(inputs: _SomaprintInputs3D, params: dict, plane: int) -> dict:
    """Per-slice full 2D somaprint + per-cell max-score aggregation across z.

    Implements the TRU-FACT 3D Soma-print algorithm:
      1. For each HCR z-slice in the outer range
         [floor(z_warp.min()) - z_band, ceil(z_warp.max()) + z_band], run
         the full 2D somaprint (round 1 + identity rounds + GMM curation)
         on the 2P cells whose own z_warp is within z_band of z (curved
         per-cell band that follows the folded sheet).
      2. For each 2P cell, pick the (z, HCR cell) with the highest final-
         round somaprint score across all slices it was scored at.

    The paper's downstream TPS-manifold refit + slab projection are skipped
    because their purpose is to produce a coherent 2D HCR reference image
    for visualisation; the paper explicitly verifies match precision/recall
    are unchanged when you stop at the per-cell argmax-z stage.

    Returns a dict of n_2p-length arrays:
      score, second, hcr_label, z, confident.
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
        round1 = _score_round1(slice_inputs, params, plane)
        final = _iterate_rounds(slice_inputs, params, round1)

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
    mode = str(params.get('mode', '3d')).lower()

    if mode == '3d':
        inputs3d = _load_inputs_3d(twop_3d_path, params_path, hcr_masks_path)
        if inputs3d is None:
            return {}
        n_2p = len(inputs3d.twop_centroids)
        n_z  = inputs3d.hcr_3d.shape[0]
        if n_2p == 0:
            rprint(f"  [yellow][somaprint] plane {plane}: no 2P cells; skipping[/yellow]")
            return {}
        rprint(f"  [somaprint] plane {plane}: {n_2p} 2P cells, 3D mode "
               f"(z_band=+/-{params['z_band']}, HCR z={n_z}), scoring...")
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
               f"(3D mode, {time.time() - t0:.1f}s)")
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
    round1 = _score_round1(inputs, params, plane)
    final = _iterate_rounds(inputs, params, round1)

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
