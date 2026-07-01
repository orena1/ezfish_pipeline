"""
Automated tile stitching for ezfish pipeline.

Replaces manual BigStitcher workflow for high-resolution 2P tiled imaging.

ALGORITHM OVERVIEW
==================
For each pair of adjacent tiles, we find the optimal (dy, dx) shift by:
1. Using the config's overlap_fraction as initial guess for expected dx
2. Searching over a range of shifts centered on that guess
3. Computing normalized correlation for each candidate shift
4. Selecting the shift with highest correlation

The search is done independently for each z-plane, then results are
combined into a consensus shift (or used per-plane if variability is high).

COORDINATE CONVENTION
=====================
- All shifts are in PIXELS
- dy: vertical offset (positive = tile2 is below tile1)
- dx: horizontal offset (positive = tile2 is to the right of tile1)
- Tiles are numbered 1, 2, 3, ... from LEFT to RIGHT
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

try:
    from .meta import rprint  # Relative import (running as part of a package)
except ImportError:
    from meta import rprint  # Absolute import (running in Jupyter notebook)


class StitchingError(Exception):
    """Custom exception for stitching failures."""
    pass


def find_horizontal_pairs(num_tiles: int) -> List[Tuple[int, int]]:
    """
    Find pairs of horizontally adjacent tiles.
    Assumes tiles are numbered sequentially from left to right.
    """
    return [(i, i+1) for i in range(1, num_tiles)]


def compute_overlap_correlation(img1, img2, dy, dx, noise_floor=15.0):
    """
    Compute normalized correlation for overlapping region at given shift.

    If tile2 is placed at position (dy, dx) relative to tile1 at origin,
    this computes how well the overlapping pixels correlate.

    Returns correlation coefficient (-1 to 1), or -inf if invalid.
    """
    h1, w1 = img1.shape
    h2, w2 = img2.shape

    # Calculate overlap region bounds
    # tile1 spans [0, h1) x [0, w1)
    # tile2 spans [dy, dy+h2) x [dx, dx+w2)
    y1_start, y1_end = max(0, dy), min(h1, h2 + dy)
    x1_start, x1_end = max(0, dx), min(w1, w2 + dx)
    y2_start, y2_end = max(0, -dy), min(h2, h1 - dy)
    x2_start, x2_end = max(0, -dx), min(w2, w1 - dx)

    # Check for valid overlap
    if y1_end <= y1_start or x1_end <= x1_start:
        return -np.inf

    # Extract overlapping regions
    overlap1 = img1[y1_start:y1_end, x1_start:x1_end]
    overlap2 = img2[y2_start:y2_end, x2_start:x2_end]

    # Mask out low-signal pixels (background)
    mask = (overlap1 > noise_floor) & (overlap2 > noise_floor)
    if mask.sum() < 100:
        return -np.inf

    # Percentile-based normalization (robust to outliers)
    o1 = overlap1[mask].astype(np.float32)
    o2 = overlap2[mask].astype(np.float32)

    p5_1, p95_1 = np.percentile(o1, [5, 95])
    p5_2, p95_2 = np.percentile(o2, [5, 95])

    o1_clipped = np.clip(o1, p5_1, p95_1)
    o2_clipped = np.clip(o2, p5_2, p95_2)

    o1_norm = (o1_clipped - o1_clipped.mean()) / (o1_clipped.std() + 1e-8)
    o2_norm = (o2_clipped - o2_clipped.mean()) / (o2_clipped.std() + 1e-8)

    return np.mean(o1_norm * o2_norm)


def find_best_shift_phase(img1, img2, expected_dx, noise_floor=15.0, dy_range=150, k=5):
    """
    Top-K phase-corr peaks scored by honest NCC on the actual non-dark overlap.
    Robust against wrong-period lock-on from repetitive features.
    """
    h, w = img1.shape
    expected_overlap = w - expected_dx
    if expected_overlap <= 20:
        return find_best_shift_grid(img1, img2, expected_dx, noise_floor, dy_range)

    strip_width = min(int(expected_overlap * 1.5), w)
    strip1 = img1[:, w - strip_width:].astype(np.float64)
    strip2 = img2[:, :strip_width].astype(np.float64)

    if (strip1 > noise_floor).sum() < 1000 or (strip2 > noise_floor).sum() < 1000:
        return find_best_shift_grid(img1, img2, expected_dx, noise_floor, dy_range)

    try:
        # Phase-correlation surface (no fftshift — wrap-index below).
        F1 = np.fft.fft2(strip1)
        F2 = np.fft.fft2(strip2)
        R = F1 * np.conj(F2)
        surface = np.fft.ifft2(R / (np.abs(R) + 1e-10)).real
    except Exception as e:
        warnings.warn(f"Phase correlation failed ({e}), falling back to grid search")
        return find_best_shift_grid(img1, img2, expected_dx, noise_floor, dy_range)

    sh, sw = surface.shape
    dx_margin = max(100, int(expected_dx * 0.5))
    dx_lo, dx_hi = max(10, expected_dx - dx_margin), expected_dx + dx_margin

    # Pull more than K from the surface — most won't pass plausibility filtering.
    flat = surface.ravel()
    pool = np.argpartition(flat, -min(k * 8, flat.size))[-min(k * 8, flat.size):]
    pool = pool[np.argsort(flat[pool])[::-1]]

    candidates = []
    seen = set()
    for idx in pool:
        ri, ci = np.unravel_index(idx, (sh, sw))
        dy_p = ri - sh if ri >= sh // 2 else ri
        dx_p = ci - sw if ci >= sw // 2 else ci
        # Wrap-around variants in case the true shift exceeds half the strip size.
        for dx_c in (dx_p, dx_p + sw, dx_p - sw):
            for dy_c in (dy_p, dy_p + sh, dy_p - sh):
                total_dx = (w - strip_width) + dx_c
                if abs(dy_c) <= dy_range and dx_lo <= total_dx <= dx_hi:
                    key = (int(round(dy_c)), int(round(total_dx)))
                    if key not in seen:
                        seen.add(key)
                        candidates.append(key)
        if len(candidates) >= k:
            break

    if not candidates:
        return find_best_shift_grid(img1, img2, expected_dx, noise_floor, dy_range)

    # Score each candidate by honest NCC on the actual overlap, ±5 px refine.
    best_corr, best_dy, best_dx = -np.inf, None, None
    for cy, cx in candidates:
        for dy_off in range(-5, 6):
            for dx_off in range(-5, 6):
                dy_t, dx_t = cy + dy_off, cx + dx_off
                corr = compute_overlap_correlation(img1, img2, dy_t, dx_t, noise_floor)
                if corr > best_corr:
                    best_corr, best_dy, best_dx = corr, dy_t, dx_t

    if best_corr < 0.1:
        return find_best_shift_grid(img1, img2, expected_dx, noise_floor, dy_range)
    return best_dy, best_dx, best_corr


def find_best_shift_grid(img1, img2, expected_dx, noise_floor=15.0, dy_range=150):
    """
    Find optimal (dy, dx) shift using coarse-to-fine grid search.
    Fallback for when phase correlation fails.
    """
    dx_margin = max(100, int(expected_dx * 0.5))
    dx_min = max(10, expected_dx - dx_margin)
    dx_max = expected_dx + dx_margin

    dy_min, dy_max = -dy_range, dy_range

    # === COARSE SEARCH (10px steps) ===
    x_coarse = np.arange(dx_min, dx_max + 1, 10)
    y_coarse = np.arange(dy_min, dy_max + 1, 10)

    best_corr = -np.inf
    best_shift = None

    for dx in x_coarse:
        for dy in y_coarse:
            corr = compute_overlap_correlation(img1, img2, dy, dx, noise_floor)
            if corr > best_corr:
                best_corr = corr
                best_shift = (dy, dx)

    if best_shift is None:
        return None, None, -np.inf

    # === FINE SEARCH (1px steps around coarse best) ===
    coarse_dy, coarse_dx = best_shift
    y_fine = range(coarse_dy - 10, coarse_dy + 11)
    x_fine = range(coarse_dx - 10, coarse_dx + 11)

    for dx in x_fine:
        for dy in y_fine:
            corr = compute_overlap_correlation(img1, img2, dy, dx, noise_floor)
            if corr > best_corr:
                best_corr = corr
                best_shift = (dy, dx)

    return best_shift[0], best_shift[1], best_corr


def find_best_shift_wide_ncc(img1, img2, noise_floor=15.0, dy_range=150,
                             dx_frac=(0.70, 0.97), coarse_step=10):
    """Wide NCC grid search — overlap ∈ [3%, 30%], dy ∈ ±150 px. No manifest prior.

    Each pair finds its own true shift independently. NCC is computed only on
    the actual non-dark overlap rectangle at each candidate shift, so wrong-period
    candidates lose to the true peak even with thin overlaps or repetitive features.
    """
    h, w = img1.shape
    dx_lo = max(10, int(w * dx_frac[0]))
    dx_hi = int(w * dx_frac[1])
    dy_step = max(coarse_step // 2, 5)
    best_corr, best_dy, best_dx = -np.inf, None, None
    # Coarse sweep
    for dx in range(dx_lo, dx_hi + 1, coarse_step):
        for dy in range(-dy_range, dy_range + 1, dy_step):
            corr = compute_overlap_correlation(img1, img2, dy, dx, noise_floor)
            if corr > best_corr:
                best_corr, best_dy, best_dx = corr, dy, dx
    if best_dy is None:
        return None, None, -np.inf
    # 1-px refine around coarse winner
    for dx in range(best_dx - coarse_step, best_dx + coarse_step + 1):
        for dy in range(best_dy - dy_step, best_dy + dy_step + 1):
            corr = compute_overlap_correlation(img1, img2, dy, dx, noise_floor)
            if corr > best_corr:
                best_corr, best_dy, best_dx = corr, dy, dx
    if best_corr < 0.1:
        return None, None, -np.inf
    return best_dy, best_dx, best_corr


def find_best_shift(img1, img2, expected_dx, noise_floor=15.0, dy_range=150):
    """Primary matcher: wide NCC grid search. `expected_dx` kept for signature
    compatibility but ignored — the matcher derives its own prior from the data."""
    return find_best_shift_wide_ncc(img1, img2, noise_floor=noise_floor, dy_range=dy_range)


def compute_pairwise_shifts(
    tile_data: Dict[int, np.ndarray],
    pairs: List[Tuple[int, int]],
    z_planes: List[int],
    overlap_fraction: float = 0.40,
    noise_floor: float = 15.0,
    stitch_channel: int = 0,
) -> Tuple[Dict[Tuple[int, int], Dict], Dict[Tuple[int, int], Dict[int, Dict]]]:
    """
    Compute pairwise shifts for all tile pairs across all z-planes.

    Args:
        tile_data: Dict mapping tile number to data array (Z, C, Y, X)
        pairs: List of (tile1, tile2) pairs
        z_planes: List of z-plane indices
        overlap_fraction: Expected overlap as fraction of tile width (from config)
        noise_floor: Minimum pixel value to consider as signal
        stitch_channel: Which channel of tile_data to use for cross-correlation
            (default 0 = green; set to 1 for mice with PMT0 off / single-PMT red)

    Returns:
        consensus_shifts: Dict with mean shift across z-planes
        per_plane_shifts: Dict with shift for each z-plane
    """
    consensus_shifts = {}
    per_plane_shifts = {}

    first_tile = tile_data[min(tile_data.keys())]
    tile_height, tile_width = first_tile.shape[2], first_tile.shape[3]

    # Expected dx based on overlap_fraction
    # overlap_fraction=0.4 means 40% overlap, so dx = 60% of tile width
    expected_dx = int(tile_width * (1 - overlap_fraction))

    for t1, t2 in tqdm(pairs, desc=f"Aligning {len(pairs)} tile pairs"):
        data1 = tile_data[t1][:, stitch_channel, :, :]
        data2 = tile_data[t2][:, stitch_channel, :, :]

        shifts_by_z = {}

        for z in z_planes:
            img1, img2 = data1[z], data2[z]

            dy, dx, corr = find_best_shift(img1, img2, expected_dx, noise_floor)

            if dy is not None and corr > 0.1:
                shifts_by_z[z] = {
                    'shift': np.array([dy, dx]),
                    'correlation': corr
                }

        per_plane_shifts[(t1, t2)] = shifts_by_z

        # Consensus across z-planes — median (not mean) so a single wrong-peak
        # lockon on one z doesn't drag the consensus off the true shift.
        if shifts_by_z:
            shifts_array = np.array([s['shift'] for s in shifts_by_z.values()])
            corrs = [s['correlation'] for s in shifts_by_z.values()]
            median_shift = np.median(shifts_array, axis=0)
            std_shift = shifts_array.std(axis=0) if len(shifts_array) > 1 else np.zeros(2)

            consensus_shifts[(t1, t2)] = {
                'shift': median_shift,
                'std': std_shift,
                'n_good_z': len(shifts_by_z),
                'mean_correlation': np.mean(corrs)
            }
        else:
            warnings.warn(f"No valid correlation for pair ({t1},{t2})")
            consensus_shifts[(t1, t2)] = {
                'shift': None, 'std': None, 'n_good_z': 0, 'mean_correlation': 0.0
            }

    return consensus_shifts, per_plane_shifts


def warn_if_shifts_nonuniform(consensus_shifts, tolerance=0.20):
    """Warn if pairwise shifts deviate from the median by >tolerance. Returns True if fired."""
    valid = [s['shift'] for s in consensus_shifts.values() if s['shift'] is not None]
    if len(valid) < 2:
        return False
    dx_med = float(np.median([s[1] for s in valid]))
    dy_med = float(np.median([s[0] for s in valid]))
    if dx_med <= 0:
        return False
    abs_tol = tolerance * dx_med
    bad = [(p, s['shift']) for p, s in consensus_shifts.items()
           if s['shift'] is not None
           and (abs(s['shift'][1] - dx_med) / dx_med > tolerance
                or abs(s['shift'][0] - dy_med) > abs_tol)]
    if not bad:
        return False
    rprint(f"[bold red]WARNING:[/bold red] non-uniform shifts "
           f"(median dy={dy_med:.1f}, dx={dx_med:.1f}px; ±{tolerance*100:.0f}%):")
    for pair, (dy, dx) in bad:
        rprint(f"  [yellow]pair {pair}: dy={dy:.1f}, dx={dx:.1f}[/yellow]")
    return True


def warn_if_cross_slice_shifts_vary(per_plane_shifts, consensus_shifts, tolerance=0.20):
    """Warn if a pair's shifts vary across z by >tolerance × median dx. Returns True if fired."""
    valid = [s['shift'] for s in consensus_shifts.values() if s['shift'] is not None]
    if not valid:
        return False
    dx_med = float(np.median([s[1] for s in valid]))
    if dx_med <= 0:
        return False
    abs_tol = tolerance * dx_med
    bad = []
    for pair, by_z in per_plane_shifts.items():
        if len(by_z) < 2:
            continue
        dys = [s['shift'][0] for s in by_z.values()]
        dxs = [s['shift'][1] for s in by_z.values()]
        dy_range = max(dys) - min(dys)
        dx_range = max(dxs) - min(dxs)
        if dx_range > abs_tol or dy_range > abs_tol:
            bad.append((pair, dy_range, dx_range, len(by_z)))
    if not bad:
        return False
    rprint(f"[bold red]WARNING:[/bold red] cross-slice shift spread "
           f"(>{tolerance*100:.0f}% of median dx={dx_med:.1f}px):")
    for pair, dy_r, dx_r, n_z in bad:
        rprint(f"  [yellow]pair {pair}: dy range={dy_r:.1f}px, dx range={dx_r:.1f}px (across {n_z} z)[/yellow]")
    return True


def warn_if_overlap_fraction_off(consensus_shifts, manifest_overlap, tile_width, tolerance=0.10):
    """Warn if manifest overlap_fraction disagrees with observed median dx. Returns True if fired."""
    valid = [s['shift'][1] for s in consensus_shifts.values() if s['shift'] is not None]
    if not valid or manifest_overlap <= 0:
        return False
    obs_overlap = 1.0 - float(np.median(valid)) / tile_width
    if abs(obs_overlap - manifest_overlap) <= tolerance * manifest_overlap:
        return False
    rprint(f"[bold red]WARNING:[/bold red] manifest overlap_fraction={manifest_overlap:.2f} "
           f"vs observed ≈{obs_overlap:.2f} — update manifest to match acquisition")
    return True


def compute_global_positions(
    num_tiles: int,
    pair_shifts: Dict[Tuple[int, int], Dict],
    tile_width: int,
) -> Dict[int, Tuple[float, float]]:
    """
    Compute global tile positions from pairwise shifts.
    Tile 1 at origin, positions accumulate sequentially.
    """
    n_valid = sum(1 for s in pair_shifts.values() if s['shift'] is not None)
    if n_valid == 0:
        raise StitchingError("No valid pairwise shifts found.")

    positions = {1: (0.0, 0.0)}

    for i in range(2, num_tiles + 1):
        pair = (i-1, i)
        prev_y, prev_x = positions[i-1]

        if pair in pair_shifts and pair_shifts[pair]['shift'] is not None:
            dy, dx = pair_shifts[pair]['shift']
            positions[i] = (prev_y + dy, prev_x + dx)
        else:
            # Fallback: nominal spacing
            nominal_dx = tile_width * 0.6
            positions[i] = (prev_y, prev_x + nominal_dx)
            warnings.warn(f"No valid shift for pair {pair}, using nominal {nominal_dx:.0f}px")

    return positions


def compute_global_positions_per_plane(
    num_tiles: int,
    per_plane_shifts: Dict[Tuple[int, int], Dict[int, Dict]],
    consensus_shifts: Dict[Tuple[int, int], Dict],
    z_planes: List[int],
    tile_width: int,
) -> Dict[int, Dict[int, Tuple[float, float]]]:
    """
    Compute global positions for each z-plane independently.
    Falls back to consensus if per-plane shift is missing OR if the per-plane
    shift looks like a phase-corr lock-on error (outlier vs the cross-pair
    median). Outlier threshold matches warn_if_cross_slice_shifts_vary().
    """
    # Cross-pair median dx — reference scale for the outlier threshold.
    valid_consensus = [s['shift'] for s in consensus_shifts.values() if s['shift'] is not None]
    cross_pair_dx_med = float(np.median([s[1] for s in valid_consensus])) if valid_consensus else 0.0
    outlier_tol_px = 0.20 * cross_pair_dx_med  # 20% of median dx

    positions_by_z = {}

    for z in z_planes:
        positions = {1: (0.0, 0.0)}

        for i in range(2, num_tiles + 1):
            pair = (i-1, i)
            prev_y, prev_x = positions[i-1]

            consensus = consensus_shifts.get(pair, {}).get('shift')
            shift = None
            if pair in per_plane_shifts and z in per_plane_shifts[pair]:
                this_z = per_plane_shifts[pair][z]['shift']
                # Replace this-z shift with consensus if it's an outlier vs the
                # consensus median (>20% of cross-pair median dx on either axis).
                if (consensus is not None and outlier_tol_px > 0
                        and max(abs(this_z[0] - consensus[0]),
                                abs(this_z[1] - consensus[1])) > outlier_tol_px):
                    shift = consensus
                else:
                    shift = this_z
            elif consensus is not None:
                shift = consensus

            if shift is not None:
                dy, dx = shift
                positions[i] = (prev_y + dy, prev_x + dx)
            else:
                # Fall back to the cross-pair median dx if we have one (matches
                # observed spacing); else nominal 60% of tile width.
                nominal_dx = cross_pair_dx_med if cross_pair_dx_med > 0 else tile_width * 0.6
                positions[i] = (prev_y, prev_x + nominal_dx)

        positions_by_z[z] = positions

    return positions_by_z


def create_weight_map(tile_shape: Tuple[int, int], overlap_px: int) -> np.ndarray:
    """Create linear feathering weight map for blending."""
    h, w = tile_shape
    weight = np.ones((h, w), dtype=np.float32)

    ramp_width = min(overlap_px, w // 2)
    for i in range(ramp_width):
        weight[:, i] = i / ramp_width
        weight[:, -(i+1)] = i / ramp_width

    return weight


def stitch_single_plane_channel(
    tile_data: Dict[int, np.ndarray],
    positions: Dict[int, Tuple[float, float]],
    z_plane: int,
    channel: int,
    overlap_fraction: float = 0.40
) -> np.ndarray:
    """Stitch a single z-plane and channel using weighted blending."""
    min_y = min(pos[0] for pos in positions.values())
    max_y = max(pos[0] for pos in positions.values())
    min_x = min(pos[1] for pos in positions.values())
    max_x = max(pos[1] for pos in positions.values())

    first_tile = tile_data[min(tile_data.keys())]
    tile_height, tile_width = first_tile.shape[2], first_tile.shape[3]

    canvas_height = int(np.ceil(max_y - min_y)) + tile_height
    canvas_width = int(np.ceil(max_x - min_x)) + tile_width

    stitched = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    weight_sum = np.zeros((canvas_height, canvas_width), dtype=np.float32)

    overlap_px = int(tile_width * overlap_fraction)
    weight_map = create_weight_map((tile_height, tile_width), overlap_px)

    for tile_num in sorted(tile_data.keys()):
        tile = tile_data[tile_num][z_plane, channel, :, :]
        y_pos, x_pos = positions[tile_num]

        y_offset = int(np.round(y_pos - min_y))
        x_offset = int(np.round(x_pos - min_x))

        y_start = max(0, y_offset)
        y_end = min(canvas_height, y_offset + tile_height)
        x_start = max(0, x_offset)
        x_end = min(canvas_width, x_offset + tile_width)

        tile_y_start = max(0, -y_offset)
        tile_y_end = tile_y_start + (y_end - y_start)
        tile_x_start = max(0, -x_offset)
        tile_x_end = tile_x_start + (x_end - x_start)

        tile_region = tile[tile_y_start:tile_y_end, tile_x_start:tile_x_end]
        weight_region = weight_map[tile_y_start:tile_y_end, tile_x_start:tile_x_end]

        stitched[y_start:y_end, x_start:x_end] += tile_region * weight_region
        weight_sum[y_start:y_end, x_start:x_end] += weight_region

    mask = weight_sum > 0
    stitched[mask] /= weight_sum[mask]

    return stitched


def stitch_volume_per_plane(
    tile_data: Dict[int, np.ndarray],
    positions_by_z: Dict[int, Dict[int, Tuple[float, float]]],
    z_planes: List[int],
    overlap_fraction: float = 0.40,
    show_progress: bool = True
) -> np.ndarray:
    """Stitch volume using per-plane positions."""
    first_tile = tile_data[min(tile_data.keys())]
    num_channels = first_tile.shape[1]
    tile_height, tile_width = first_tile.shape[2], first_tile.shape[3]

    # Find max canvas size across all planes
    max_canvas_height, max_canvas_width = 0, 0
    for z in z_planes:
        positions = positions_by_z[z]
        min_y = min(pos[0] for pos in positions.values())
        max_y = max(pos[0] for pos in positions.values())
        min_x = min(pos[1] for pos in positions.values())
        max_x = max(pos[1] for pos in positions.values())

        canvas_height = int(np.ceil(max_y - min_y)) + tile_height
        canvas_width = int(np.ceil(max_x - min_x)) + tile_width

        max_canvas_height = max(max_canvas_height, canvas_height)
        max_canvas_width = max(max_canvas_width, canvas_width)

    stitched = np.zeros((len(z_planes), num_channels, max_canvas_height, max_canvas_width), dtype=np.float32)

    iterator = [(z_idx, z, c) for z_idx, z in enumerate(z_planes) for c in range(num_channels)]
    if show_progress:
        iterator = tqdm(iterator, desc="Stitching volume")

    for z_idx, z, c in iterator:
        positions = positions_by_z[z]
        plane_stitched = stitch_single_plane_channel(tile_data, positions, z, c, overlap_fraction)
        h, w = plane_stitched.shape
        stitched[z_idx, c, :h, :w] = plane_stitched

    return stitched


def auto_stitch_tiles(
    tile_dir: Path,
    num_tiles: int,
    output_path: Path,
    use_unwarped: bool = False,
    overlap_fraction: float = 0.40,
    noise_floor: float = 15.0,
    min_signal_frac: float = 0.01,
    upsample_factor: int = 10,
    stitch_channel: int = 0,
) -> np.ndarray:
    """
    Main entry point for automated tile stitching.

    Args:
        tile_dir: Directory containing warped/ or unwarped/ subdirs
        num_tiles: Number of tiles to stitch
        output_path: Where to save stitched output
        use_unwarped: Use unwarped/ instead of warped/
        overlap_fraction: Expected overlap as fraction of tile width (from config)
        noise_floor: Minimum pixel value for signal
        min_signal_frac: (unused, for API compatibility)
        upsample_factor: (unused, for API compatibility)
        stitch_channel: Channel of the warped tiles used for cross-correlation
            (default 0; set to 1 when PMT0 is off so ch0 has no signal).

    Returns:
        Stitched volume (Z, C, Y, X)
    """
    from tifffile import imread as tif_imread, imwrite as tif_imwrite

    rprint("[bold]Automated tile stitching[/bold]")

    # Load tiles
    source_dir = tile_dir / ('unwarped' if use_unwarped else 'warped')
    tile_data = {}

    for i in tqdm(range(1, num_tiles + 1), desc=f"Loading {num_tiles} tiles"):
        tile_path = source_dir / f"stack_{'un' if use_unwarped else ''}warped_C12_{i:03d}.tiff"

        if not tile_path.exists():
            raise StitchingError(f"Missing tile {i}/{num_tiles}: {tile_path}")

        data = tif_imread(tile_path)

        if data.ndim != 4:
            raise StitchingError(f"Tile {i} has {data.ndim} dimensions, expected 4 (ZCYX)")
        if data.shape[1] != 2:
            raise StitchingError(f"Tile {i} has {data.shape[1]} channels, expected 2")

        tile_data[i] = data

    # Verify consistent z-planes
    z_counts = [d.shape[0] for d in tile_data.values()]
    if len(set(z_counts)) > 1:
        raise StitchingError(f"Inconsistent z-planes: {z_counts}")

    num_z = z_counts[0]
    z_planes = list(range(num_z))
    tile_width = tile_data[1].shape[3]

    # Find pairs and compute shifts
    pairs = find_horizontal_pairs(num_tiles)

    consensus_shifts, per_plane_shifts = compute_pairwise_shifts(
        tile_data, pairs, z_planes, overlap_fraction, noise_floor,
        stitch_channel=stitch_channel,
    )

    # Informational: matcher doesn't depend on manifest overlap_fraction anymore.
    warn_if_overlap_fraction_off(consensus_shifts, overlap_fraction, tile_width)
    # Gating: these two indicate something the matcher couldn't resolve cleanly.
    alarms = (
        warn_if_shifts_nonuniform(consensus_shifts, tolerance=0.20)
        | warn_if_cross_slice_shifts_vary(per_plane_shifts, consensus_shifts, tolerance=0.20)
    )
    if alarms:
        resp = input("Continue stitching despite warnings? [y/N] ").strip().lower()
        if resp != 'y':
            raise StitchingError("User aborted after stitching warnings")

    # Compute positions
    positions_by_z = compute_global_positions_per_plane(
        num_tiles, per_plane_shifts, consensus_shifts, z_planes, tile_width
    )

    consensus_positions = compute_global_positions(num_tiles, consensus_shifts, tile_width)

    # Stitch
    stitched = stitch_volume_per_plane(
        tile_data, positions_by_z, z_planes, overlap_fraction, show_progress=True
    )

    # Save — single ZCYX file. Per-z-plane slices are not written separately:
    # downstream code always extracts the functional plane from this stack.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tif_imwrite(output_path, stitched.astype(np.float32), imagej=True, metadata={'axes': 'ZCYX'})

    rprint(f"[green]Stitching complete:[/green] {output_path}")

    return stitched
