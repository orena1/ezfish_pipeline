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
from rich import print as rprint


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


def find_best_shift(img1, img2, expected_dx, noise_floor=15.0):
    """
    Find optimal (dy, dx) shift using coarse-to-fine grid search.

    Args:
        img1, img2: 2D tile images
        expected_dx: Expected horizontal shift based on overlap_fraction
        noise_floor: Minimum pixel value for signal

    Returns:
        (best_dy, best_dx, best_correlation) or (None, None, -inf) if failed
    """
    # Search range: centered on expected_dx, with generous margins
    # dx: expected ± 50% of expected (but at least ±100px)
    dx_margin = max(100, int(expected_dx * 0.5))
    dx_min = max(10, expected_dx - dx_margin)
    dx_max = expected_dx + dx_margin

    # dy: expect ~0 for horizontal tiling, search ±100px
    dy_min, dy_max = -100, 100

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


def compute_pairwise_shifts(
    tile_data: Dict[int, np.ndarray],
    pairs: List[Tuple[int, int]],
    z_planes: List[int],
    overlap_fraction: float = 0.40,
    noise_floor: float = 15.0,
) -> Tuple[Dict[Tuple[int, int], Dict], Dict[Tuple[int, int], Dict[int, Dict]]]:
    """
    Compute pairwise shifts for all tile pairs across all z-planes.

    Uses channel 0 (green/GCaMP) for registration.

    Args:
        tile_data: Dict mapping tile number to data array (Z, C, Y, X)
        pairs: List of (tile1, tile2) pairs
        z_planes: List of z-plane indices
        overlap_fraction: Expected overlap as fraction of tile width (from config)
        noise_floor: Minimum pixel value to consider as signal

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
        data1 = tile_data[t1][:, 0, :, :]  # Channel 0
        data2 = tile_data[t2][:, 0, :, :]

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

        # Consensus across z-planes
        if shifts_by_z:
            shifts_array = np.array([s['shift'] for s in shifts_by_z.values()])
            corrs = [s['correlation'] for s in shifts_by_z.values()]
            mean_shift = shifts_array.mean(axis=0)
            std_shift = shifts_array.std(axis=0) if len(shifts_array) > 1 else np.zeros(2)

            consensus_shifts[(t1, t2)] = {
                'shift': mean_shift,
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
    Falls back to consensus if per-plane shift is missing.
    """
    positions_by_z = {}

    for z in z_planes:
        positions = {1: (0.0, 0.0)}

        for i in range(2, num_tiles + 1):
            pair = (i-1, i)
            prev_y, prev_x = positions[i-1]

            # Try per-plane shift first, then consensus
            shift = None
            if pair in per_plane_shifts and z in per_plane_shifts[pair]:
                shift = per_plane_shifts[pair][z]['shift']
            elif pair in consensus_shifts and consensus_shifts[pair]['shift'] is not None:
                shift = consensus_shifts[pair]['shift']

            if shift is not None:
                dy, dx = shift
                positions[i] = (prev_y + dy, prev_x + dx)
            else:
                nominal_dx = tile_width * 0.6
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
    upsample_factor: int = 10
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
        tile_data, pairs, z_planes, overlap_fraction, noise_floor
    )

    # Compute positions
    positions_by_z = compute_global_positions_per_plane(
        num_tiles, per_plane_shifts, consensus_shifts, z_planes, tile_width
    )

    consensus_positions = compute_global_positions(num_tiles, consensus_shifts, tile_width)

    # Stitch
    stitched = stitch_volume_per_plane(
        tile_data, positions_by_z, z_planes, overlap_fraction, show_progress=True
    )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tif_imwrite(output_path, stitched.astype(np.float32), imagej=True, metadata={'axes': 'ZCYX'})

    for plane in range(stitched.shape[0]):
        plane_dir = output_path.parent / f'plane{plane}'
        plane_dir.mkdir(exist_ok=True)
        tif_imwrite(plane_dir / output_path.name, stitched[plane].astype(np.float32),
                   imagej=True, metadata={'axes': 'CYX'})

    rprint(f"[green]Stitching complete:[/green] {output_path}")

    return stitched
