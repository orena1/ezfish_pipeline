"""
Analysis utilities for 2P-to-HCR integration.

This module provides reusable functions for:
- Loading and validating pipeline outputs
- Neuropil subtraction
- Deduplication of HCR-2P matches
- Stimulus response analysis (DF/F, excited/inhibited classification)
- Gene classification (mode-shift, threshold-based)
- Visualization functions
- Output persistence

Usage:
    from analysis_utils import (
        load_analysis_manifest, load_feature_tables,
        subtract_neuropil_with_qc, classify_responses,
        plot_dff_heatmap, save_analysis_results
    )
"""

import json
import hjson
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from scipy.io import loadmat
from scipy.ndimage import median_filter
import warnings

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from rich import print as rprint
    from rich.console import Console
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    rprint = print


# =============================================================================
# UTILITIES
# =============================================================================

def get_col(df, col_name):
    """Get column name handling both flat and MultiIndex DataFrames.

    For MultiIndex columns, finds the first column containing col_name.
    For flat columns, returns col_name unchanged.
    """
    if isinstance(df.columns, pd.MultiIndex):
        for col in df.columns:
            if col_name in str(col):
                return col
    return col_name


# =============================================================================
# CONSTANTS
# =============================================================================

# Genes to exclude from analysis (housekeeping/controls)
EXCLUDE_GENES = ['DAPI', 'GCAMP', 'CHRIMSON']

# Known "noisy" genes that need higher thresholds
DEFAULT_NOISY_GENES = ('GPR101', 'CCK', 'BRS3')


# =============================================================================
# DATACLASSES FOR CONFIGURATION
# =============================================================================

@dataclass
class AnalysisConfig:
    """
    All configurable parameters for analysis.

    Set these once at the top of your notebook, then pass to functions.
    Framerate is determined empirically from SBX info file, not set here.
    """
    # Paths
    manifest_path: Path = None
    output_dir: Path = None

    # Stimulus response
    stim_run: str = None             # run to use for ephys/stimulus (None = use functional_run)
    concatenated_runs: list = None   # runs concatenated in Suite2p, e.g. ['001', '002', '003']
    pre_stim: float = 10.0           # seconds before stimulus
    post_stim: float = 20.0          # seconds after stimulus
    sort_window: float = 5.0         # seconds for response sorting
    p_threshold: float = 0.05        # p-value for excited/inhibited (same for both)
    stim_detection_threshold: float = 0.05
    min_stim_interval_frames: int = 100
    min_onset_interval_sec: float = 10.0  # min seconds between train onsets

    # Gene classification
    normal_sigma: float = 2.0        # mode-shift threshold for normal genes
    noisy_sigma: float = 4.0         # mode-shift threshold for noisy genes
    noisy_genes: tuple = DEFAULT_NOISY_GENES

    # Artifact filtering
    y_artifact_threshold: float = 100  # pixels - remove neurons with Y < this (top artifact)
    edge_margin: int = 20  # pixels - remove neurons within this distance from FOV edges

    # Feature selection
    feature_type: str = 'mean_medflt_3x4x4'
    neuropil_type: str = 'medflt_3x4x4_neuropil_mean_nr50_nb_1'
    neuropil_weight: float = 1.0  # weight for neuropil subtraction (0-1, typically 0.7-1.0)

    # Output
    save_figures: bool = True
    figure_formats: list = field(default_factory=lambda: ['png', 'pdf'])
    figure_dpi: int = 150


# =============================================================================
# PATH UTILITIES
# =============================================================================

def normalize_path(path: Union[str, Path]) -> Path:
    """
    Normalize path for cross-platform compatibility.

    Handles conversion between:
    - Linux: /mnt/nasquatch/data/...
    - Windows: //nasquatch/data/... or \\\\nasquatch\\data\\...

    Automatically detects current platform and converts accordingly.
    """
    import platform
    path_str = str(path)

    # Detect platform
    is_windows = platform.system() == 'Windows'

    if is_windows:
        # Convert Linux mount to Windows UNC
        if path_str.startswith('/mnt/nasquatch/'):
            path_str = '//nasquatch/' + path_str[15:]  # len('/mnt/nasquatch/') = 15
        elif path_str.startswith('/mnt/nasquatch'):
            path_str = '//nasquatch' + path_str[14:]
    else:
        # Convert Windows UNC to Linux mount
        if path_str.startswith('//nasquatch/'):
            path_str = '/mnt/nasquatch/' + path_str[12:]
        elif path_str.startswith('\\\\nasquatch\\'):
            path_str = '/mnt/nasquatch/' + path_str[12:].replace('\\', '/')

    return Path(path_str)


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_analysis_manifest(manifest_path: Union[str, Path]) -> dict:
    """
    Load and parse manifest for analysis, extracting key fields.

    Args:
        manifest_path: Path to HJSON manifest file

    Returns:
        dict with keys:
        - 'manifest': Full parsed manifest
        - 'base_path': Path to mouse data directory
        - 'mouse_name': str
        - 'planes': List of all plane numbers (sorted)
        - 'reference_plane': int (the functional_plane)
        - 'additional_planes': List of additional plane numbers
        - 'rounds': List of HCR round dicts with 'round' and 'channels'
        - 'reference_round': str (e.g., '01')
        - 'genes': List of all gene names (excluding DAPI, GCAMP, CHRIMSON)
        - 'session': Session dict from manifest
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, 'r') as f:
        manifest = hjson.load(f)

    data = manifest['data']
    base_path = normalize_path(data['base_path'])
    mouse_name = data['mouse_name']

    # Extract session info
    # backward compat: accept old key name
    if 'two_photons_imaging' in data and 'two_photon_imaging' not in data:
        data['two_photon_imaging'] = data['two_photons_imaging']
    session = data['two_photon_imaging']['sessions'][0]

    # Get planes
    reference_plane = int(session['functional_plane'][0])
    additional_planes = [int(p) for p in session.get('additional_functional_planes', [])]
    all_planes = sorted(set([reference_plane] + additional_planes))

    # Get HCR rounds and genes
    rounds = data['HCR_confocal_imaging']['rounds']
    reference_round = data['HCR_confocal_imaging']['reference_round']

    # Collect all genes (excluding housekeeping)
    all_genes = []
    for round_info in rounds:
        for channel in round_info['channels']:
            if channel not in EXCLUDE_GENES and channel not in all_genes:
                all_genes.append(channel)

    result = {
        'manifest': manifest,
        'manifest_path': manifest_path,
        'base_path': base_path,
        'mouse_name': mouse_name,
        'planes': all_planes,
        'reference_plane': reference_plane,
        'additional_planes': additional_planes,
        'rounds': rounds,
        'reference_round': reference_round,
        'genes': all_genes,
        'session': session,
    }

    # Print summary
    rprint(f"[bold]Loaded manifest for {mouse_name}[/bold]")
    rprint(f"  Base path: {base_path}")
    rprint(f"  Planes: {all_planes} (reference: {reference_plane})")
    rprint(f"  HCR rounds: {len(rounds)} (reference: {reference_round})")
    rprint(f"  Genes: {len(all_genes)} ({', '.join(all_genes[:5])}{'...' if len(all_genes) > 5 else ''})")

    return result


def load_feature_tables(
    base_path: Path,
    mouse_name: str,
    planes: List[int],
    feature_type: str = 'mean_medflt_3x4x4'
) -> Dict[int, pd.DataFrame]:
    """
    Load feature tables for all planes.

    Args:
        base_path: Path to mouse data directory
        mouse_name: Mouse name
        planes: List of plane numbers to load
        feature_type: Feature type string (default: 'mean_medflt_3x4x4')

    Returns:
        Dict mapping plane number to DataFrame
    """
    base_path = Path(base_path)
    output_dir = base_path / mouse_name / 'OUTPUT' / 'MERGED' / 'aligned_extracted_features'

    tables = {}
    for plane in planes:
        filename = f'full_table_{feature_type}_twop_plane{plane}.pkl'
        filepath = output_dir / filename

        if not filepath.exists():
            rprint(f"[yellow]Warning: Feature table not found: {filepath}[/yellow]")
            continue

        df = pd.read_pickle(filepath)
        tables[plane] = df
        rprint(f"  Plane {plane}: {len(df)} cells, {len(df.columns)} columns")

    if not tables:
        raise FileNotFoundError(f"No feature tables found in {output_dir}")

    return tables


def load_neuropil_tables(
    base_path: Path,
    mouse_name: str,
    planes: List[int],
    neuropil_type: str = 'medflt_3x4x4_neuropil_mean_nr50_nb_1'
) -> Dict[int, pd.DataFrame]:
    """
    Load neuropil feature tables for all planes.

    Args:
        base_path: Path to mouse data directory
        mouse_name: Mouse name
        planes: List of plane numbers to load
        neuropil_type: Neuropil type string

    Returns:
        Dict mapping plane number to DataFrame
    """
    base_path = Path(base_path)
    output_dir = base_path / mouse_name / 'OUTPUT' / 'MERGED' / 'aligned_extracted_features'

    tables = {}
    for plane in planes:
        filename = f'full_table_{neuropil_type}_twop_plane{plane}.pkl'
        filepath = output_dir / filename

        if not filepath.exists():
            rprint(f"[yellow]Warning: Neuropil table not found: {filepath}[/yellow]")
            continue

        df = pd.read_pickle(filepath)
        tables[plane] = df

    return tables


def load_hcr_coordinates(
    base_path: Path,
    mouse_name: str,
    reference_round: int = 1
) -> pd.DataFrame:
    """
    Load HCR cell coordinates from extract_intensities.

    The extract_intensities files contain X, Y, Z coordinates for each mask,
    repeated for each channel. This function loads the reference round and
    returns unique coordinates per mask_id.

    Args:
        base_path: Path to mouse data directory
        mouse_name: Mouse name
        reference_round: HCR round number to load coordinates from (default: 1)

    Returns:
        DataFrame with columns: mask_id, hcr_x, hcr_y, hcr_z
    """
    base_path = Path(base_path)
    extract_dir = base_path / mouse_name / 'OUTPUT' / 'HCR' / 'extract_intensities'

    # Try to find the reference round file (zero-padded, e.g., HCR01)
    pkl_path = extract_dir / f'HCR{reference_round:02d}_probs_intensities.pkl'

    if not pkl_path.exists():
        rprint(f"[yellow]Warning: HCR coordinates file not found: {pkl_path}[/yellow]")
        return pd.DataFrame(columns=['mask_id', 'hcr_x', 'hcr_y', 'hcr_z'])

    # Load the pickle file
    df = pd.read_pickle(pkl_path)

    # Extract unique coordinates per mask_id (they're repeated per channel)
    # Group by mask_id and take first X, Y, Z (they should all be the same)
    coords = df.groupby('mask_id').agg({
        'X': 'first',
        'Y': 'first',
        'Z': 'first'
    }).reset_index()

    # Rename columns for clarity
    coords = coords.rename(columns={
        'X': 'hcr_x',
        'Y': 'hcr_y',
        'Z': 'hcr_z'
    })

    rprint(f"  Loaded HCR coordinates: {len(coords)} cells from round {reference_round}")

    return coords


def load_2p_spatial_data(
    base_path: Path,
    mouse_name: str,
    planes: List[int]
) -> Dict[int, dict]:
    """
    Load 2P spatial data (masks, centroids, mean images) for all planes.

    Args:
        base_path: Path to mouse data directory
        mouse_name: Mouse name
        planes: List of plane numbers to load

    Returns:
        Dict mapping plane number to dict with:
        - 'mean_frames': 2D array of mean image
        - 'masks_locs': dict mapping cell_id to coordinate array
        - 'x_centroids': array of x positions
        - 'y_centroids': array of y positions
        - 'cell_ids': list of cell IDs
    """
    base_path = Path(base_path)
    suite2p_dir = base_path / mouse_name / 'OUTPUT' / '2P' / 'suite2p'

    plane_data = {}
    for plane in planes:
        filename = f'lowres_meanImg_C0_plane{plane}.pkl'
        filepath = suite2p_dir / filename

        if not filepath.exists():
            rprint(f"[yellow]Warning: 2P data not found: {filepath}[/yellow]")
            continue

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        # Extract mask locations and compute centroids
        masks_locs_raw = data.get('masks_locs', {})
        mean_frames = data.get('mean_frames')

        # Filter out cell_0 which is typically a background/artifact mask
        # This is standard practice - cell_0 often covers the entire FOV
        masks_locs = {k: v for k, v in masks_locs_raw.items() if k != 'cell_0'}

        x_centroids = []
        y_centroids = []
        cell_ids = []

        # Process masks in sorted order to ensure alignment with mean_frames
        # mean_frames rows are ordered by cell number (cell_1 -> row 0, cell_2 -> row 1, etc.)
        # after excluding cell_0
        for cell_id in sorted(masks_locs.keys(), key=lambda x: int(x.split('_')[1])):
            coords = masks_locs[cell_id]
            if coords is not None and len(coords) > 0:
                # coords is a tuple of (y_indices, x_indices) arrays
                # representing pixel locations of the mask
                if isinstance(coords, tuple) and len(coords) == 2:
                    y_indices, x_indices = coords
                    centroid_y = np.mean(y_indices)
                    centroid_x = np.mean(x_indices)
                else:
                    # Fallback for (N, 2) array format
                    centroid_y = np.mean(coords[:, 0])
                    centroid_x = np.mean(coords[:, 1])
                x_centroids.append(centroid_x)
                y_centroids.append(centroid_y)
                cell_ids.append(cell_id)

        plane_data[plane] = {
            'mean_frames': mean_frames,
            'masks_locs': masks_locs,
            'x_centroids': np.array(x_centroids),
            'y_centroids': np.array(y_centroids),
            'cell_ids': cell_ids,
        }

        rprint(f"  Plane {plane}: {len(cell_ids)} cells with masks (excluded cell_0)")

    return plane_data


def get_framerate(sbx_path: Path, nplanes: int = 4) -> float:
    """
    Calculate framerate from SBX info file based on line scanning parameters.

    Standard framerate for 512-line scanning is 15.63 Hz.
    Adjusted for number of lines and planes: framerate = 15.63 / (sz / 512) / nplanes

    Args:
        sbx_path: Path to .sbx file or directory containing info file
        nplanes: Number of imaging planes (default: 4)

    Returns:
        Framerate in Hz (per plane)
    """
    # Base framerate for 512-line scanning
    BASE_FRAMERATE = 15.63
    BASE_LINES = 512

    sbx_path = Path(sbx_path)

    # Find info file
    if sbx_path.is_dir():
        info_files = list(sbx_path.glob('*.mat'))
        if not info_files:
            rprint(f"[yellow]Warning: No .mat info file found in {sbx_path}, using default framerate[/yellow]")
            return BASE_FRAMERATE / nplanes
        info_path = info_files[0]
    else:
        info_path = sbx_path.with_suffix('.mat')

    if not info_path.exists():
        rprint(f"[yellow]Warning: Info file not found: {info_path}, using default framerate[/yellow]")
        return BASE_FRAMERATE / nplanes

    # Load info and extract sz (number of lines)
    try:
        mat_data = loadmat(info_path, squeeze_me=True)
        info = mat_data.get('info', None)

        if info is None:
            rprint(f"[yellow]Warning: No 'info' field in {info_path}, using default[/yellow]")
            return BASE_FRAMERATE / nplanes

        # Extract sz (number of lines per frame)
        try:
            sz = info['sz'][0][0][0][0]
        except (IndexError, TypeError, KeyError):
            try:
                sz = info['sz'][0][0]
                if hasattr(sz, '__len__') and len(sz) > 0:
                    sz = sz[0]
            except (IndexError, TypeError, KeyError):
                try:
                    sz = info['sz']
                    if hasattr(sz, 'item'):
                        sz = sz.item()
                    if hasattr(sz, '__len__'):
                        sz = sz[0]
                except:
                    rprint(f"[yellow]Warning: Could not extract 'sz', using default 512[/yellow]")
                    sz = BASE_LINES

        # Ensure sz is a scalar
        if hasattr(sz, '__len__'):
            sz = sz[0] if len(sz) > 0 else BASE_LINES
        sz = int(sz)

        # Calculate adjusted framerate
        # framerate = base_rate / (lines / base_lines) / nplanes
        adjusted_framerate = BASE_FRAMERATE / (sz / BASE_LINES) / nplanes

        rprint(f"  SBX info - sz: {sz}, base framerate: {BASE_FRAMERATE}, planes: {nplanes}")
        rprint(f"  Adjusted framerate per plane: {adjusted_framerate:.2f} Hz")

        return adjusted_framerate

    except Exception as e:
        rprint(f"[yellow]Warning: Error reading SBX info ({e}), using default framerate[/yellow]")
        return BASE_FRAMERATE / nplanes


def get_gene_column(df: pd.DataFrame, gene: str, round_num: str = '01',
                    feature_type: str = 'mean_medflt_3x4x4') -> Union[str, tuple]:
    """
    Get the correct column name for a gene in the current format.

    Handles three column formats:
    1. MultiIndex columns (before smart_merge_planes flattening)
    2. Flattened MultiIndex columns (after smart_merge_planes)
    3. Simple flat columns (legacy format)

    MultiIndex format (current pipeline):
    - Round 01: ('mean_medflt_3x4x4', 'PDYN')
    - Round 02: ('mean_medflt_3x4x4_round_02', 'TAC1')

    Flattened MultiIndex format (after smart_merge_planes):
    - Round 01: 'mean_medflt_3x4x4_PDYN'
    - Round 02: 'mean_medflt_3x4x4_round_02_TAC1'

    Simple flat format (legacy):
    - Round 01: 'PDYN'
    - Round 02: 'TAC1_round_02'

    Args:
        df: DataFrame with gene columns
        gene: Gene name (e.g., 'PDYN', 'TAC1')
        round_num: Round number as string (e.g., '01', '02')
        feature_type: Feature prefix for MultiIndex (default: 'mean_medflt_3x4x4')

    Returns:
        Column name (string for flat, tuple for MultiIndex)

    Raises:
        KeyError if column not found
    """
    # Check if DataFrame has MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        # MultiIndex format
        if round_num == '01':
            col = (feature_type, gene)
        else:
            col = (f'{feature_type}_round_{round_num}', gene)

        if col in df.columns:
            return col

        # Try without feature_type prefix for some columns
        alt_cols = [
            (feature_type, gene),
            (f'{feature_type}_round_{round_num}', gene),
        ]

        for alt in alt_cols:
            if alt in df.columns:
                return alt

        raise KeyError(f"Column for gene '{gene}' round '{round_num}' not found. "
                      f"Tried: {alt_cols}. Available level-1: {list(df.columns.get_level_values(1).unique())}")
    else:
        # Flat column format - check multiple naming conventions
        # Build list of alternatives in order of preference
        alternatives = []

        # 1. Flattened MultiIndex format (from smart_merge_planes)
        # e.g., 'mean_medflt_3x4x4_PDYN' or 'mean_medflt_3x4x4_round_02_TAC1'
        if round_num == '01':
            alternatives.append(f'{feature_type}_{gene}')
        else:
            alternatives.append(f'{feature_type}_round_{round_num}_{gene}')

        # 2. Simple gene name (round 01 only)
        alternatives.append(gene)

        # 3. Gene with round suffix
        alternatives.append(f'{gene}_round_{round_num}')

        # 4. Legacy 'mean_' prefix formats
        alternatives.append(f'mean_{gene}')
        alternatives.append(f'mean_round_{round_num}_{gene}')

        # 5. Also try matching any column that ends with the gene name
        # (handles various prefixes we might not anticipate)
        for col in df.columns:
            if isinstance(col, str) and col.endswith(f'_{gene}'):
                # Check round matches if not round 01
                if round_num == '01':
                    if 'round_' not in col or f'round_{round_num}' in col:
                        alternatives.append(col)
                else:
                    if f'round_{round_num}' in col:
                        alternatives.append(col)

        # Remove duplicates while preserving order
        seen = set()
        unique_alternatives = []
        for alt in alternatives:
            if alt not in seen:
                seen.add(alt)
                unique_alternatives.append(alt)

        for alt in unique_alternatives:
            if alt in df.columns:
                return alt

        raise KeyError(f"Column for gene '{gene}' round '{round_num}' not found. "
                      f"Tried: {unique_alternatives[:10]}...")


def exclude_genes(genes: List[str], exclude: List[str] = None) -> List[str]:
    """
    Filter out housekeeping/control genes from analysis.

    Args:
        genes: List of gene names
        exclude: List of genes to exclude (default: DAPI, GCAMP, CHRIMSON)

    Returns:
        Filtered list of genes
    """
    if exclude is None:
        exclude = EXCLUDE_GENES
    return [g for g in genes if g not in exclude]


# =============================================================================
# PREPROCESSING FUNCTIONS
# =============================================================================

def subtract_neuropil_with_qc(
    signal_df: pd.DataFrame,
    neuropil_df: pd.DataFrame,
    genes: List[str],
    rounds: List[dict],
    clip_zero: bool = True,
    signal_feature_type: str = 'mean_medflt_3x4x4',
    neuropil_feature_type: str = 'medflt_3x4x4_neuropil_mean_nr50_nb_1',
    neuropil_weight: float = 1.0
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Subtract neuropil from neuron intensities with QC.

    Returns three versions:
    - original: Raw signal values (unchanged)
    - subtracted: Signal - weight*neuropil, clipped to 0 (for visualization)
    - unclipped: Signal - weight*neuropil, can be negative (for mode classification)

    Args:
        signal_df: DataFrame with signal values
        neuropil_df: DataFrame with neuropil values
        genes: List of gene names to process
        rounds: List of round info dicts
        clip_zero: If True, clip subtracted values to minimum of 0
        signal_feature_type: Feature type prefix for signal columns
        neuropil_feature_type: Feature type prefix for neuropil columns
        neuropil_weight: Weight for neuropil subtraction (0-1, typically 0.7-1.0)

    Returns:
        Tuple of (original_df, subtracted_df, unclipped_df)
    """
    # Work on copies
    original_df = signal_df.copy()
    subtracted_df = signal_df.copy()
    unclipped_df = signal_df.copy()

    # Filter genes
    genes = exclude_genes(genes)

    # Process each round
    for round_info in rounds:
        round_num = round_info['round']
        channels = [c for c in round_info['channels'] if c not in EXCLUDE_GENES]

        for gene in channels:
            if gene not in genes:
                continue

            try:
                signal_col = get_gene_column(signal_df, gene, round_num, signal_feature_type)
                neuropil_col = get_gene_column(neuropil_df, gene, round_num, neuropil_feature_type)
            except KeyError as e:
                rprint(f"[yellow]Warning: Skipping {gene} round {round_num}: {e}[/yellow]")
                continue

            # Get values
            signal_vals = signal_df[signal_col].values
            neuropil_vals = neuropil_df[neuropil_col].values

            # Subtract with weight: signal - weight * neuropil
            diff = signal_vals - neuropil_weight * neuropil_vals

            # Store unclipped version (for mode-shift classification)
            unclipped_df[signal_col] = diff

            # Store clipped version (for visualization/other analysis)
            if clip_zero:
                subtracted_df[signal_col] = np.clip(diff, 0, None)
            else:
                subtracted_df[signal_col] = diff

    return original_df, subtracted_df, unclipped_df


def select_best_match(group: pd.DataFrame) -> pd.Series:
    """
    Select the row with highest IoU from a group.

    Helper function for deduplication.
    """
    return group.loc[group['twoP_iou'].idxmax()]


def deduplicate_2p_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the best HCR-to-2P match per 2P cell.

    When multiple HCR cells match the same 2P cell (twoP_mask column),
    keep the one with highest twoP_iou.

    Args:
        df: DataFrame with 'twoP_mask' and 'twoP_iou' columns

    Returns:
        Deduplicated DataFrame
    """
    # Check required columns exist
    if 'twoP_mask' not in df.columns:
        rprint("[yellow]Warning: 'twoP_mask' column not found, skipping deduplication[/yellow]")
        return df

    if 'twoP_iou' not in df.columns:
        rprint("[yellow]Warning: 'twoP_iou' column not found, skipping deduplication[/yellow]")
        return df

    # Filter to only cells with 2P matches
    has_2p = df['twoP_mask'].notna()
    df_with_2p = df[has_2p].copy()
    df_without_2p = df[~has_2p].copy()

    n_before = len(df_with_2p)

    if n_before == 0:
        rprint("[yellow]No cells with 2P matches found[/yellow]")
        return df

    # Group by twoP_mask and keep best match
    deduplicated = df_with_2p.groupby('twoP_mask', group_keys=False).apply(
        select_best_match
    ).reset_index(drop=True)

    n_after = len(deduplicated)
    n_removed = n_before - n_after

    if n_removed > 0:
        rprint(f"  Deduplication: {n_before} -> {n_after} cells ({n_removed} duplicates removed)")

    # Combine back with cells without 2P matches
    result = pd.concat([deduplicated, df_without_2p], ignore_index=True)

    return result


def merge_planes(plane_dfs: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Simple merge: concatenate DataFrames from multiple planes, adding plane index.

    NOTE: This creates duplicate HCR cells (one per plane). For smart merging
    that keeps one row per HCR cell, use smart_merge_planes() instead.

    Args:
        plane_dfs: Dict mapping plane number to DataFrame

    Returns:
        Combined DataFrame with 'plane' column
    """
    merged_dfs = []

    for plane, df in sorted(plane_dfs.items()):
        df_copy = df.copy()
        df_copy['plane'] = plane
        merged_dfs.append(df_copy)

    if not merged_dfs:
        raise ValueError("No DataFrames to merge")

    merged = pd.concat(merged_dfs, ignore_index=True)
    rprint(f"  Merged {len(plane_dfs)} planes: {len(merged)} total rows (simple concat)")

    return merged


def smart_merge_planes(
    plane_dfs: Dict[int, pd.DataFrame],
    reference_plane: int,
    mask_id_col: str = 'mask_id_main',
    twop_mask_col: str = 'twoP_mask',
    twop_iou_col: str = 'twoP_iou',
    hcr_coords: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Smart merge: keep ONE row per unique HCR cell, selecting best 2P match across planes.

    Each plane's table contains the SAME HCR cells, but with different 2P matches.
    This function:
    1. Groups by HCR cell ID (mask_id_main)
    2. For each HCR cell, selects the best plane based on:
       - Priority 1: Reference plane (if has 2P match there)
       - Priority 2: Highest IoU match from any plane
       - Priority 3: Reference plane (if no 2P match anywhere)

    Args:
        plane_dfs: Dict mapping plane number to DataFrame
        reference_plane: The reference/functional plane number
        mask_id_col: Column name for HCR cell ID
        twop_mask_col: Column name for 2P mask ID
        twop_iou_col: Column name for IoU score
        hcr_coords: Optional DataFrame with HCR coordinates (from load_hcr_coordinates)
                    with columns: mask_id, hcr_x, hcr_y, hcr_z

    Returns:
        DataFrame with one row per HCR cell, plus columns:
        - 'best_plane': which plane the row came from
        - 'has_2p_match': boolean
        - 'match_type': 'none', 'single', or 'multi'
        - 'hcr_x', 'hcr_y', 'hcr_z': HCR coordinates (if hcr_coords provided)
    """
    import time
    t0 = time.time()

    # First concat all planes
    all_dfs = []
    for plane, df in sorted(plane_dfs.items()):
        df_copy = df.copy()
        df_copy['plane'] = plane
        all_dfs.append(df_copy)

    combined = pd.concat(all_dfs, ignore_index=True)

    # Find actual column names
    mask_col = get_col(combined, mask_id_col)
    twop_col = get_col(combined, twop_mask_col)
    iou_col = get_col(combined, twop_iou_col)

    rprint(f"  Starting smart merge: {len(combined)} rows from {len(plane_dfs)} planes")
    rprint(f"  Reference plane: {reference_plane}")

    # ========== FULLY VECTORIZED SELECTION ==========
    # Create helper columns for sorting priority
    combined['_has_match'] = combined[twop_col].notna()
    combined['_is_ref_plane'] = (combined['plane'] == reference_plane)
    combined['_iou_filled'] = combined[iou_col].fillna(-1)  # -1 for no match

    # Priority score (higher = better):
    # - Reference plane with match: 3
    # - Non-reference plane with match: 2 + iou (iou is 0-1, so max ~2.99)
    # - Reference plane without match: 1
    # - Non-reference plane without match: 0
    combined['_priority'] = (
        combined['_is_ref_plane'].astype(int) * 1 +  # +1 for ref plane
        combined['_has_match'].astype(int) * 2 +      # +2 for having match
        combined['_iou_filled'] * combined['_has_match'].astype(int) * 0.5  # +iou*0.5 for tiebreaker
    )

    # Sort by mask_id and priority (descending), then take first per group
    rprint("  Selecting best match per HCR cell (vectorized)...")
    combined_sorted = combined.sort_values(
        by=[mask_col, '_priority'],
        ascending=[True, False]
    )

    # Drop duplicates - keeps first (highest priority) per mask_id
    result = combined_sorted.drop_duplicates(subset=[mask_col], keep='first').copy()

    # Clean up temp columns
    result = result.drop(columns=['_has_match', '_is_ref_plane', '_iou_filled', '_priority'])
    result = result.reset_index(drop=True)

    # Add metadata columns - use flat column names even if DataFrame has MultiIndex
    # For MultiIndex DataFrames, single-string assignment creates ('', 'col') or similar
    # We'll flatten the columns at the end to avoid confusion

    # Determine match type (single vs multi) - count matches per HCR cell across all planes
    match_counts = combined[combined[twop_col].notna()].groupby(mask_col).size()

    # Build metadata as separate Series, then assign
    best_plane_series = result['plane'].copy()
    has_2p_match_series = result[twop_col].notna()
    match_count_series = result[mask_col].map(match_counts).fillna(0).astype(int)

    # Compute match_type as a Series
    match_type_series = pd.Series('none', index=result.index)
    match_type_series[has_2p_match_series & (match_count_series == 1)] = 'single'
    match_type_series[has_2p_match_series & (match_count_series > 1)] = 'multi'

    # If columns are MultiIndex, flatten them first to avoid issues
    if isinstance(result.columns, pd.MultiIndex):
        # Flatten MultiIndex columns to single strings
        result.columns = ['_'.join(str(c) for c in col).strip('_') if isinstance(col, tuple) else str(col)
                          for col in result.columns]
        rprint("  (Flattened MultiIndex columns to single-level)")

    # Now assign metadata columns (will be simple string keys)
    result['best_plane'] = best_plane_series.values
    result['has_2p_match'] = has_2p_match_series.values
    result['match_type'] = match_type_series.values

    # Summary stats
    n_total = len(result)
    n_matched = result['has_2p_match'].sum()
    n_single = (result['match_type'] == 'single').sum()
    n_multi = (result['match_type'] == 'multi').sum()

    elapsed = time.time() - t0
    rprint(f"\n  Smart merge complete in {elapsed:.2f}s:")
    rprint(f"    Total HCR cells: {n_total}")
    rprint(f"    With 2P match: {n_matched} ({100*n_matched/n_total:.1f}%)")
    rprint(f"      - Single plane match: {n_single}")
    rprint(f"      - Multi plane match: {n_multi}")
    rprint(f"    Without 2P match: {n_total - n_matched}")

    # Plane distribution
    plane_counts = result['best_plane'].value_counts().sort_index()
    rprint(f"    Rows from each plane:")
    for plane, count in plane_counts.items():
        marker = " (reference)" if plane == reference_plane else ""
        rprint(f"      Plane {plane}: {count}{marker}")

    # Merge HCR coordinates if provided
    if hcr_coords is not None and len(hcr_coords) > 0:
        # Find the actual mask_id column name after flattening
        mask_col_flat = None
        for col in result.columns:
            if 'mask_id_main' in str(col):
                mask_col_flat = col
                break

        if mask_col_flat is not None:
            # Merge coordinates on mask_id
            result = result.merge(
                hcr_coords,
                left_on=mask_col_flat,
                right_on='mask_id',
                how='left'
            )
            # Drop the duplicate mask_id column from coords
            if 'mask_id' in result.columns and mask_col_flat != 'mask_id':
                result = result.drop(columns=['mask_id'])

            n_with_coords = result['hcr_x'].notna().sum()
            rprint(f"    With HCR coordinates: {n_with_coords}/{n_total}")
        else:
            rprint(f"  [yellow]Warning: Could not find mask_id_main column for coordinate merge[/yellow]")

    return result


def filter_artifacts(
    df: pd.DataFrame,
    plane_data: Dict[int, dict],
    y_threshold: float = 100
) -> pd.DataFrame:
    """
    Remove neurons in artifact regions (e.g., top edge of FOV).

    Uses centroid Y position from 2P masks.

    Args:
        df: DataFrame with 'plane' and 'twoP_mask' columns
        plane_data: Dict from load_2p_spatial_data
        y_threshold: Minimum Y position (pixels)

    Returns:
        Filtered DataFrame
    """
    if 'plane' not in df.columns or 'twoP_mask' not in df.columns:
        rprint("[yellow]Warning: Required columns not found, skipping artifact filter[/yellow]")
        return df

    n_before = len(df)

    # Build lookup of cell positions
    keep_mask = np.ones(len(df), dtype=bool)

    for idx, row in df.iterrows():
        plane = row['plane']
        twop_mask = row['twoP_mask']

        if pd.isna(twop_mask) or plane not in plane_data:
            continue

        pdata = plane_data[plane]
        cell_ids = pdata['cell_ids']
        y_centroids = pdata['y_centroids']

        # Find this cell's position
        try:
            cell_idx = cell_ids.index(twop_mask) if isinstance(cell_ids, list) else np.where(cell_ids == twop_mask)[0][0]
            y_pos = y_centroids[cell_idx]

            if y_pos < y_threshold:
                keep_mask[idx] = False
        except (ValueError, IndexError):
            continue

    filtered = df[keep_mask].copy()
    n_removed = n_before - len(filtered)

    if n_removed > 0:
        rprint(f"  Artifact filter: removed {n_removed} cells with Y < {y_threshold}")

    return filtered


def create_artifact_mask(
    plane_data: Dict[int, dict],
    y_artifact_threshold: float = 100,
    edge_margin: int = 20,
    fov_shape: Tuple[int, int] = None
) -> Dict[int, np.ndarray]:
    """
    Create boolean masks indicating which 2P neurons should be kept (not artifacts).

    Filters neurons based on:
    1. Y artifact threshold: removes neurons with Y < threshold (top of FOV artifact)
    2. Edge margin: removes neurons within `edge_margin` pixels of any FOV edge

    Args:
        plane_data: Dict from load_2p_spatial_data mapping plane -> spatial data
        y_artifact_threshold: Minimum Y position (pixels). Neurons with centroid Y
                              below this are removed (artifact at top of image)
        edge_margin: Pixels from FOV edge. Neurons within this margin of any edge
                     are removed.
        fov_shape: (height, width) of FOV in pixels. If None, inferred from
                   maximum centroid values (with 10% buffer).

    Returns:
        Dict mapping plane -> boolean mask array (True = keep, False = filter out)
        The mask arrays align with plane_data[plane]['cell_ids']
    """
    artifact_masks = {}

    for plane, pdata in plane_data.items():
        x_centroids = pdata['x_centroids']
        y_centroids = pdata['y_centroids']
        n_cells = len(x_centroids)

        if n_cells == 0:
            artifact_masks[plane] = np.array([], dtype=bool)
            continue

        # Infer FOV shape if not provided
        if fov_shape is not None:
            fov_height, fov_width = fov_shape
        else:
            # Estimate from centroid range with buffer
            fov_width = int(np.max(x_centroids) * 1.1)
            fov_height = int(np.max(y_centroids) * 1.1)

        # Start with all cells kept
        keep_mask = np.ones(n_cells, dtype=bool)

        # Filter 1: Y artifact threshold (top of image = low Y values)
        # In image coordinates, Y=0 is at top, Y increases going down
        artifact_y_mask = y_centroids < y_artifact_threshold

        # Filter 2: Edge margin (all four edges)
        edge_mask = (
            (x_centroids < edge_margin) |  # left edge
            (x_centroids > fov_width - edge_margin) |  # right edge
            (y_centroids < edge_margin) |  # top edge
            (y_centroids > fov_height - edge_margin)  # bottom edge
        )

        # Combine: remove if in artifact region OR on edge
        remove_mask = artifact_y_mask | edge_mask
        keep_mask = ~remove_mask

        artifact_masks[plane] = keep_mask

        n_artifact = np.sum(artifact_y_mask)
        n_edge = np.sum(edge_mask & ~artifact_y_mask)  # edge-only (not also artifact)
        n_removed = np.sum(remove_mask)

        if n_removed > 0:
            rprint(f"  Plane {plane}: filtering {n_removed}/{n_cells} neurons "
                   f"(Y<{y_artifact_threshold}: {n_artifact}, edge: {n_edge})")

    return artifact_masks


def apply_artifact_filter_to_responses(
    all_responses: Dict[int, dict],
    artifact_masks: Dict[int, np.ndarray]
) -> Dict[int, dict]:
    """
    Apply artifact filtering to response classifications.

    Sets excited_mask and inhibited_mask to False for neurons that fail
    the artifact filter, effectively excluding them from response analysis.

    Args:
        all_responses: Dict mapping plane -> response dict with 'excited_mask',
                       'inhibited_mask', 'response_magnitudes', 'p_values'
        artifact_masks: Dict mapping plane -> boolean mask (True = keep)

    Returns:
        Modified all_responses dict with filtered masks
    """
    filtered_responses = {}

    total_excited_before = 0
    total_excited_after = 0
    total_inhibited_before = 0
    total_inhibited_after = 0

    for plane, responses in all_responses.items():
        if plane not in artifact_masks:
            filtered_responses[plane] = responses.copy()
            continue

        keep_mask = artifact_masks[plane]

        # Copy response dict
        filtered = {k: v.copy() if isinstance(v, np.ndarray) else v
                    for k, v in responses.items()}

        # Track counts before filtering
        excited_before = np.sum(filtered['excited_mask'])
        inhibited_before = np.sum(filtered['inhibited_mask'])
        total_excited_before += excited_before
        total_inhibited_before += inhibited_before

        # Apply filter: set masks to False for neurons to exclude
        # This preserves array lengths and indexing
        filtered['excited_mask'] = filtered['excited_mask'] & keep_mask
        filtered['inhibited_mask'] = filtered['inhibited_mask'] & keep_mask

        # Track counts after filtering
        excited_after = np.sum(filtered['excited_mask'])
        inhibited_after = np.sum(filtered['inhibited_mask'])
        total_excited_after += excited_after
        total_inhibited_after += inhibited_after

        filtered_responses[plane] = filtered

    # Summary
    exc_removed = total_excited_before - total_excited_after
    inh_removed = total_inhibited_before - total_inhibited_after

    if exc_removed > 0 or inh_removed > 0:
        rprint(f"  Artifact filter removed: {exc_removed} excited, {inh_removed} inhibited")
        rprint(f"  After filtering: {total_excited_after} excited, {total_inhibited_after} inhibited")

    return filtered_responses


# =============================================================================
# SPATIAL SELECTION FUNCTIONS
# =============================================================================

def create_spatial_mask_from_bounds(
    plane_data: Dict[int, dict],
    x_min: float = None,
    x_max: float = None,
    y_min: float = None,
    y_max: float = None,
    polygon_vertices: List[Tuple[float, float]] = None
) -> Dict[int, np.ndarray]:
    """
    Create spatial masks from bounding box or polygon vertices (non-interactive).

    This filters 2P cells based on their centroid positions in 2P pixel coordinates.
    Same XY region applies to all planes since they share the same FOV.

    Args:
        plane_data: Dict from load_2p_spatial_data mapping plane -> spatial data.
                   Each plane dict must have 'x_centroids' and 'y_centroids' arrays.
        x_min, x_max: X coordinate bounds (2P pixels). None = no limit.
        y_min, y_max: Y coordinate bounds (2P pixels). None = no limit.
                      Note: Y increases downward in image coordinates.
        polygon_vertices: Optional list of (x, y) tuples defining polygon vertices.
                         If provided, overrides bounding box parameters.
                         Vertices should be in 2P pixel coordinates.

    Returns:
        Dict mapping plane -> boolean mask array (True = inside region).
        Masks align with plane_data[plane]['cell_ids'].

    Index Alignment:
        spatial_masks[plane][i] corresponds to:
        - plane_data[plane]['cell_ids'][i]
        - plane_data[plane]['x_centroids'][i]
        - plane_data[plane]['y_centroids'][i]

    Examples:
        # Bounding box - exclude top 100 pixels (artifact region)
        masks = create_spatial_mask_from_bounds(plane_data, y_min=100)

        # Rectangular ROI
        masks = create_spatial_mask_from_bounds(plane_data, x_min=100, x_max=400,
                                                y_min=100, y_max=400)

        # Polygon ROI
        vertices = [(100, 100), (400, 100), (400, 400), (100, 400)]
        masks = create_spatial_mask_from_bounds(plane_data, polygon_vertices=vertices)
    """
    spatial_masks = {}

    # Use polygon if provided
    if polygon_vertices is not None:
        from matplotlib.path import Path as MplPath
        polygon_path = MplPath(polygon_vertices)

        for plane, pdata in plane_data.items():
            x_centroids = pdata['x_centroids']
            y_centroids = pdata['y_centroids']
            n_cells = len(x_centroids)

            if n_cells == 0:
                spatial_masks[plane] = np.array([], dtype=bool)
                continue

            points = np.column_stack((x_centroids, y_centroids))
            spatial_masks[plane] = polygon_path.contains_points(points)

    else:
        # Use bounding box
        for plane, pdata in plane_data.items():
            x_centroids = pdata['x_centroids']
            y_centroids = pdata['y_centroids']
            n_cells = len(x_centroids)

            if n_cells == 0:
                spatial_masks[plane] = np.array([], dtype=bool)
                continue

            # Start with all cells selected
            keep_mask = np.ones(n_cells, dtype=bool)

            # Apply bounds
            if x_min is not None:
                keep_mask &= (x_centroids >= x_min)
            if x_max is not None:
                keep_mask &= (x_centroids <= x_max)
            if y_min is not None:
                keep_mask &= (y_centroids >= y_min)
            if y_max is not None:
                keep_mask &= (y_centroids <= y_max)

            spatial_masks[plane] = keep_mask

    # Summary
    total_cells = sum(len(m) for m in spatial_masks.values())
    selected_cells = sum(np.sum(m) for m in spatial_masks.values())
    rprint(f"Spatial selection: {selected_cells}/{total_cells} 2P cells in region")

    return spatial_masks


def create_spatial_mask_from_path(
    selection_path,  # matplotlib.path.Path
    plane_data: Dict[int, dict]
) -> Dict[int, np.ndarray]:
    """
    Create spatial masks from a matplotlib Path object (from interactive selection).

    This applies Path.contains_points() to check which cell centroids fall within
    the selection region for each plane.

    Args:
        selection_path: matplotlib.path.Path object defining the selection region
                       in 2P pixel coordinates. Typically created by LassoSelector.
        plane_data: Dict from load_2p_spatial_data mapping plane -> spatial data.

    Returns:
        Dict mapping plane -> boolean mask array (True = inside region).
        Masks align with plane_data[plane]['cell_ids'].

    Index Alignment:
        spatial_masks[plane][i] corresponds to:
        - plane_data[plane]['cell_ids'][i]
        - plane_data[plane]['x_centroids'][i]
        - plane_data[plane]['y_centroids'][i]
    """
    spatial_masks = {}

    for plane, pdata in plane_data.items():
        x_centroids = pdata['x_centroids']
        y_centroids = pdata['y_centroids']
        n_cells = len(x_centroids)

        if n_cells == 0:
            spatial_masks[plane] = np.array([], dtype=bool)
            continue

        points = np.column_stack((x_centroids, y_centroids))
        spatial_masks[plane] = selection_path.contains_points(points)

    # Summary
    total_cells = sum(len(m) for m in spatial_masks.values())
    selected_cells = sum(np.sum(m) for m in spatial_masks.values())
    rprint(f"Spatial selection: {selected_cells}/{total_cells} 2P cells in region")

    return spatial_masks


def apply_spatial_selection_to_dataframe(
    merged_df: pd.DataFrame,
    spatial_masks: Dict[int, np.ndarray],
    plane_data: Dict[int, dict],
    include_hcr_only: bool = True,
    plane_col: str = 'best_plane',
    twop_mask_col: str = 'twoP_mask'
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Apply spatial selection masks to HCR merged DataFrame.

    This filters the HCR DataFrame to only include cells whose 2P centroids
    fall within the spatial selection region. Handles the complex index mapping
    between DataFrame rows and per-plane spatial data.

    Args:
        merged_df: DataFrame from smart_merge_planes with HCR cells.
                  Must have columns for plane ID and 2P mask ID.
        spatial_masks: Dict from create_spatial_mask_* functions.
                      Maps plane -> boolean mask aligned with plane_data cell_ids.
        plane_data: Dict from load_2p_spatial_data. Required for cell_id lookup.
        include_hcr_only: If True (default), keep cells without 2P match.
                         If False, exclude cells that don't have a 2P match.
        plane_col: Column name for plane ID (default: 'best_plane').
        twop_mask_col: Column name for 2P mask ID (default: 'twoP_mask').

    Returns:
        Tuple of:
        - Filtered DataFrame (copy with only selected cells)
        - Boolean mask aligned to original DataFrame rows (True = kept)

    Index Mapping (CRITICAL):
        For each row in merged_df:
        1. Get plane from row[plane_col]
        2. Get twoP_mask from row[twop_mask_col] (e.g., 42.0)
        3. Convert to cell_id: f'cell_{int(twop_mask)}' -> 'cell_42'
        4. Find index in plane_data[plane]['cell_ids']
        5. Check spatial_masks[plane][index]

    Note:
        - Rows with NaN twoP_mask (HCR-only cells) pass through if include_hcr_only=True
        - Uses original DataFrame indices via .loc[] to preserve row identity
    """
    n_rows = len(merged_df)
    keep_mask = np.zeros(n_rows, dtype=bool)

    # Build reverse lookup: (plane, cell_id) -> local_index
    # This avoids O(n) list.index() calls
    plane_cell_to_idx = {}
    for plane, pdata in plane_data.items():
        cell_ids = pdata['cell_ids']
        for idx, cell_id in enumerate(cell_ids):
            plane_cell_to_idx[(int(plane), cell_id)] = idx

    # Track statistics
    n_hcr_only = 0
    n_no_match_in_spatial = 0
    n_selected = 0
    n_excluded = 0

    for i, (df_idx, row) in enumerate(merged_df.iterrows()):
        # Handle multi-index columns (e.g., ('best_plane', ''))
        if isinstance(merged_df.columns, pd.MultiIndex):
            plane_val = row.get((plane_col, ''), row.get(plane_col))
            twop_mask_val = row.get((twop_mask_col, ''), row.get(twop_mask_col))
        else:
            plane_val = row.get(plane_col)
            twop_mask_val = row.get(twop_mask_col)

        # HCR-only cell (no 2P match)
        if pd.isna(twop_mask_val):
            n_hcr_only += 1
            keep_mask[i] = include_hcr_only
            continue

        # Check if plane is valid
        if pd.isna(plane_val) or int(plane_val) not in spatial_masks:
            n_no_match_in_spatial += 1
            keep_mask[i] = False
            continue

        plane = int(plane_val)

        # Convert twoP_mask (42.0) to cell_id ('cell_42')
        cell_id = f'cell_{int(twop_mask_val)}'
        key = (plane, cell_id)

        if key in plane_cell_to_idx:
            local_idx = plane_cell_to_idx[key]
            is_in_region = spatial_masks[plane][local_idx]
            keep_mask[i] = is_in_region
            if is_in_region:
                n_selected += 1
            else:
                n_excluded += 1
        else:
            # Cell ID not found in plane_data (shouldn't happen normally)
            n_no_match_in_spatial += 1
            keep_mask[i] = False

    # Filter DataFrame using boolean mask
    # Use iloc to get row positions, then loc to preserve proper indexing
    filtered_df = merged_df.iloc[keep_mask].copy()

    # Summary
    rprint(f"Spatial selection applied to DataFrame:")
    rprint(f"  Total rows: {n_rows}")
    rprint(f"  2P cells selected: {n_selected}")
    rprint(f"  2P cells excluded: {n_excluded}")
    rprint(f"  HCR-only cells: {n_hcr_only} ({'kept' if include_hcr_only else 'excluded'})")
    rprint(f"  Final rows: {len(filtered_df)}")

    return filtered_df, keep_mask


class SpatialSelector:
    """
    Interactive lasso selector for 2P cells. Use with %matplotlib widget backend.

    Usage (two cells):
        # Cell 1: Create selector and draw your selection
        selector = SpatialSelector(plane_data, reference_plane, merged_df=merged_subtracted)
        # Draw lasso on the plot, then press Enter to confirm

        # Cell 2: After confirming, get the result
        selection_path = selector.get_selection()
        if selection_path is not None:
            spatial_masks = create_spatial_mask_from_path(selection_path, plane_data)
    """

    def __init__(
        self,
        plane_data: Dict[int, dict],
        reference_plane: int,
        merged_df: pd.DataFrame = None,
        highlight_matched: bool = True,
        figsize: Tuple[int, int] = (14, 7)
    ):
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for interactive selection")

        from matplotlib.widgets import LassoSelector
        from matplotlib.path import Path as MplPath

        if reference_plane not in plane_data:
            raise ValueError(f"Reference plane {reference_plane} not found in plane_data")

        self.selected_path = None
        self.selected_indices = []
        self.confirmed = False
        self.cancelled = False

        ref_data = plane_data[reference_plane]
        self.x_coords = ref_data['x_centroids']
        self.y_coords = ref_data['y_centroids']
        cell_ids = ref_data['cell_ids']
        n_cells = len(self.x_coords)

        if n_cells == 0:
            rprint("[yellow]Warning: No cells in reference plane[/yellow]")
            return

        # Determine cell colors based on HCR match status
        if highlight_matched and merged_df is not None:
            colors = np.full(n_cells, 0.3)

            if isinstance(merged_df.columns, pd.MultiIndex):
                plane_col = ('best_plane', '') if ('best_plane', '') in merged_df.columns else 'best_plane'
                mask_col = ('twoP_mask', '') if ('twoP_mask', '') in merged_df.columns else 'twoP_mask'
            else:
                plane_col = 'best_plane'
                mask_col = 'twoP_mask'

            plane_rows = merged_df[merged_df[plane_col] == reference_plane]
            matched_masks = set(plane_rows[mask_col].dropna().astype(int))

            for i, cell_id in enumerate(cell_ids):
                try:
                    mask_id = int(cell_id.split('_')[1])
                    if mask_id in matched_masks:
                        colors[i] = 1.0
                except (ValueError, IndexError):
                    continue
        else:
            if 'mean_frames' in ref_data and ref_data['mean_frames'] is not None:
                mean_activity = np.mean(ref_data['mean_frames'], axis=1)
                colors = (mean_activity - np.percentile(mean_activity, 5)) / \
                         (np.percentile(mean_activity, 95) - np.percentile(mean_activity, 5) + 1e-10)
                colors = np.clip(colors, 0, 1)
            else:
                colors = np.ones(n_cells) * 0.5

        # Create figure
        self.fig, self.ax = plt.subplots(1, 1, figsize=figsize)

        scatter = self.ax.scatter(self.x_coords, self.y_coords, c=colors, cmap='viridis', s=15, alpha=0.7)
        self.ax.set_title(f"Reference Plane {reference_plane} - Draw selection region\n"
                     f"(Click+drag to select, Enter to confirm, Escape to cancel)")
        self.ax.set_xlabel("X (2P pixels)")
        self.ax.set_ylabel("Y (2P pixels)")
        self.ax.invert_yaxis()
        self.ax.set_aspect('equal')

        cbar = self.fig.colorbar(scatter, ax=self.ax)
        if highlight_matched and merged_df is not None:
            cbar.set_label('HCR match (bright=matched)')
        else:
            cbar.set_label('Activity (normalized)')

        # Status text
        self.status = self.ax.text(0.5, 0.02, "No selection yet - draw a lasso around cells",
                             ha='center', va='bottom', transform=self.ax.transAxes,
                             bbox=dict(boxstyle="round", facecolor='white', alpha=0.9))

        # Lasso selector
        self.lasso = LassoSelector(self.ax, onselect=self._on_select, button=1)

        # Key callback
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        # Store MplPath for later use
        self._MplPath = MplPath

        plt.show()
        rprint("[cyan]Draw a lasso around cells, then press Enter to confirm (Escape to cancel)[/cyan]")

    def _on_select(self, verts):
        path = self._MplPath(verts)
        points = np.column_stack((self.x_coords, self.y_coords))
        self.selected_indices = np.nonzero(path.contains_points(points))[0]
        self.selected_path = path

        self.status.set_text(f"Selected {len(self.selected_indices)} cells. Press Enter to confirm.")
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key == 'enter':
            if self.selected_path is not None:
                self.confirmed = True
                rprint(f"[green]Selection confirmed: {len(self.selected_indices)} cells[/green]")
                plt.close(self.fig)
            else:
                self.status.set_text("No selection made. Draw a region first.")
                self.fig.canvas.draw_idle()
        elif event.key == 'escape':
            self.cancelled = True
            self.selected_path = None
            rprint("[yellow]Selection cancelled[/yellow]")
            plt.close(self.fig)

    def get_selection(self):
        """Get the confirmed selection path. Returns None if not confirmed or cancelled."""
        if self.confirmed and self.selected_path is not None:
            return self.selected_path
        elif self.cancelled:
            rprint("[yellow]Selection was cancelled[/yellow]")
            return None
        elif self.selected_path is not None and not self.confirmed:
            rprint("[yellow]Selection not yet confirmed. Press Enter on the plot to confirm.[/yellow]")
            return None
        else:
            rprint("[yellow]No selection made[/yellow]")
            return None


def create_spatial_selection_interactive(
    plane_data: Dict[int, dict],
    reference_plane: int,
    merged_df: pd.DataFrame = None,
    highlight_matched: bool = True,
    figsize: Tuple[int, int] = (14, 7)
):
    """
    Create interactive lasso selector on reference plane 2P FOV.

    DEPRECATED: Use SpatialSelector class instead for reliable widget backend support.

    User draws polygon on 2P pixel coordinates. Same region applies to all planes
    since they share the same FOV.

    IMPORTANT: Requires '%matplotlib widget' backend (ipympl) for interactivity.

    Args:
        plane_data: Dict from load_2p_spatial_data (maps plane -> spatial data)
        reference_plane: Plane number to display for interactive selection
        merged_df: Optional HCR DataFrame to highlight HCR-matched cells
        highlight_matched: If True and merged_df provided, color HCR-matched cells
        figsize: Figure size (width, height)

    Returns:
        SpatialSelector object. Call .get_selection() in a separate cell after confirming.

    Usage:
        %matplotlib widget

        # Cell 1:
        selector = create_spatial_selection_interactive(plane_data, reference_plane)
        # Draw region and press Enter

        # Cell 2:
        selection_path = selector.get_selection()
        if selection_path is not None:
            spatial_masks = create_spatial_mask_from_path(selection_path, plane_data)
    """
    return SpatialSelector(
        plane_data, reference_plane, merged_df, highlight_matched, figsize
    )


# =============================================================================
# STIMULUS RESPONSE ANALYSIS
# =============================================================================

def build_frame_mapping(
    base_path: Path,
    mouse_name: str,
    date: str,
    runs: list,
    nplanes: int
) -> Tuple[dict, dict, int]:
    """Build cumulative frame offsets for concatenated runs.

    When multiple runs are concatenated in Suite2p, this function determines
    where each run starts in the concatenated trace array.

    Parameters
    ----------
    base_path : Path
        Base path to data (e.g., /mnt/nasquatch/data/2p/jonna/EASI_FISH/pipeline)
    mouse_name : str
        Mouse name (e.g., 'PS274_1R')
    date : str
        Session date (e.g., '250826')
    runs : list
        List of run IDs in concatenation order (e.g., ['001', '002', '003'])
    nplanes : int
        Number of imaging planes

    Returns
    -------
    cumulative_frames : dict
        {run_id: start_frame} - where each run begins in concatenated array
    frame_counts : dict
        {run_id: n_frames} - frames per plane in each run
    total_frames : int
        Total frames in concatenated movie
    """
    import scipy.io as sio

    cumulative_frames = {}
    frame_counts = {}
    total = 0

    for run_id in runs:
        run_dir = base_path / mouse_name / '2P' / f'{mouse_name}_{date}_{run_id}'
        mat_path = run_dir / f'{mouse_name}_{date}_{run_id}.mat'

        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        if not mat_path.exists():
            raise FileNotFoundError(f"MAT file not found: {mat_path}")

        mat_data = sio.loadmat(str(mat_path))
        total_frames_in_file = int(mat_data['info']['config'][0][0]['frames'][0][0])
        frames_per_plane = total_frames_in_file // nplanes

        cumulative_frames[run_id] = total
        frame_counts[run_id] = frames_per_plane
        total += frames_per_plane

    return cumulative_frames, frame_counts, total


def load_ephys_stimulus(
    ephys_path: Path,
    nchannels: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load stimulus and frame channels from .ephys file.

    The .ephys file contains multi-channel DAQ recordings:
    - Channel 1 (index 1): Frame sync TTL pulses
    - Channel 2 (index 2): Stimulus TTL pulses

    Args:
        ephys_path: Path to .ephys file
        nchannels: Number of channels in ephys recording (default: 5)

    Returns:
        Tuple of (stim_channel, frame_channel) as 1D arrays
    """
    ephys_path = Path(ephys_path)

    if not ephys_path.exists():
        raise FileNotFoundError(f"Ephys file not found: {ephys_path}")

    # Load as single-precision float (np.single)
    with open(ephys_path, 'rb') as fid:
        data = np.fromfile(fid, np.single)

    # Reshape with Fortran order (channel-first layout)
    remainder = data.size % nchannels
    if remainder != 0:
        data = np.pad(data, (0, nchannels - remainder), mode='constant')

    data_channeled = data.reshape(nchannels, -1, order='F')

    # Extract channels: channel 1 = frame sync, channel 2 = stimulus
    frame_channel = data_channeled[1, :]
    stim_channel = data_channeled[2, :]

    # Clean stimulus channel (remove negative values)
    stim_channel[stim_channel < 0] = 0

    rprint(f"  Loaded ephys: {len(stim_channel)} samples, {nchannels} channels")

    return stim_channel, frame_channel


def detect_stimulus_onsets(
    run_path: Path,
    nplanes: int = 4,
    nchannels: int = 5,
    stim_threshold: float = 0.05,
    min_stim_separation: int = 1000,
    min_onset_interval_sec: float = None,
    framerate: float = None,
    frame_ttl_threshold: float = 1.0,
    min_frame_separation: int = 30,
    smoothing_window: int = 200,
    expected_frames_per_plane: int = None
) -> Tuple[np.ndarray, dict]:
    """
    Detect stimulus onsets from .ephys file with proper frame conversion.

    For multi-plane imaging, converts raw ephys sample times to per-plane
    frame indices by:
    1. Detecting all frame TTL pulses
    2. Mapping stimulus times to closest frame index
    3. Dividing by nplanes for per-plane index
    4. (Optional) Filtering to train onsets with minimum time interval

    Args:
        run_path: Path to run directory (contains .ephys file) or .ephys file directly
        nplanes: Number of imaging planes (for frame conversion)
        nchannels: Number of channels in ephys recording
        stim_threshold: Threshold for stimulus detection (0-1)
        min_stim_separation: Minimum separation between stimuli (in ephys samples)
        min_onset_interval_sec: Minimum interval between train onsets in seconds.
            Use this to filter out within-train pulses and keep only train starts.
            Requires framerate to be set. If None, no secondary filtering is applied.
        framerate: Per-plane framerate in Hz. Required if min_onset_interval_sec is set.
        frame_ttl_threshold: Threshold for frame TTL detection
        min_frame_separation: Minimum separation between frames (in ephys samples)
        smoothing_window: Window size for stimulus smoothing
        expected_frames_per_plane: If provided, validates that ephys frame count matches
            the actual trace length. If there's a mismatch, rescales onset indices.

    Returns:
        Tuple of (stim_frames, info_dict):
        - stim_frames: Array of stimulus onset frame indices (per-plane)
        - info_dict: Diagnostic information including raw stim_channel
    """
    from scipy.ndimage import uniform_filter1d

    run_path = Path(run_path)

    # Find ephys file
    if run_path.is_dir():
        ephys_files = list(run_path.glob('*.ephys'))
        if not ephys_files:
            raise FileNotFoundError(f"No .ephys file found in {run_path}")
        ephys_path = ephys_files[0]
    else:
        ephys_path = run_path

    # Load ephys data
    stim_channel, frame_channel = load_ephys_stimulus(ephys_path, nchannels)

    # Smooth stimulus channel for more robust detection
    smoothed_stim = uniform_filter1d(stim_channel, size=smoothing_window)

    # Detect stimulus onsets (rising edge crossings)
    all_stim_times = np.where(
        (smoothed_stim[:-1] < stim_threshold) &
        (smoothed_stim[1:] >= stim_threshold)
    )[0]

    # Filter by minimum separation
    stim_times = []
    last_crossing = -min_stim_separation
    for stim in all_stim_times:
        if stim - last_crossing >= min_stim_separation:
            stim_times.append(stim)
            last_crossing = stim
    stim_times = np.array(stim_times)

    # Detect frame TTL pulses (for time-to-frame mapping)
    frame_times = []
    last_crossing = -min_frame_separation
    for i in range(1, len(frame_channel)):
        if frame_channel[i - 1] < frame_ttl_threshold and frame_channel[i] >= frame_ttl_threshold:
            if i - last_crossing >= min_frame_separation:
                frame_times.append(i)
                last_crossing = i
    frame_times = np.array(frame_times)

    # Map stimulus times to frame indices
    stim_frames_raw = []
    for stim in stim_times:
        frame_index = np.argmin(np.abs(frame_times - stim))
        stim_frames_raw.append(frame_index)

    # Convert to per-plane frame indices
    stim_frames = np.array(stim_frames_raw) // nplanes
    n_events_before_filter = len(stim_frames)

    # Apply minimum interval filter in frame space (for train onset detection)
    min_interval_frames = None
    rprint(f"  [DEBUG] min_onset_interval_sec={min_onset_interval_sec}, framerate={framerate}")
    if min_onset_interval_sec is not None and framerate is not None:
        min_interval_frames = int(min_onset_interval_sec * framerate)
        filtered_frames = []
        last_frame = -min_interval_frames
        for frame in stim_frames:
            if frame - last_frame >= min_interval_frames:
                filtered_frames.append(frame)
                last_frame = frame
        stim_frames = np.array(filtered_frames)

    # Check for frame count mismatch and rescale if needed
    ephys_frames_per_plane = len(frame_times) // nplanes
    scale_factor = 1.0

    if expected_frames_per_plane is not None:
        if ephys_frames_per_plane != expected_frames_per_plane:
            scale_factor = expected_frames_per_plane / ephys_frames_per_plane
            rprint(f"  [yellow]Frame count mismatch detected![/yellow]")
            rprint(f"    Ephys reports: {ephys_frames_per_plane} frames/plane")
            rprint(f"    Traces have: {expected_frames_per_plane} frames/plane")
            rprint(f"    Rescaling onset indices by {scale_factor:.4f}")

            # Rescale the onset indices
            stim_frames = (stim_frames * scale_factor).astype(int)

    # Build info dict for diagnostics
    info = {
        'stim_channel': stim_channel,
        'smoothed_stim': smoothed_stim,
        'frame_channel': frame_channel,
        'stim_times_raw': stim_times,
        'frame_times': frame_times,
        'stim_frames_raw': np.array(stim_frames_raw),
        'nplanes': nplanes,
        'n_total_frames': len(frame_times),
        'n_frames_per_plane': ephys_frames_per_plane,
        'expected_frames_per_plane': expected_frames_per_plane,
        'scale_factor': scale_factor,
        'min_onset_interval_sec': min_onset_interval_sec,
        'min_interval_frames': min_interval_frames,
        'n_events_before_filter': n_events_before_filter,
    }

    rprint(f"  Detected {n_events_before_filter} stimulus events")
    if min_onset_interval_sec is not None and framerate is not None:
        rprint(f"  Filtered to {len(stim_frames)} train onsets (min {min_onset_interval_sec}s / {min_interval_frames} frames interval)")
    rprint(f"  Frame conversion: {len(frame_times)} total frames -> {ephys_frames_per_plane} per plane")
    if expected_frames_per_plane is not None and scale_factor != 1.0:
        rprint(f"  [green]Rescaled to trace length: {expected_frames_per_plane} frames[/green]")
    if len(stim_frames) > 1:
        intervals = np.diff(stim_frames)
        rprint(f"  Inter-stimulus intervals (frames): min={intervals.min()}, max={intervals.max()}")

    return stim_frames, info


def compute_dff(
    traces: np.ndarray,
    onsets: np.ndarray,
    framerate: float,
    config: AnalysisConfig
) -> dict:
    """
    Compute trial-aligned DF/F.

    Args:
        traces: (n_neurons, n_frames) array of fluorescence traces
        onsets: Array of stimulus onset frame indices
        framerate: Imaging framerate in Hz
        config: AnalysisConfig with pre_stim, post_stim

    Returns:
        dict with:
        - 'dff_trials': List of (n_neurons, n_frames) arrays per trial
        - 'dff_avg': (n_neurons, n_frames) trial-averaged DF/F
        - 'baseline_f': (n_neurons,) baseline fluorescence
        - 'time_axis': Array of time points relative to stimulus
    """
    pre_frames = int(config.pre_stim * framerate)
    post_frames = int(config.post_stim * framerate)
    trial_length = pre_frames + post_frames

    n_neurons = traces.shape[0]
    n_frames = traces.shape[1]

    # Diagnostic: check if onsets are in valid range
    valid_range_min = pre_frames
    valid_range_max = n_frames - post_frames
    n_before = np.sum(onsets < valid_range_min)
    n_after = np.sum(onsets > valid_range_max)
    n_valid = len(onsets) - n_before - n_after

    if n_before > 0 or n_after > 0:
        rprint(f"  [yellow]Warning: {len(onsets)} onsets provided, but only {n_valid} are within valid frame range[/yellow]")
        rprint(f"    Trace length: {n_frames} frames, valid onset range: [{valid_range_min}, {valid_range_max}]")
        rprint(f"    Onset range: [{onsets.min()}, {onsets.max()}]")
        rprint(f"    Skipped: {n_before} too early, {n_after} too late")

    dff_trials = []
    skipped_early = 0
    skipped_late = 0

    for onset in onsets:
        start = onset - pre_frames
        end = onset + post_frames

        # Skip if trial extends beyond recording
        if start < 0:
            skipped_early += 1
            continue
        if end > n_frames:
            skipped_late += 1
            continue

        trial_data = traces[:, start:end]

        # Calculate baseline from pre-stimulus period
        baseline = np.mean(trial_data[:, :pre_frames], axis=1, keepdims=True)
        baseline[baseline == 0] = 1e-10  # Avoid division by zero

        # Compute DF/F
        dff = (trial_data - baseline) / baseline
        dff_trials.append(dff)

    if not dff_trials:
        raise ValueError("No valid trials found")

    # Average across trials
    dff_avg = np.mean(np.stack(dff_trials), axis=0)

    # Overall baseline
    baseline_f = np.mean(traces[:, :pre_frames], axis=1)

    # Time axis
    time_axis = np.arange(-pre_frames, post_frames) / framerate

    result = {
        'dff_trials': dff_trials,
        'dff_avg': dff_avg,
        'baseline_f': baseline_f,
        'time_axis': time_axis,
        'n_trials': len(dff_trials),
        'pre_frames': pre_frames,
        'post_frames': post_frames,
    }

    rprint(f"  Computed DF/F: {n_neurons} neurons, {len(dff_trials)} trials")
    return result


def classify_responses(
    dff_trials: List[np.ndarray],
    framerate: float,
    config: AnalysisConfig
) -> dict:
    """
    Classify neurons as excited, inhibited, or non-responsive.

    Uses paired t-test comparing pre-stim baseline to post-stim response.

    Args:
        dff_trials: List of (n_neurons, n_frames) arrays per trial
        framerate: Imaging framerate in Hz
        config: AnalysisConfig with p_threshold, pre_stim, sort_window

    Returns:
        dict with:
        - 'excited_mask': bool array of excited neurons
        - 'inhibited_mask': bool array of inhibited neurons
        - 'p_values': float array of p-values
        - 'response_magnitudes': float array of response magnitudes
        - 'mean_responses': float array of mean post-stim responses
    """
    pre_frames = int(config.pre_stim * framerate)
    sort_frames = int(config.sort_window * framerate)

    n_neurons = dff_trials[0].shape[0]
    n_trials = len(dff_trials)

    # Collect baseline and response values for each neuron across trials
    baseline_values = np.zeros((n_neurons, n_trials))
    response_values = np.zeros((n_neurons, n_trials))

    for t, trial in enumerate(dff_trials):
        # Baseline: mean of pre-stimulus period
        baseline_values[:, t] = np.mean(trial[:, :pre_frames], axis=1)
        # Response: mean of sort_window after stimulus
        response_values[:, t] = np.mean(trial[:, pre_frames:pre_frames + sort_frames], axis=1)

    # Paired t-test for each neuron
    p_values = np.zeros(n_neurons)
    t_stats = np.zeros(n_neurons)

    for i in range(n_neurons):
        t_stat, p_val = stats.ttest_rel(response_values[i, :], baseline_values[i, :])
        p_values[i] = p_val
        t_stats[i] = t_stat

    # Response magnitude (mean across trials)
    response_magnitudes = np.mean(response_values - baseline_values, axis=1)
    mean_responses = np.mean(response_values, axis=1)

    # Classify
    significant = p_values < config.p_threshold
    excited_mask = significant & (response_magnitudes > 0)
    inhibited_mask = significant & (response_magnitudes < 0)

    n_excited = np.sum(excited_mask)
    n_inhibited = np.sum(inhibited_mask)
    n_responsive = n_excited + n_inhibited

    rprint(f"  Classification (p < {config.p_threshold}):")
    rprint(f"    Excited: {n_excited} ({100*n_excited/n_neurons:.1f}%)")
    rprint(f"    Inhibited: {n_inhibited} ({100*n_inhibited/n_neurons:.1f}%)")
    rprint(f"    Non-responsive: {n_neurons - n_responsive} ({100*(n_neurons - n_responsive)/n_neurons:.1f}%)")

    return {
        'excited_mask': excited_mask,
        'inhibited_mask': inhibited_mask,
        'p_values': p_values,
        't_stats': t_stats,
        'response_magnitudes': response_magnitudes,
        'mean_responses': mean_responses,
    }


# =============================================================================
# ARTIFACT FILTERING
# =============================================================================

def create_artifact_mask(
    plane_data: Dict,
    y_artifact_threshold: float = 100,
    edge_margin: float = 20
) -> Dict[int, np.ndarray]:
    """
    Create boolean masks identifying neurons to KEEP (True) vs filter (False).

    Filters out neurons that are:
    - In artifact region: Y < y_artifact_threshold (top of FOV in image coords where Y=0 is TOP)
    - Near edges: within edge_margin pixels of any FOV edge

    Args:
        plane_data: Dict[plane, dict] with 'x_centroids', 'y_centroids' arrays
        y_artifact_threshold: Neurons with Y < this are filtered (image coords)
        edge_margin: Neurons within this many pixels of FOV edge are filtered

    Returns:
        Dict[plane, np.ndarray[bool]] - True = keep, False = filter
    """
    masks = {}
    for plane, pdata in plane_data.items():
        x = np.array(pdata['x_centroids'])
        y = np.array(pdata['y_centroids'])

        # FOV bounds (estimate from data)
        x_max = np.max(x)
        y_max = np.max(y)

        # Keep neurons that pass all filters
        keep = (
            (y >= y_artifact_threshold) &  # Not in top artifact region
            (x >= edge_margin) &            # Not too close to left edge
            (x <= x_max - edge_margin) &    # Not too close to right edge
            (y <= y_max - edge_margin)      # Not too close to bottom edge
        )
        masks[plane] = keep

    return masks


def apply_artifact_filter_to_responses(
    all_responses: Dict,
    artifact_masks: Dict[int, np.ndarray]
) -> Dict:
    """
    Apply artifact masks to response classifications.

    Sets excited_mask and inhibited_mask to False for neurons in artifact regions.
    Other response data (p_values, magnitudes) are preserved unchanged.

    Args:
        all_responses: Dict[plane, dict] with 'excited_mask', 'inhibited_mask', etc.
        artifact_masks: Dict[plane, np.ndarray[bool]] from create_artifact_mask

    Returns:
        Dict with same structure as all_responses, masks updated
    """
    filtered = {}
    for plane, responses in all_responses.items():
        if plane not in artifact_masks:
            filtered[plane] = responses
            continue

        keep_mask = artifact_masks[plane]

        # Ensure mask lengths match
        if len(keep_mask) != len(responses['excited_mask']):
            rprint(f"  [WARNING] Plane {plane}: artifact mask length ({len(keep_mask)}) != "
                   f"response mask length ({len(responses['excited_mask'])}), skipping filter")
            filtered[plane] = responses
            continue

        filtered[plane] = {
            'excited_mask': responses['excited_mask'] & keep_mask,
            'inhibited_mask': responses['inhibited_mask'] & keep_mask,
            'p_values': responses.get('p_values'),
            't_stats': responses.get('t_stats'),
            'response_magnitudes': responses.get('response_magnitudes'),
            'mean_responses': responses.get('mean_responses'),
        }

    return filtered


# =============================================================================
# GENE CLASSIFICATION
# =============================================================================

def classify_gene_mode_shift(
    values: np.ndarray,
    sigma_multiplier: float = 3.5
) -> Tuple[np.ndarray, float, dict]:
    """
    Classify gene expression using mode-shift method.

    Finds mode of distribution, estimates noise from values NEAR the mode
    (not below, to avoid including true signal), sets threshold at
    mode + sigma_multiplier * noise_std.

    Args:
        values: Array of expression values
        sigma_multiplier: Multiplier for noise std to set threshold

    Returns:
        Tuple of (positive_mask, threshold, stats_dict)
    """
    # Remove NaN values for calculation
    clean_vals = values[~np.isnan(values)]

    if len(clean_vals) < 50:
        # Not enough data for reliable classification
        return np.zeros(len(values), dtype=bool), np.nan, {'error': 'insufficient_data'}

    # Find mode using histogram with more bins for better resolution
    hist, bins = np.histogram(clean_vals, bins=100)
    mode_idx = np.argmax(hist)
    mode_value = (bins[mode_idx] + bins[mode_idx + 1]) / 2

    # Estimate noise std from values NEAR the mode (within 0.5 std)
    # This avoids including true positive signal in noise estimate
    overall_std = np.std(clean_vals)
    near_mode = clean_vals[np.abs(clean_vals - mode_value) <= overall_std * 0.5]

    # If not enough values near mode, widen the window
    if len(near_mode) < 10:
        near_mode = clean_vals[np.abs(clean_vals - mode_value) <= overall_std]

    noise_std = np.std(near_mode) if len(near_mode) > 0 else overall_std * 0.5

    # Set threshold
    threshold = mode_value + sigma_multiplier * noise_std

    # Classify - need to handle NaN positions correctly
    positive_mask = np.zeros(len(values), dtype=bool)
    non_nan_mask = ~np.isnan(values)
    positive_mask[non_nan_mask] = values[non_nan_mask] > threshold

    n_positive = np.sum(positive_mask)
    n_valid = np.sum(non_nan_mask)

    stats = {
        'mode': mode_value,
        'noise_std': noise_std,
        'threshold': threshold,
        'n_positive': n_positive,
        'n_valid': n_valid,
        'pct_positive': 100 * n_positive / n_valid if n_valid > 0 else 0,
    }

    return positive_mask, threshold, stats


@dataclass
class ClassificationConfig:
    """
    Configuration for gene classification method.

    Allows explicit specification of each component:
    - normalization: how to transform values before classification
    - classifier: how to determine threshold
    - clip_negative: whether to clip negative values to 0 before classification
    - noisy_genes: genes that need higher sigma (more conservative threshold)

    Example usage in notebook:
        clf_config = ClassificationConfig(
            normalization='robust_iqr',  # 'none', 'log1p', 'robust_iqr'
            classifier='mode_shift',     # 'mode_shift', 'gmm'
            clip_negative=False,         # keep negatives from neuropil subtraction
            base_sigma=3.5,              # base sigma for mode_shift
            noisy_sigma=5.0,             # sigma for noisy genes
            noisy_genes=('PDYN',),       # genes needing higher sigma
        )

    Note: noisy_genes matches by gene name, so if a gene appears in multiple
    rounds (re-staining), all instances will use noisy_sigma.
    """
    normalization: str = 'none'       # 'none', 'log1p', 'robust_iqr'
    classifier: str = 'mode_shift'    # 'mode_shift', 'gmm'
    clip_negative: bool = False       # clip negative values to 0
    base_sigma: float = 3.5           # base sigma for mode_shift
    noisy_sigma: float = 5.0          # sigma for noisy genes
    noisy_genes: tuple = ()           # gene names needing higher sigma


def normalize_log1p(values: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Log transform: log(1+x). Compresses high values, handles zeros.

    For negative values (from neuropil subtraction), shifts data to positive first.

    Args:
        values: Array of expression values (can include negatives and NaN)

    Returns:
        Tuple of (normalized_values, stats_dict)
    """
    min_val = np.nanmin(values)
    if min_val < 0:
        shifted = values - min_val + 1e-6
    else:
        shifted = values
    return np.log1p(shifted), {'method': 'log1p', 'shift': min_val if min_val < 0 else 0}


def normalize_robust_iqr(values: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Robust scaling: (x - median) / IQR.

    This normalization is robust to outliers and works well for gene expression
    data that may have heavy tails or skewed distributions.

    Args:
        values: Array of expression values (can include negatives and NaN)

    Returns:
        Tuple of (normalized_values, stats_dict)
    """
    clean = values[~np.isnan(values)]
    median = np.median(clean)
    q75, q25 = np.percentile(clean, [75, 25])
    iqr = q75 - q25
    if iqr < 1e-6:
        # Fallback if IQR is too small
        iqr = np.std(clean) + 1e-6
    scaled = (values - median) / iqr
    return scaled, {'method': 'robust_iqr', 'median': median, 'iqr': iqr}


def classify_gmm(
    values: np.ndarray,
    n_components: int = 2
) -> Tuple[np.ndarray, float, dict]:
    """
    2-component Gaussian Mixture Model classification.

    Fits a GMM and uses posterior probability for classification.
    The positive component is the one with higher mean.

    Args:
        values: Array of expression values
        n_components: Number of GMM components (default 2)

    Returns:
        Tuple of (positive_mask, threshold, stats_dict)
    """
    from sklearn.mixture import GaussianMixture

    clean = values[~np.isnan(values)]
    if len(clean) < 50:
        return np.zeros(len(values), dtype=bool), np.nan, {'error': 'insufficient_data'}

    # Fit GMM
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        reg_covar=1e-4,
        n_init=5,
        random_state=42
    )
    gmm.fit(clean.reshape(-1, 1))

    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_

    # Identify positive component (higher mean)
    pos_idx = np.argmax(means)
    neg_idx = 1 - pos_idx

    # Threshold: 2 std above negative mean
    threshold = means[neg_idx] + 2 * stds[neg_idx]

    # Use posterior probability for classification
    mask = np.zeros(len(values), dtype=bool)
    valid = ~np.isnan(values)
    probs = gmm.predict_proba(values[valid].reshape(-1, 1))
    mask[valid] = probs[:, pos_idx] > 0.5

    n_positive = np.sum(mask)
    n_valid = np.sum(valid)

    stats = {
        'method': 'gmm',
        'threshold': threshold,
        'n_positive': n_positive,
        'n_valid': n_valid,
        'pct_positive': 100 * n_positive / n_valid if n_valid > 0 else 0,
        'means': means.tolist(),
        'stds': stds.tolist(),
        'weights': weights.tolist(),
        'pos_component': int(pos_idx),
    }

    return mask, threshold, stats


def classify_gene_expression(
    values: np.ndarray,
    clf_config: ClassificationConfig = None,
    gene_name: str = None
) -> Tuple[np.ndarray, float, dict]:
    """
    Unified gene classification function with configurable pipeline.

    Applies the full classification pipeline:
    1. Optional clipping (negative values -> 0)
    2. Optional normalization (log1p, robust_iqr)
    3. Classification (mode_shift, gmm) with per-gene sigma support

    Args:
        values: Raw expression values (can include negatives from neuropil subtraction)
        clf_config: ClassificationConfig specifying the method
        gene_name: Optional gene name for noisy_genes lookup (matches by name,
                   so re-stained genes in multiple rounds will all use noisy_sigma)

    Returns:
        Tuple of (positive_mask, threshold, stats_dict)
    """
    if clf_config is None:
        clf_config = ClassificationConfig()

    # Determine sigma based on whether gene is in noisy_genes list
    # Matches by gene name only, so same gene in multiple rounds all get same treatment
    is_noisy = gene_name is not None and gene_name in clf_config.noisy_genes
    sigma = clf_config.noisy_sigma if is_noisy else clf_config.base_sigma

    stats = {'config': {
        'normalization': clf_config.normalization,
        'classifier': clf_config.classifier,
        'clip_negative': clf_config.clip_negative,
        'base_sigma': clf_config.base_sigma,
        'noisy_sigma': clf_config.noisy_sigma,
        'noisy_genes': clf_config.noisy_genes,
    }}

    # Step 1: Optional clipping
    working_values = values.copy()
    if clf_config.clip_negative:
        working_values = np.where(working_values < 0, 0, working_values)
        stats['n_clipped'] = np.sum(values < 0)

    # Step 2: Optional normalization
    if clf_config.normalization == 'log1p':
        working_values, norm_stats = normalize_log1p(working_values)
        stats.update(norm_stats)
    elif clf_config.normalization == 'robust_iqr':
        working_values, norm_stats = normalize_robust_iqr(working_values)
        stats.update(norm_stats)
    # else: 'none' - use raw values

    # Step 3: Classification
    if clf_config.classifier == 'gmm':
        mask, threshold, clf_stats = classify_gmm(working_values)
    else:  # 'mode_shift' (default)
        mask, threshold, clf_stats = classify_gene_mode_shift(working_values, sigma)

    stats.update(clf_stats)
    stats['sigma_used'] = sigma
    stats['is_noisy_gene'] = is_noisy
    stats['method'] = f"{clf_config.normalization}|{clf_config.classifier}"
    if clf_config.clip_negative:
        stats['method'] = f"clipped|{stats['method']}"

    return mask, threshold, stats


def compute_gene_thresholds(
    df: pd.DataFrame,
    genes: List[str],
    rounds: List[dict],
    config: AnalysisConfig,
    custom_thresholds: Dict[str, float] = None,
    method: str = 'mode_shift',
    clf_config: ClassificationConfig = None
) -> Dict[str, dict]:
    """
    Compute thresholds for all genes WITHOUT classifying.

    Use this to preview thresholds before running classification.
    Allows custom threshold overrides for specific genes.

    Args:
        df: DataFrame with gene expression columns
        genes: List of gene names
        rounds: List of round info dicts
        config: AnalysisConfig with normal_sigma, noisy_sigma, noisy_genes
        custom_thresholds: Optional dict mapping gene names to custom thresholds
        method: Classification method - 'mode_shift' or 'robust_adaptive' (legacy)
        clf_config: ClassificationConfig for full control over classification pipeline.
                   If provided, takes precedence over 'method' parameter.

    Returns:
        Dict mapping gene name to stats dict (mode, threshold, noise_std, etc.)
    """
    if custom_thresholds is None:
        custom_thresholds = {}

    stats_dict = {}

    for round_info in rounds:
        round_num = round_info['round']
        channels = [c for c in round_info['channels'] if c not in EXCLUDE_GENES]

        for gene in channels:
            if gene not in genes:
                continue

            try:
                col = get_gene_column(df, gene, round_num)
            except KeyError:
                continue

            values = df[col].values

            # Compute stats based on configuration
            if clf_config is not None:
                # Use new unified classification (pass gene name for noisy_genes lookup)
                _, auto_threshold, stats = classify_gene_expression(values, clf_config, gene_name=gene)
            else:
                # Legacy: Use standard mode_shift with sigma from config
                sigma = config.noisy_sigma if gene in config.noisy_genes else config.normal_sigma
                _, auto_threshold, stats = classify_gene_mode_shift(values, sigma)
                stats['sigma_used'] = sigma

            # Use custom threshold if provided
            if gene in custom_thresholds:
                threshold = custom_thresholds[gene]
                # Recompute counts with custom threshold
                clean_vals = values[~np.isnan(values)]
                n_positive = np.sum(clean_vals > threshold)
                n_valid = len(clean_vals)
                stats['threshold'] = threshold
                stats['n_positive'] = n_positive
                stats['pct_positive'] = 100 * n_positive / n_valid if n_valid > 0 else 0
                stats['custom_threshold'] = True
                stats['auto_threshold'] = auto_threshold
            else:
                stats['custom_threshold'] = False

            stats['round'] = round_num
            stats_dict[gene] = stats

    return stats_dict


def classify_all_genes(
    df: pd.DataFrame,
    genes: List[str],
    rounds: List[dict],
    config: AnalysisConfig,
    custom_thresholds: Dict[str, float] = None,
    method: str = 'mode_shift',
    clf_config: ClassificationConfig = None
) -> pd.DataFrame:
    """
    Classify all genes across all rounds.

    Args:
        df: DataFrame with gene expression columns
        genes: List of gene names
        rounds: List of round info dicts
        config: AnalysisConfig with normal_sigma, noisy_sigma, noisy_genes
        custom_thresholds: Optional dict mapping gene names to custom thresholds.
                          Overrides auto-computed thresholds.
        method: Classification method - 'mode_shift' or 'robust_adaptive' (legacy)
        clf_config: ClassificationConfig for full control over classification pipeline.
                   If provided, takes precedence over 'method' parameter.

    Returns:
        DataFrame with new '{gene}_positive' columns for each gene
    """
    if custom_thresholds is None:
        custom_thresholds = {}

    result_df = df.copy()
    classification_stats = {}

    for round_info in rounds:
        round_num = round_info['round']
        channels = [c for c in round_info['channels'] if c not in EXCLUDE_GENES]

        for gene in channels:
            if gene not in genes:
                continue

            try:
                col = get_gene_column(df, gene, round_num)
            except KeyError:
                continue

            values = df[col].values

            # Classify based on configuration
            if clf_config is not None:
                # Use new unified classification (pass gene name for noisy_genes lookup)
                positive_mask, auto_threshold, stats = classify_gene_expression(values, clf_config, gene_name=gene)
                sigma = stats.get('sigma_used', clf_config.base_sigma)
            else:
                # Legacy: Use standard mode_shift with sigma from config
                sigma = config.noisy_sigma if gene in config.noisy_genes else config.normal_sigma
                positive_mask, auto_threshold, stats = classify_gene_mode_shift(values, sigma)

            # Override with custom threshold if provided
            if gene in custom_thresholds:
                threshold = custom_thresholds[gene]
                # Recompute mask with custom threshold
                positive_mask = np.zeros(len(values), dtype=bool)
                non_nan_mask = ~np.isnan(values)
                positive_mask[non_nan_mask] = values[non_nan_mask] > threshold
                stats['threshold'] = threshold
                stats['n_positive'] = np.sum(positive_mask)
                stats['pct_positive'] = 100 * stats['n_positive'] / stats['n_valid'] if stats['n_valid'] > 0 else 0
                stats['custom_threshold'] = True
                stats['auto_threshold'] = auto_threshold
            else:
                stats['custom_threshold'] = False

            # Store result
            pos_col = f'{gene}_positive'
            result_df[pos_col] = positive_mask

            # Determine method string for display
            if clf_config is not None:
                method_str = stats.get('method', str(clf_config))
            else:
                method_str = method

            classification_stats[gene] = {
                **stats,
                'round': round_num,
                'sigma_used': sigma,
            }

    # Print summary
    rprint("\n  Gene classification summary:")
    for gene, stats in classification_stats.items():
        custom_marker = " (custom)" if stats.get('custom_threshold') else ""
        method_info = f" [{stats.get('method', 'mode_shift')}"
        if stats.get('sigma_used'):
            method_info += f", σ={stats['sigma_used']:.1f}"
        if stats.get('is_noisy_gene'):
            method_info += ", noisy"
        method_info += "]"
        rprint(f"    {gene}: {stats['n_positive']} positive ({stats['pct_positive']:.1f}%), "
               f"threshold={stats['threshold']:.2f}{custom_marker}{method_info}")

    # Store stats as attribute
    result_df.attrs['classification_stats'] = classification_stats

    return result_df


# =============================================================================
# OVERLAP ANALYSIS
# =============================================================================

def align_masks_to_hcr_table(
    responses: dict,
    plane_data: Dict[int, dict],
    hcr_df: pd.DataFrame,
    all_responses: Dict[int, dict] = None,
    all_plane_results: Dict[int, dict] = None
) -> dict:
    """
    Align 2P-based response masks to HCR table indices.

    Maps from 2P cell indices to HCR table rows using twoP_mask and plane columns.
    The HCR table has one row per HCR cell, with twoP_mask indicating the matched
    2P cell ID (or NaN if no match).

    Args:
        responses: Dict with combined excited_mask, inhibited_mask (concatenated across planes)
                   OR can be empty if all_responses is provided
        plane_data: Dict from load_2p_spatial_data (maps plane -> {cell_ids, ...})
        hcr_df: DataFrame with 'twoP_mask' and 'best_plane' (or 'plane') columns
        all_responses: Optional dict mapping plane -> response dict with per-plane masks.
                       If provided, uses this instead of combined responses for more
                       accurate per-plane indexing.
        all_plane_results: Optional dict mapping plane -> {cell_ids, ...} from artifact-filtered
                          response analysis. If provided with all_responses, uses these cell_ids
                          for indexing (critical when artifact filtering was applied).

    Returns:
        dict with 'excited_aligned', 'inhibited_aligned' arrays matching HCR table
    """
    n_rows = len(hcr_df)
    excited_aligned = np.zeros(n_rows, dtype=bool)
    inhibited_aligned = np.zeros(n_rows, dtype=bool)

    # Find the actual column names
    twop_mask_col = get_col(hcr_df, 'twoP_mask')
    plane_col = get_col(hcr_df, 'best_plane')
    if plane_col not in hcr_df.columns:
        plane_col = get_col(hcr_df, 'plane')

    if twop_mask_col not in hcr_df.columns:
        rprint(f"[yellow]Warning: twoP_mask column not found, cannot align responses[/yellow]")
        return {'excited_aligned': excited_aligned, 'inhibited_aligned': inhibited_aligned}

    if plane_col not in hcr_df.columns:
        rprint(f"[yellow]Warning: plane/best_plane column not found, cannot align responses[/yellow]")
        return {'excited_aligned': excited_aligned, 'inhibited_aligned': inhibited_aligned}

    # Strategy: Use per-plane responses if available, otherwise use combined
    if all_responses is not None and len(all_responses) > 0:
        # Per-plane alignment (more accurate)
        # Build lookup: (plane, cell_id) -> response index within that plane
        # CRITICAL: If all_plane_results is provided, use its cell_ids because
        # all_responses masks are indexed relative to the filtered cell list,
        # not the full plane_data cell list
        plane_cell_to_idx = {}
        if all_plane_results is not None:
            # Use filtered cell IDs from response analysis (correct indexing)
            for plane, presult in all_plane_results.items():
                cell_ids = presult.get('cell_ids', [])
                for idx, cell_id in enumerate(cell_ids):
                    plane_cell_to_idx[(plane, cell_id)] = idx
        else:
            # Fallback to plane_data (assumes no artifact filtering)
            for plane, pdata in plane_data.items():
                cell_ids = pdata.get('cell_ids', [])
                for idx, cell_id in enumerate(cell_ids):
                    plane_cell_to_idx[(plane, cell_id)] = idx

        # Count HCR rows with 2P matches
        has_2p_mask = hcr_df[twop_mask_col].notna()
        n_with_2p = has_2p_mask.sum()

        n_matched = 0
        n_excited = 0
        n_inhibited = 0

        # Determine cell_id format from plane_data
        # Check if cell_ids are strings like 'cell_1' or integers/other
        sample_cell_id = list(plane_cell_to_idx.keys())[0][1] if plane_cell_to_idx else None
        uses_cell_prefix = isinstance(sample_cell_id, str) and sample_cell_id.startswith('cell_')

        for hcr_idx, row in hcr_df.iterrows():
            twop_mask = row[twop_mask_col]
            plane = row[plane_col]

            if pd.isna(twop_mask) or pd.isna(plane):
                continue

            plane = int(plane)

            # Convert twoP_mask to match plane_data cell_id format
            if uses_cell_prefix:
                # twoP_mask is numeric (e.g., 486.0), convert to 'cell_486'
                cell_id = f'cell_{int(twop_mask)}'
            else:
                # Keep as-is or convert to int
                cell_id = int(twop_mask) if isinstance(twop_mask, float) else twop_mask

            key = (plane, cell_id)

            if key in plane_cell_to_idx and plane in all_responses:
                cell_idx = plane_cell_to_idx[key]
                plane_resp = all_responses[plane]

                if cell_idx < len(plane_resp.get('excited_mask', [])):
                    # Get the row's position in the DataFrame index
                    # hcr_idx is the original index, we need position
                    pos = hcr_df.index.get_loc(hcr_idx)
                    excited_aligned[pos] = plane_resp['excited_mask'][cell_idx]
                    inhibited_aligned[pos] = plane_resp['inhibited_mask'][cell_idx]
                    n_matched += 1
                    if plane_resp['excited_mask'][cell_idx]:
                        n_excited += 1
                    if plane_resp['inhibited_mask'][cell_idx]:
                        n_inhibited += 1

        rprint(f"  Aligned {n_matched}/{n_with_2p} HCR cells with 2P matches to responses")
        rprint(f"    Excited: {n_excited}, Inhibited: {n_inhibited}")

    elif 'excited_mask' in responses and len(responses['excited_mask']) > 0:
        # Fallback: combined responses (assumes plane-sorted order)
        # Build cumulative offset for each plane
        planes_sorted = sorted(plane_data.keys())
        plane_offsets = {}
        offset = 0
        for plane in planes_sorted:
            plane_offsets[plane] = offset
            offset += len(plane_data[plane].get('cell_ids', []))

        # Build lookup: (plane, cell_id) -> combined index
        combined_idx_lookup = {}
        for plane in planes_sorted:
            cell_ids = plane_data[plane].get('cell_ids', [])
            base_offset = plane_offsets[plane]
            for local_idx, cell_id in enumerate(cell_ids):
                combined_idx_lookup[(plane, cell_id)] = base_offset + local_idx

        n_matched = 0
        n_excited = 0
        n_inhibited = 0
        combined_excited = np.array(responses['excited_mask'])
        combined_inhibited = np.array(responses['inhibited_mask'])

        # Determine cell_id format from plane_data
        sample_cell_id = list(combined_idx_lookup.keys())[0][1] if combined_idx_lookup else None
        uses_cell_prefix = isinstance(sample_cell_id, str) and sample_cell_id.startswith('cell_')

        for hcr_idx, row in hcr_df.iterrows():
            twop_mask = row[twop_mask_col]
            plane = row[plane_col]

            if pd.isna(twop_mask) or pd.isna(plane):
                continue

            plane = int(plane)

            # Convert twoP_mask to match plane_data cell_id format
            if uses_cell_prefix:
                cell_id = f'cell_{int(twop_mask)}'
            else:
                cell_id = int(twop_mask) if isinstance(twop_mask, float) else twop_mask

            key = (plane, cell_id)

            if key in combined_idx_lookup:
                combined_idx = combined_idx_lookup[key]
                if combined_idx < len(combined_excited):
                    pos = hcr_df.index.get_loc(hcr_idx)
                    excited_aligned[pos] = combined_excited[combined_idx]
                    inhibited_aligned[pos] = combined_inhibited[combined_idx]
                    n_matched += 1
                    if combined_excited[combined_idx]:
                        n_excited += 1
                    if combined_inhibited[combined_idx]:
                        n_inhibited += 1

        rprint(f"  Aligned {n_matched} HCR cells to 2P responses (combined mode)")
        rprint(f"    Excited: {n_excited}, Inhibited: {n_inhibited}")
    else:
        rprint(f"[yellow]Warning: No response masks provided[/yellow]")

    return {
        'excited_aligned': excited_aligned,
        'inhibited_aligned': inhibited_aligned,
    }


def compute_overlaps(
    classified_df: pd.DataFrame,
    aligned_masks: dict,
    genes: List[str] = None
) -> pd.DataFrame:
    """
    Compute overlap between gene+ neurons and excited/inhibited neurons.

    IMPORTANT: This function computes statistics ONLY for 2P-matched neurons
    (where excited/inhibited classification is available). Gene+ counts are
    restricted to this subset for accurate percentage calculations.

    Args:
        classified_df: DataFrame with '{gene}_positive' columns
        aligned_masks: Dict with 'excited_aligned', 'inhibited_aligned' arrays
        genes: List of genes to analyze (default: all with '_positive' suffix)

    Returns:
        DataFrame with overlap statistics per gene:
        - n_positive_2p: Gene+ cells in 2P-matched subset (used for % calculations)
        - n_positive_total: Gene+ cells in full volume (for reference)
        - n_excited, n_inhibited: Total excited/inhibited cells
        - n_pos_excited, n_pos_inhibited: Overlap counts
        - pct_pos_in_excited: % of 2P-matched gene+ that are excited
        - pct_pos_in_inhibited: % of 2P-matched gene+ that are inhibited
        - pct_excited_in_pos: % of excited that are gene+
        - pct_inhibited_in_pos: % of inhibited that are gene+
    """
    excited = aligned_masks['excited_aligned']
    inhibited = aligned_masks['inhibited_aligned']

    # Find all classified genes
    if genes is None:
        genes = [col.replace('_positive', '') for col in classified_df.columns
                 if col.endswith('_positive')]

    results = []

    n_excited_total = np.sum(excited)
    n_inhibited_total = np.sum(inhibited)

    # Create mask for 2P-matched neurons (where excited/inhibited are defined)
    # These are neurons where the excited/inhibited masks are meaningful
    has_2p_match = (excited | inhibited | ~excited)  # All neurons in aligned_masks
    # Actually, we need neurons that COULD be excited/inhibited (i.e., have 2P data)
    # The aligned_masks arrays are already restricted to 2P-matched cells,
    # but classified_df may include non-matched cells with NaN/False for excited
    # We identify 2P-matched cells as those where has_2p_match column is True
    if 'has_2p_match' in classified_df.columns:
        twop_matched = classified_df['has_2p_match'].fillna(False).astype(bool).values
    else:
        # Fallback: assume all cells in excited/inhibited arrays are 2P-matched
        twop_matched = np.ones(len(classified_df), dtype=bool)

    # Ensure arrays are same length
    min_len = min(len(twop_matched), len(excited), len(inhibited))
    twop_matched = twop_matched[:min_len]
    exc = excited[:min_len]
    inh = inhibited[:min_len]

    for gene in genes:
        pos_col = f'{gene}_positive'
        if pos_col not in classified_df.columns:
            continue

        positive = classified_df[pos_col].values[:min_len]

        # Total gene+ in full volume
        n_positive_total = np.sum(positive)

        # Gene+ in 2P-matched subset only (for accurate % calculations)
        positive_2p = positive & twop_matched
        n_positive_2p = np.sum(positive_2p)

        # Overlaps (by definition, excited/inhibited are in 2P-matched subset)
        n_pos_excited = np.sum(positive & exc)
        n_pos_inhibited = np.sum(positive & inh)

        # Percentages - use 2P-matched gene+ count for "% of gene+ that are excited"
        pct_pos_in_excited = 100 * n_pos_excited / n_positive_2p if n_positive_2p > 0 else 0
        pct_pos_in_inhibited = 100 * n_pos_inhibited / n_positive_2p if n_positive_2p > 0 else 0

        # % of excited/inhibited neurons that are gene+
        pct_excited_in_pos = 100 * n_pos_excited / n_excited_total if n_excited_total > 0 else 0
        pct_inhibited_in_pos = 100 * n_pos_inhibited / n_inhibited_total if n_inhibited_total > 0 else 0

        results.append({
            'gene': gene,
            'n_positive_2p': n_positive_2p,  # Gene+ in 2P-matched subset
            'n_positive_total': n_positive_total,  # Gene+ in full volume
            'n_positive': n_positive_2p,  # For backward compatibility (use 2P subset)
            'n_excited': n_excited_total,
            'n_inhibited': n_inhibited_total,
            'n_pos_excited': n_pos_excited,
            'n_pos_inhibited': n_pos_inhibited,
            'n_twop_matched': np.sum(twop_matched),  # Total 2P-matched neurons
            'pct_pos_in_excited': pct_pos_in_excited,
            'pct_pos_in_inhibited': pct_pos_in_inhibited,
            'pct_excited_in_pos': pct_excited_in_pos,
            'pct_inhibited_in_pos': pct_inhibited_in_pos,
        })

    overlap_df = pd.DataFrame(results)

    # Print summary
    if len(overlap_df) > 0:
        n_2p = overlap_df['n_twop_matched'].iloc[0] if len(overlap_df) > 0 else 0
        rprint(f"\n  Overlap summary ({n_excited_total} excited, {n_inhibited_total} inhibited, {n_2p} 2P-matched):")
        for _, row in overlap_df.iterrows():
            rprint(f"    {row['gene']}: {row['n_positive_2p']}/{row['n_positive_total']} positive (2P/total), "
                   f"{row['n_pos_excited']} excited ({row['pct_pos_in_excited']:.1f}%), "
                   f"{row['n_pos_inhibited']} inhibited ({row['pct_pos_in_inhibited']:.1f}%)")

    return overlap_df


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_dff_heatmap(
    dff: np.ndarray,
    framerate: float,
    pre_stim: float,
    title: str = None,
    cmap: str = 'RdBu_r',
    vmin: float = None,
    vmax: float = None,
    ax: plt.Axes = None
) -> plt.Axes:
    """
    Plot DF/F heatmap with stimulus onset line.

    Args:
        dff: (n_neurons, n_frames) array of DF/F values
        framerate: Imaging framerate in Hz
        pre_stim: Pre-stimulus time in seconds
        title: Plot title
        cmap: Colormap name
        vmin, vmax: Color scale limits
        ax: Matplotlib axes (creates new figure if None)

    Returns:
        Matplotlib Axes
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    n_neurons, n_frames = dff.shape

    # Time axis
    time = np.arange(n_frames) / framerate - pre_stim

    # Auto-scale if not provided
    if vmin is None or vmax is None:
        vmax = np.percentile(np.abs(dff), 98)
        vmin = -vmax

    # Plot
    im = ax.imshow(dff, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[time[0], time[-1], n_neurons, 0])

    # Stimulus onset line
    ax.axvline(0, color='black', linestyle='--', linewidth=1)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neuron')

    if title:
        ax.set_title(title)

    plt.colorbar(im, ax=ax, label='DF/F')

    return ax


def plot_spatial_response(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    cmap: str = 'RdBu_r',
    title: str = None,
    vmin: float = None,
    vmax: float = None,
    size: float = 20,
    ax: plt.Axes = None
) -> plt.Axes:
    """
    Plot spatial distribution of response values.

    Args:
        x, y: Coordinate arrays
        values: Values to plot (e.g., response magnitude)
        cmap: Colormap name
        title: Plot title
        vmin, vmax: Color scale limits
        size: Marker size
        ax: Matplotlib axes

    Returns:
        Matplotlib Axes
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Auto-scale
    if vmin is None or vmax is None:
        vmax = np.percentile(np.abs(values), 98)
        vmin = -vmax

    scatter = ax.scatter(x, y, c=values, cmap=cmap, s=size, vmin=vmin, vmax=vmax)

    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_aspect('equal')

    if title:
        ax.set_title(title)

    plt.colorbar(scatter, ax=ax)

    return ax


def plot_gene_histogram(
    values: np.ndarray,
    threshold: float,
    gene_name: str,
    n_bins: int = 50,
    ax: plt.Axes = None
) -> plt.Axes:
    """
    Plot gene expression histogram with threshold.

    Args:
        values: Expression values
        threshold: Classification threshold
        gene_name: Gene name for title
        n_bins: Number of histogram bins
        ax: Matplotlib axes

    Returns:
        Matplotlib Axes
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Remove NaN
    valid_values = values[~np.isnan(values)]

    # Plot histogram
    ax.hist(valid_values, bins=n_bins, alpha=0.7, edgecolor='black')

    # Threshold line
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2,
               label=f'Threshold: {threshold:.2f}')

    # Count positive
    n_positive = np.sum(valid_values > threshold)
    pct_positive = 100 * n_positive / len(valid_values)

    ax.set_xlabel('Expression')
    ax.set_ylabel('Count')
    ax.set_title(f'{gene_name}: {n_positive} positive ({pct_positive:.1f}%)')
    ax.legend()

    return ax


def plot_gene_cdf_by_response(
    values: np.ndarray,
    excited_mask: np.ndarray,
    inhibited_mask: np.ndarray,
    gene_name: str,
    ax: plt.Axes = None,
    p_values: np.ndarray = None,
    p_cutoffs: Tuple[float, ...] = (0.1, 0.05, 0.01),
    min_neurons_per_category: int = 10,
    clip_percentile: float = 99.0,
    use_log_scale: bool = False
) -> plt.Axes:
    """
    Plot CDF of gene expression split by response type.

    Grey line shows NON-SIGNIFICANT neurons (neither excited nor inhibited).
    If p_values provided, shows excited/inhibited split by p-value ranges
    with overlapping sets (e.g., p<0.05 includes all neurons with p<0.05,
    including those with p<0.01).

    Args:
        values: Expression values
        excited_mask: Boolean mask of excited neurons (p < most lenient cutoff)
        inhibited_mask: Boolean mask of inhibited neurons (p < most lenient cutoff)
        gene_name: Gene name for title
        ax: Matplotlib axes
        p_values: Per-neuron p-values (required for p-value stratification)
        p_cutoffs: Tuple of p-value cutoffs (e.g., (0.1, 0.05, 0.01))
        min_neurons_per_category: Minimum neurons to plot a category
        clip_percentile: Clip x-axis to this percentile of data (default 99)
        use_log_scale: Use log scale for x-axis (useful for skewed data)

    Returns:
        Matplotlib Axes
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    def plot_cdf(data, label, color, linewidth=2, alpha=1.0, linestyle='-'):
        clean_data = data[~np.isnan(data)]
        if len(clean_data) == 0:
            return 0
        sorted_data = np.sort(clean_data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax.plot(sorted_data, cdf, label=label, color=color, linewidth=linewidth,
                alpha=alpha, linestyle=linestyle)
        return len(clean_data)

    # Non-significant neurons (neither excited nor inhibited)
    non_sig_mask = ~excited_mask & ~inhibited_mask
    n_non_sig = np.sum(non_sig_mask & ~np.isnan(values))
    if n_non_sig > 0:
        plot_cdf(values[non_sig_mask], f'Non-sig (n={n_non_sig})', 'gray')

    # Color palettes for p-value stratification (darker = more significant)
    red_shades = ['#FFAAAA', '#FF6666', '#CC0000']  # light to dark red
    blue_shades = ['#AAAAFF', '#6666FF', '#0000CC']  # light to dark blue

    # Sort p_cutoffs from most lenient to most stringent
    sorted_cutoffs = sorted(p_cutoffs, reverse=True)

    if p_values is not None and len(p_values) == len(values):
        # Plot excited neurons by p-value (overlapping sets)
        for i, p_cut in enumerate(sorted_cutoffs):
            # Excited at this p-value (includes all more significant too)
            excited_at_p = excited_mask & (p_values < p_cut)
            n_exc = np.sum(excited_at_p & ~np.isnan(values))

            if n_exc >= min_neurons_per_category:
                color = red_shades[min(i, len(red_shades)-1)]
                plot_cdf(values[excited_at_p], f'Exc p<{p_cut} (n={n_exc})', color)

        # Plot inhibited neurons by p-value (overlapping sets)
        for i, p_cut in enumerate(sorted_cutoffs):
            inhibited_at_p = inhibited_mask & (p_values < p_cut)
            n_inh = np.sum(inhibited_at_p & ~np.isnan(values))

            if n_inh >= min_neurons_per_category:
                color = blue_shades[min(i, len(blue_shades)-1)]
                plot_cdf(values[inhibited_at_p], f'Inh p<{p_cut} (n={n_inh})', color)
    else:
        # No p-values provided - use simple excited/inhibited masks
        n_excited = np.sum(excited_mask & ~np.isnan(values))
        if n_excited >= min_neurons_per_category:
            plot_cdf(values[excited_mask], f'Excited (n={n_excited})', 'red')

        n_inhibited = np.sum(inhibited_mask & ~np.isnan(values))
        if n_inhibited >= min_neurons_per_category:
            plot_cdf(values[inhibited_mask], f'Inhibited (n={n_inhibited})', 'blue')

    # Clip x-axis to percentile to handle outliers/skewed data
    clean_values = values[~np.isnan(values)]
    if len(clean_values) > 0 and clip_percentile < 100:
        x_max = np.percentile(clean_values, clip_percentile)
        x_min = np.percentile(clean_values, 100 - clip_percentile) if (100 - clip_percentile) > 0 else clean_values.min()
        ax.set_xlim(x_min, x_max)

    if use_log_scale:
        ax.set_xscale('log')
        ax.set_xlabel('Expression (log scale)')
    else:
        ax.set_xlabel('Expression')

    ax.set_ylabel('Cumulative probability')
    ax.set_title(f'{gene_name} expression by response type')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)

    return ax


def plot_gene_histogram_by_response(
    values: np.ndarray,
    excited_mask: np.ndarray,
    inhibited_mask: np.ndarray,
    gene_name: str,
    ax: plt.Axes = None,
    p_values: np.ndarray = None,
    p_cutoffs: Tuple[float, ...] = (0.1, 0.05, 0.01),
    min_neurons_per_category: int = 10,
    n_bins: int = 50,
    threshold: float = None
) -> plt.Axes:
    """
    Plot histogram of gene expression with response categories overlaid.

    Shows full population as grey filled histogram, with excited/inhibited neurons
    overlaid as step outlines (probability-normalized) in red/blue shades by p-value.

    Args:
        values: Expression values for all neurons
        excited_mask: Boolean mask of excited neurons (at most lenient p threshold)
        inhibited_mask: Boolean mask of inhibited neurons (at most lenient p threshold)
        gene_name: Gene name for title
        ax: Matplotlib axes
        p_values: Per-neuron p-values (required for p-value stratification)
        p_cutoffs: Tuple of p-value cutoffs (e.g., (0.1, 0.05, 0.01))
        min_neurons_per_category: Minimum neurons to plot a category (default 10)
        n_bins: Number of histogram bins
        threshold: Optional gene-positive threshold to show as vertical line

    Returns:
        Matplotlib Axes
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Clean data
    valid_mask = ~np.isnan(values)
    clean_values = values[valid_mask]

    if len(clean_values) == 0:
        ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(gene_name)
        return ax

    # Compute histogram range (clip to 1-99 percentile for visualization)
    vmin, vmax = np.percentile(clean_values, [1, 99])
    bins = np.linspace(vmin, vmax, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Plot full population histogram (grey filled, normalized to probability)
    counts_all, _ = np.histogram(clean_values, bins=bins)
    prob_all = counts_all / counts_all.sum()
    ax.fill_between(bin_centers, 0, prob_all, alpha=0.3, color='grey', step='mid',
                    label=f'All (n={len(clean_values)})')
    ax.step(bin_centers, prob_all, where='mid', color='grey', linewidth=1, alpha=0.7)

    # Color palettes for p-value stratification (darker = more significant)
    red_shades = ['#FFAAAA', '#FF6666', '#CC0000']  # light to dark red
    blue_shades = ['#AAAAFF', '#6666FF', '#0000CC']  # light to dark blue

    # Sort p_cutoffs from most lenient to most stringent
    sorted_cutoffs = sorted(p_cutoffs, reverse=True)

    # Align masks to valid_mask
    excited_valid = excited_mask[valid_mask] if len(excited_mask) == len(values) else excited_mask
    inhibited_valid = inhibited_mask[valid_mask] if len(inhibited_mask) == len(values) else inhibited_mask
    p_valid = p_values[valid_mask] if p_values is not None and len(p_values) == len(values) else p_values

    if p_valid is not None and len(p_valid) == len(clean_values):
        # Plot excited neurons by p-value as step outlines
        for i, p_cut in enumerate(sorted_cutoffs):
            excited_at_p = excited_valid & (p_valid < p_cut)
            n_exc = np.sum(excited_at_p)

            if n_exc >= min_neurons_per_category:
                color = red_shades[min(i, len(red_shades)-1)]
                counts_exc, _ = np.histogram(clean_values[excited_at_p], bins=bins)
                prob_exc = counts_exc / counts_exc.sum() if counts_exc.sum() > 0 else counts_exc
                linewidth = 2.5 - i * 0.5  # thicker for more significant
                ax.step(bin_centers, prob_exc, where='mid', color=color,
                        linewidth=linewidth, label=f'Exc p<{p_cut} (n={n_exc})')

        # Plot inhibited neurons by p-value as step outlines
        for i, p_cut in enumerate(sorted_cutoffs):
            inhibited_at_p = inhibited_valid & (p_valid < p_cut)
            n_inh = np.sum(inhibited_at_p)

            if n_inh >= min_neurons_per_category:
                color = blue_shades[min(i, len(blue_shades)-1)]
                counts_inh, _ = np.histogram(clean_values[inhibited_at_p], bins=bins)
                prob_inh = counts_inh / counts_inh.sum() if counts_inh.sum() > 0 else counts_inh
                linewidth = 2.5 - i * 0.5
                ax.step(bin_centers, prob_inh, where='mid', color=color,
                        linewidth=linewidth, label=f'Inh p<{p_cut} (n={n_inh})')
    else:
        # No p-values - use simple masks
        n_excited = np.sum(excited_valid)
        if n_excited >= min_neurons_per_category:
            counts_exc, _ = np.histogram(clean_values[excited_valid], bins=bins)
            prob_exc = counts_exc / counts_exc.sum() if counts_exc.sum() > 0 else counts_exc
            ax.step(bin_centers, prob_exc, where='mid', color='red',
                    linewidth=2, label=f'Excited (n={n_excited})')

        n_inhibited = np.sum(inhibited_valid)
        if n_inhibited >= min_neurons_per_category:
            counts_inh, _ = np.histogram(clean_values[inhibited_valid], bins=bins)
            prob_inh = counts_inh / counts_inh.sum() if counts_inh.sum() > 0 else counts_inh
            ax.step(bin_centers, prob_inh, where='mid', color='blue',
                    linewidth=2, label=f'Inhibited (n={n_inhibited})')

    # Add threshold line if provided
    if threshold is not None and vmin <= threshold <= vmax:
        ax.axvline(threshold, color='green', linestyle='--', linewidth=2,
                   label=f'Thresh ({threshold:.2f})')

    ax.set_xlabel('Expression')
    ax.set_ylabel('Probability')
    ax.set_title(f'{gene_name}')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_xlim(vmin, vmax)

    return ax


def plot_gene_response_spatial(
    classified_df: pd.DataFrame,
    aligned_masks: dict,
    genes: List[str],
    x_col: str = 'hcr_x',
    y_col: str = 'hcr_y',
    n_cols: int = 4,
    figsize_per_gene: Tuple[float, float] = (4, 4),
    analysis_mask: np.ndarray = None,
    response_type: str = 'excited'
) -> plt.Figure:
    """
    Create spatial plots showing neurons colored by category for each gene.

    Categories (for excited):
    - Grey: Neither (not gene+, not responding)
    - Green: Responding only (responding but not gene+)
    - Magenta: Gene+ only (gene+ but not responding)
    - Yellow: Both (gene+ AND responding)

    For inhibited, Green becomes Blue.

    Args:
        classified_df: DataFrame with gene classification and coordinates
        aligned_masks: Dict with 'excited_aligned' and 'inhibited_aligned' arrays
        genes: List of gene names
        x_col, y_col: Coordinate column names
        n_cols: Number of columns in grid
        figsize_per_gene: Size per gene subplot
        analysis_mask: Optional boolean array indicating which cells were included in
                       the response analysis (2P-matched + non-artifact). If None,
                       uses has_2p_match column.
        response_type: 'excited' or 'inhibited'

    Returns:
        Matplotlib Figure
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")

    # Get response mask based on type
    mask_key = f'{response_type}_aligned'
    if mask_key not in aligned_masks:
        rprint(f"[yellow]Response type '{response_type}' not found in aligned_masks[/yellow]")
        return None
    response_mask = aligned_masks[mask_key]

    # Color scheme: excited=green, inhibited=blue for response-only cells
    response_color = 'green' if response_type == 'excited' else 'blue'

    # Auto-detect coordinate columns
    x_alternatives = [x_col, 'hcr_x', 'centroid_x_main', 'x_main', 'centroid_x', 'X']
    y_alternatives = [y_col, 'hcr_y', 'centroid_y_main', 'y_main', 'centroid_y', 'Y']

    actual_x_col = None
    actual_y_col = None
    for alt in x_alternatives:
        if alt in classified_df.columns:
            actual_x_col = alt
            break
    for alt in y_alternatives:
        if alt in classified_df.columns:
            actual_y_col = alt
            break

    if actual_x_col is None or actual_y_col is None:
        rprint("[yellow]Could not find coordinate columns for spatial plot[/yellow]")
        return None

    # Use provided analysis_mask or fall back to has_2p_match
    if analysis_mask is not None:
        valid_mask = analysis_mask
    elif 'has_2p_match' in classified_df.columns:
        valid_mask = classified_df['has_2p_match'].fillna(False).astype(bool).values
    else:
        valid_mask = np.ones(len(classified_df), dtype=bool)

    # Ensure same length
    min_len = min(len(valid_mask), len(response_mask))
    valid_mask = valid_mask[:min_len]
    resp = response_mask[:min_len]

    x_all = classified_df[actual_x_col].values[:min_len]
    y_all = classified_df[actual_y_col].values[:min_len]

    # Only plot cells that were in the analysis
    x = x_all[valid_mask]
    y = y_all[valid_mask]
    resp_2p = resp[valid_mask]

    n_rows = (len(genes) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(figsize_per_gene[0] * n_cols, figsize_per_gene[1] * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for i, gene in enumerate(genes):
        ax = axes[i]
        pos_col = f'{gene}_positive'

        if pos_col not in classified_df.columns:
            ax.text(0.5, 0.5, f'{gene}\nNot found', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title(gene)
            continue

        positive = classified_df[pos_col].values[:min_len][valid_mask]

        # Create category masks
        neither = ~positive & ~resp_2p
        resp_only = ~positive & resp_2p
        gene_only = positive & ~resp_2p
        both = positive & resp_2p

        # Labels based on response type
        resp_label = 'Exc' if response_type == 'excited' else 'Inh'

        # Plot in order: neither (background), then categories
        # Grey - neither
        if np.sum(neither) > 0:
            ax.scatter(x[neither], y[neither], c='lightgrey', s=8, alpha=0.3,
                      label=f'Neither ({np.sum(neither)})', rasterized=True)

        # Response only (green for excited, blue for inhibited)
        if np.sum(resp_only) > 0:
            ax.scatter(x[resp_only], y[resp_only], c=response_color, s=20, alpha=0.8,
                      label=f'{resp_label} only ({np.sum(resp_only)})', rasterized=True)

        # Magenta - gene+ only
        if np.sum(gene_only) > 0:
            ax.scatter(x[gene_only], y[gene_only], c='magenta', s=20, alpha=0.8,
                      label=f'{gene}+ only ({np.sum(gene_only)})', rasterized=True)

        # Yellow - both (on top)
        if np.sum(both) > 0:
            ax.scatter(x[both], y[both], c='gold', s=30, alpha=1.0,
                      edgecolors='black', linewidths=0.5,
                      label=f'Both ({np.sum(both)})', rasterized=True)

        ax.set_title(f'{gene}', fontsize=11, fontweight='bold')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.legend(fontsize=7, loc='upper right', framealpha=0.8)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    title_resp = 'Excited' if response_type == 'excited' else 'Inhibited'
    fig.suptitle(f'Spatial Distribution: Gene+ vs {title_resp}', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig


def plot_overlap_bars(
    overlap_df: pd.DataFrame,
    ax: plt.Axes = None
) -> plt.Figure:
    """
    Create 2x2 bar plots for gene-response overlaps.

    Layout:
    - Top row: % of Gene+ neurons that are Excited/Inhibited
    - Bottom row: % of Excited/Inhibited neurons that are Gene+

    Args:
        overlap_df: DataFrame from compute_overlaps
        ax: Ignored (creates its own figure)

    Returns:
        Matplotlib Figure
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")

    if len(overlap_df) == 0:
        rprint("[yellow]No overlap data to plot[/yellow]")
        return None

    # Sort genes alphabetically
    overlap_df = overlap_df.sort_values('gene')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    x = np.arange(len(overlap_df))
    genes = overlap_df['gene'].values

    # ========== TOP ROW: % of Gene+ that are Excited/Inhibited ==========

    # Plot 1: % of gene+ neurons that are excited
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x, overlap_df['pct_pos_in_excited'], color='green', alpha=0.7)
    ax1.set_ylabel('% of Gene+ Neurons', fontsize=11)
    ax1.set_title('% of Gene+ Neurons that are Excited', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(genes, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    max_val = overlap_df['pct_pos_in_excited'].max()
    ax1.set_ylim(0, max(max_val * 1.2, 10) if not np.isnan(max_val) else 10)

    # Add value labels
    for bar, val, n in zip(bars1, overlap_df['pct_pos_in_excited'], overlap_df['n_pos_excited']):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{val:.1f}%\n(n={int(n)})', ha='center', va='bottom', fontsize=8)

    # Plot 2: % of gene+ neurons that are inhibited
    ax2 = axes[0, 1]
    bars2 = ax2.bar(x, overlap_df['pct_pos_in_inhibited'], color='blue', alpha=0.7)
    ax2.set_ylabel('% of Gene+ Neurons', fontsize=11)
    ax2.set_title('% of Gene+ Neurons that are Inhibited', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(genes, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    max_val = overlap_df['pct_pos_in_inhibited'].max()
    ax2.set_ylim(0, max(max_val * 1.2, 10) if not np.isnan(max_val) else 10)

    for bar, val, n in zip(bars2, overlap_df['pct_pos_in_inhibited'], overlap_df['n_pos_inhibited']):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{val:.1f}%\n(n={int(n)})', ha='center', va='bottom', fontsize=8)

    # ========== BOTTOM ROW: % of Excited/Inhibited that are Gene+ ==========

    # Plot 3: % of excited neurons that are gene+
    ax3 = axes[1, 0]
    bars3 = ax3.bar(x, overlap_df['pct_excited_in_pos'], color='darkgreen', alpha=0.7)
    ax3.set_ylabel('% of Excited Neurons', fontsize=11)
    ax3.set_title('% of Excited Neurons that are Gene+', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(genes, rotation=45, ha='right')
    ax3.grid(axis='y', alpha=0.3)
    max_val = overlap_df['pct_excited_in_pos'].max()
    ax3.set_ylim(0, max(max_val * 1.2, 10) if not np.isnan(max_val) else 10)

    for bar, val, n in zip(bars3, overlap_df['pct_excited_in_pos'], overlap_df['n_pos_excited']):
        if val > 0:
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{val:.1f}%\n(n={int(n)})', ha='center', va='bottom', fontsize=8)

    # Plot 4: % of inhibited neurons that are gene+
    ax4 = axes[1, 1]
    bars4 = ax4.bar(x, overlap_df['pct_inhibited_in_pos'], color='darkblue', alpha=0.7)
    ax4.set_ylabel('% of Inhibited Neurons', fontsize=11)
    ax4.set_title('% of Inhibited Neurons that are Gene+', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(genes, rotation=45, ha='right')
    ax4.grid(axis='y', alpha=0.3)
    max_val = overlap_df['pct_inhibited_in_pos'].max()
    ax4.set_ylim(0, max(max_val * 1.2, 10) if not np.isnan(max_val) else 10)

    for bar, val, n in zip(bars4, overlap_df['pct_inhibited_in_pos'], overlap_df['n_pos_inhibited']):
        if val > 0:
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{val:.1f}%\n(n={int(n)})', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    return fig


def save_figure(
    fig: plt.Figure,
    output_path: Path,
    formats: List[str] = ['png', 'pdf'],
    dpi: int = 150
) -> List[Path]:
    """
    Save figure to multiple formats.

    Args:
        fig: Matplotlib figure
        output_path: Base path (without extension)
        formats: List of format extensions
        dpi: Resolution for raster formats

    Returns:
        List of saved file paths
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for fmt in formats:
        path = output_path.with_suffix(f'.{fmt}')
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        saved_paths.append(path)

    return saved_paths


# =============================================================================
# DIAGNOSTIC VISUALIZATION FUNCTIONS
# =============================================================================

def plot_2p_loading_diagnostic(
    plane_data: Dict[int, dict],
    plane: int,
    figsize: Tuple[int, int] = (14, 5),
    reference_plane: int = None
) -> plt.Figure:
    """
    Diagnostic plot for 2P data loading: mean image with mask overlays + activity heatmap.

    The pkl file contains:
    - 'mean_frames': (n_neurons, n_frames) fluorescence traces for each neuron
    - 'masks_locs': dict mapping cell_id to (y_indices, x_indices) pixel coordinates

    This function:
    - Left panel: Creates a mean image from traces and overlays cell masks
    - Right panel: Shows row-normalized activity heatmap

    Args:
        plane_data: Dict from load_2p_spatial_data containing mean_frames and masks_locs
        plane: Which plane to visualize
        figsize: Figure size
        reference_plane: If provided and matches plane, adds "(Reference)" to title

    Returns:
        Matplotlib Figure
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")

    if plane not in plane_data:
        raise ValueError(f"Plane {plane} not in plane_data")

    pdata = plane_data[plane]
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    is_reference = (reference_plane is not None and plane == reference_plane)
    ref_label = " (Reference)" if is_reference else ""

    # Get the traces data (n_neurons, n_frames)
    traces = pdata.get('mean_frames')
    masks_locs = pdata.get('masks_locs', {})
    n_cells = len(pdata.get('cell_ids', []))

    # ========== Left: Mean image with mask overlays ==========
    ax1 = axes[0]

    # Determine image dimensions from mask coordinates
    if masks_locs:
        max_y, max_x = 0, 0
        for coords in masks_locs.values():
            if coords is not None and len(coords) > 0:
                if isinstance(coords, tuple) and len(coords) == 2:
                    y_idx, x_idx = coords
                    max_y = max(max_y, np.max(y_idx) + 1)
                    max_x = max(max_x, np.max(x_idx) + 1)
                elif hasattr(coords, 'shape') and len(coords.shape) == 2:
                    max_y = max(max_y, int(np.max(coords[:, 0])) + 1)
                    max_x = max(max_x, int(np.max(coords[:, 1])) + 1)

        # Create background: mean activity per neuron mapped to mask pixels
        mean_image = np.zeros((max_y, max_x), dtype=np.float32)
        mask_image = np.zeros((max_y, max_x), dtype=np.float32)

        cell_ids = pdata.get('cell_ids', [])
        if traces is not None and len(traces) > 0:
            # Compute mean activity per neuron
            mean_activity = np.mean(traces, axis=1)
            # Normalize to 0-1 for display
            mean_activity_norm = (mean_activity - np.percentile(mean_activity, 5)) / \
                                 (np.percentile(mean_activity, 95) - np.percentile(mean_activity, 5) + 1e-10)
            mean_activity_norm = np.clip(mean_activity_norm, 0, 1)
        else:
            mean_activity_norm = np.ones(len(cell_ids))

        # Fill in the images
        for i, cell_id in enumerate(cell_ids):
            if cell_id in masks_locs:
                coords = masks_locs[cell_id]
                if coords is not None and len(coords) > 0:
                    if isinstance(coords, tuple) and len(coords) == 2:
                        y_idx, x_idx = coords
                    elif hasattr(coords, 'shape') and len(coords.shape) == 2:
                        y_idx = coords[:, 0].astype(int)
                        x_idx = coords[:, 1].astype(int)
                    else:
                        continue

                    # Clip to bounds
                    y_idx = np.clip(y_idx, 0, max_y - 1)
                    x_idx = np.clip(x_idx, 0, max_x - 1)

                    # Fill mean activity
                    if i < len(mean_activity_norm):
                        mean_image[y_idx, x_idx] = mean_activity_norm[i]
                    # Fill mask
                    mask_image[y_idx, x_idx] = 1

        # Display mean activity image
        ax1.imshow(mean_image, cmap='gray', aspect='equal', vmin=0, vmax=1)

        # Create colored mask overlay (only boundaries for cleaner look)
        from scipy import ndimage
        # Dilate and subtract to get boundaries
        mask_dilated = ndimage.binary_dilation(mask_image > 0, iterations=1)
        mask_boundary = mask_dilated.astype(float) - (mask_image > 0).astype(float)
        mask_boundary = np.clip(mask_boundary, 0, 1)

        # Overlay boundaries in red
        overlay_rgba = np.zeros((max_y, max_x, 4), dtype=np.float32)
        overlay_rgba[..., 0] = mask_boundary  # Red channel
        overlay_rgba[..., 3] = mask_boundary * 0.8  # Alpha
        ax1.imshow(overlay_rgba, aspect='equal')

        ax1.set_title(f'Plane {plane}{ref_label}: 2P Masks ({n_cells} cells)')
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
    else:
        ax1.text(0.5, 0.5, 'No mask data available',
                ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        ax1.set_title(f'Plane {plane}{ref_label}: 2P Masks (no data)')

    # ========== Right: Activity heatmap (row-normalized) ==========
    ax2 = axes[1]
    if traces is not None and len(traces) > 0 and traces.ndim == 2:
        # Subsample if too many neurons for visibility
        n_show = min(500, len(traces))
        idx = np.linspace(0, len(traces) - 1, n_show, dtype=int)
        traces_sub = traces[idx, :].astype(np.float64)

        # Row-normalize each neuron to 0-1 range
        row_min = traces_sub.min(axis=1, keepdims=True)
        row_max = traces_sub.max(axis=1, keepdims=True)
        row_range = row_max - row_min
        row_range[row_range == 0] = 1  # Avoid division by zero
        traces_normalized = (traces_sub - row_min) / row_range

        im2 = ax2.imshow(traces_normalized, aspect='auto', cmap='viridis',
                        vmin=0, vmax=1)
        ax2.set_xlabel('Frame')
        ax2.set_ylabel(f'Neuron (showing {n_show}/{len(traces)})')
        ax2.set_title(f'Plane {plane}{ref_label}: Activity (row-normalized)')
        cbar = plt.colorbar(im2, ax=ax2)
        cbar.set_label('Normalized F')
    else:
        ax2.text(0.5, 0.5, 'No trace data available',
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title(f'Plane {plane}{ref_label}: Activity (no data)')

    # Stats in suptitle
    n_frames = traces.shape[1] if traces is not None and traces.ndim == 2 else 0
    fig.suptitle(f'Cells: {n_cells}, Frames: {n_frames}', fontsize=10, y=1.02)

    plt.tight_layout()
    return fig


def plot_neuropil_subtraction_qc(
    original_df: pd.DataFrame,
    subtracted_df: pd.DataFrame,
    genes: List[str],
    rounds: List[dict],
    feature_type: str = 'mean_medflt_3x4x4',
    n_cols: int = 4,
    figsize_per_gene: Tuple[float, float] = (3.5, 3.0)
) -> plt.Figure:
    """
    QC plot for neuropil subtraction: before/after histograms for each gene.

    Args:
        original_df: DataFrame with original (pre-subtraction) values
        subtracted_df: DataFrame with neuropil-subtracted values
        genes: List of gene names to plot
        rounds: List of round info dicts
        feature_type: Feature type prefix for columns
        n_cols: Number of columns in subplot grid
        figsize_per_gene: Size per subplot

    Returns:
        Matplotlib Figure with grid of histograms
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")

    # Build gene-to-round mapping
    gene_round_map = {}
    for round_info in rounds:
        round_num = round_info['round']
        for gene in round_info['channels']:
            if gene not in EXCLUDE_GENES:
                gene_round_map[gene] = round_num

    # Filter to available genes
    plot_genes = [g for g in genes if g in gene_round_map]
    n_genes = len(plot_genes)

    if n_genes == 0:
        rprint("[yellow]No genes to plot for neuropil QC[/yellow]")
        return None

    n_rows = int(np.ceil(n_genes / n_cols))
    figsize = (figsize_per_gene[0] * n_cols, figsize_per_gene[1] * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()

    summary_stats = []

    for idx, gene in enumerate(plot_genes):
        ax = axes[idx]
        round_num = gene_round_map[gene]

        try:
            col = get_gene_column(original_df, gene, round_num, feature_type)
        except KeyError:
            ax.text(0.5, 0.5, f'{gene}\nColumn not found',
                   ha='center', va='center', transform=ax.transAxes)
            continue

        orig_vals = original_df[col].values
        sub_vals = subtracted_df[col].values

        # Remove NaN for histograms
        orig_clean = orig_vals[~np.isnan(orig_vals)]
        sub_clean = sub_vals[~np.isnan(sub_vals)]

        # Compute stats
        mean_shift = np.mean(orig_clean) - np.mean(sub_clean)
        n_negative = np.sum(sub_clean < 0)

        summary_stats.append({
            'gene': gene,
            'mean_shift': mean_shift,
            'n_negative': n_negative,
            'pct_negative': 100 * n_negative / len(sub_clean) if len(sub_clean) > 0 else 0
        })

        # Plot histograms with adaptive axis limits
        # Use 1st-99th percentile for axis limits to avoid outlier-stretched axes
        data_combined = np.concatenate([orig_clean, sub_clean])
        if len(data_combined) > 0:
            xlim_low = np.percentile(data_combined, 1)
            xlim_high = np.percentile(data_combined, 99)
            # Ensure we at least show the data range if everything is same value
            if xlim_low == xlim_high:
                xlim_low = data_combined.min() - 0.1
                xlim_high = data_combined.max() + 0.1
        else:
            xlim_low, xlim_high = 0, 1

        bins = np.linspace(xlim_low, xlim_high, 50)

        ax.hist(orig_clean, bins=bins, alpha=0.5, label='Original', color='blue')
        ax.hist(sub_clean, bins=bins, alpha=0.5, label='Subtracted', color='orange')

        # Set tight adaptive axis limits
        ax.set_xlim(xlim_low, xlim_high)

        ax.set_title(f'{gene}\nshift={mean_shift:.1f}', fontsize=9)
        ax.set_xlabel('Intensity', fontsize=8)
        ax.tick_params(labelsize=7)

        if idx == 0:
            ax.legend(fontsize=7)

    # Hide unused axes
    for idx in range(n_genes, len(axes)):
        axes[idx].set_visible(False)

    # Print summary
    rprint("\n  Neuropil subtraction QC:")
    for s in summary_stats:
        rprint(f"    {s['gene']}: mean shift = {s['mean_shift']:.2f}, "
               f"{s['n_negative']} negative ({s['pct_negative']:.1f}%)")

    plt.tight_layout()
    fig.suptitle('Neuropil Subtraction: Original vs Subtracted', y=1.02, fontsize=12)
    return fig


def plot_stimulus_in_video(
    stim_info: dict,
    framerate: float,
    title: str = "Stimulus Events in Recording",
    figsize: Tuple[int, int] = (14, 4)
) -> plt.Figure:
    """
    Diagnostic plot: stimulus trace with detected onset markers.

    Args:
        stim_info: Dict from detect_stimulus_onsets containing 'smoothed_stim',
                   'stim_times_raw', 'stim_frames', 'n_frames_per_plane'
        framerate: Framerate in Hz (per plane)
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=False)

    # Top: Raw stimulus channel trace with onset markers (in ephys samples)
    ax1 = axes[0]
    stim_trace = stim_info.get('smoothed_stim', stim_info.get('stim_channel'))
    stim_times_raw = stim_info.get('stim_times_raw', np.array([]))

    # Downsample for plotting if too many points
    n_samples = len(stim_trace)
    if n_samples > 100000:
        step = n_samples // 100000
        stim_trace_plot = stim_trace[::step]
        x_samples = np.arange(0, n_samples, step)
    else:
        stim_trace_plot = stim_trace
        x_samples = np.arange(len(stim_trace))

    ax1.plot(x_samples, stim_trace_plot, 'b-', linewidth=0.5, alpha=0.7)

    # Mark detected onsets
    for stim_time in stim_times_raw:
        ax1.axvline(stim_time, color='red', linestyle='--', alpha=0.7, linewidth=1)

    ax1.set_xlabel('Ephys sample')
    ax1.set_ylabel('Stimulus signal')
    ax1.set_title(f'Raw Stimulus Channel ({len(stim_times_raw)} events detected)')
    ax1.grid(True, alpha=0.3)

    # Bottom: Stimulus events in per-plane frame space
    ax2 = axes[1]
    stim_frames = stim_info.get('stim_frames_raw', np.array([])) // stim_info.get('nplanes', 1)
    n_frames = stim_info.get('n_frames_per_plane', 0)

    # Create timeline
    time_axis = np.arange(n_frames) / framerate

    # Plot as event markers
    ax2.eventplot([stim_frames], lineoffsets=0.5, linelengths=0.8, colors='red')

    ax2.set_xlim(0, n_frames)
    ax2.set_xlabel(f'Frame (per plane, {framerate:.2f} Hz)')
    ax2.set_ylabel('Events')
    ax2.set_title(f'Stimulus Onsets in Per-Plane Frame Space ({len(stim_frames)} events)')
    ax2.set_yticks([])

    # Add time axis on top
    ax2_twin = ax2.twiny()
    ax2_twin.set_xlim(0, n_frames / framerate)
    ax2_twin.set_xlabel('Time (seconds)')

    # Stats
    if len(stim_frames) > 1:
        intervals_frames = np.diff(stim_frames)
        intervals_sec = intervals_frames / framerate
        stats_text = (f"Inter-stimulus intervals: "
                     f"{intervals_sec.min():.1f}-{intervals_sec.max():.1f} sec "
                     f"(mean={intervals_sec.mean():.1f} sec)")
        ax2.text(0.02, 0.9, stats_text, transform=ax2.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fig.suptitle(title, y=1.02, fontsize=12)
    return fig


def plot_spatial_match_summary(
    plane_data: Dict[int, dict],
    merged_df: pd.DataFrame,
    reference_plane: int,
    mask_id_col: str = 'mask_id_main',
    twop_mask_col: str = 'twoP_mask',
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Diagnostic plot: spatial match summary showing 2P and HCR coordinate spaces.

    2-panel layout:
    1. Original 2P space: All cells (gray) + matched cells (red, plotted last)
    2. HCR space: By match type (unmatched gray, single blue, multi orange)

    Args:
        plane_data: Dict from load_2p_spatial_data
        merged_df: DataFrame from smart_merge_planes with 'has_2p_match', 'match_type'
        reference_plane: Reference plane number
        mask_id_col: Column for HCR cell ID
        twop_mask_col: Column for 2P mask ID
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    twop_col = get_col(merged_df, twop_mask_col)

    # Colors
    c_unmatched = 'lightgray'
    c_matched = 'orange'  # Matched 2P cells in first panel
    c_single = 'dodgerblue'
    c_multi = 'orange'

    # ========== Panel 1: Original 2P Space - All (gray) + Matched (orange) ==========
    ax1 = axes[0]
    if reference_plane in plane_data:
        pdata = plane_data[reference_plane]
        cell_ids = pdata['cell_ids']
        x_all = np.array(pdata['x_centroids'])
        y_all = np.array(pdata['y_centroids'])

        # Build lookup from cell_id to centroid position
        cell_to_idx = {cid: i for i, cid in enumerate(cell_ids)}

        # Get matched 2P cells from merged_df
        has_2p_col = get_col(merged_df, 'has_2p_match')
        best_plane_col = get_col(merged_df, 'best_plane')

        has_2p_arr = merged_df[has_2p_col].values if has_2p_col in merged_df.columns else np.zeros(len(merged_df), dtype=bool)
        best_plane_arr = merged_df[best_plane_col].values if best_plane_col in merged_df.columns else np.full(len(merged_df), -1)
        twop_arr = merged_df[twop_col].values

        # Build boolean masks
        is_matched = np.array([bool(x) if pd.notna(x) else False for x in has_2p_arr])
        is_ref_plane = (best_plane_arr == reference_plane)
        ref_matched_mask = is_matched & is_ref_plane

        # Get matched cell positions
        # Note: twop_arr contains mask IDs (int), cell_ids are strings like "plane0_123"
        # Build lookup from mask_id (int) to centroid index
        mask_to_idx = {}
        for i, cid in enumerate(cell_ids):
            try:
                # Extract mask ID from cell_id string (e.g., "plane0_123" -> 123)
                mask_id = int(cid.split('_')[1])
                mask_to_idx[mask_id] = i
            except (ValueError, IndexError):
                continue

        ref_indices = np.where(ref_matched_mask)[0]
        x_matched, y_matched = [], []
        for idx in ref_indices:
            twop_id = twop_arr[idx]
            if pd.notna(twop_id):
                twop_id_int = int(twop_id)
                if twop_id_int in mask_to_idx:
                    cent_idx = mask_to_idx[twop_id_int]
                    x_matched.append(x_all[cent_idx])
                    y_matched.append(y_all[cent_idx])

        # Plot all cells first (gray), then matched (orange) on top
        ax1.scatter(x_all, y_all, c=c_unmatched, s=10, alpha=0.5, label=f'All ({len(x_all)})')
        if len(x_matched) > 0:
            ax1.scatter(x_matched, y_matched, c=c_matched, s=15, alpha=0.7,
                       label=f'Matched ({len(x_matched)})')
        ax1.legend(fontsize=8, loc='upper right')
        ax1.set_title(f'Original 2P Space\n(plane {reference_plane}, pre-registration)')
    else:
        ax1.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Original 2P Space')
    ax1.set_xlabel('X (2P pixels)')
    ax1.set_ylabel('Y (2P pixels)')
    ax1.invert_yaxis()  # Image coordinates
    ax1.set_aspect('equal')

    # ========== Panel 2: HCR Space - By Match Type ==========
    ax2 = axes[1]

    # Try to get HCR coordinates from merged_df
    hcr_x_col = get_col(merged_df, 'centroid_x_main') if 'centroid_x_main' in str(merged_df.columns) else None
    hcr_y_col = get_col(merged_df, 'centroid_y_main') if 'centroid_y_main' in str(merged_df.columns) else None

    # Alternative column names
    for alt_x in ['hcr_x', 'x_main', 'centroid_x', 'x', 'X']:
        if hcr_x_col is None:
            try:
                hcr_x_col = get_col(merged_df, alt_x)
                if hcr_x_col in merged_df.columns:
                    break
                hcr_x_col = None
            except:
                pass

    for alt_y in ['hcr_y', 'y_main', 'centroid_y', 'y', 'Y']:
        if hcr_y_col is None:
            try:
                hcr_y_col = get_col(merged_df, alt_y)
                if hcr_y_col in merged_df.columns:
                    break
                hcr_y_col = None
            except:
                pass

    if hcr_x_col is not None and hcr_y_col is not None:
        hcr_x_arr = merged_df[hcr_x_col].values
        hcr_y_arr = merged_df[hcr_y_col].values

        # Get match info
        has_2p_col = get_col(merged_df, 'has_2p_match')
        match_type_col = get_col(merged_df, 'match_type')

        if has_2p_col in merged_df.columns:
            has_2p_arr = merged_df[has_2p_col].values
            has_match_mask = np.array([bool(x) if pd.notna(x) else False for x in has_2p_arr])
        else:
            has_match_mask = np.zeros(len(merged_df), dtype=bool)

        if match_type_col in merged_df.columns:
            match_type_arr = merged_df[match_type_col].values
            is_single_mask = (match_type_arr == 'single')
            is_multi_mask = (match_type_arr == 'multi')
        else:
            is_single_mask = np.zeros(len(merged_df), dtype=bool)
            is_multi_mask = np.zeros(len(merged_df), dtype=bool)

        # Unmatched (gray, plotted first)
        unmatched_mask = ~has_match_mask
        ax2.scatter(hcr_x_arr[unmatched_mask], hcr_y_arr[unmatched_mask],
                   c=c_unmatched, s=5, alpha=0.3, label=f'No match ({unmatched_mask.sum()})')

        # Single match (blue)
        if is_single_mask.sum() > 0:
            ax2.scatter(hcr_x_arr[is_single_mask], hcr_y_arr[is_single_mask],
                       c=c_single, s=10, alpha=0.7, label=f'Single ({is_single_mask.sum()})')

        # Multi match (orange)
        if is_multi_mask.sum() > 0:
            ax2.scatter(hcr_x_arr[is_multi_mask], hcr_y_arr[is_multi_mask],
                       c=c_multi, s=10, alpha=0.7, label=f'Multi ({is_multi_mask.sum()})')

        ax2.legend(fontsize=8, loc='upper right')
        ax2.set_title('HCR Space: By Match Type')
    else:
        ax2.text(0.5, 0.5, 'HCR coords not found\nCheck column names',
                ha='center', va='center', transform=ax2.transAxes, fontsize=9)
        ax2.set_title('HCR Space: By Match Type')
    ax2.set_xlabel('X (HCR)')
    ax2.set_ylabel('Y (HCR)')
    ax2.set_aspect('equal')

    # Summary stats
    n_total = len(merged_df)
    has_2p_col = get_col(merged_df, 'has_2p_match')
    match_type_col = get_col(merged_df, 'match_type')

    if has_2p_col in merged_df.columns:
        has_2p_arr = merged_df[has_2p_col].values
        n_matched = sum(1 for x in has_2p_arr if pd.notna(x) and bool(x))
    else:
        n_matched = 0

    if match_type_col in merged_df.columns:
        match_type_arr = merged_df[match_type_col].values
        n_single = (match_type_arr == 'single').sum()
        n_multi = (match_type_arr == 'multi').sum()
    else:
        n_single = 0
        n_multi = 0

    stats_text = (f"Total HCR cells: {n_total} | "
                 f"With 2P match: {n_matched} ({100*n_matched/n_total:.1f}%) | "
                 f"Single: {n_single} | Multi: {n_multi}")
    fig.suptitle(stats_text, y=1.02, fontsize=11)

    plt.tight_layout()
    return fig


def plot_response_validation(
    dff_data: dict,
    responses: dict,
    x: np.ndarray,
    y: np.ndarray,
    framerate: float,
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Diagnostic plot: validate response classification with spatial + temporal views.

    Left: Spatial scatter colored by response type (red=excited, blue=inhibited, gray=none)
    Right: Average DF/F traces for each response type

    Args:
        dff_data: Dict from compute_dff with 'dff_avg', 'time_axis'
        responses: Dict from classify_responses with 'excited_mask', 'inhibited_mask'
        x, y: Coordinate arrays for spatial plot
        framerate: Framerate in Hz
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    excited = responses['excited_mask']
    inhibited = responses['inhibited_mask']
    non_resp = ~excited & ~inhibited

    dff_avg = dff_data['dff_avg']
    time_axis = dff_data['time_axis']
    pre_frames = dff_data['pre_frames']

    # Validate array lengths match
    n_responses = len(excited)
    n_coords = len(x)
    if n_responses != n_coords:
        rprint(f"  [yellow]Warning: response masks ({n_responses}) don't match coordinates ({n_coords})[/yellow]")
        rprint(f"    Truncating to minimum length for visualization")
        min_len = min(n_responses, n_coords)
        excited = excited[:min_len]
        inhibited = inhibited[:min_len]
        non_resp = non_resp[:min_len]
        x = x[:min_len]
        y = y[:min_len]
        dff_avg = dff_avg[:min_len, :]

    # ========== Left: Spatial scatter ==========
    ax1 = axes[0]

    # Plot non-responsive first (background)
    ax1.scatter(x[non_resp], y[non_resp], c='lightgray', s=10, alpha=0.3,
               label=f'Non-resp ({np.sum(non_resp)})')

    # Excited and inhibited on top
    ax1.scatter(x[inhibited], y[inhibited], c='blue', s=20, alpha=0.7,
               label=f'Inhibited ({np.sum(inhibited)})')
    ax1.scatter(x[excited], y[excited], c='red', s=20, alpha=0.7,
               label=f'Excited ({np.sum(excited)})')

    ax1.set_xlabel('X position')
    ax1.set_ylabel('Y position')
    ax1.set_title('Spatial Distribution of Response Types')
    ax1.set_aspect('equal')
    ax1.legend(loc='upper right', fontsize=9)

    # ========== Right: Average traces ==========
    ax2 = axes[1]

    # Compute average traces per group
    if np.sum(non_resp) > 0:
        trace_non = np.mean(dff_avg[non_resp, :], axis=0)
        ax2.plot(time_axis, trace_non, 'gray', linewidth=2, alpha=0.7,
                label=f'Non-resp (n={np.sum(non_resp)})')

    if np.sum(inhibited) > 0:
        trace_inh = np.mean(dff_avg[inhibited, :], axis=0)
        ax2.plot(time_axis, trace_inh, 'blue', linewidth=2,
                label=f'Inhibited (n={np.sum(inhibited)})')

    if np.sum(excited) > 0:
        trace_exc = np.mean(dff_avg[excited, :], axis=0)
        ax2.plot(time_axis, trace_exc, 'red', linewidth=2,
                label=f'Excited (n={np.sum(excited)})')

    # Stimulus onset line
    ax2.axvline(0, color='black', linestyle='--', linewidth=1, label='Stim onset')
    ax2.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    ax2.set_xlabel('Time from stimulus (s)')
    ax2.set_ylabel('DF/F')
    ax2.set_title('Average DF/F by Response Type')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Summary stats
    n_total = len(excited)
    stats_text = (f"Total: {n_total} | "
                 f"Excited: {np.sum(excited)} ({100*np.sum(excited)/n_total:.1f}%) | "
                 f"Inhibited: {np.sum(inhibited)} ({100*np.sum(inhibited)/n_total:.1f}%)")
    fig.suptitle(stats_text, y=1.02, fontsize=11)

    plt.tight_layout()
    return fig


def plot_zflat_expression(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    gene: str,
    pctile: Tuple[float, float] = (5, 95),
    cmap: str = 'viridis',
    size: float = 10,
    ax: plt.Axes = None
) -> plt.Axes:
    """
    Spatial scatter plot with z-ordering (highest expression plotted last).

    Values are normalized to percentile range for consistent visualization.

    Args:
        x, y: Coordinate arrays
        values: Expression values
        gene: Gene name for title
        pctile: Percentile range for normalization (default: 5-95)
        cmap: Colormap
        size: Marker size
        ax: Matplotlib axes

    Returns:
        Matplotlib Axes
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Remove NaN
    valid = ~np.isnan(values)
    x_valid = x[valid]
    y_valid = y[valid]
    v_valid = values[valid]

    # Normalize to percentile range
    vmin = np.percentile(v_valid, pctile[0])
    vmax = np.percentile(v_valid, pctile[1])

    # Sort by value (lowest first, highest last)
    sort_idx = np.argsort(v_valid)
    x_sorted = x_valid[sort_idx]
    y_sorted = y_valid[sort_idx]
    v_sorted = v_valid[sort_idx]

    scatter = ax.scatter(x_sorted, y_sorted, c=v_sorted, cmap=cmap, s=size,
                        vmin=vmin, vmax=vmax, alpha=0.8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'{gene}')
    ax.set_aspect('equal')

    plt.colorbar(scatter, ax=ax, label='Expression', shrink=0.8)

    return ax


def plot_expression_grid(
    merged_df: pd.DataFrame,
    genes: List[str],
    rounds: List[dict],
    x_col: str = 'hcr_x',
    y_col: str = 'hcr_y',
    feature_type: str = 'mean_medflt_3x4x4',
    n_cols: int = 4,
    figsize_per_gene: Tuple[float, float] = (4, 4),
    pctile: Tuple[float, float] = (5, 95)
) -> plt.Figure:
    """
    Grid of z-flattened expression maps for multiple genes.

    Args:
        merged_df: DataFrame with gene expression and coordinate columns
        genes: List of gene names to plot
        rounds: List of round info dicts
        x_col, y_col: Column names for coordinates (auto-detects alternatives)
        feature_type: Feature type prefix for gene columns
        n_cols: Number of columns in grid
        figsize_per_gene: Size per subplot
        pctile: Percentile range for normalization

    Returns:
        Matplotlib Figure
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")

    # Build gene-to-round mapping
    gene_round_map = {}
    for round_info in rounds:
        round_num = round_info['round']
        for gene in round_info['channels']:
            if gene not in EXCLUDE_GENES:
                gene_round_map[gene] = round_num

    # Filter to available genes
    plot_genes = [g for g in genes if g in gene_round_map]
    n_genes = len(plot_genes)

    if n_genes == 0:
        rprint("[yellow]No genes to plot in expression grid[/yellow]")
        return None

    # Auto-detect coordinate columns with fallback alternatives
    x_col_actual = None
    y_col_actual = None

    # Try the specified column first, then alternatives
    x_alternatives = [x_col, 'hcr_x', 'centroid_x_main', 'x_main', 'centroid_x', 'X']
    y_alternatives = [y_col, 'hcr_y', 'centroid_y_main', 'y_main', 'centroid_y', 'Y']

    for alt_x in x_alternatives:
        candidate = get_col(merged_df, alt_x)
        if candidate in merged_df.columns:
            x_col_actual = candidate
            break

    for alt_y in y_alternatives:
        candidate = get_col(merged_df, alt_y)
        if candidate in merged_df.columns:
            y_col_actual = candidate
            break

    # Check if coordinate columns exist
    if x_col_actual is None or y_col_actual is None:
        rprint(f"[yellow]Coordinate columns not found. Tried: {x_alternatives[:3]}... / {y_alternatives[:3]}...[/yellow]")
        rprint(f"[yellow]Available columns: {list(merged_df.columns)[:10]}...[/yellow]")
        return None

    x = merged_df[x_col_actual].values
    y = merged_df[y_col_actual].values

    n_rows = int(np.ceil(n_genes / n_cols))
    figsize = (figsize_per_gene[0] * n_cols, figsize_per_gene[1] * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()

    for idx, gene in enumerate(plot_genes):
        ax = axes[idx]
        round_num = gene_round_map[gene]

        try:
            col = get_gene_column(merged_df, gene, round_num, feature_type)
            values = merged_df[col].values
            plot_zflat_expression(x, y, values, gene, pctile=pctile, ax=ax)
        except KeyError as e:
            ax.text(0.5, 0.5, f'{gene}\nColumn not found',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(gene)

    # Hide unused axes
    for idx in range(n_genes, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    fig.suptitle('Z-Flattened Gene Expression Maps (highest plotted last)', y=1.02, fontsize=12)
    return fig


# =============================================================================
# GENE CLASSIFICATION DIAGNOSTICS
# =============================================================================

def plot_gene_classification_diagnostic(
    merged_df: pd.DataFrame,
    gene: str,
    round_num: str,
    classification_stats: dict,
    planes: List[int],
    reference_plane: int = None,
    x_col: str = 'hcr_x',
    y_col: str = 'hcr_y',
    feature_type: str = 'mean_medflt_3x4x4',
    neuropil_subtracted: bool = True,
    axes: List = None,
    pctile: Tuple[float, float] = (2, 98),
    custom_threshold: float = None
) -> List:
    """
    Create a 3-panel diagnostic row for a single gene.

    Panel 1: Spatial map with normalized gene expression (colormap, 0-99th pctile)
    Panel 2: Spatial map showing gene-positive (magenta) vs gene-negative (grey)
    Panel 3: Distribution histogram with threshold line (x-axis 0 to 99th pctile)

    Args:
        merged_df: DataFrame with gene expression and classification columns
        gene: Gene name to plot
        round_num: Round number for this gene
        classification_stats: Dict with stats for this gene (mode, threshold, etc.)
        planes: List of plane numbers
        reference_plane: Reference plane (highlighted in display)
        x_col, y_col: Column names for coordinates
        feature_type: Feature type used (for labeling)
        neuropil_subtracted: Whether data is neuropil-subtracted (for labeling)
        axes: List of 3 matplotlib Axes (if None, creates new figure)
        pctile: Percentile range for normalization
        custom_threshold: Optional threshold to use instead of classification_stats threshold.
                         If provided, positive mask is recomputed using this threshold.

    Returns:
        List of 3 matplotlib Axes
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")

    # Get gene column
    try:
        gene_col = get_gene_column(merged_df, gene, round_num, feature_type)
    except KeyError:
        if axes is not None:
            for ax in axes:
                ax.text(0.5, 0.5, f'{gene}\nColumn not found',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(gene)
        return axes

    # Create figure if axes not provided
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax1, ax2, ax3 = axes

    # Get gene values
    gene_values = merged_df[gene_col].values

    # Determine threshold to use (custom overrides stored stats)
    stats = classification_stats.get(gene, {})
    stored_threshold = stats.get('threshold', None)

    if custom_threshold is not None:
        threshold = custom_threshold
    else:
        threshold = stored_threshold

    # Compute positive mask based on threshold
    # Always recompute if custom_threshold provided, otherwise use stored column if available
    if custom_threshold is not None or f'{gene}_positive' not in merged_df.columns:
        if threshold is not None:
            # Check if normalization was applied during classification
            # If so, we need to apply the same normalization before thresholding
            norm_method = stats.get('config', {}).get('normalization', 'none')
            if norm_method == 'none':
                norm_method = stats.get('method', 'mode_shift')
                # Check if method string contains normalization info (e.g., 'robust_iqr|adaptive_sigma')
                if '|' in str(norm_method):
                    norm_method = norm_method.split('|')[0]
                else:
                    norm_method = 'none'

            # Apply normalization if needed
            if norm_method == 'robust_iqr':
                normalized_values, _ = normalize_robust_iqr(gene_values)
                positive_mask = normalized_values > threshold
            elif norm_method == 'log1p':
                normalized_values, _ = normalize_log1p(gene_values)
                positive_mask = normalized_values > threshold
            else:
                # No normalization - compare raw values
                positive_mask = gene_values > threshold
        else:
            positive_mask = np.zeros(len(merged_df), dtype=bool)
    else:
        positive_mask = merged_df[f'{gene}_positive'].values.astype(bool)

    # Auto-detect coordinate columns
    x_col_actual = None
    y_col_actual = None

    x_alternatives = [x_col, 'hcr_x', 'centroid_x_main', 'x_main', 'centroid_x', 'X']
    y_alternatives = [y_col, 'hcr_y', 'centroid_y_main', 'y_main', 'centroid_y', 'Y']

    for alt in x_alternatives:
        if alt in merged_df.columns:
            x_col_actual = alt
            break

    for alt in y_alternatives:
        if alt in merged_df.columns:
            y_col_actual = alt
            break

    # Get coordinates
    if x_col_actual and y_col_actual:
        x = merged_df[x_col_actual].values
        y = merged_df[y_col_actual].values
        valid_coords = ~(np.isnan(x) | np.isnan(y))
    else:
        x = y = None
        valid_coords = np.zeros(len(merged_df), dtype=bool)

    # Valid values for normalization
    valid_values = gene_values[~np.isnan(gene_values)]
    if len(valid_values) > 0:
        vmin = 0
        vmax = np.percentile(valid_values, 99)
    else:
        vmin, vmax = 0, 1

    # ========== PANEL 1: Spatial expression map (normalized colormap) ==========

    if x is not None and np.sum(valid_coords) > 0:
        # Only plot neurons with valid coords and non-nan expression
        plot_mask = valid_coords & ~np.isnan(gene_values)

        scatter = ax1.scatter(
            x[plot_mask], y[plot_mask],
            c=gene_values[plot_mask],
            cmap='viridis',
            vmin=vmin, vmax=vmax,
            s=10, alpha=0.7,
            rasterized=True
        )
        plt.colorbar(scatter, ax=ax1, shrink=0.8, label='Expression')
        ax1.set_title(f'{gene}: Expression (0-99th pctile)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.invert_yaxis()
        ax1.set_aspect('equal')
    else:
        ax1.text(0.5, 0.5, 'Coordinates\nnot found', ha='center', va='center',
                transform=ax1.transAxes)
        ax1.set_title(f'{gene}: Expression')

    # ========== PANEL 2: Spatial positive/negative map ==========

    if x is not None and np.sum(valid_coords) > 0:
        # Negative neurons (grey background)
        neg_mask = valid_coords & ~positive_mask & ~np.isnan(gene_values)
        ax2.scatter(x[neg_mask], y[neg_mask], c='grey', s=5, alpha=0.3,
                   label=f'Negative ({np.sum(neg_mask)})', rasterized=True)

        # Positive neurons (magenta foreground, plotted last)
        pos_mask = valid_coords & positive_mask & ~np.isnan(gene_values)
        ax2.scatter(x[pos_mask], y[pos_mask], c='magenta', s=15, alpha=0.8,
                   label=f'Positive ({np.sum(pos_mask)})', rasterized=True)

        n_pos = np.sum(pos_mask)
        n_total = np.sum(neg_mask) + n_pos
        pct = 100 * n_pos / n_total if n_total > 0 else 0

        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title(f'{gene}: {n_pos} positive ({pct:.1f}%)')
        ax2.invert_yaxis()
        ax2.set_aspect('equal')
        ax2.legend(loc='upper right', fontsize=8, markerscale=2)
    else:
        ax2.text(0.5, 0.5, 'Coordinates\nnot found', ha='center', va='center',
                transform=ax2.transAxes)
        ax2.set_title(f'{gene}: Classification')

    # ========== PANEL 3: Histogram with threshold ==========

    if len(valid_values) > 0:
        # Check if normalization was used - if so, show normalized histogram
        norm_method = stats.get('config', {}).get('normalization', 'none')
        if norm_method == 'none':
            method_str = stats.get('method', 'mode_shift')
            if '|' in str(method_str):
                norm_method = method_str.split('|')[0]
                if norm_method == 'clipped':
                    parts = method_str.split('|')
                    norm_method = parts[1] if len(parts) > 1 else 'none'
            else:
                norm_method = 'none'

        # Determine what values to plot in histogram
        if norm_method == 'robust_iqr':
            hist_values, _ = normalize_robust_iqr(gene_values)
            hist_valid = hist_values[~np.isnan(hist_values)]
            hist_label = f'{gene} (robust_iqr normalized)'
            # For normalized data, use symmetric range around 0
            hist_vmax = np.percentile(np.abs(hist_valid), 99)
            hist_vmin = -hist_vmax * 0.5  # Allow some negative values to show
            bins = np.linspace(hist_vmin, hist_vmax, 75)
        elif norm_method == 'log1p':
            hist_values, _ = normalize_log1p(gene_values)
            hist_valid = hist_values[~np.isnan(hist_values)]
            hist_label = f'{gene} (log1p normalized)'
            hist_vmin = 0
            hist_vmax = np.percentile(hist_valid, 99)
            bins = np.linspace(hist_vmin, hist_vmax, 75)
        else:
            hist_valid = valid_values
            hist_label = gene
            hist_vmin = 0
            hist_vmax = vmax
            bins = np.linspace(0, vmax, 75)

        ax3.hist(hist_valid[(hist_valid >= hist_vmin) & (hist_valid <= hist_vmax)],
                bins=bins, alpha=0.7, edgecolor='black', linewidth=0.5, color='steelblue')

        # Use stats from earlier (threshold already set based on custom_threshold)
        mode = stats.get('mode', np.nan)
        noise_std = stats.get('noise_std', np.nan)
        sigma_used = stats.get('sigma_used', stats.get('adaptive_sigma', 2.0))
        n_positive = np.sum(positive_mask)
        pct_positive = 100 * n_positive / len(valid_values) if len(valid_values) > 0 else 0

        # Threshold line (use the threshold we computed earlier - it's in normalized space if normalized)
        if threshold is not None and hist_vmin <= threshold <= hist_vmax:
            threshold_label = f'Threshold: {threshold:.3f}'
            if custom_threshold is not None:
                threshold_label += ' (custom)'
            ax3.axvline(threshold, color='red', linestyle='--', linewidth=2,
                       label=threshold_label)

            # Shade positive region
            ylim = ax3.get_ylim()
            ax3.axvspan(threshold, hist_vmax, alpha=0.15, color='magenta')
            ax3.set_ylim(ylim)

        # Mode line (also in normalized space if normalized)
        if not np.isnan(mode) and hist_vmin <= mode <= hist_vmax:
            ax3.axvline(mode, color='grey', linestyle=':', linewidth=1.5,
                       label=f'Mode: {mode:.3f}')

        # Set x-axis limits
        ax3.set_xlim(hist_vmin, hist_vmax)

        # Text box with metadata
        neuropil_str = "subtracted" if neuropil_subtracted else "raw"
        threshold_val = threshold if threshold is not None else np.nan
        text_lines = [
            f"n = {len(valid_values)}",
            f"Mode: {mode:.3f}" if not np.isnan(mode) else "",
            f"Threshold: {threshold_val:.3f}" if not np.isnan(threshold_val) else "",
            f"Positive: {n_positive} ({pct_positive:.1f}%)"
        ]
        if norm_method != 'none':
            text_lines.insert(1, f"[{norm_method}]")
        if custom_threshold is not None:
            text_lines.insert(2, "(custom threshold)")
        text_str = '\n'.join([l for l in text_lines if l])

        ax3.text(0.97, 0.97, text_str, transform=ax3.transAxes,
                fontsize=8, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                family='monospace')

        xlabel = 'Expression (normalized)' if norm_method != 'none' else 'Expression value'
        ax3.set_xlabel(xlabel)
        ax3.set_ylabel('Count')
        ax3.set_title(f'{gene}: Distribution')
        ax3.legend(loc='upper left', fontsize=8)
    else:
        ax3.text(0.5, 0.5, 'No valid values', ha='center', va='center',
                transform=ax3.transAxes)
        ax3.set_title(f'{gene}: Distribution')
        ax3.set_title(f'{gene}: Spatial')

    return axes


def plot_gene_classification_grid(
    merged_df: pd.DataFrame,
    genes: List[str],
    rounds: List[dict],
    classification_stats: dict,
    planes: List[int],
    reference_plane: int = None,
    x_col: str = 'hcr_x',
    y_col: str = 'hcr_y',
    feature_type: str = 'mean_medflt_3x4x4',
    neuropil_subtracted: bool = True,
    figsize_per_row: Tuple[float, float] = (18, 4),
    custom_thresholds: Dict[str, float] = None,
    full_hcr_df: pd.DataFrame = None,
    density_bins: int = 50
) -> plt.Figure:
    """
    Create comprehensive gene classification diagnostic grid organized by round.

    Each gene gets a row with 3 panels (or 5 if full_hcr_df provided):
    - Panel 1: Spatial expression map (2P-matched neurons)
    - Panel 2: Spatial classification map (2P-matched neurons)
    - Panel 3: Distribution histogram with threshold
    - Panel 4 (optional): Full HCR volume classification scatter
    - Panel 5 (optional): Full HCR positive cell density heatmap

    Genes are grouped by HCR round with clear separators.

    Args:
        merged_df: DataFrame with gene expression and classification columns (2P-matched subset)
        genes: List of gene names to plot
        rounds: List of round info dicts (with 'round' and 'channels' keys)
        classification_stats: Dict from classified_df.attrs['classification_stats']
        planes: List of plane numbers
        reference_plane: Reference plane (highlighted in expression strips)
        x_col, y_col: Column names for coordinates
        feature_type: Feature type used (for labeling)
        neuropil_subtracted: Whether data is neuropil-subtracted (for labeling)
        figsize_per_row: Figure size per gene row
        custom_thresholds: Optional dict mapping gene names to custom thresholds.
                          Overrides classification_stats thresholds for visualization.
        full_hcr_df: Optional DataFrame with full HCR volume (all cells, not just 2P-matched).
                    If provided, adds panels 4 and 5 showing classification across the entire sample.
        density_bins: Number of bins for density heatmap (default 50)

    Returns:
        Matplotlib Figure
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")

    # Build gene-to-round mapping and group genes by round
    # Use a set to track genes we've already added (avoid duplicates)
    gene_round_map = {}
    genes_by_round = {}
    seen_genes = set()

    for round_info in rounds:
        round_num = round_info['round']
        genes_by_round[round_num] = []
        for channel in round_info['channels']:
            if channel not in EXCLUDE_GENES and channel in genes and channel not in seen_genes:
                gene_round_map[channel] = round_num
                genes_by_round[round_num].append(channel)
                seen_genes.add(channel)

    # Count total genes to plot (should match seen_genes)
    n_genes = len(seen_genes)

    if n_genes == 0:
        rprint("[yellow]No genes to plot in classification grid[/yellow]")
        return None

    # Determine number of columns (3, or 5 if full_hcr_df provided)
    n_cols = 5 if full_hcr_df is not None else 3

    # Create figure with extra space for round headers
    n_rounds = len([r for r in genes_by_round.values() if len(r) > 0])
    total_height = figsize_per_row[1] * n_genes + 0.5 * n_rounds  # Extra for headers
    # Adjust width for 5 columns
    fig_width = figsize_per_row[0] * (n_cols / 3)
    fig = plt.figure(figsize=(fig_width, total_height))

    # Use GridSpec for precise control
    from matplotlib.gridspec import GridSpec

    # Calculate rows needed (genes + round separators)
    gs_rows = n_genes
    gs = GridSpec(gs_rows, n_cols, figure=fig, hspace=0.4, wspace=0.3)

    # Pre-compute full HCR coordinates if provided
    twop_fov_bounds = None  # Will store (x_min, x_max, y_min, y_max) of 2P FOV
    if full_hcr_df is not None:
        # Auto-detect coordinate columns for full HCR
        full_x_col = None
        full_y_col = None
        x_alternatives = [x_col, 'hcr_x', 'centroid_x_main', 'x_main', 'centroid_x', 'X']
        y_alternatives = [y_col, 'hcr_y', 'centroid_y_main', 'y_main', 'centroid_y', 'Y']

        for alt in x_alternatives:
            if alt in full_hcr_df.columns:
                full_x_col = alt
                break
        for alt in y_alternatives:
            if alt in full_hcr_df.columns:
                full_y_col = alt
                break

        if full_x_col and full_y_col:
            full_x = full_hcr_df[full_x_col].values
            full_y = full_hcr_df[full_y_col].values
            full_valid_coords = ~(np.isnan(full_x) | np.isnan(full_y))

            # Compute 2P FOV bounds from cells with 2P matches
            # Use min/max (not percentiles) to get the actual FOV extent
            if 'has_2p_match' in full_hcr_df.columns:
                twop_matched = full_hcr_df['has_2p_match'].fillna(False).astype(bool)
                twop_x = full_x[twop_matched & full_valid_coords]
                twop_y = full_y[twop_matched & full_valid_coords]
                if len(twop_x) > 10:
                    # Use actual min/max of matched cells (this is the true 2P FOV extent)
                    twop_fov_bounds = (
                        np.min(twop_x),   # x_min
                        np.max(twop_x),   # x_max
                        np.min(twop_y),   # y_min
                        np.max(twop_y),   # y_max
                    )
        else:
            full_x = full_y = None
            full_valid_coords = None

    row_idx = 0

    for round_num in sorted(genes_by_round.keys()):
        round_genes = genes_by_round[round_num]
        if len(round_genes) == 0:
            continue

        # Add round header annotation
        if row_idx > 0:
            # Add some visual separation
            pass

        for gene in round_genes:
            if gene not in gene_round_map:
                continue

            # Safety check to avoid index out of bounds
            if row_idx >= n_genes:
                rprint(f"[yellow]Warning: row_idx {row_idx} >= n_genes {n_genes}, skipping {gene}[/yellow]")
                continue

            # Create axes for this gene
            ax1 = fig.add_subplot(gs[row_idx, 0])
            ax2 = fig.add_subplot(gs[row_idx, 1])
            ax3 = fig.add_subplot(gs[row_idx, 2])

            # Add round label to first panel
            if gene == round_genes[0]:
                ax1.annotate(f'Round {round_num}', xy=(0, 1.15), xycoords='axes fraction',
                           fontsize=11, fontweight='bold', color='darkblue')

            # Plot diagnostic for this gene
            # Check for custom threshold for this gene
            custom_thresh = None
            if custom_thresholds is not None and gene in custom_thresholds:
                custom_thresh = custom_thresholds[gene]

            plot_gene_classification_diagnostic(
                merged_df=merged_df,
                gene=gene,
                round_num=round_num,
                classification_stats=classification_stats,
                planes=planes,
                reference_plane=reference_plane,
                x_col=x_col,
                y_col=y_col,
                feature_type=feature_type,
                neuropil_subtracted=neuropil_subtracted,
                axes=[ax1, ax2, ax3],
                pctile=(2, 98),
                custom_threshold=custom_thresh
            )

            # Panel 4 & 5: Full HCR volume classification (if provided)
            if full_hcr_df is not None and n_cols == 5:
                ax4 = fig.add_subplot(gs[row_idx, 3])
                ax5 = fig.add_subplot(gs[row_idx, 4])

                if full_x is not None and full_valid_coords is not None:
                    # Get gene values from full HCR
                    try:
                        full_gene_col = get_gene_column(full_hcr_df, gene, round_num, feature_type)
                        full_gene_values = full_hcr_df[full_gene_col].values

                        # Get threshold and normalization method
                        stats = classification_stats.get(gene, {})
                        threshold = custom_thresh if custom_thresh is not None else stats.get('threshold', None)

                        if threshold is not None:
                            # Check if normalization was applied during classification
                            norm_method = stats.get('config', {}).get('normalization', 'none')
                            if norm_method == 'none':
                                method_str = stats.get('method', 'mode_shift')
                                if '|' in str(method_str):
                                    norm_method = method_str.split('|')[0]
                                    if norm_method == 'clipped':
                                        # Handle 'clipped|robust_iqr|...' format
                                        parts = method_str.split('|')
                                        norm_method = parts[1] if len(parts) > 1 else 'none'
                                else:
                                    norm_method = 'none'

                            # Apply same normalization before thresholding
                            if norm_method == 'robust_iqr':
                                normalized_values, _ = normalize_robust_iqr(full_gene_values)
                                full_positive = normalized_values > threshold
                            elif norm_method == 'log1p':
                                normalized_values, _ = normalize_log1p(full_gene_values)
                                full_positive = normalized_values > threshold
                            else:
                                full_positive = full_gene_values > threshold

                            # ===== PANEL 4: Scatter plot =====
                            # Plot negatives first (grey, small)
                            neg_mask = full_valid_coords & ~full_positive & ~np.isnan(full_gene_values)
                            ax4.scatter(full_x[neg_mask], full_y[neg_mask],
                                       c='lightgrey', s=1, alpha=0.3, rasterized=True)

                            # Plot positives on top (magenta, larger)
                            pos_mask = full_valid_coords & full_positive & ~np.isnan(full_gene_values)
                            ax4.scatter(full_x[pos_mask], full_y[pos_mask],
                                       c='magenta', s=3, alpha=0.7, rasterized=True)

                            # Add 2P FOV rectangle
                            if twop_fov_bounds is not None:
                                from matplotlib.patches import Rectangle
                                x_min, x_max, y_min, y_max = twop_fov_bounds
                                rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                                linewidth=2, edgecolor='cyan', facecolor='none',
                                                linestyle='--', label='2P FOV')
                                ax4.add_patch(rect)

                            n_pos = np.sum(pos_mask)
                            n_total = np.sum(full_valid_coords & ~np.isnan(full_gene_values))
                            pct_pos = 100 * n_pos / n_total if n_total > 0 else 0

                            ax4.set_title(f'Full HCR: {n_pos}/{n_total} ({pct_pos:.1f}%)')
                            ax4.invert_yaxis()
                            ax4.set_aspect('equal')
                            ax4.set_xlabel('X')
                            ax4.set_ylabel('Y')

                            # ===== PANEL 5: Density heatmap of positive cells =====
                            pos_x = full_x[pos_mask]
                            pos_y = full_y[pos_mask]

                            if len(pos_x) > 10:
                                # Create 2D histogram (density)
                                x_range = (np.nanmin(full_x[full_valid_coords]),
                                          np.nanmax(full_x[full_valid_coords]))
                                y_range = (np.nanmin(full_y[full_valid_coords]),
                                          np.nanmax(full_y[full_valid_coords]))

                                # Compute density of positive cells
                                H, xedges, yedges = np.histogram2d(
                                    pos_x, pos_y,
                                    bins=density_bins,
                                    range=[x_range, y_range]
                                )

                                # Plot as heatmap (transpose H for correct orientation)
                                extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]]  # Note: y inverted
                                im = ax5.imshow(H.T, extent=extent, origin='upper',
                                               cmap='magma', aspect='auto', interpolation='gaussian')

                                # Add colorbar
                                cbar = plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
                                cbar.set_label('Count', fontsize=8)

                                # Add 2P FOV rectangle
                                if twop_fov_bounds is not None:
                                    x_min, x_max, y_min, y_max = twop_fov_bounds
                                    rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                                    linewidth=2, edgecolor='cyan', facecolor='none',
                                                    linestyle='--')
                                    ax5.add_patch(rect)

                                ax5.set_title(f'Positive Density')
                                ax5.set_xlabel('X')
                                ax5.set_ylabel('Y')
                            else:
                                ax5.text(0.5, 0.5, 'Too few positive cells', ha='center', va='center',
                                        transform=ax5.transAxes)
                                ax5.set_title('Positive Density')

                        else:
                            ax4.text(0.5, 0.5, 'No threshold', ha='center', va='center',
                                    transform=ax4.transAxes)
                            ax4.set_title(f'{gene}: Full HCR')
                            ax5.text(0.5, 0.5, 'No threshold', ha='center', va='center',
                                    transform=ax5.transAxes)
                            ax5.set_title('Positive Density')

                    except KeyError:
                        ax4.text(0.5, 0.5, 'Column not found', ha='center', va='center',
                                transform=ax4.transAxes)
                        ax4.set_title(f'{gene}: Full HCR')
                        ax5.text(0.5, 0.5, 'Column not found', ha='center', va='center',
                                transform=ax5.transAxes)
                        ax5.set_title('Positive Density')
                else:
                    ax4.text(0.5, 0.5, 'No coordinates', ha='center', va='center',
                            transform=ax4.transAxes)
                    ax4.set_title(f'{gene}: Full HCR')
                    ax5.text(0.5, 0.5, 'No coordinates', ha='center', va='center',
                            transform=ax5.transAxes)
                    ax5.set_title('Positive Density')

            row_idx += 1

    # Overall title
    title_suffix = ' (with Full HCR + Density)' if full_hcr_df is not None else ''
    fig.suptitle(f'Gene Classification Diagnostics by Round{title_suffix}',
                fontsize=14, fontweight='bold', y=1.01)

    plt.tight_layout()
    return fig


# =============================================================================
# OUTPUT PERSISTENCE
# =============================================================================

def save_analysis_results(
    results: dict,
    output_dir: Path,
    config: AnalysisConfig = None
) -> dict:
    """
    Save analysis results to disk.

    Saves:
    - config.json: Analysis parameters
    - summary.json: Summary statistics
    - data/processed_table.pkl: Full processed DataFrame
    - data/classifications.pkl: Gene classifications
    - data/overlaps.csv: Overlap statistics

    Args:
        results: Dict with keys like 'processed_df', 'classifications', 'overlaps'
        output_dir: Output directory path
        config: AnalysisConfig (optional, will be saved if provided)

    Returns:
        Dict of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'data').mkdir(exist_ok=True)
    (output_dir / 'figures').mkdir(exist_ok=True)

    saved_paths = {}

    # Save config
    if config is not None:
        config_path = output_dir / 'config.json'
        config_dict = {
            'manifest_path': str(config.manifest_path) if config.manifest_path else None,
            'output_dir': str(config.output_dir) if config.output_dir else None,
            'pre_stim': config.pre_stim,
            'post_stim': config.post_stim,
            'sort_window': config.sort_window,
            'p_threshold': config.p_threshold,
            'stim_detection_threshold': config.stim_detection_threshold,
            'min_stim_interval_frames': config.min_stim_interval_frames,
            'normal_sigma': config.normal_sigma,
            'noisy_sigma': config.noisy_sigma,
            'noisy_genes': list(config.noisy_genes),
            'y_artifact_threshold': config.y_artifact_threshold,
            'feature_type': config.feature_type,
            'neuropil_type': config.neuropil_type,
        }
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        saved_paths['config'] = config_path

    # Save processed DataFrame
    if 'processed_df' in results:
        df_path = output_dir / 'data' / 'processed_table.pkl'
        results['processed_df'].to_pickle(df_path)
        saved_paths['processed_df'] = df_path

    # Save classifications
    if 'classifications' in results:
        cls_path = output_dir / 'data' / 'classifications.pkl'
        results['classifications'].to_pickle(cls_path)
        saved_paths['classifications'] = cls_path

    # Save overlaps as CSV (human-readable)
    if 'overlaps' in results:
        overlap_path = output_dir / 'data' / 'overlaps.csv'
        results['overlaps'].to_csv(overlap_path, index=False)
        saved_paths['overlaps'] = overlap_path

    # Save summary
    summary = {
        'n_cells': len(results.get('processed_df', [])),
        'n_genes': len(results.get('overlaps', [])),
    }
    if 'responses' in results:
        summary['n_excited'] = int(np.sum(results['responses'].get('excited_mask', [])))
        summary['n_inhibited'] = int(np.sum(results['responses'].get('inhibited_mask', [])))

    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    saved_paths['summary'] = summary_path

    rprint(f"\n  Saved results to {output_dir}")
    for key, path in saved_paths.items():
        rprint(f"    {key}: {path.name}")

    return saved_paths


def load_analysis_results(output_dir: Path) -> dict:
    """
    Load previously saved analysis results.

    Args:
        output_dir: Directory containing saved results

    Returns:
        Dict with loaded data
    """
    output_dir = Path(output_dir)

    results = {}

    # Load config
    config_path = output_dir / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            results['config'] = json.load(f)

    # Load processed DataFrame
    df_path = output_dir / 'data' / 'processed_table.pkl'
    if df_path.exists():
        results['processed_df'] = pd.read_pickle(df_path)

    # Load classifications
    cls_path = output_dir / 'data' / 'classifications.pkl'
    if cls_path.exists():
        results['classifications'] = pd.read_pickle(cls_path)

    # Load overlaps
    overlap_path = output_dir / 'data' / 'overlaps.csv'
    if overlap_path.exists():
        results['overlaps'] = pd.read_csv(overlap_path)

    # Load summary
    summary_path = output_dir / 'summary.json'
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            results['summary'] = json.load(f)

    return results
