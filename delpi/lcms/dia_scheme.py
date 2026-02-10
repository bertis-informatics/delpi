"""
DIA scheme detection and disjoint-window binning.

Supports:
- Fixed-window DIA (uniform window width)
- Variable-window DIA (varying window width)
- Overlapping-window DIA (windows overlap)
- Staggered DIA (interleaved cycles with offset windows)
- Scanning DIA (continuous quadrupole scan)

Not supported (raises ValueError):
- MSX-DIA (multiplexed isolation)
- DDA (data-dependent acquisition)
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
import polars as pl


@dataclass(frozen=True)
class SchemeConfig:
    """Configuration for DIA scheme detection."""

    # Window clustering tolerance (for handling floating-point noise)
    cluster_eps_ratio: float = 0.01  # relative to median window width
    cluster_eps_min: float = 0.02  # absolute minimum (m/z)
    cluster_eps_max: float = 0.20  # absolute maximum (m/z)

    # Staggered detection
    staggered_overlap_ratio: float = 0.4  # overlap > width * ratio => staggered

    # Fixed/variable window classification
    fixed_cv_threshold: float = 0.05  # CV(width) < 5% => fixed-window


def _cluster_values(values: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster 1D array by proximity after sorting.

    Returns:
        centers: cluster medians (sorted)
        labels: cluster assignment for each input value (original order)
    """
    if values.size == 0:
        return np.array([]), np.array([], dtype=np.int32)

    order = np.argsort(values)
    sorted_vals = values[order]

    # Find cluster boundaries
    boundaries = [0]
    for i in range(1, len(sorted_vals)):
        if sorted_vals[i] - sorted_vals[i - 1] > eps:
            boundaries.append(i)
    boundaries.append(len(sorted_vals))

    # Compute cluster centers and labels
    centers = []
    labels_sorted = np.empty(len(sorted_vals), dtype=np.int32)

    for k in range(len(boundaries) - 1):
        start, end = boundaries[k], boundaries[k + 1]
        centers.append(float(np.median(sorted_vals[start:end])))
        labels_sorted[start:end] = k

    # Reorder labels to match original order
    labels = np.empty(len(values), dtype=np.int32)
    labels[order] = labels_sorted

    return np.array(centers), labels


def _extract_unique_windows(
    ms2_df: pl.DataFrame, eps: float
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Extract unique isolation windows from MS2 scans.

    Returns:
        unique_windows: DataFrame with columns [win_id, lb, ub, width, n_scans]
        ms2_with_ids: Original ms2_df with added win_id column
    """
    lb_arr = ms2_df["isolation_min_mz"].to_numpy().astype(np.float64)
    ub_arr = ms2_df["isolation_max_mz"].to_numpy().astype(np.float64)

    # Cluster lower and upper bounds independently
    _, lb_labels = _cluster_values(lb_arr, eps)
    _, ub_labels = _cluster_values(ub_arr, eps)

    # Combine to create unique window IDs
    win_key = lb_labels * (ub_labels.max() + 2) + ub_labels
    _, win_ids = np.unique(win_key, return_inverse=True)

    # Add window IDs to dataframe
    ms2_with_ids = ms2_df.with_columns(pl.Series("win_id", win_ids, dtype=pl.UInt32))

    # Aggregate to get unique windows
    unique_windows = (
        ms2_with_ids.group_by("win_id")
        .agg(
            pl.col("isolation_min_mz").median().alias("lb"),
            pl.col("isolation_max_mz").median().alias("ub"),
            pl.len().alias("n_scans"),
        )
        .with_columns((pl.col("ub") - pl.col("lb")).alias("width"))
        .sort("lb")
    )

    return unique_windows, ms2_with_ids


def _detect_scheme_type(
    unique_windows: pl.DataFrame, cfg: SchemeConfig, eps: float
) -> str:
    """
    Detect DIA scheme type from unique windows.

    Logic:
    1. Check overlap existence (most fundamental distinction)
    2. No overlap → fixed vs variable (by width CV)
    3. Has overlap → scanning vs staggered vs overlapping

    Returns:
        "scanning"
        "staggered"
        "overlapping"
        "fixed" or "variable"
    """
    n_windows = unique_windows.height

    if n_windows < 2:
        return "fixed"

    # Check for DDA: many unique windows with minimal repetition
    # DDA typically has hundreds/thousands of unique precursors, each appearing 1-3 times
    # DIA typically has 10-100 unique windows, each appearing 50-200+ times
    n_scans_arr = unique_windows["n_scans"].to_numpy()
    scans_per_window_med = float(np.median(n_scans_arr))

    if n_windows > 100 and scans_per_window_med < 5:
        raise ValueError(
            f"This data seems not DIA. Unable to determine DIA scheme: {n_windows} unique windows with median "
            f"{scans_per_window_med:.1f} scans/window."
        )

    lb_arr = unique_windows["lb"].to_numpy()
    ub_arr = unique_windows["ub"].to_numpy()
    width_arr = unique_windows["width"].to_numpy()

    width_med = float(np.median(width_arr))
    width_cv = float(np.std(width_arr) / width_med) if width_med > 0 else 0.0

    # Step 1: Compute overlaps between consecutive windows
    overlaps = ub_arr[:-1] - lb_arr[1:]
    overlap_positive = overlaps > eps

    # Step 2: No overlap → fixed or variable window
    if not np.any(overlap_positive):
        return "fixed" if width_cv < cfg.fixed_cv_threshold else "variable"

    # Step 3: Has overlap → classify by overlap ratio
    # Compute overlap ratio for positive overlaps
    overlap_ratio = overlaps[overlap_positive] / width_arr[:-1][overlap_positive]
    overlap_ratio_med = float(np.median(overlap_ratio))

    # 1) Staggered: overlap ratio > 40%
    if overlap_ratio_med > cfg.staggered_overlap_ratio:
        return "staggered"

    # 2) Scanning vs Overlapping: compare overlap to non-overlap portion
    # - Scanning DIA: overlap > non-overlap (e.g., 4 m/z window, 2 m/z step → 50% overlap)
    # - Overlapping DIA: overlap < non-overlap (e.g., 20 m/z window, 5 m/z overlap → 25%)
    elif overlap_ratio_med > 0.5:
        return "scanning"
    else:
        return "overlapping"


def _bin_fixed_variable(
    unique_windows: pl.DataFrame, ms2_with_ids: pl.DataFrame
) -> pl.DataFrame:
    """
    Bin fixed or variable-window DIA (no overlap).

    Each unique window becomes one bin.
    """
    return (
        unique_windows.select(
            pl.col("win_id").alias("isolation_win_idx"),
            pl.col("lb").alias("isolation_min_mz"),
            pl.col("ub").alias("isolation_max_mz"),
        )
        .join(
            ms2_with_ids.select("win_id", "frame_num"),
            left_on="isolation_win_idx",
            right_on="win_id",
            how="left",
        )
        .group_by("isolation_win_idx")
        .agg(
            pl.col("isolation_min_mz").first(),
            pl.col("isolation_max_mz").first(),
            pl.col("frame_num"),
        )
        .sort("isolation_win_idx")
    )


def _bin_overlapping(
    unique_windows: pl.DataFrame, ms2_with_ids: pl.DataFrame
) -> pl.DataFrame:
    """
    Bin overlapping-window DIA.

    Split overlaps at midpoint between consecutive window bounds.
    """
    lb_arr = unique_windows["lb"].to_numpy()
    ub_arr = unique_windows["ub"].to_numpy()

    # Compute midpoints for overlap splitting
    midpoints = 0.5 * (ub_arr[:-1] + lb_arr[1:])

    # Adjust bounds: each window gets [lb, midpoint] or [midpoint, ub]
    new_lb = lb_arr.copy()
    new_ub = ub_arr.copy()
    new_ub[:-1] = midpoints
    new_lb[1:] = midpoints

    return (
        unique_windows.select(pl.col("win_id"))
        .with_columns(
            isolation_min_mz=pl.Series(new_lb),
            isolation_max_mz=pl.Series(new_ub),
        )
        .join(ms2_with_ids.select("win_id", "frame_num"), on="win_id", how="left")
        .group_by("win_id")
        .agg(
            pl.col("isolation_min_mz").first(),
            pl.col("isolation_max_mz").first(),
            pl.col("frame_num"),
        )
        .rename({"win_id": "isolation_win_idx"})
        .sort("isolation_win_idx")
    )


def _bin_scanning(
    unique_windows: pl.DataFrame, ms2_with_ids: pl.DataFrame
) -> pl.DataFrame:
    """
    Bin scanning DIA.

    Uses the shift delta between consecutive windows as bin width.
    Creates uniform bins covering [min_lb, max_ub] with delta width.
    """
    lb_arr = unique_windows["lb"].to_numpy()
    ub_arr = unique_windows["ub"].to_numpy()

    # Compute delta (shift between consecutive windows)
    deltas = np.diff(lb_arr)
    delta = float(np.median(deltas))

    # Compute overall range
    mz_min = float(lb_arr.min())
    mz_max = float(ub_arr.max())

    # Create uniform bins with delta width
    n_bins = int(np.ceil((mz_max - mz_min) / delta))
    bin_edges = mz_min + np.arange(n_bins + 1, dtype=np.float64) * delta

    # Assign each MS2 frame to bin by isolation center
    ms2_lb = ms2_with_ids.join(
        unique_windows.select("win_id", pl.col("lb").alias("iso_lb")),
        on="win_id",
        how="left",
    )
    ms2_ub = ms2_lb.join(
        unique_windows.select("win_id", pl.col("ub").alias("iso_ub")),
        on="win_id",
        how="left",
    )

    centers = 0.5 * (ms2_ub["iso_lb"].to_numpy() + ms2_ub["iso_ub"].to_numpy())
    bin_idx = np.clip(
        np.searchsorted(bin_edges, centers, side="right") - 1, 0, n_bins - 1
    )

    # Build result dataframe
    ms2_binned = ms2_with_ids.with_columns(
        pl.Series("isolation_win_idx", bin_idx.astype(np.int32))
    )

    # Create bin boundaries
    bin_df = pl.DataFrame(
        {
            "isolation_win_idx": np.arange(n_bins, dtype=np.int32),
            "isolation_min_mz": bin_edges[:-1],
            "isolation_max_mz": bin_edges[1:],
        }
    )

    return (
        bin_df.join(
            ms2_binned.select("isolation_win_idx", "frame_num"),
            on="isolation_win_idx",
            how="left",
        )
        .group_by("isolation_win_idx")
        .agg(
            pl.col("isolation_min_mz").first(),
            pl.col("isolation_max_mz").first(),
            pl.col("frame_num"),
        )
        .sort("isolation_win_idx")
    )


def _bin_staggered(
    unique_windows: pl.DataFrame, ms2_with_ids: pl.DataFrame
) -> pl.DataFrame:
    """
    Bin staggered DIA.

    Uses the logic from original DIARun.determine_dia_scheme:
    - First bin: [window_0], lb_0, ub_0
    - Middle bins: [window_i, window_i+1], prev_ub, ub_i
    - Last bin: [window_n], prev_ub, ub_n
    """
    n = unique_windows.height
    win_ids = unique_windows["win_id"].to_list()
    lb_arr = unique_windows["lb"].to_numpy()
    ub_arr = unique_windows["ub"].to_numpy()

    # Build bin structure
    bins = []

    # First bin: only first window
    bins.append(
        {
            "win_ids": [win_ids[0]],
            "isolation_min_mz": float(lb_arr[0]),
            "isolation_max_mz": float(ub_arr[0]),
        }
    )

    prev_ub = float(ub_arr[0])

    # Middle bins: pairs of consecutive windows
    for i in range(1, n - 1):
        bins.append(
            {
                "win_ids": [win_ids[i], win_ids[i + 1]],
                "isolation_min_mz": prev_ub,
                "isolation_max_mz": float(ub_arr[i]),
            }
        )
        prev_ub = float(ub_arr[i])

    # Last bin: only last window
    if n > 1:
        bins.append(
            {
                "win_ids": [win_ids[-1]],
                "isolation_min_mz": prev_ub,
                "isolation_max_mz": float(ub_arr[-1]),
            }
        )

    # Convert to DataFrame
    bin_df = pl.DataFrame(
        [
            {
                "isolation_win_idx": i,
                "win_ids": b["win_ids"],
                "isolation_min_mz": b["isolation_min_mz"],
                "isolation_max_mz": b["isolation_max_mz"],
            }
            for i, b in enumerate(bins)
        ]
    )

    # Map frames to bins through window IDs
    return (
        bin_df.explode("win_ids")
        .join(
            ms2_with_ids.select("win_id", "frame_num"),
            left_on="win_ids",
            right_on="win_id",
            how="left",
        )
        .group_by("isolation_win_idx")
        .agg(
            pl.col("isolation_min_mz").first(),
            pl.col("isolation_max_mz").first(),
            pl.col("frame_num"),
        )
        .sort("isolation_win_idx")
    )


def determine_dia_scheme(
    meta_df: pl.DataFrame, cfg: SchemeConfig = None
) -> pl.DataFrame:
    """
    Determine DIA scheme and return disjoint isolation bins.

    Args:
        meta_df: Metadata DataFrame with columns:
            - frame_num
            - ms_level
            - isolation_min_mz
            - isolation_max_mz
        cfg: Configuration (uses defaults if None)

    Returns:
        DataFrame with columns:
            - isolation_win_idx: bin index
            - isolation_min_mz: bin lower bound
            - isolation_max_mz: bin upper bound
            - frame_num: list of frame numbers in this bin

    Raises:
        ValueError: For unsupported schemes (DDA, MSX-DIA)
    """
    if cfg is None:
        cfg = SchemeConfig()

    # Validate input
    required_cols = {"frame_num", "ms_level", "isolation_min_mz", "isolation_max_mz"}
    missing = required_cols - set(meta_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Extract MS2 scans
    ms2_df = (
        meta_df.filter(pl.col("ms_level") == 2)
        .select(["frame_num", "isolation_min_mz", "isolation_max_mz"])
        .sort("frame_num")
    )

    if ms2_df.height == 0:
        raise ValueError("No MS2 scans found")

    # Compute clustering tolerance
    widths = (ms2_df["isolation_max_mz"] - ms2_df["isolation_min_mz"]).to_numpy()
    width_med = float(np.median(widths))

    if not np.isfinite(width_med) or width_med <= 0:
        raise ValueError("Invalid isolation window widths")

    eps = float(
        np.clip(
            cfg.cluster_eps_ratio * width_med, cfg.cluster_eps_min, cfg.cluster_eps_max
        )
    )

    # Extract unique windows
    unique_windows, ms2_with_ids = _extract_unique_windows(ms2_df, eps)

    # Detect scheme type
    scheme_type = _detect_scheme_type(unique_windows, cfg, eps)

    # Bin according to scheme
    if scheme_type in ("fixed", "variable"):
        dia_scheme_df = _bin_fixed_variable(unique_windows, ms2_with_ids)
    elif scheme_type == "overlapping":
        dia_scheme_df = _bin_overlapping(unique_windows, ms2_with_ids)
    elif scheme_type == "staggered":
        dia_scheme_df = _bin_staggered(unique_windows, ms2_with_ids)
    elif scheme_type == "scanning":
        dia_scheme_df = _bin_scanning(unique_windows, ms2_with_ids)
    else:
        raise ValueError(f"Unsupported DIA scheme: {scheme_type}")

    # Ensure correct types
    return scheme_type, dia_scheme_df.select(
        pl.col("isolation_win_idx").cast(pl.UInt32),
        pl.col("isolation_min_mz").cast(pl.Float32),
        pl.col("isolation_max_mz").cast(pl.Float32),
        pl.col("frame_num"),
    )


def get_test_meta_df(
    scheme: str = "fixed",
    mz_min: float = 400.0,
    mz_max: float = 800.0,
    window_width: float = 8.0,
    num_cycles: int = 100,
) -> pl.DataFrame:
    """
    Generate test meta_df for different DIA schemes.

    Parameters
    ----------
    scheme : str
        DIA scheme type: "fixed", "variable", "overlapping", "staggered", "scanning"
    mz_min : float
        Minimum m/z value
    mz_max : float
        Maximum m/z value
    window_width : float
        Base window width (m/z). For variable schemes, this is the median width.
    num_cycles : int
        Number of acquisition cycles

    Returns
    -------
    pl.DataFrame
        Meta dataframe with columns: frame_num, isolation_min_mz, isolation_max_mz, ms_level

    Examples
    --------
    >>> # Fixed-window DIA (uniform 20 m/z windows)
    >>> meta_df = get_test_meta_df("fixed", window_width=20.0)

    >>> # Variable-window DIA (wider windows at high m/z)
    >>> meta_df = get_test_meta_df("variable")

    >>> # Overlapping-window DIA (5 m/z overlap)
    >>> meta_df = get_test_meta_df("overlapping")

    >>> # Staggered DIA (2x window width, interleaved)
    >>> meta_df = get_test_meta_df("staggered")

    >>> # Scanning DIA (sliding 2 m/z steps)
    >>> meta_df = get_test_meta_df("scanning")
    """
    scheme = scheme.lower()

    if scheme == "fixed":
        # Fixed-window: uniform width across m/z range
        lower_bounds = np.arange(mz_min, mz_max, window_width)
        upper_bounds = lower_bounds + window_width

    elif scheme == "variable":
        # Variable-window: non-overlapping windows with increasing width
        # e.g., 8, 12, 16, 20, 24 m/z (wider at high m/z, mimics real instruments)
        lower_bounds = []
        upper_bounds = []

        current_mz = mz_min
        window_idx = 0

        while current_mz < mz_max:
            # Linear increase: width = base * (1 + idx * 0.5)
            # This gives: 8, 12, 16, 20, 24, ... for base=8
            width = window_width * (1 + window_idx * 0.5)

            next_mz = min(current_mz + width, mz_max)
            lower_bounds.append(current_mz)
            upper_bounds.append(next_mz)

            current_mz = next_mz
            window_idx += 1

        lower_bounds = np.array(lower_bounds)
        upper_bounds = np.array(upper_bounds)

    elif scheme == "overlapping":
        # Overlapping-window: 25% overlap (5 m/z overlap for 20 m/z windows)
        overlap = window_width * 0.25
        step = window_width - overlap
        lower_bounds = np.arange(mz_min, mz_max, step)
        upper_bounds = lower_bounds + window_width

    elif scheme == "staggered":
        # Staggered: 2x window width, alternating odd/even cycles
        wide_width = window_width * 2
        lower_bounds_odd = np.arange(mz_min, mz_max, wide_width)
        lower_bounds_even = lower_bounds_odd + window_width

        # Interleave odd and even windows
        lower_bounds = np.sort(np.concatenate([lower_bounds_odd, lower_bounds_even]))
        upper_bounds = lower_bounds + wide_width

    elif scheme == "scanning":
        # Scanning: small steps (2 m/z) with narrow windows (4 m/z)
        # Creates ~150 unique windows for 350-1650 m/z range
        scan_width = 4.0
        scan_step = 2.0
        lower_bounds = np.arange(mz_min, mz_max, scan_step)
        upper_bounds = lower_bounds + scan_width

    else:
        raise ValueError(
            f"Unknown scheme '{scheme}'. "
            "Supported: fixed, variable, overlapping, staggered, scanning"
        )

    # Clip to valid m/z range
    mask = upper_bounds <= mz_max + 0.1  # small tolerance
    lower_bounds = lower_bounds[mask]
    upper_bounds = upper_bounds[mask]

    # Generate meta_df with repeated cycles
    n_windows = len(lower_bounds)
    frame_num = 0
    meta_list = []

    for cycle in range(num_cycles):
        for i in range(n_windows):
            meta_list.append([frame_num, lower_bounds[i], upper_bounds[i]])
            frame_num += 1

    meta_df = pl.DataFrame(
        meta_list,
        schema={
            "frame_num": pl.UInt32,
            "isolation_min_mz": pl.Float32,
            "isolation_max_mz": pl.Float32,
        },
        orient="row",
    ).with_columns(ms_level=pl.lit(2, dtype=pl.UInt8))

    return meta_df
