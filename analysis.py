"""analysis.py

Backend analysis utilities for the Raman GUI.

Design goals
- Pure-Python backend that can be used from BOTH:
  (1) wx_gui.py (interactive GUI)
  (2) a standalone CLI (future-proofing)

Current scope (as agreed)
1) Similar filename detection from a seed file.
2) Decide whether to perform cosmic-ray correction.
3) Decide whether to use manual laser wavelength or fit/auto.
   - CLI shape: --manual-laser-nm <wavelength>
4) Do NOT rename originals. Only embed corrected values into the merged output.
5) Normalize is deferred.
6) --laser-peak-plot and --cosmic-peak-plot control whether diagnostic plots are shown.
   - Defaults: laser_peak_plot=False, cosmic_peak_plot=True
   - IMPORTANT: Even if cosmic_peak_plot is False, analysis must still print
     numeric evidence and request confirmation (y/n) in CLI mode.

This file intentionally provides a clear data-contract via dataclasses.
GUI should only populate options and consume results.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# ============================================================================
# Data contracts (shared between GUI and backend)
# ============================================================================


@dataclass
class MergeDiscoverOptions:
    """Stage-1: similar-file detection + raw matrix build (wavelength axis)."""

    seed_file: str
    assume_xgrid_consistent: bool = True


@dataclass
class MergePreviewOptions:
    """Stage-2: apply corrections / compute Raman-shift axis for preview."""

    seed_file: str
    files: List[str]

    cosmic_enable: bool = True

    # If not None -> manual laser wavelength is used.
    manual_laser_nm: Optional[float] = None
    use_raman_x: bool = False
    raman_x_mode: Optional[str] = "cm-1"

    laser_peak_plot: bool = False # Only for CLI; GUI always shows laser plot
    cosmic_peak_plot: bool = True # Only for CLI; GUI has separate cosmic plot

    # New options for GUI/CLI safety
    interactive_confirm: bool = False
    require_laser_nm: bool = False


# ============================================================================
# Cosmic-ray GUI Data Contracts
# ============================================================================


@dataclass
class CosmicPeak:
    """A detected cosmic-ray candidate in a single spectrum row.

    Cosmic rays in this project are modeled as narrow spikes within ONE angle row
    (i.e., a single file / xxxx), localized in wavelength.
    """

    row_index: int
    col_index: int
    is_confirmed_cosmic: bool = False

    xxxx: Optional[str] = None
    yyyy: Optional[str] = None

    angle_deg: Optional[float] = None
    center_wavelength_nm: float = float("nan")
    intensity: float = float("nan")
    fwhm_nm: float = float("nan")
    test_results: Dict[str, Union[str, float, bool]] = field(default_factory=dict)



@dataclass
class MergeDiscoverResult:
    """Result of Stage-1: raw preview on wavelength axis."""

    files: List[str]
    pattern_hint: str
    unique_xxxx: List[str]
    unique_yyyy: List[str]

    wavelength_nm: np.ndarray  # shape (nx,)
    angle_values: np.ndarray  # shape (ny,)
    intensity_matrix: np.ndarray  # shape (ny, nx)

    title: str = "Preview (raw, wavelength axis)"
    candidate_laser_nm: Optional[float] = None
    
    # Store raw data for cosmic ray detection etc.
    raw_files: List[str] = field(default_factory=list, repr=False)
    raw_xxxx: List[Optional[str]] = field(default_factory=list, repr=False) # xxxx from original files
    raw_yyyy: List[Optional[str]] = field(default_factory=list, repr=False) # yyyy from original files
    raw_angle_values: Optional[np.ndarray] = field(default=None, repr=False)  # shape (n_raw,)
    raw_intensity_matrix: Optional[np.ndarray] = field(default=None, repr=False)  # shape (n_raw, nx)
    raw_rows_by_xxxx: Optional[Dict[str, List[int]]] = field(default=None, repr=False) # Map xxxx to list of raw row indices

    primitive_xxxx: List[str] = field(default_factory=list) # xxxx values after grouping, e.g., ["0001", "0002"] or ["RR", "RL"]
    primitive_matrix: Optional[np.ndarray] = field(default=None)  # shape (n_xxxx, nx)

    cosmic_matrix: Optional[np.ndarray] = field(default=None) # Mask applied to original raw_intensity_matrix
    
    is_polarization_merge: bool = False # New flag to indicate a polarization merge


@dataclass
class MergePreviewResult:
    """Result of Stage-2: corrected preview on Raman-shift axis."""

    raman_shift_cm1: np.ndarray  # shape (nx,)
    energy_ev: Optional[np.ndarray]  # shape (nx,) or None
    angle_values: np.ndarray  # shape (ny,)
    intensity_matrix: np.ndarray  # shape (ny, nx)

    # cosmic_summary is removed, as this function no longer handles cosmic correction.
    title: str = "Preview (corrected, Raman shift axis)"

@dataclass
class MergeCosmicOptions(MergeDiscoverResult):
    dark_value: float = 600.0
    intensity_thresh: float = 1200.0
    comparison_factor: float = 20.0
    z_thresh_fallback: float = 8.0

@dataclass
class MergeCosmicResult(MergeCosmicOptions):
    peaks: List[CosmicPeak] = field(default_factory=list)
    evidence: Dict[str, object] = field(default_factory=dict)


# ============================================================================
# User interaction hooks (CLI vs GUI)
# ============================================================================


class UserPrompt:
    """Abstract prompt interface.

    - CLI implementation uses print/input.
    - GUI implementation should replace ask_yes_no/show_plot to use dialogs.

    This interface exists so analysis logic never directly depends on wx.
    """

    def info(self, msg: str) -> None:
        raise NotImplementedError

    def ask_yes_no(self, question: str, default: bool = True) -> bool:
        raise NotImplementedError

    def show_plot(self, kind: Literal["laser", "cosmic"], payload: object) -> None:
        """Optional diagnostic plot visualization.

        payload is intentionally opaque here.
        In the future, pass matplotlib Figure or a small dict of arrays.
        """
        return


class CliPrompt(UserPrompt):
    def info(self, msg: str) -> None:
        print(msg)

    def ask_yes_no(self, question: str, default: bool = True) -> bool:
        suffix = "[Y/n]" if default else "[y/N]"
        while True:
            ans = input(f"{question} {suffix} ").strip().lower()
            if ans == "" and default is not None:
                return bool(default)
            if ans in ("y", "yes"):
                return True
            if ans in ("n", "no"):
                return False
            print("Please answer y/n.")


# NullPrompt for GUI-safe backend (no interaction, stores messages)
class NullPrompt(UserPrompt):
    def __init__(self):
        self.messages: list[str] = []
    def info(self, msg: str) -> None:
        self.messages.append(msg)
    def ask_yes_no(self, question: str, default: bool = True) -> bool:
        raise RuntimeError("Interactive prompt requested but no prompt handler was provided")
    def show_plot(self, kind: Literal["laser", "cosmic"], payload: object) -> None:
        return


# ============================================================================
# Filename detection helpers
POLARIZATION_TYPES = ["RR", "LL", "RL", "LR"]
POLARIZATION_TYPES_RE = "|".join(POLARIZATION_TYPES)

# Regex for different filename patterns

POLARIZATION_IDX_RE = re.compile(

    r"^(?P<prefix>.*?)_(?P<pol_type>RR|RL|LR|LL)(?P<middle>.*)_(?P<yyyy>\d{4})(?P<ext>\.[^.]+)$"

)

NORMAL_IDX_TWO_RE = re.compile(r"^(?P<prefix>.+)_(?P<xxxx>\d{4})_(?P<yyyy>\d{4})(?P<ext>\.[^.]+)$")

NORMAL_IDX_ONE_RE = re.compile(r"^(?P<prefix>.+)_(?P<xxxx>\d{4})(?P<ext>\.[^.]+)$")

# ============================================================================





def detect_similar_files(seed_file: str) -> Tuple[List[str], str, List[str], List[str], bool]:

    """Detect similar files in the same directory."""

    seed_file = os.path.abspath(seed_file)

    d = os.path.dirname(seed_file)

    base = os.path.basename(seed_file)



    # Determine the pattern from the seed file

    seed_xxxx, seed_yyyy, is_pol = _parse_xxxx_yyyy_from_path(base)

    

    if seed_xxxx is None: # Not a parsable file

        return [seed_file], base, [], [], False



    cands = []

    

    # Find a common prefix/pattern based on the seed file type

    if is_pol:

        m_seed = POLARIZATION_IDX_RE.match(base)

        if not m_seed: return [seed_file], base, [], [], False

        

        seed_prefix = m_seed.group("prefix")

        seed_middle = m_seed.group("middle")

        seed_ext = m_seed.group("ext")

        pattern_hint = f"{seed_prefix}_{{pol_type}}{seed_middle}_{{yyyy}}{seed_ext}"

        

        for fn in os.listdir(d):

            m = POLARIZATION_IDX_RE.match(fn)

            if m and m.group("prefix") == seed_prefix and m.group("middle") == seed_middle and m.group("ext") == seed_ext:

                cands.append((os.path.join(d, fn), m.group("pol_type"), m.group("yyyy")))

        

        cands.sort(key=lambda t: (t[2], t[1]))

        files = [p for p, _, _ in cands]

        unique_xxxx = sorted(list({c[1] for c in cands}))

        unique_yyyy = sorted(list({c[2] for c in cands}))

        return files, pattern_hint, unique_xxxx, unique_yyyy, True



    else: # Normal numeric merge

        m_seed = NORMAL_IDX_TWO_RE.match(base) or NORMAL_IDX_ONE_RE.match(base)

        if not m_seed: return [seed_file], base, [], [], False



        seed_prefix = m_seed.group("prefix")

        seed_ext = m_seed.group("ext")

        

        # Check both one and two-index files

        for fn in os.listdir(d):

            m2 = NORMAL_IDX_TWO_RE.match(fn)

            if m2 and m2.group("prefix") == seed_prefix and m2.group("ext") == seed_ext:

                cands.append((os.path.join(d, fn), m2.group("xxxx"), m2.group("yyyy")))

                continue



            m1 = NORMAL_IDX_ONE_RE.match(fn)

            if m1 and m1.group("prefix") == seed_prefix and m1.group("ext") == seed_ext:

                 cands.append((os.path.join(d, fn), m1.group("xxxx"), None))

        

        if m_seed.groupdict().get("yyyy"):

             pattern_hint = f"{seed_prefix}_xxxx_yyyy{seed_ext}"

        else:

             pattern_hint = f"{seed_prefix}_xxxx{seed_ext}"



        cands.sort(key=lambda t: (t[1], t[2] or ""))

        files = [p for p, _, _ in cands]

        unique_xxxx = sorted(list({c[1] for c in cands}))

        unique_yyyy = sorted(list({c[2] for c in cands if c[2] is not None}))

        return files, pattern_hint, unique_xxxx, unique_yyyy, False

        

    return [seed_file], base, [], [], False



def load_1d_xy(path: str) -> Tuple[np.ndarray, np.ndarray]:

    """Load a 1D spectrum file as (x, y).



    Current prototype assumes a CSV-like file with at least 2 columns.

    Delimiter is inferred by numpy.genfromtxt for common cases.

    NOTE: This is intentionally conservative and will be improved later to

    follow the project's Run loader conventions (data_structure.py).

    """

    arr = np.genfromtxt(path, delimiter=None, dtype=float)

    if arr.ndim == 1:

        raise ValueError(f"File {path} does not look like a 2-column table.")

    if arr.shape[1] < 2:

        raise ValueError(f"File {path} has <2 columns.")



    x = np.asarray(arr[:, 0], float)

    y = np.asarray(arr[:, 1], float)

    m = np.isfinite(x) & np.isfinite(y)

    x = x[m]

    y = y[m]

    if x.size < 2:

        raise ValueError(f"File {path} has too few finite points.")

    return x, y







# Helper: parse angle from filename using step_deg and xxxx, else fallback to default

def _angle_deg_from_filename(path: str, xxxx: Optional[str], default: float) -> Union[float, str]:

    base = os.path.basename(path)

    if xxxx in POLARIZATION_TYPES:

        return xxxx

    if xxxx is not None:

        try:

            xxxx_int = int(xxxx)

        except Exception:

            xxxx_int = None

    else:

        xxxx_int = None

    mstep = re.search(r"(-?\d+(?:\.\d+)?)deg", base)

    if mstep and xxxx_int is not None:

        try:

            step_deg = abs(float(mstep.group(1)))

            return float(step_deg * (xxxx_int - 1))

        except Exception:

            return float(default)

    return float(default)





def build_raw_matrix_wavelength_axis(

    files: Sequence[str],

    *,

    assume_xgrid_consistent: bool = True,

) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    """Build (wavelength_nm, raw_angle_values, raw_intensity_matrix) for all raw files.

    - angle derived from filename step_deg and xxxx (see _angle_deg_from_filename).

    - Returns (wavelength_nm, raw_angle_values, raw_intensity_matrix) where:

        wavelength_nm: (nx,)

        raw_angle_values: (n_raw,)

        raw_intensity_matrix: (n_raw, nx)

    """

    xs: Optional[np.ndarray] = None

    rows: List[np.ndarray] = []

    angles: List[float] = []

    for idx, p in enumerate(files):

        x, y = load_1d_xy(p)

        if xs is None:

            xs = x

        if assume_xgrid_consistent:

            n = min(xs.size, x.size, y.size)

            if n < xs.size:

                xs = xs[:n]

                # Trim all existing rows to the new minimum size

                rows = [r[:n] for r in rows]

            rows.append(np.asarray(y[:n], float))

        else:

            n = min(xs.size, x.size, y.size)

            if n < xs.size:

                xs = xs[:n]

                rows = [r[:n] for r in rows]

            rows.append(np.asarray(y[:n], float))

        xxxx, yyyy, _ = _parse_xxxx_yyyy_from_path(p)

        ang = _angle_deg_from_filename(p, xxxx, default=idx)

        angles.append(ang)

    if xs is None or not rows:

        raise ValueError("No valid files were loaded.")

    wavelength_nm = np.asarray(xs, float)

    angle_values = np.asarray(angles, float)

    intensity_matrix = np.vstack(rows)

    return wavelength_nm, angle_values, intensity_matrix





# Helper: build primitive matrix averaged over yyyy for each xxxx

def build_primitive_matrix_by_xxxx(

    files: Sequence[str],

    wavelength_nm: np.ndarray,

    raw_angle_values: np.ndarray,

    raw_intensity_matrix: np.ndarray,

    dark_value: float = 0.0,

) -> Tuple[List[str], np.ndarray, np.ndarray, Dict[str, List[int]]]:

    """

    - Parse xxxx, yyyy for each row.

    - Build raw_rows_by_xxxx mapping.

    - Sort primitive_xxxx numerically where possible.

    - For each xxxx group, compute mean over rows to create primitive row.

    - angle for primitive row: mean of raw_angle_values for that group.

    - Returns (primitive_xxxx, primitive_angle_values, primitive_matrix, raw_rows_by_xxxx)

    """

    n_raw = len(files)

    raw_xxxx: List[Optional[str]] = []

    raw_yyyy: List[Optional[str]] = []

    for p in files:

        x, y, _ = _parse_xxxx_yyyy_from_path(p)

        raw_xxxx.append(x)

        raw_yyyy.append(y)

    # Group rows by xxxx

    raw_rows_by_xxxx: Dict[str, List[int]] = {}

    for r, x in enumerate(raw_xxxx):

        key = x if x is not None else f"row{r:04d}"

        raw_rows_by_xxxx.setdefault(key, []).append(r)

    # Sort primitive_xxxx numerically if possible

    def sort_key(xx):

        try:

            return int(xx)

        except Exception:

            return xx

    primitive_xxxx = sorted(raw_rows_by_xxxx.keys(), key=sort_key)

    primitive_angle_values: List[float] = []

    primitive_matrix_rows: List[np.ndarray] = []

    for xx in primitive_xxxx:

        rows = raw_rows_by_xxxx[xx]

        block = raw_intensity_matrix[rows, :]

        mean_row = np.mean(block, axis=0)

        primitive_matrix_rows.append(mean_row)

        mean_angle = float(np.mean(raw_angle_values[rows]))

        primitive_angle_values.append(mean_angle)

    primitive_matrix = np.vstack(primitive_matrix_rows)

    

    if dark_value != 0.0:

        primitive_matrix -= dark_value

        primitive_matrix[primitive_matrix < 0] = 0

        

    primitive_angle_values_arr = np.asarray(primitive_angle_values, float)

    return primitive_xxxx, primitive_angle_values_arr, primitive_matrix, raw_rows_by_xxxx





# ============================================================================

# Physics conversions

# ============================================================================





def raman_shift_cm1_from_wavelength_nm(

    wavelength_nm: np.ndarray,

    laser_nm: float,

) -> np.ndarray:

    """Compute Raman shift (cm^-1) from wavelength axis and laser wavelength.



    Convention: shift = (1/laser - 1/lambda) * 1e7 with nm -> cm^-1.



    Assumes wavelength_nm and laser_nm are in nanometers.

    """



    wl = np.asarray(wavelength_nm, float)

    laser_nm = float(laser_nm)

    if laser_nm <= 0:

        raise ValueError("laser_nm must be > 0")

    if np.any(wl <= 0):

        raise ValueError("wavelength_nm must be > 0")



    # 1/nm -> 1/cm factor is 1e7

    shift = (1.0 / laser_nm - 1.0 / wl) * 1.0e7

    return shift





def ev_from_cm1(shift_cm1: np.ndarray) -> np.ndarray:

    """Convert Raman shift in cm^-1 to energy in eV."""



    # 1 eV ≈ 8065.544005 cm^-1

    return np.asarray(shift_cm1, float) / 8065.544005



def cm1_from_ev(ev: np.ndarray) -> np.ndarray:

    """Convert Raman shift in cm^-1 to energy in eV."""



    # 1 eV ≈ 8065.544005 cm^-1

    return np.asarray(ev, float) * 8065.544005





# ============================================================================

# Cosmic-ray correction (placeholder)

# ============================================================================





def cosmic_detect_evidence(intensity_matrix: np.ndarray) -> Dict[str, object]:

    """Return numeric evidence for cosmic-ray detection.



    Placeholder implementation:

    - reports max value and max-to-median ratio

    - reports whether max exceeds a heuristic threshold (500)



    This will be replaced by the project's real cosmic detection logic.

    """



    I = np.asarray(intensity_matrix, float)

    mx = float(np.nanmax(I))

    med = float(np.nanmedian(I))

    ratio = float(mx / med) if med != 0 else float("inf")

    threshold = 500.0

    exceeds = bool(mx > threshold)

    return {

        "max": mx,

        "median": med,

        "max_over_median": ratio,

        "threshold": threshold,

        "exceeds_threshold": exceeds,

        "note": "placeholder evidence; replace with robust cosmic logic",

    }





# ============================================================================

# Cosmic-ray GUI-Oriented Helpers

# ============================================================================





_PARSE_CACHE: Dict[str, Tuple[Optional[str], Optional[str], bool]] = {}



def _parse_xxxx_yyyy_from_path(path: str) -> Tuple[Optional[str], Optional[str], bool]:

    base = os.path.basename(path)

    if base in _PARSE_CACHE:

        return _PARSE_CACHE[base]



    m_pol = POLARIZATION_IDX_RE.match(base)

    if m_pol:

        xxxx = m_pol.group("pol_type")

        yyyy = m_pol.group("yyyy")

        _PARSE_CACHE[base] = (xxxx, yyyy, True)

        return xxxx, yyyy, True

    

    m_normal = NORMAL_IDX_TWO_RE.match(base)

    if m_normal:

        xxxx = m_normal.group("xxxx")

        yyyy = m_normal.group("yyyy")

        _PARSE_CACHE[base] = (xxxx, yyyy, False)

        return xxxx, yyyy, False



    m_normal = NORMAL_IDX_ONE_RE.match(base)

    if m_normal:

        xxxx = m_normal.group("xxxx")

        yyyy = None

        _PARSE_CACHE[base] = (xxxx, yyyy, False)

        return xxxx, yyyy, False

    

    _PARSE_CACHE[base] = (None, None, False)

    return None, None, False


def _estimate_fwhm_nm(x_nm: np.ndarray, y: np.ndarray, i0: int) -> float:
    """Estimate FWHM around a peak index i0 (simple half-max crossing).

    Returns NaN if it cannot be estimated robustly.
    """

    x_nm = np.asarray(x_nm, float)
    y = np.asarray(y, float)
    n = y.size
    if i0 < 0 or i0 >= n:
        return float("nan")
    y0 = float(y[i0])
    if not np.isfinite(y0):
        return float("nan")

    # local baseline: use median of the row
    base = float(np.nanmedian(y))
    half = base + 0.5 * (y0 - base)

    # search left
    il = i0
    while il > 0 and np.isfinite(y[il]) and float(y[il]) > half:
        il -= 1
    # search right
    ir = i0
    while ir < n - 1 and np.isfinite(y[ir]) and float(y[ir]) > half:
        ir += 1

    if il == i0 or ir == i0:
        return float("nan")

    # Linear interpolation for edge positions
    def interp_edge(i_a: int, i_b: int) -> float:
        # i_a is inside (>half), i_b is outside (<=half) OR vice versa
        ya = float(y[i_a])
        yb = float(y[i_b])
        xa = float(x_nm[i_a])
        xb = float(x_nm[i_b])
        if not (np.isfinite(ya) and np.isfinite(yb) and np.isfinite(xa) and np.isfinite(xb)):
            return float("nan")
        if ya == yb:
            return 0.5 * (xa + xb)
        t = (half - ya) / (yb - ya)
        return xa + t * (xb - xa)

    # Choose neighbors for interpolation
    x_left = interp_edge(il, min(il + 1, n - 1)) if il < i0 else float("nan")
    x_right = interp_edge(ir, max(ir - 1, 0)) if ir > i0 else float("nan")

    if not (np.isfinite(x_left) and np.isfinite(x_right)):
        return float("nan")
    return float(abs(x_right - x_left))


def detect_cosmic_peaks_comparative(
    discover_result: "MergeDiscoverResult",
    *,
    intensity_thresh: float = 1200.0,
    comparison_factor: float = 20.0,
    z_thresh_fallback: float = 8.0,
    dark_value: float = 600.0,
) -> List[CosmicPeak]:
    """Detects cosmic rays using a comparative method.
    1. Finds peaks above `intensity_thresh`.
    2. Compares them to adjacent/repeated measurements.
    3. Falls back to a statistical z-score test if no reference is available.
    """
    # Prefer raw matrix for access to all yyyy repetitions
    use_raw = (
        discover_result.raw_intensity_matrix is not None
        and discover_result.raw_angle_values is not None
        and discover_result.raw_files
        and discover_result.raw_intensity_matrix.shape[0] == len(discover_result.raw_files)
    )

    if use_raw:
        I = np.asarray(discover_result.raw_intensity_matrix, dtype=float).copy()
        I -= dark_value
        I[I < 0] = 0
        angles = discover_result.raw_angle_values
        files = discover_result.raw_files
    else:
        # Fallback to primitive matrix if raw is not available/valid.
        I = np.asarray(discover_result.intensity_matrix, dtype=float).copy()
        I -= dark_value
        I[I < 0] = 0
        angles = discover_result.angle_values
        files = discover_result.files

    x_nm = discover_result.wavelength_nm
    ny, nx = I.shape
    peaks: List[CosmicPeak] = []

    # Structure for reference lookups
    raw_rows_by_xxxx = discover_result.raw_rows_by_xxxx or {}
    primitive_xxxx = discover_result.primitive_xxxx or []
    
    # Ensure primitive_matrix is also dark-subtracted if used as reference
    primitive_matrix = None
    if discover_result.primitive_matrix is not None:
        primitive_matrix = np.asarray(discover_result.primitive_matrix, dtype=float).copy()
        primitive_matrix -= dark_value
        primitive_matrix[primitive_matrix < 0] = 0

    # 1. Find all candidate peaks above the intensity threshold
    for r in range(ny):
        y = I[r, :] # Already dark-subtracted

        # Find local maxima (y[i] > y[i-1] and y[i] >= y[i+1])
        y0 = np.nan_to_num(y, nan=-np.inf)
        is_max = np.zeros(nx, dtype=bool)
        is_max[1:-1] = (y0[1:-1] > y0[:-2]) & (y0[1:-1] >= y0[2:])

        # Filter by intensity threshold
        cand_cols = np.where(is_max & (y > intensity_thresh))[0]

        if cand_cols.size == 0:
            continue

        row_xxxx, row_yyyy, _ = _parse_xxxx_yyyy_from_path(files[r])

        for c in cand_cols:
            peak_intensity = y[c]
            test_results: Dict[str, Union[str, float, bool]] = {}

            # 2. Find reference intensity for comparison
            reference_intensity = np.nan
            reference_type = "none"

            # Priority 1: Same xxxx, different yyyy (repetitions) from raw data
            if use_raw and row_xxxx and row_xxxx in raw_rows_by_xxxx:
                rep_rows = [rr for rr in raw_rows_by_xxxx.get(row_xxxx, []) if rr != r]
                if rep_rows:
                    reference_intensity = np.nanmean(I[rep_rows, c]) # Already dark-subtracted
                    reference_type = "repetition"

            # Priority 2 & 3: Neighboring xxxx from primitive matrix
            if not np.isfinite(reference_intensity) and row_xxxx and primitive_matrix is not None:
                try:
                    prim_idx = primitive_xxxx.index(row_xxxx)
                    
                    # Try with neighbors at distance 1
                    neighbor_prim_indices = []
                    if prim_idx > 0: neighbor_prim_indices.append(prim_idx - 1)
                    if prim_idx < len(primitive_xxxx) - 1: neighbor_prim_indices.append(prim_idx + 1)

                    if neighbor_prim_indices:
                        reference_intensity = np.nanmean(primitive_matrix[neighbor_prim_indices, c]) # Already dark-subtracted
                        reference_type = "neighbor"

                    # If still no good reference, try neighbors at distance 2
                    if not np.isfinite(reference_intensity) or reference_intensity <= 0:
                        neighbor_prim_indices_2 = []
                        if prim_idx > 1: neighbor_prim_indices_2.append(prim_idx - 2)
                        if prim_idx < len(primitive_xxxx) - 2: neighbor_prim_indices_2.append(prim_idx + 2)
                        
                        if neighbor_prim_indices_2:
                            reference_intensity = np.nanmean(primitive_matrix[neighbor_prim_indices_2, c]) # Already dark-subtracted
                            reference_type = "neighbor_dist_2"

                except (ValueError, IndexError):
                    pass
            
            if reference_intensity <= 0:
                reference_intensity = 0.1 # prevent division by zero
            # 3. Filter the peak and set the confirmation flag
            is_cosmic = False
            if np.isfinite(reference_intensity) and reference_intensity > 0:
                ratio = peak_intensity / reference_intensity
                is_cosmic = ratio > comparison_factor
                test_results = {
                    "method": "comparative",
                    "reference_type": reference_type,
                    "reference_intensity": float(reference_intensity),
                    "peak_intensity": float(peak_intensity),
                    "comparison_factor": comparison_factor,
                    "ratio": ratio,
                    "failed": is_cosmic,
                }
            else:
                # Priority 4: Fallback to statistical z-score test
                med = float(np.nanmedian(y))
                mad = float(np.nanmedian(np.abs(y - med)))
                sigma = 1.4826 * mad
                z = float("inf")
                if sigma > 0:
                    z = (peak_intensity - med) / sigma
                
                is_cosmic = z > z_thresh_fallback
                test_results = {
                    "method": "z-score",
                    "z_score": z,
                    "z_threshold": z_thresh_fallback,
                    "median": med,
                    "sigma": sigma,
                    "failed": is_cosmic,
                }

            # 4. Create a CosmicPeak object for EVERY candidate > thresh
            fwhm = _estimate_fwhm_nm(x_nm, y, int(c))
            peaks.append(
                CosmicPeak(
                    row_index=int(r),
                    col_index=int(c),
                    is_confirmed_cosmic=is_cosmic,
                    xxxx=row_xxxx,
                    yyyy=row_yyyy,
                    angle_deg=float(angles[r]) if np.isfinite(angles[r]) else None,
                    center_wavelength_nm=float(x_nm[c]),
                    intensity=float(peak_intensity),
                    fwhm_nm=float(fwhm),
                    test_results=test_results,
                )
            )
    return peaks


def _apply_cosmic_removal_logic(
    discover_result: "MergeDiscoverResult",
    peaks: Sequence[CosmicPeak],
    remove_mask: Sequence[bool],
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Core logic for applying a placeholder cosmic removal by replacing peak neighborhoods.

    This function extracts the common logic for modifying the intensity matrix
    and is used internally by `apply_cosmic_removal`.

    Replacement rule (as requested):
    - For a peak at (row r, center wavelength), compute a wavelength window
      [center - 5.5*FWHM, center + 5.5*FWHM].
    - Replace values column-wise within that window.
    - Preferred donor rows: same xxxx, different yyyy.
    - Fallback if only one yyyy exists for that xxxx: use rows from xxxx-1 and xxxx+1
      (nearest neighbors in xxxx), averaging donors column-wise.

    Returns (corrected_matrix, summary).
    """

    # 1) Choose working matrix and file list
    use_raw = (
        getattr(discover_result, "raw_intensity_matrix", None) is not None
        and getattr(discover_result, "raw_files", None)
        and len(getattr(discover_result, "raw_files", [])) > 0
        and getattr(discover_result, "raw_intensity_matrix", None).shape[0] == len(getattr(discover_result, "raw_files", []))
    )
    if use_raw:
        I = np.asarray(discover_result.raw_intensity_matrix, float)
        files = list(discover_result.raw_files)
        applied_on = "raw"
    else:
        I = np.asarray(discover_result.intensity_matrix, float)
        files = list(discover_result.files)
        applied_on = "legacy"

    x = np.asarray(discover_result.wavelength_nm, float)

    ny, nx = I.shape

    # 4) Build row_meta and rows_by_xxxx from selected files
    row_meta: List[Tuple[Optional[str], Optional[str], bool]] = [
        _parse_xxxx_yyyy_from_path(p) for p in files
    ]
    rows_by_xxxx: Dict[str, List[int]] = {}
    for r, (xxxx, yyyy, _) in enumerate(row_meta):
        if xxxx is None:
            continue
        rows_by_xxxx.setdefault(xxxx, []).append(r)

    # Also keep sorted unique xxxx for neighbor lookup
    unique_xxxx = sorted(rows_by_xxxx.keys())

    I2 = I.copy()
    applied = 0

    for pk, do_remove in zip(peaks, remove_mask):
        if not do_remove:
            continue
        r = int(pk.row_index)
        c0 = int(pk.col_index)
        if r < 0 or r >= ny or c0 < 0 or c0 >= nx:
            continue

        center = float(pk.center_wavelength_nm)
        fwhm = float(pk.fwhm_nm)
        if not np.isfinite(center) or not np.isfinite(fwhm) or fwhm <= 0:
            # If FWHM is not available, fall back to a small fixed window (3 points)
            lo = max(c0 - 1, 0)
            hi = min(c0 + 2, nx)
            cols = np.arange(lo, hi)
        else:
            lo_nm = center - 5.5 * fwhm
            hi_nm = center + 5.5 * fwhm
            cols = np.where((x >= lo_nm) & (x <= hi_nm))[0]
            if cols.size == 0:
                cols = np.array([c0], dtype=int)

        # Determine donor rows
        xxxx = pk.xxxx
        yyyy = pk.yyyy

        donors: List[int] = []
        if xxxx is not None and xxxx in rows_by_xxxx:
            # same xxxx, different yyyy
            for rr in rows_by_xxxx[xxxx]:
                if rr == r:
                    continue
                _, y2, _ = row_meta[rr]
                if yyyy is None or y2 != yyyy:
                    donors.append(rr)

        if not donors:
            # fallback: neighbor xxxx-1 and xxxx+1
            if xxxx is not None and xxxx in unique_xxxx:
                idx = unique_xxxx.index(xxxx)
                neigh = []
                if idx - 1 >= 0:
                    neigh.append(unique_xxxx[idx - 1])
                if idx + 1 < len(unique_xxxx):
                    neigh.append(unique_xxxx[idx + 1])
                for xx in neigh:
                    donors.extend(rows_by_xxxx.get(xx, []))

        if not donors:
            # no donors available; skip
            continue

        # Column-wise replacement: average donors at those wavelengths
        donor_block = I[np.array(donors)[:, None], cols[None, :]]  # shape (nd, ncols)
        repl = np.nanmean(donor_block, axis=0)

        # Apply replacement for each selected wavelength column
        I2[r, cols] = repl
        applied += 1

    summary = {
        "num_candidates": int(len(peaks)),
        "num_removed": int(np.sum(np.asarray(remove_mask, dtype=bool))),
        "num_applied": int(applied),
        "note": "placeholder replacement; replace with robust cosmic logic later",
        "applied_on": applied_on,
    }
    return I2, summary

def apply_cosmic_removal(
    cosmic_result: MergeCosmicResult,
    remove_mask: Sequence[bool],
) -> MergeDiscoverResult:
    """Apply cosmic removals selected by GUI and return an updated MergeDiscoverResult.
    """
    discover_result = cosmic_result # MergeCosmicResult inherits from MergeDiscoverResult
    peaks = cosmic_result.peaks

    I2, summary = _apply_cosmic_removal_logic(
        discover_result,
        peaks,
        remove_mask,
    )

    # If I2 is RAW (multiple repetitions per angle), we MUST average it
    # down to a primitive matrix to maintain dimension consistency with angle_values.
    is_raw = (
        getattr(discover_result, "raw_intensity_matrix", None) is not None
        and I2.shape[0] == len(getattr(discover_result, "raw_files", []))
        and len(getattr(discover_result, "raw_files", [])) > len(discover_result.unique_xxxx)
    )

    if is_raw:
        # Re-build primitive matrix from the corrected RAW matrix I2
        raw_files = discover_result.raw_files
        raw_angles = discover_result.raw_angle_values
        _, primitive_angle_values_new, primitive_matrix, _ = build_primitive_matrix_by_xxxx(
            raw_files,
            discover_result.wavelength_nm,
            raw_angles,
            I2,
            dark_value=0.0 # dark_value already subtracted or handled
        )
        intensity_matrix_out = primitive_matrix
        angle_values_out = primitive_angle_values_new
    else:
        intensity_matrix_out = I2
        angle_values_out = discover_result.angle_values

    # Construct and return a new MergeDiscoverResult
    return MergeDiscoverResult(
        files=discover_result.files,
        pattern_hint=discover_result.pattern_hint,
        unique_xxxx=discover_result.unique_xxxx,
        unique_yyyy=discover_result.unique_yyyy,
        wavelength_nm=discover_result.wavelength_nm,
        angle_values=angle_values_out,
        # Use the corrected (and potentially averaged) intensity matrix
        intensity_matrix=intensity_matrix_out,
        title=f"{discover_result.title} (cosmic removed)",
        candidate_laser_nm=discover_result.candidate_laser_nm,
        raw_files=discover_result.raw_files,
        raw_xxxx=discover_result.raw_xxxx,
        raw_yyyy=discover_result.raw_yyyy,
        raw_angle_values=discover_result.raw_angle_values,
        raw_intensity_matrix=I2, # Keep the corrected RAW matrix here
        raw_rows_by_xxxx=discover_result.raw_rows_by_xxxx,
        primitive_xxxx=discover_result.primitive_xxxx,
        primitive_matrix=intensity_matrix_out,
        cosmic_matrix=I2, # Store corrected RAW matrix here if raw, else primitive
    )


# ============================================================================
# 2D Map Fitting Logic
# ============================================================================

from scipy.optimize import curve_fit

def _deg2rad(x): return np.deg2rad(x)
def _abs2(x): return np.abs(x)**2

class SelectionRules:
    @staticmethod
    def D2h_Ag(theta, config, a, b, phi):
        th = _deg2rad(theta - phi)
        if config == 'parallel':
            return _abs2(a * np.cos(th)**2 + b * np.sin(th)**2)
        else:
            return _abs2(0.5 * (a - b) * np.sin(2 * th))

    @staticmethod
    def D2h_B1g(theta, config, d, phi):
        th = _deg2rad(theta - phi)
        if config == 'parallel':
            return _abs2(d * np.sin(2 * th))
        else:
            return _abs2(d * np.cos(2 * th))

    @staticmethod
    def D6h_A1g(theta, config, a):
        if config == 'parallel': return np.full_like(theta, a**2)
        return np.zeros_like(theta)

    @staticmethod
    def D6h_E2g(theta, config, d):
        return np.full_like(theta, d**2)

    @staticmethod
    def Linear_Background(theta, offset, slope):
        return offset + slope * theta

RULE_METADATA = {
    "D2h_Ag":  {"func": SelectionRules.D2h_Ag,  "params": ["a", "b", "phi"]},
    "D2h_B1g": {"func": SelectionRules.D2h_B1g, "params": ["d", "phi"]},
    "D6h_A1g": {"func": SelectionRules.D6h_A1g, "params": ["a"]},
    "D6h_E2g": {"func": SelectionRules.D6h_E2g, "params": ["d"]},
}

def lorentzian_normalized(x, x0, gamma):
    g = np.abs(gamma) + 1e-9
    return (1 / np.pi) * (g / ((x - x0)**2 + g**2))

class MapFittingEngine:
    def __init__(self):
        self.datasets = [
            {"label": "Parallel (XX)", "config": "parallel", "x": None, "ang": None, "z": None, "nickname": "XX"},
            {"label": "Cross (YX)",    "config": "cross",    "x": None, "ang": None, "z": None, "nickname": "YX"}
        ]
        self.peaks = [] 
        self.bg_params = {
            "offset": [0.0, -np.inf, np.inf],
            "slope":  [0.0, -np.inf, np.inf]
        }
        self.x_min_limit = -np.inf
        self.x_max_limit = np.inf
        
    def set_data(self, index, x, ang, z, nickname="Run"):
        self.datasets[index]["x"] = x
        self.datasets[index]["ang"] = ang
        self.datasets[index]["z"] = z
        self.datasets[index]["nickname"] = nickname
        
        if index == 0:
            self.x_min_limit = x.min()
            self.x_max_limit = x.max()

    def add_peak(self, name="New Peak", rule_name="D2h_B1g", center=None):
        if center is None:
            center = 300.0
            if self.datasets[0]["x"] is not None:
                center = np.median(self.datasets[0]["x"])
            
        spec_params = { "x0": [center, 0.0, 5000.0], "gamma": [2.0, 0.1, 50.0] }
        ang_params = {}
        for p in RULE_METADATA[rule_name]["params"]:
            ang_params[p] = [10.0, -np.inf, np.inf]
            if p == "phi": ang_params[p] = [0.0, -180, 180]

        self.peaks.append({ "name": name, "rule": rule_name, "spec_params": spec_params, "ang_params": ang_params })

    def get_mask(self, dataset_index):
        ds = self.datasets[dataset_index]
        if ds["x"] is None: return None
        return (ds["x"] >= self.x_min_limit) & (ds["x"] <= self.x_max_limit)

    def reconstruct(self, dataset_index):
        ds = self.datasets[dataset_index]
        if ds["z"] is None: return np.zeros((10, 10))
        XX, YY = np.meshgrid(ds["x"], ds["ang"])
        flat_params = self.flatten_params(which_val=0)
        z_flat = self._calc_single_config(XX.ravel(), YY.ravel(), ds["config"], flat_params)
        return z_flat.reshape(ds["z"].shape)

    def _calc_single_config(self, x_flat, theta_flat, config_mode, params):
        bg_off, bg_slope = params[0], params[1]
        intensity = SelectionRules.Linear_Background(theta_flat, bg_off, bg_slope)
        idx = 2
        for peak in self.peaks:
            x0, gamma = params[idx], params[idx+1]
            idx += 2
            rule_def = RULE_METADATA[peak["rule"]]
            n_ang = len(rule_def["params"])
            ang_p = params[idx : idx+n_ang]
            idx += n_ang
            I_val = rule_def["func"](theta_flat, config_mode, *ang_p)
            area = I_val / (np.abs(gamma) + 1e-9)
            intensity += area * lorentzian_normalized(x_flat, x0, gamma)
        return intensity

    def _joint_model_func(self, xy_tuple, *params):
        x1, th1, x2, th2 = xy_tuple
        z1 = self._calc_single_config(x1, th1, "parallel", params)
        z2 = self._calc_single_config(x2, th2, "cross", params)
        return np.concatenate([z1, z2])

    def flatten_params(self, which_val=0): 
        p = [self.bg_params["offset"] [which_val], self.bg_params["slope"] [which_val]]
        for peak in self.peaks:
            p.extend([peak["spec_params"]["x0"] [which_val], peak["spec_params"]["gamma"] [which_val]])
            for name in RULE_METADATA[peak["rule"]]["params"]:
                p.append(peak["ang_params"] [name] [which_val])
        return p

    def update_params_from_fit(self, popt):
        idx = 0
        self.bg_params["offset"] [0], self.bg_params["slope"] [0] = popt[idx], popt[idx+1]
        idx = 2
        for peak in self.peaks:
            peak["spec_params"]["x0"] [0], peak["spec_params"]["gamma"] [0] = popt[idx], popt[idx+1]
            idx += 2
            for name in RULE_METADATA[peak["rule"]]["params"]:
                peak["ang_params"] [name] [0] = popt[idx]; idx += 1

    def run_optimization(self):
        active = [
            (idx, ds)
            for idx, ds in enumerate(self.datasets)
            if ds["z"] is not None and ds["x"] is not None and ds["ang"] is not None
        ]
        if not active:
            return False, "Data missing"

        p0, lower, upper = self.flatten_params(0), self.flatten_params(1), self.flatten_params(2)

        try:
            if len(active) == 1:
                ds_idx, ds = active[0]
                XX, YY = np.meshgrid(ds["x"], ds["ang"])
                mask = np.tile(self.get_mask(ds_idx), (len(ds["ang"]), 1)).ravel()
                x_fit = XX.ravel()[mask]
                th_fit = YY.ravel()[mask]
                z_fit = ds["z"].ravel()[mask]
                if len(z_fit) == 0:
                    return False, "No data points in range."

                def single_model(xy_tuple, *params):
                    x_vals, th_vals = xy_tuple
                    return self._calc_single_config(x_vals, th_vals, ds["config"], params)

                popt, _ = curve_fit(
                    single_model,
                    (x_fit, th_fit),
                    z_fit,
                    p0=p0,
                    bounds=(lower, upper),
                    maxfev=5000,
                )
            else:
                d1 = self.datasets[0]
                XX1, YY1 = np.meshgrid(d1["x"], d1["ang"])
                mask1 = np.tile(self.get_mask(0), (len(d1["ang"]), 1)).ravel()
                x1_fit = XX1.ravel()[mask1]
                th1_fit = YY1.ravel()[mask1]
                z1_fit = d1["z"].ravel()[mask1]

                d2 = self.datasets[1]
                XX2, YY2 = np.meshgrid(d2["x"], d2["ang"])
                mask2 = np.tile(self.get_mask(1), (len(d2["ang"]), 1)).ravel()
                x2_fit = XX2.ravel()[mask2]
                th2_fit = YY2.ravel()[mask2]
                z2_fit = d2["z"].ravel()[mask2]

                if len(z1_fit) == 0 or len(z2_fit) == 0:
                    return False, "No data points in range for one or both datasets."

                z_combined = np.concatenate([z1_fit, z2_fit])
                popt, _ = curve_fit(
                    self._joint_model_func,
                    (x1_fit, th1_fit, x2_fit, th2_fit),
                    z_combined,
                    p0=p0,
                    bounds=(lower, upper),
                    maxfev=5000,
                )
            self.update_params_from_fit(popt)
            return True, "Success"
        except Exception as e:
            return False, str(e)

    def validate_row_by_row(self):
        """Perform row-by-row fitting and return results."""
        if not any(ds["z"] is not None for ds in self.datasets):
            return False, "No Data", None
        
        validation_results = []
        
        for ds_idx, ds in enumerate(self.datasets):
            if ds["z"] is None or ds["x"] is None or ds["ang"] is None:
                continue
            rows_params = []
            rec_matrix = []
            
            config_mode = ds["config"]
            x_full = ds["x"]
            mask = self.get_mask(ds_idx)
            x_fit = x_full[mask]
            
            headers = ["Angle", "BG_Const"]
            for i, p in enumerate(self.peaks):
                headers.extend([f"P{i+1}_Area", f"P{i+1}_Gamma", f"P{i+1}_Height"])
            
            for r in range(len(ds["ang"])):
                angle = ds["ang"][r]
                z_row = ds["z"][r, :]
                z_fit_data = z_row[mask]
                
                if len(z_fit_data) == 0:
                    rows_params.append([angle, np.nan] + [np.nan]*(3*len(self.peaks)))
                    rec_matrix.append(np.zeros_like(x_full))
                    continue

                bg_guess = self.bg_params["offset"][0] + self.bg_params["slope"][0] * angle
                p0_row = [bg_guess]
                bounds_low = [-np.inf]
                bounds_high = [np.inf]
                
                for peak in self.peaks:
                    g_guess = peak["spec_params"]["gamma"][0]
                    g_min = peak["spec_params"]["gamma"][1]
                    g_max = peak["spec_params"]["gamma"][2]
                    
                    rule_def = RULE_METADATA[peak["rule"]]
                    ang_p = [peak["ang_params"][pn][0] for pn in rule_def["params"]]
                    I_val = rule_def["func"](np.array([angle]), config_mode, *ang_p)[0]
                    area_guess = I_val / (np.abs(g_guess) + 1e-9)
                    
                    p0_row.extend([area_guess, g_guess])
                    bounds_low.extend([0, g_min])
                    bounds_high.extend([np.inf, g_max])
                
                def fit_func(x, *p):
                    y = np.full_like(x, p[0])
                    idx = 1
                    for k in range(len(self.peaks)):
                        area, gamma = p[idx], p[idx+1]
                        idx += 2
                        x0 = self.peaks[k]["spec_params"]["x0"][0] 
                        y += area * lorentzian_normalized(x, x0, gamma)
                    return y

                try:
                    popt, _ = curve_fit(fit_func, x_fit, z_fit_data, p0=p0_row, bounds=(bounds_low, bounds_high), maxfev=2000)
                    row_res = [angle, popt[0]]
                    idx = 1
                    for k in range(len(self.peaks)):
                        area, gamma = popt[idx], popt[idx+1]
                        idx += 2
                        height = area * (1.0 / (np.pi * gamma))
                        row_res.extend([area, gamma, height])
                    rows_params.append(row_res)
                    z_rec_row = fit_func(x_full, *popt)
                    rec_matrix.append(z_rec_row)
                except Exception as e:
                    row_res = [angle, np.nan] + [np.nan]*(3*len(self.peaks))
                    rows_params.append(row_res)
                    rec_matrix.append(np.zeros_like(x_full))

            validation_results.append({
                "label": ds["label"],
                "x": ds["x"],
                "ang": ds["ang"],
                "z_raw": ds["z"],
                "z_rec": np.array(rec_matrix),
                "params": rows_params,
                "headers": headers
            })
            
        return True, "Validation Complete", validation_results

    def get_peak_reconstructions(self):
        """Generate matrices for individual peaks and background."""
        results = []
        flat_params = self.flatten_params(which_val=0) 

        for ds_idx, ds in enumerate(self.datasets):
            if ds["z"] is None or ds["x"] is None or ds["ang"] is None:
                continue
                
            XX, YY = np.meshgrid(ds["x"], ds["ang"])
            x_flat = XX.ravel()
            theta_flat = YY.ravel()
            
            # Background
            bg_off, bg_slope = flat_params[0], flat_params[1]
            z_bg_flat = SelectionRules.Linear_Background(theta_flat, bg_off, bg_slope)
            z_bg = z_bg_flat.reshape(ds["z"].shape)
            
            results.append({
                "dataset_idx": ds_idx,
                "name": "Background",
                "matrix": z_bg
            })

            current_idx = 2
            for i, peak in enumerate(self.peaks):
                x0 = flat_params[current_idx]
                gamma = flat_params[current_idx+1]
                current_idx += 2
                
                rule_def = RULE_METADATA[peak["rule"]]
                n_ang = len(rule_def["params"])
                ang_p = flat_params[current_idx : current_idx+n_ang]
                current_idx += n_ang
                
                I_val = rule_def["func"](theta_flat, ds["config"], *ang_p)
                area = I_val / (np.abs(gamma) + 1e-9)
                z_peak_flat = area * lorentzian_normalized(x_flat, x0, gamma)
                z_peak = z_peak_flat.reshape(ds["z"].shape)
                
                results.append({
                    "dataset_idx": ds_idx,
                    "name": peak["name"],
                    "matrix": z_peak
                })
        
        return results

    def to_dict(self):
        """Return engine state as a dictionary."""
        return {
            "peaks": self.peaks,
            "bg_params": self.bg_params,
            "x_min_limit": float(self.x_min_limit) if np.isfinite(self.x_min_limit) else -1e9,
            "x_max_limit": float(self.x_max_limit) if np.isfinite(self.x_max_limit) else 1e9
        }

    def from_dict(self, data):
        """Load engine state from a dictionary."""
        self.peaks = data.get("peaks", [])
        self.bg_params = data.get("bg_params", self.bg_params)
        self.x_min_limit = data.get("x_min_limit", -np.inf)
        self.x_max_limit = data.get("x_max_limit", np.inf)

    def get_peak_polar_areas(self, dataset_index):
        """Return a 2D map of peak areas (angle vs peak)."""
        ds = self.datasets[dataset_index]
        if ds["ang"] is None or ds["z"] is None:
            return None
        
        n_peaks = len(self.peaks)
        if n_peaks == 0: return np.zeros((len(ds["ang"]), 0))
        
        n_angles = len(ds["ang"])
        areas = np.zeros((n_angles, n_peaks))
        
        flat_params = self.flatten_params(which_val=0)
        idx = 2
        for p_idx, peak in enumerate(self.peaks):
            gamma = flat_params[idx+1]
            idx += 2
            rule_def = RULE_METADATA[peak["rule"]]
            n_ang = len(rule_def["params"])
            ang_p = flat_params[idx : idx+n_ang]
            idx += n_ang
            
            I_val = rule_def["func"](ds["ang"], ds["config"], *ang_p)
            areas[:, p_idx] = I_val / (np.abs(gamma) + 1e-9)
            
        return areas

    def export_parameters_text(self):
        """Return a string summary of parameters."""
        lines = []
        lines.append(f"# Joint Fit Export")
        lines.append(f"# Range: {self.x_min_limit} - {self.x_max_limit}")
        lines.append(f"[Background]")
        lines.append(f"Offset: {self.bg_params['offset'][0]} [{self.bg_params['offset'][1]}, {self.bg_params['offset'][2]}]")
        lines.append(f"Slope: {self.bg_params['slope'][0]} [{self.bg_params['slope'][1]}, {self.bg_params['slope'][2]}]")
        lines.append("")
        for i, p in enumerate(self.peaks):
            lines.append(f"[Peak {i+1}: {p['name']}]")
            lines.append(f"Rule: {p['rule']}")
            lines.append(f"Center (x0): {p['spec_params']['x0'][0]} [{p['spec_params']['x0'][1]}, {p['spec_params']['x0'][2]}]")
            lines.append(f"Width (Gamma): {p['spec_params']['gamma'][0]} [{p['spec_params']['gamma'][1]}, {p['spec_params']['gamma'][2]}]")
            for k, v in p['ang_params'].items(): 
                lines.append(f"{k}: {v[0]} [{v[1]}, {v[2]}]")
            lines.append("")
        return "\n".join(lines)

    def import_parameters_text(self, text):
        """Parse a string summary of parameters and update engine state."""
        self.peaks = []
        
        current_peak = None
        
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"): continue
            
            if line.startswith("[Background]"):
                current_peak = "BG"
                continue
            
            m_peak = re.match(r"\[Peak \d+: (.*)\]", line)
            if m_peak:
                name = m_peak.group(1)
                current_peak = {"name": name, "spec_params": {}, "ang_params": {}}
                self.peaks.append(current_peak)
                continue
            
            if current_peak == "BG":
                m_val = re.match(r"(Offset|Slope): ([\d.-e]+) \[([\d.-e]+), ([\d.-e]+)\]", line)
                if m_val:
                    key = m_val.group(1).lower()
                    self.bg_params[key] = [float(m_val.group(2)), float(m_val.group(3)), float(m_val.group(4))]
            elif isinstance(current_peak, dict):
                if line.startswith("Rule: "):
                    current_peak["rule"] = line[len("Rule: "):]
                else:
                    m_val = re.match(r"(.*): ([\d.-e]+) \[([\d.-e]+), ([\d.-e]+)\]", line)
                    if m_val:
                        key = m_val.group(1)
                        vals = [float(m_val.group(2)), float(m_val.group(3)), float(m_val.group(4))]
                        if key == "Center (x0)": current_peak["spec_params"]["x0"] = vals
                        elif key == "Width (Gamma)": current_peak["spec_params"]["gamma"] = vals
                        else: current_peak["ang_params"][key] = vals


# ============================================================================
# Stage-1 and Stage-2 API
# ============================================================================


def discover_merge(options: MergeDiscoverOptions) -> MergeDiscoverResult:
    files, pattern_hint, unique_xxxx, unique_yyyy, is_polarization_merge = detect_similar_files(options.seed_file)
    wavelength_nm, raw_angle_values, raw_intensity_matrix = build_raw_matrix_wavelength_axis(
        files, assume_xgrid_consistent=options.assume_xgrid_consistent
    )
    # Parse raw_xxxx, raw_yyyy
    raw_xxxx = []
    raw_yyyy = []
    for p in files:
        x, y, _ = _parse_xxxx_yyyy_from_path(p)
        raw_xxxx.append(x)
        raw_yyyy.append(y)
    # Build primitive matrix (averaged over yyyy for each xxxx)
    primitive_xxxx, primitive_angle_values, primitive_matrix, raw_rows_by_xxxx = build_primitive_matrix_by_xxxx(
        files, wavelength_nm, raw_angle_values, raw_intensity_matrix
    )
    # For backward compatibility: angle_values = primitive_angle_values, intensity_matrix = primitive_matrix

    # Parse laser wavelength from seed filename, with a default fallback.
    parsed_laser_nm: Optional[float] = None
    m = re.search(r"(\d+(?:[\.,]\d+)?)nm", os.path.basename(options.seed_file))
    if m:
        try:
            parsed_laser_nm = float(m.group(1).replace(",", "."))
        except Exception:
            parsed_laser_nm = None

    if parsed_laser_nm is None:
        parsed_laser_nm = 515.0

    # Initialize candidate_laser_nm with the parsed value as a fallback.
    candidate_laser_nm = parsed_laser_nm
    
    # Calculate candidate laser nm by finding the peak in a trimmed-mean spectrum
    # around the parsed laser wavelength.
    # Find the indices for the laser domain using searchsorted for efficiency,
    # assuming wavelength_nm is sorted.
    start_index = np.searchsorted(wavelength_nm, parsed_laser_nm - 1.0, side='left')
    end_index = np.searchsorted(wavelength_nm, parsed_laser_nm + 1.0, side='right')

    if start_index < end_index:
        # Slicing is more memory-efficient than boolean masking for large arrays
        domain_wavelengths = wavelength_nm[start_index:end_index]
        domain_intensities = raw_intensity_matrix[:, start_index:end_index]

        # Calculate trimmed mean for each column (wavelength point)
        n_rows = domain_intensities.shape[0]
        trim_count = int(n_rows * 0.1)

        # Sort each column's intensities
        sorted_intensities = np.sort(domain_intensities, axis=0)
        
        # Trim top and bottom 10% to remove cosmic ray influence
        if trim_count > 0:
            trimmed_intensities = sorted_intensities[trim_count:-trim_count, :]
        else:
            trimmed_intensities = sorted_intensities
        
        # Average the trimmed intensities to get a clean spectrum
        averaged_spectrum = np.mean(trimmed_intensities, axis=0)
        
        # Find the most prominent peak in the averaged spectrum
        if averaged_spectrum.size > 0:
            peak_index_in_domain = np.argmax(averaged_spectrum)
            candidate_laser_nm = domain_wavelengths[peak_index_in_domain]

    return MergeDiscoverResult(
        files=files,
        pattern_hint=pattern_hint,
        unique_xxxx=unique_xxxx,
        unique_yyyy=unique_yyyy,
        wavelength_nm=wavelength_nm,
        angle_values=primitive_angle_values,
        intensity_matrix=primitive_matrix,
        title="Preview (raw, wavelength axis)",
        candidate_laser_nm=candidate_laser_nm,
        raw_files=list(files),
        raw_xxxx=raw_xxxx,
        raw_yyyy=raw_yyyy,
        raw_angle_values=raw_angle_values,
        raw_intensity_matrix=raw_intensity_matrix,
        raw_rows_by_xxxx=raw_rows_by_xxxx,
        primitive_xxxx=primitive_xxxx,
        primitive_matrix=primitive_matrix,
        cosmic_matrix=None,
        is_polarization_merge=is_polarization_merge,
    )


def preview_merge(
    discover_result: MergeDiscoverResult,
    options: MergePreviewOptions,
    prompt: Optional[UserPrompt] = None,
) -> MergePreviewResult:
    """Stage-2 preview: compute Raman-shift axis from a given intensity matrix.

    This function is now purely for converting the axis from wavelength to
    Raman shift. It does NOT perform cosmic-ray correction. Any correction
    should be applied to the `discover_result.intensity_matrix` BEFORE
    calling this function.

    Behavior:
    - Laser wavelength: use manual if provided, else try to parse from seed_file.
      If parsing fails:
        - If require_laser_nm is True: raise.
        - Else: set laser_nm=None, return NaN raman_shift_cm1 and energy_ev=No    """

    if options.interactive_confirm and prompt is None:
        # This check is kept for potential future interactive features, but
        # it no longer pertains to cosmic-ray confirmation here.
        raise RuntimeError("interactive_confirm is True but no prompt handler was provided")

    # The input matrix is used directly.
    I2 = np.asarray(discover_result.intensity_matrix, float)

    if options.use_raman_x:
        if options.raman_x_mode == "meV":
            return MergePreviewResult(
            raman_shift_cm1=cm1_from_ev(discover_result.wavelength_nm),
            energy_ev=discover_result.wavelength_nm,
            angle_values=np.asarray(discover_result.angle_values, float),
            intensity_matrix=I2,
            title="Preview (Raman shift axis)",
            )
        elif options.raman_x_mode == "cm-1":
            return MergePreviewResult(
            raman_shift_cm1=discover_result.wavelength_nm,
            energy_ev=ev_from_cm1(discover_result.wavelength_nm),
            angle_values=np.asarray(discover_result.angle_values, float),
            intensity_matrix=I2,
            title="Preview (Raman shift axis)",
            )

    # Determine laser wavelength
    laser_nm: Optional[float] = None
    if options.manual_laser_nm is not None:
        laser_nm = float(options.manual_laser_nm)
    else:
        # Use candidate laser wavelength from discovery stage
        candidate_laser_nm = discover_result.candidate_laser_nm
        if candidate_laser_nm is not None:
            laser_nm = candidate_laser_nm
        else:
            if options.require_laser_nm:
                raise ValueError(
                    "manual_laser_nm not provided and could not parse '<number>nm' from seed filename. "
                    "Set require_laser_nm=False to allow plotting only wavelength axis."
                )
            else:
                laser_nm = None

    if laser_nm is not None:
        shift_cm1 = raman_shift_cm1_from_wavelength_nm(discover_result.wavelength_nm, laser_nm)
        energy = ev_from_cm1(shift_cm1)
    else:
        shift_cm1 = np.full_like(discover_result.wavelength_nm, np.nan)
        energy = None

    return MergePreviewResult(
        raman_shift_cm1=shift_cm1,
        energy_ev=energy,
        angle_values=np.asarray(discover_result.angle_values, float),
        intensity_matrix=I2,
        title="Preview (Raman shift axis)",
    )


# ============================================================================
# GUI-facing Cosmic-ray API
# ============================================================================


def discover_cosmics(
    options: MergeCosmicOptions,
) -> MergeCosmicResult:
    """Detect cosmic rays and return a GUI-friendly result.

    The GUI should render the raw/preview matrix and provide checkboxes.
    The GUI then calls `cosmic_apply_for_gui(...)` with the chosen mask.
    """
    peaks = detect_cosmic_peaks_comparative(
        options,
        intensity_thresh=options.intensity_thresh,
        comparison_factor=options.comparison_factor,
        z_thresh_fallback=options.z_thresh_fallback,
        dark_value=options.dark_value
    )
    # Use the raw matrix for evidence calculation if available, as that is what
    # the detection is primarily run on.
    matrix_for_evidence = (
        options.raw_intensity_matrix
        if options.raw_intensity_matrix is not None
        else options.intensity_matrix
    )
    ev = cosmic_detect_evidence(matrix_for_evidence)

    return MergeCosmicResult(
        files=options.files,
        pattern_hint=options.pattern_hint,
        unique_xxxx=options.unique_xxxx,
        unique_yyyy=options.unique_yyyy,
        wavelength_nm=options.wavelength_nm,
        angle_values=options.angle_values,
        intensity_matrix=options.intensity_matrix,
        title=options.title,
        candidate_laser_nm=options.candidate_laser_nm,
        raw_files=options.raw_files,
        raw_xxxx=options.raw_xxxx,
        raw_yyyy=options.raw_yyyy,
        raw_angle_values=options.raw_angle_values,
        raw_intensity_matrix=options.raw_intensity_matrix,
        raw_rows_by_xxxx=options.raw_rows_by_xxxx,
        primitive_xxxx=options.primitive_xxxx,
        primitive_matrix=options.primitive_matrix,
        cosmic_matrix=options.cosmic_matrix,
        dark_value=options.dark_value,
        intensity_thresh=options.intensity_thresh,
        comparison_factor=options.comparison_factor,
        z_thresh_fallback=options.z_thresh_fallback,
        peaks=peaks,
        evidence=ev,
    )


def cosmic_apply_for_gui(
    discover_result: MergeDiscoverResult,
    cosmic_result: MergeCosmicResult,
    remove_mask: Sequence[bool],
) -> MergePreviewResult:
    """Apply placeholder cosmic removals selected by GUI and return a preview result.

    This returns a MergePreviewResult on the *wavelength axis* intensity update.
    The Raman-shift axis computation remains handled by `preview_merge`.
    In the current pipeline, wx_gui may call this and then call `preview_merge`
    with the corrected matrix embedded into a temporary discover_result.
    """

    I2, summary = _apply_cosmic_removal_logic(
        discover_result,
        cosmic_result.peaks,
        remove_mask,
    )

    # Determine if I2 is RAW or primitive:
    is_raw = (
        getattr(discover_result, "raw_intensity_matrix", None) is not None
        and isinstance(discover_result.raw_intensity_matrix, np.ndarray)
        and discover_result.raw_intensity_matrix.shape[0] == I2.shape[0]
        and getattr(discover_result, "raw_files", None) is not None
        and len(discover_result.raw_files) == I2.shape[0]
    )

    if is_raw:
        # RAW: rebuild primitive matrix by averaging over yyyy for each xxxx
        raw_files = discover_result.raw_files if getattr(discover_result, "raw_files", None) else discover_result.files
        raw_angles = discover_result.raw_angle_values if getattr(discover_result, "raw_angle_values", None) is not None else np.arange(len(raw_files), dtype=float)
        primitive_xxxx, primitive_angle_values, primitive_matrix, raw_rows_by_xxxx = build_primitive_matrix_by_xxxx(
            raw_files,
            discover_result.wavelength_nm,
            raw_angles,
            I2,
        )
        angle_values_out = primitive_angle_values
        matrix_out = primitive_matrix
    else:
        # Already primitive
        angle_values_out = np.asarray(discover_result.angle_values, float)
        matrix_out = I2

    return MergePreviewResult(
        raman_shift_cm1=np.asarray(discover_result.wavelength_nm, float) * np.nan,
        energy_ev=None,
        angle_values=angle_values_out,
        intensity_matrix=matrix_out,
        title="Preview (cosmic-selected, wavelength axis)",
    )


# ============================================================================
# CLI entrypoint (prototype)
# ============================================================================


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Prototype merge backend (analysis.py).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("seed_file", help="Seed 1D run file")
    p.add_argument(
        "--no-cosmic",
        action="store_true",
        help="Disable cosmic correction (equivalent to cosmic_enable=False)",
    )
    p.add_argument(
        "--manual-laser-nm",
        type=float,
        default=None,
        help="Manually set laser wavelength in nm",
    )
    p.add_argument(
        "--laser-peak-plot",
        action="store_true",
        help="Show laser peak diagnostic plot (placeholder)",
    )
    p.add_argument(
        "--no-cosmic-peak-plot",
        action="store_true",
        help="Disable cosmic peak plot (still prints evidence and asks y/n)",
    )

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)

    seed = os.path.abspath(args.seed_file)
    if not os.path.exists(seed):
        print(f"Seed file not found: {seed}")
        return 2

    discover_opts = MergeDiscoverOptions(seed_file=seed, assume_xgrid_consistent=True)
    disc = discover_merge(discover_opts)

    print(f"Detected {len(disc.files)} files")
    print(f"Pattern: {disc.pattern_hint}")
    if disc.unique_xxxx:
        print(f"unique xxxx: {len(disc.unique_xxxx)}")
    if disc.unique_yyyy:
        print(f"unique yyyy: {len(disc.unique_yyyy)}")

    prev_opts = MergePreviewOptions(
        seed_file=seed,
        files=disc.files,
        cosmic_enable=(not args.no_cosmic),
        manual_laser_nm=args.manual_laser_nm,
        interactive_confirm=True,
        require_laser_nm=True,
    )

    prompt = CliPrompt()

    # --- Cosmic ray detection and correction ---
    active_disc_result = disc # This will hold the result, potentially with cosmic corrections
    
    if prev_opts.cosmic_enable:
        prompt.info("\n--- Cosmic Ray Detection ---")
        cosmic_options = MergeCosmicOptions(**asdict(active_disc_result),
                                            dark_value=0.0, # Defaulting, will be set by GUI
                                            intensity_thresh=1200.0, # Defaulting, will be set by GUI
                                            comparison_factor=20.0, # Defaulting, will be set by GUI
                                            z_thresh_fallback=8.0 # Defaulting, will be set by GUI
                                            )
        cosmic_result = discover_cosmics(cosmic_options)
        
        if not cosmic_result.peaks:
            prompt.info("No cosmic ray peaks detected.")
        else:
            prompt.info(f"Detected {len(cosmic_result.peaks)} cosmic ray candidates.")
            prompt.info(f"Evidence: {cosmic_result.evidence}")
            
            remove_mask: List[bool] = []
            confirmed_peaks_indices = [i for i, peak in enumerate(cosmic_result.peaks) if peak.is_confirmed_cosmic]

            if confirmed_peaks_indices:
                for i in confirmed_peaks_indices:
                    peak = cosmic_result.peaks[i]
                    prompt.info(f"  Candidate {i+1}: Angle={peak.angle_deg:.1f} deg, Wavelength={peak.center_wavelength_nm:.3f} nm, Intensity={peak.intensity:.3g}")
                    if prompt.ask_yes_no(f"    Confirm removal of candidate {i+1}?", default=True):
                        remove_mask.append(True)
                    else:
                        remove_mask.append(False)
            
            # Create a full-length remove_mask
            full_remove_mask = [False] * len(cosmic_result.peaks)
            for i, mask_val in zip(confirmed_peaks_indices, remove_mask):
                full_remove_mask[i] = mask_val

            if any(full_remove_mask):
                prompt.info("Applying cosmic ray corrections...")
                # Apply corrections to the active_disc_result.intensity_matrix
                active_disc_result = apply_cosmic_removal(cosmic_result, full_remove_mask)
                
                # Note: cosmic_apply_for_gui returns a MergePreviewResult which has intensity_matrix on wavelength axis
                # We need to transfer this corrected matrix back into a MergeDiscoverResult structure
                # to be consumed by the final preview_merge which calculates Raman shift.
                prompt.info("Cosmic ray corrections applied.")
            else:
                prompt.info("No cosmic ray corrections confirmed by user.")

    prev = preview_merge(active_disc_result, prev_opts, prompt=prompt)

    print("\n--- Final Preview Result ---")
    print(f"  matrix shape: {prev.intensity_matrix.shape}")
    if not np.all(np.isnan(prev.raman_shift_cm1)):
        print(f"  shift range (cm^-1): [{np.nanmin(prev.raman_shift_cm1):.3g}, {np.nanmax(prev.raman_shift_cm1):.3g}]")
    if prev.energy_ev is not None and not np.all(np.isnan(prev.energy_ev)):
        print(f"  energy range (eV): [{np.nanmin(prev.energy_ev*1000):.3g}, {np.nanmax(prev.energy_ev*1000):.3g}]")

    # --- Export to CSV ---
    base = os.path.basename(seed)
    m = POLARIZATION_IDX_RE.match(base)
    if not m:
        m = NORMAL_IDX_RE.match(base)
    if m:
        prefix = m.group("prefix")
    else:
        prefix, _ = os.path.splitext(base)

    dir_path = os.path.dirname(seed)
    output_cm_filename = os.path.join(dir_path, f"{prefix}_cm-1.csv")
    output_mev_filename = os.path.join(dir_path, f"{prefix}_meV.csv")

    # cm-1 export
    if not np.all(np.isnan(prev.raman_shift_cm1)):
        df_cm1 = pd.DataFrame(
            data=prev.intensity_matrix,
            index=prev.angle_values,
            columns=prev.raman_shift_cm1,
        )
        df_cm1.index.name = "Angle (deg)"
        df_cm1.columns.name = "Raman shift (cm-1)"
        df_cm1.to_csv(output_cm_filename)
        prompt.info(f"Exported cm-1 data to {output_cm_filename}")

    # meV export
    if prev.energy_ev is not None and not np.all(np.isnan(prev.energy_ev)):
        energy_mev = prev.energy_ev * 1000
        df_mev = pd.DataFrame(
            data=prev.intensity_matrix,
            index=prev.angle_values,
            columns=energy_mev,
        )
        df_mev.index.name = "Angle (deg)"
        df_mev.columns.name = "Energy (meV)"
        df_mev.to_csv(output_mev_filename)
        prompt.info(f"Exported meV data to {output_mev_filename}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
