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
import copy
import json
import os
import re
import sys
import traceback
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
    intensity_thresh: float = 500.0
    comparison_factor: float = 5.0
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
POLARIZATION_ROW_VALUES = {pol: float(idx) for idx, pol in enumerate(POLARIZATION_TYPES)}


def polarization_family_prefix(prefix: str) -> str:
    """Return the polarization merge prefix after the leading category token.

    Lab polarization sets may use different category/date tokens for each
    polarization, e.g. ``26042414_..._RR`` and ``26042413_..._LL``. The
    measurement identity starts after that first underscore.
    """
    text = str(prefix or "")
    return text.split("_", 1)[1] if "_" in text else text

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
        seed_family_prefix = polarization_family_prefix(seed_prefix)

        seed_middle = m_seed.group("middle")

        seed_ext = m_seed.group("ext")

        pattern_hint = f"*_ {seed_family_prefix}_{{pol_type}}{seed_middle}_{{yyyy}}{seed_ext}".replace("*_ ", "*_")



        for fn in os.listdir(d):

            m = POLARIZATION_IDX_RE.match(fn)

            if (
                m
                and polarization_family_prefix(m.group("prefix")) == seed_family_prefix
                and m.group("middle") == seed_middle
                and m.group("ext") == seed_ext
            ):

                cands.append((os.path.join(d, fn), m.group("pol_type"), m.group("yyyy")))

        found_pols = {c[1] for c in cands}
        cands.sort(
            key=lambda t: (
                t[2],
                POLARIZATION_TYPES.index(t[1]) if t[1] in POLARIZATION_TYPES else len(POLARIZATION_TYPES),
                t[1],
            )
        )

        files = [p for p, _, _ in cands]

        unique_xxxx = [pol for pol in POLARIZATION_TYPES if pol in found_pols]

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

    Accept comma CSV, whitespace TXT, simple header rows, and 2D matrix CSVs
    whose column headers are the spectral x-axis. Matrix input is reduced to a
    representative mean spectrum so merge discovery keeps the real x values.
    """

    def _from_project_loader() -> Optional[Tuple[np.ndarray, np.ndarray]]:
        try:
            from data_structure import load_table

            arrays, _labels = load_table(path)
        except Exception:
            return None
        if len(arrays) == 2:
            return None
        if len(arrays) == 3:
            x_axis, _y_axis, matrix = arrays
            x_arr = np.asarray(x_axis, dtype=float)
            z = np.asarray(matrix, dtype=float)
            if z.ndim != 2 or x_arr.ndim != 1:
                return None
            if z.shape[1] != x_arr.size and z.shape[0] == x_arr.size:
                z = z.T
            if z.shape[1] != x_arr.size:
                return None
            with np.errstate(invalid="ignore"):
                y_arr = np.nanmean(z, axis=0)
            return x_arr, y_arr
        return None

    loaded = _from_project_loader()
    if loaded is not None:
        x, y = loaded
    else:
        arr = None
        for delimiter in (None, ",", "\t", ";"):
            try:
                candidate = np.genfromtxt(path, delimiter=delimiter, dtype=float, comments="#")
            except Exception:
                candidate = None
            if candidate is None or np.size(candidate) == 0:
                continue
            candidate = np.asarray(candidate, dtype=float)
            if np.isfinite(candidate).any():
                arr = candidate
                break

        if arr is None or np.size(arr) == 0 or not np.isfinite(np.asarray(arr, dtype=float)).any():
            frames = []
            for kwargs in (
                {"sep": None, "header": "infer"},
                {"sep": r"\s+", "header": None},
                {"sep": ",", "header": None},
            ):
                try:
                    df = pd.read_csv(path, comment="#", engine="python", **kwargs)
                    frames.append(df)
                except Exception:
                    pass
            arr = None
            for df in frames:
                numeric = df.apply(pd.to_numeric, errors="coerce").dropna(how="all").dropna(axis=1, how="all")
                if numeric.shape[1] < 2:
                    continue
                numeric = numeric.dropna(subset=[numeric.columns[0], numeric.columns[1]])
                if not numeric.empty:
                    arr = numeric.iloc[:, :2].to_numpy(dtype=float)
                    break
            if arr is None:
                raise ValueError(f"File {path} has <2 numeric columns.")

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

        return POLARIZATION_ROW_VALUES.get(xxxx, float(default))

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
    polarization_as_xxxx: bool = True,

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

        xxxx, yyyy, _ = _parse_merge_xxxx_yyyy_from_path(p, polarization_as_xxxx=polarization_as_xxxx)

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
    polarization_as_xxxx: bool = True,

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

        x, y, _ = _parse_merge_xxxx_yyyy_from_path(p, polarization_as_xxxx=polarization_as_xxxx)

        raw_xxxx.append(x)

        raw_yyyy.append(y)

    # Group rows by xxxx

    raw_rows_by_xxxx: Dict[str, List[int]] = {}

    for r, x in enumerate(raw_xxxx):

        key = x if x is not None else f"row{r:04d}"

        raw_rows_by_xxxx.setdefault(key, []).append(r)

    # Sort primitive_xxxx numerically if possible

    def sort_key(xx):
        if xx in POLARIZATION_ROW_VALUES:
            return (0, POLARIZATION_ROW_VALUES[xx])

        try:

            return (1, int(xx))

        except Exception:

            return (2, str(xx))

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


def _parse_merge_xxxx_yyyy_from_path(path: str, *, polarization_as_xxxx: bool = True) -> Tuple[Optional[str], Optional[str], bool]:
    """Parse merge row/repetition indices.

    Multi-polarization merges use ``RR/RL/LR/LL`` as row keys. Single-pol
    angular series such as ``..._RR_300K_0001.txt`` use the trailing numeric
    token as the row key.
    """
    xxxx, yyyy, is_pol = _parse_xxxx_yyyy_from_path(path)
    if is_pol and not polarization_as_xxxx:
        return yyyy, None, False
    return xxxx, yyyy, is_pol


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
        and I2.shape[0] != len(getattr(discover_result, "angle_values", []))
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
            dark_value=0.0, # dark_value already subtracted or handled
            polarization_as_xxxx=bool(getattr(discover_result, "is_polarization_merge", False)),
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
        is_polarization_merge=bool(getattr(discover_result, "is_polarization_merge", False)),
    )


# ============================================================================
# 2D Map Fitting Logic
# ============================================================================

from scipy.optimize import curve_fit
from scipy.interpolate import BSpline

def _deg2rad(x): return np.deg2rad(x)
def _abs2(x):
    with np.errstate(over="ignore", invalid="ignore"):
        return np.abs(x)**2

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
    def D6h_E2g_anisotropy(theta, config, a, b, phi):
        """Incoherent anisotropic E2g pair with a constant crossed response.

        The pair consists of diag(a, b) and an off-diagonal tensor whose
        coefficient is (a - b) / 2. At b = -a it reduces exactly to the
        ordinary, isotropic D6h E2g response.
        """
        th = _deg2rad(theta - phi)
        if config == 'parallel':
            return _abs2(a * np.cos(th)) + _abs2(b * np.sin(th))
        return np.full_like(theta, _abs2(0.5 * (a - b)), dtype=float)

    @staticmethod
    def Linear_Background(*args):
        if len(args) == 3:
            theta, offset, slope = args
            return offset + slope * theta
        if len(args) == 5:
            x, theta, offset, slope_x, slope_theta = args
            return offset + slope_x * x + slope_theta * theta
        raise TypeError("Linear_Background expects (theta, offset, slope) or (x, theta, offset, slope_x, slope_theta).")

RULE_METADATA = {
    "D2h_Ag":  {"func": SelectionRules.D2h_Ag,  "params": ["a", "b", "phi"]},
    "D2h_B1g": {"func": SelectionRules.D2h_B1g, "params": ["d", "phi"]},
    "D6h_A1g": {"func": SelectionRules.D6h_A1g, "params": ["a"]},
    "D6h_E2g": {"func": SelectionRules.D6h_E2g, "params": ["d"]},
    "D6h_E2g(anisotropy)": {
        "func": SelectionRules.D6h_E2g_anisotropy,
        "params": ["a", "b", "phi"],
        "auto_select": False,
    },
}

def lorentzian_normalized(x, x0, gamma):
    g = np.abs(gamma) + 1e-9
    return (1 / np.pi) * (g / ((x - x0)**2 + g**2))


def normalized_b1g_basis(theta, config, phi, normalization_theta=None):
    raw = SelectionRules.D2h_B1g(theta, config, d=1.0, phi=phi)
    reference_theta = theta if normalization_theta is None else normalization_theta
    reference_theta = np.asarray(reference_theta, dtype=float)
    finite_reference = reference_theta[np.isfinite(reference_theta)]
    if finite_reference.size == 0:
        return raw
    reference_raw = SelectionRules.D2h_B1g(finite_reference, config, d=1.0, phi=phi)
    mean = np.nanmean(reference_raw)
    if not np.isfinite(mean) or abs(mean) < 1e-12:
        return raw
    return raw / mean


class SafeBSpline:
    def __init__(self, t, c, k):
        t_arr = np.asarray(t, dtype=float)
        c_arr = np.asarray(c, dtype=float)
        if t_arr.size == 0 or c_arr.size == 0:
            raise ValueError("B-spline knot/coefficient arrays are empty.")
        if not np.all(np.isfinite(t_arr)):
            bad = int(np.where(~np.isfinite(t_arr))[0][0])
            raise ValueError(f"B-spline knots contain a non-finite value at index {bad}.")
        if not np.all(np.isfinite(c_arr)):
            bad = int(np.where(~np.isfinite(c_arr))[0][0])
            raise ValueError(f"B-spline coefficients contain a non-finite value at index {bad}.")
        self.spline = BSpline(t_arr, c_arr, int(k), extrapolate=False)
        self.x_min = float(np.nanmin(t_arr))
        self.x_max = float(np.nanmax(t_arr))

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(self.spline(x), nan=0.0, posinf=0.0, neginf=0.0)


def _orient_2d_arrays(shift, angles, intensity):
    shift = np.asarray(shift, dtype=float)
    angles = np.asarray(angles, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    if intensity.shape != (angles.size, shift.size):
        if intensity.shape == (shift.size, angles.size):
            intensity = intensity.T
        else:
            raise ValueError(f"2D data shape {intensity.shape} does not match angle/shift axes.")
    return shift, angles, intensity


def _fold_angle_delta(values, anchor, period=360.0):
    return np.mod(np.asarray(values, dtype=float) - float(anchor), float(period))


def _unwrapped_from_anchor(values, anchor, period=360.0):
    return float(anchor) + _fold_angle_delta(values, anchor, period)


def _angle_window_mask(angles, start, end, period=360.0):
    """Select an inclusive angular window without silently dropping a row."""
    angles = np.asarray(angles, dtype=float)
    eps = max(1e-9, abs(float(period)) * 1e-9)
    return (angles >= float(start) - eps) & (angles <= float(end) + eps)


def angle_rotation_coordinate_offset(settings):
    """Return the angular coordinate shift represented by rotation metadata.

    Cyclic rotation keeps the displayed angle axis fixed while rolling data, so
    tensor ``phi`` must follow its effective offset.  Unwrapped rotation moves
    the angle coordinates with the rows and therefore has no tensor offset.
    """
    if not settings or not settings.get("enabled") or not settings.get("cyclic_rotate", True):
        return 0.0
    try:
        offset = float(settings.get("effective_angle_offset_deg", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return offset if np.isfinite(offset) else 0.0


def angle_rotation_transition_delta(previous_settings, new_settings):
    """Return the folded tensor-coordinate change between two rotations."""
    old_offset = angle_rotation_coordinate_offset(previous_settings)
    new_offset = angle_rotation_coordinate_offset(new_settings)
    period_source = (new_settings or {}).get("period_deg", (previous_settings or {}).get("period_deg", 360.0))
    try:
        period = abs(float(period_source))
    except (TypeError, ValueError):
        period = 360.0
    if not np.isfinite(period) or period <= 0:
        period = 360.0
    return float(((new_offset - old_offset + period / 2.0) % period) - period / 2.0)


def angle_rotation_summary(settings):
    if not settings or not settings.get("enabled"):
        return "Rotation disabled"
    anchor = float(settings.get("anchor_deg", 0.0))
    roll = int(settings.get("roll_rows", 0))
    mode = "cyclic" if settings.get("cyclic_rotate", True) else "unwrapped"
    start = settings.get("window_start_deg")
    end = settings.get("window_end_deg")
    if start is not None and end is not None:
        return f"Rotated to {anchor:.6g} deg anchor, rows rolled {roll}, {mode}, window {float(start):.6g}-{float(end):.6g} deg"
    return f"Rotated to {anchor:.6g} deg anchor, rows rolled {roll}, {mode}"


def build_angle_rotation_settings(angles, *, anchor_deg=0.0, window_start_deg=None, period_deg=360.0, cyclic_rotate=True):
    """Build non-destructive angular rotation metadata for a 2D run."""
    angles = np.asarray(angles, dtype=float)
    finite = angles[np.isfinite(angles)]
    if finite.size == 0:
        raise ValueError("Angle axis is empty.")

    period = float(period_deg)
    anchor = float(anchor_deg)
    amin = float(np.nanmin(finite))
    amax = float(np.nanmax(finite))
    span = amax - amin

    eps = max(1e-9, abs(period) * 1e-9)
    if window_start_deg is None:
        if span > period + 1e-9:
            # Pick a complete available period containing the requested anchor.
            # Center it where possible, then clamp to the measured range.
            window_start = min(max(anchor - period / 2.0, amin), amax - period)
        else:
            window_start = amin
    else:
        window_start = float(window_start_deg)
    window_end = window_start + period

    use_window = span > period + eps or window_start_deg is not None
    mask = np.ones(angles.shape, dtype=bool)
    if use_window:
        mask = _angle_window_mask(angles, window_start, window_end, period)
    if not np.any(mask):
        raise ValueError("The selected 360-degree window does not contain any rows.")

    selected = angles[mask]
    anchor_idx = int(np.argmin(_fold_angle_delta(selected, anchor, period)))
    roll = -anchor_idx
    rolled_selected = np.roll(selected, roll)
    offset = 0.0
    if selected.size:
        offset = float(((selected[0] - rolled_selected[0] + 180.0) % 360.0) - 180.0)
    settings = {
        "enabled": True,
        "period_deg": period,
        "anchor_deg": anchor,
        "window_start_deg": float(window_start) if use_window else None,
        "window_end_deg": float(window_end) if use_window else None,
        "window_end_inclusive": True,
        "roll_rows": roll,
        "cyclic_rotate": bool(cyclic_rotate),
        "effective_angle_offset_deg": offset,
        "included_rows": int(selected.size),
        "excluded_rows": int(angles.size - selected.size),
    }
    settings["summary"] = angle_rotation_summary(settings)
    return settings


def apply_angle_rotation_with_acquisition_angles(angles, intensity, settings):
    """Return display data plus the original angle attached to every rolled row.

    ``display_angles`` remain the coordinate used by Raman selection rules and
    plots.  ``acquisition_angles`` follow the intensity rows through the roll
    and preserve the pre-rotation zero used by time-accumulating backgrounds.
    """
    angles = np.asarray(angles, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    if intensity.ndim != 2 or intensity.shape[0] != angles.size:
        raise ValueError("Intensity rows must match angle axis before rotation.")
    if not settings or not settings.get("enabled"):
        return angles, intensity, np.array(angles, copy=True)

    period = float(settings.get("period_deg", 360.0))
    anchor = float(settings.get("anchor_deg", 0.0))
    start = settings.get("window_start_deg")
    end = settings.get("window_end_deg")
    mask = np.ones(angles.shape, dtype=bool)
    if start is not None and end is not None:
        mask = _angle_window_mask(angles, start, end, period)
    if not np.any(mask):
        return angles, intensity, np.array(angles, copy=True)

    selected_angles = angles[mask]
    selected_intensity = intensity[mask, :]
    if selected_angles.size == 0:
        return selected_angles, selected_intensity, selected_angles

    roll = settings.get("roll_rows")
    if roll is None:
        anchor_idx = int(np.argmin(_fold_angle_delta(selected_angles, anchor, period)))
        roll = -anchor_idx
    roll = int(roll)
    rotated_intensity = np.roll(selected_intensity, roll, axis=0)
    acquisition_angles = np.roll(selected_angles, roll)
    if settings.get("cyclic_rotate", True):
        rotated_angles = selected_angles
    else:
        rotated_angles = _unwrapped_from_anchor(acquisition_angles, anchor, period)
    return rotated_angles, rotated_intensity, acquisition_angles


def apply_angle_rotation(angles, intensity, settings):
    """Return display angles/intensity after optional 360-degree windowing and rolling."""
    rotated_angles, rotated_intensity, _acquisition_angles = apply_angle_rotation_with_acquisition_angles(
        angles, intensity, settings
    )
    return rotated_angles, rotated_intensity


def display_2d_from_run(run):
    """Return (shift_cm1, display_angles, display_intensity) for GUI/plot exports."""
    if run.shift_cm1 is None or run.angle_values is None or run.intensity_2d is None:
        raise ValueError(f"Run '{getattr(run, 'nickname', 'run')}' is not a complete 2D run.")
    shift, angles, intensity = _orient_2d_arrays(run.shift_cm1, run.angle_values, run.intensity_2d)
    settings = (getattr(run, "metadata", None) or {}).get("angle_rotation")
    angles, intensity = apply_angle_rotation(angles, intensity, settings)
    return shift, angles, intensity


def display_2d_with_acquisition_angles_from_run(run):
    """Return display data and each row's pre-rotation acquisition angle.

    The first three return values intentionally match :func:`display_2d_from_run`.
    The fourth array is used only for linear background accumulation in 2D fits.
    """
    if run.shift_cm1 is None or run.angle_values is None or run.intensity_2d is None:
        raise ValueError(f"Run '{getattr(run, 'nickname', 'run')}' is not a complete 2D run.")
    shift, angles, intensity = _orient_2d_arrays(run.shift_cm1, run.angle_values, run.intensity_2d)
    settings = (getattr(run, "metadata", None) or {}).get("angle_rotation")
    display_angles, display_intensity, acquisition_angles = apply_angle_rotation_with_acquisition_angles(
        angles, intensity, settings
    )
    return shift, display_angles, display_intensity, acquisition_angles


def _fold_fit_phi(phi):
    return ((float(phi) + 180.0) % 360.0) - 180.0


def _scaled_basis_fit(y, basis):
    y = np.asarray(y, dtype=float)
    basis = np.asarray(basis, dtype=float)
    mask = np.isfinite(y) & np.isfinite(basis)
    if np.count_nonzero(mask) < 3:
        return None
    ym = y[mask]
    bm = basis[mask]
    den = float(np.dot(bm, bm))
    if den > 1e-18:
        scale = max(0.0, float(np.dot(ym, bm) / den))
        fit = scale * bm
    else:
        scale = 0.0
        fit = np.zeros_like(ym)
    resid = ym - fit
    sse = float(np.dot(resid, resid))
    centered = ym - float(np.mean(ym))
    sst = float(np.dot(centered, centered))
    denom = sst if sst > 1e-18 else float(np.dot(ym, ym))
    if denom <= 1e-18:
        score = 1.0 if sse <= 1e-18 else 0.0
    else:
        score = 1.0 - (sse / denom)
    return scale, float(score)


def _score_symmetry_rules(angles, angular, gamma, *, config_mode="parallel", fallback_rule="D2h_B1g"):
    angles = np.asarray(angles, dtype=float)
    angular = np.asarray(angular, dtype=float)
    finite = np.isfinite(angles) & np.isfinite(angular)
    if np.count_nonzero(finite) < 3:
        return fallback_rule, {}, {}, "Auto: insufficient angular data"

    low = float(np.nanpercentile(angular[finite], 10.0))
    y = angular - low
    y = np.where(np.isfinite(y), y, 0.0)
    y = np.clip(y, 0.0, None)
    if not np.any(y[finite] > 0):
        min_val = float(np.nanmin(angular[finite]))
        y = np.where(np.isfinite(angular), angular - min_val, 0.0)

    peak_angle = 0.0
    if np.any(np.isfinite(y)):
        try:
            peak_angle = float(angles[int(np.nanargmax(y))])
        except Exception:
            peak_angle = 0.0

    phi_grid = np.linspace(-180.0, 180.0, 73, endpoint=False)
    phi_hints = [peak_angle, peak_angle - 45.0, peak_angle + 45.0, 0.0]
    phi_candidates = np.array([_fold_fit_phi(p) for p in np.concatenate([phi_grid, phi_hints])])
    phi_candidates = np.unique(np.round(phi_candidates, decimals=6))
    ratio_candidates = np.array([-2.0, -1.5, -1.0, -0.75, -0.5, -0.25,
                                 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    gamma = float(gamma) if np.isfinite(gamma) and gamma > 0 else 2.0
    column_to_area_scale = np.pi * gamma * gamma
    penalties = {
        "D2h_Ag": 0.020,
        "D2h_B1g": 0.010,
        "D6h_A1g": 0.000,
        "D6h_E2g": 0.000,
    }
    rule_scores = {}
    rule_params = {}
    best = None

    def consider(rule, score, params):
        nonlocal best
        if score is None or not np.isfinite(score):
            return
        rule_scores[rule] = max(float(score), rule_scores.get(rule, -np.inf))
        rule_params[rule] = params
        choice_score = float(score) - penalties.get(rule, 0.0)
        if best is None or choice_score > best["choice_score"]:
            best = {
                "rule": rule,
                "score": float(score),
                "choice_score": choice_score,
                "params": params,
            }

    for rule_name, rule_def in RULE_METADATA.items():
        if not rule_def.get("auto_select", True):
            continue
        func = rule_def["func"]
        params = rule_def["params"]
        if params == ["a"]:
            fit = _scaled_basis_fit(y, func(angles, config_mode, 1.0))
            if fit is None:
                continue
            scale, score = fit
            val = float(np.sqrt(max(scale * column_to_area_scale, 1e-18)))
            consider(rule_name, score, {"a": [val, -np.inf, np.inf]})
        elif params == ["d"]:
            fit = _scaled_basis_fit(y, func(angles, config_mode, 1.0))
            if fit is None:
                continue
            scale, score = fit
            val = float(np.sqrt(max(scale * column_to_area_scale, 1e-18)))
            consider(rule_name, score, {"d": [val, -np.inf, np.inf]})
        elif params == ["d", "phi"]:
            local_best = None
            for phi in phi_candidates:
                fit = _scaled_basis_fit(y, func(angles, config_mode, 1.0, phi))
                if fit is None:
                    continue
                scale, score = fit
                if local_best is None or score > local_best[0]:
                    val = float(np.sqrt(max(scale * column_to_area_scale, 1e-18)))
                    local_best = (float(score), {"d": [val, -np.inf, np.inf], "phi": [_fold_fit_phi(phi), -180.0, 180.0]})
            if local_best is not None:
                consider(rule_name, local_best[0], local_best[1])
        elif params == ["a", "b", "phi"]:
            local_best = None
            for phi in phi_candidates:
                for ratio in ratio_candidates:
                    fit = _scaled_basis_fit(y, func(angles, config_mode, 1.0, ratio, phi))
                    if fit is None:
                        continue
                    scale, score = fit
                    if local_best is None or score > local_best[0]:
                        a_val = float(np.sqrt(max(scale * column_to_area_scale, 1e-18)))
                        b_val = float(ratio * a_val)
                        local_best = (
                            float(score),
                            {
                                "a": [a_val, -np.inf, np.inf],
                                "b": [b_val, -np.inf, np.inf],
                                "phi": [_fold_fit_phi(phi), -180.0, 180.0],
                            },
                        )
            if local_best is not None:
                consider(rule_name, local_best[0], local_best[1])

    if best is None:
        return fallback_rule, {}, {}, "Auto: insufficient angular data"

    best_rule = best["rule"]
    best_score = best["score"]
    confidence = "high" if best_score >= 0.80 else "medium" if best_score >= 0.50 else "low"
    summary = f"Auto: {best_rule}, score {best_score:.2f} ({confidence})"
    return best_rule, rule_params.get(best_rule, {}), rule_scores, summary


def _nearest_index(values, target, fallback=0):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0
    if target is None or not np.isfinite(float(target)):
        return int(np.clip(fallback, 0, arr.size - 1))
    return int(np.nanargmin(np.abs(arr - float(target))))


def _default_ang_params_for_rule(rule_name, angular, angles, gamma):
    rule_name = rule_name if rule_name in RULE_METADATA else "D2h_B1g"
    finite = np.isfinite(angular)
    if np.any(finite):
        amp = max(0.0, float(np.nanmax(angular[finite]) - np.nanpercentile(angular[finite], 10.0)))
        phi = float(angles[int(np.nanargmax(np.where(finite, angular, -np.inf)))])
    else:
        amp = 1.0
        phi = 0.0
    val = float(np.sqrt(max(amp * max(abs(gamma), 1e-9), 1e-12)))
    if rule_name == "D6h_E2g(anisotropy)":
        return {
            "a": [val, -np.inf, np.inf],
            "b": [-0.8 * val, -np.inf, np.inf],
            "phi": [_fold_fit_phi(phi), -180.0, 180.0],
        }
    params = {}
    for name in RULE_METADATA[rule_name]["params"]:
        if name == "phi":
            params[name] = [_fold_fit_phi(phi), -180.0, 180.0]
        elif name == "b" and rule_name == "D2h_Ag":
            params[name] = [0.5 * val, -np.inf, np.inf]
        else:
            params[name] = [val, -np.inf, np.inf]
    return params


def estimate_peak_defaults_for_datasets(
    datasets,
    *,
    selected_dataset_index=0,
    x_value=None,
    angle_value=None,
    rule_name="D2h_B1g",
    preserve_x0=False,
    force_rule=False,
):
    if rule_name in RULE_METADATA:
        force_rule = force_rule or not RULE_METADATA[rule_name].get("auto_select", True)
    active = [
        (idx, ds) for idx, ds in enumerate(datasets or [])
        if ds and ds.get("x") is not None and ds.get("ang") is not None and ds.get("z") is not None
    ]
    if not active:
        return {}
    if not any(idx == selected_dataset_index for idx, _ds in active):
        selected_dataset_index = active[0][0]
    primary = next(ds for idx, ds in active if idx == selected_dataset_index)
    x = np.asarray(primary["x"], dtype=float)
    angles = np.asarray(primary["ang"], dtype=float)
    ix = _nearest_index(x, x_value, x.size // 2)
    iy = _nearest_index(angles, angle_value, angles.size // 2)
    result = estimate_peak_defaults(
        x,
        angles,
        primary["z"],
        ix=ix,
        iy=iy,
        rule_name=rule_name,
        config_mode=primary.get("config", "parallel"),
        preserve_x0=preserve_x0,
        force_rule=force_rule,
    )
    if len(active) <= 1 or not result or force_rule:
        result["selected_dataset_index"] = selected_dataset_index
        return result

    x0 = float(x[ix]) if preserve_x0 and x.size else result.get("x0", x[ix] if x.size else 0.0)
    gamma = result.get("gamma", 2.0)
    combined_scores = {}
    score_counts = {}
    primary_angular = None
    primary_angles = None
    for _idx, ds in active:
        sx = np.asarray(ds["x"], dtype=float)
        sang = np.asarray(ds["ang"], dtype=float)
        sz = np.asarray(ds["z"], dtype=float)
        if sz.shape != (sang.size, sx.size):
            if sz.shape == (sx.size, sang.size):
                sz = sz.T
            else:
                continue
        six = _nearest_index(sx, x0, sx.size // 2)
        angular = np.asarray(sz[:, six], dtype=float)
        if _idx == selected_dataset_index:
            primary_angular = angular
            primary_angles = sang
        _best, _params, scores, _summary = _score_symmetry_rules(
            sang,
            angular,
            gamma,
            config_mode=ds.get("config", "parallel"),
            fallback_rule=rule_name,
        )
        for rule, score in scores.items():
            if np.isfinite(score):
                combined_scores[rule] = combined_scores.get(rule, 0.0) + float(score)
                score_counts[rule] = score_counts.get(rule, 0) + 1

    if combined_scores:
        avg_scores = {rule: combined_scores[rule] / max(1, score_counts[rule]) for rule in combined_scores}
        penalties = {"D2h_Ag": 0.020, "D2h_B1g": 0.010, "D6h_A1g": 0.0, "D6h_E2g": 0.0}
        best_rule = max(avg_scores, key=lambda r: avg_scores[r] - penalties.get(r, 0.0))
        result["best_rule"] = best_rule
        result["rule_scores"] = {k: float(v) for k, v in avg_scores.items()}
        result["auto_rule_score"] = float(avg_scores[best_rule])
        confidence = "high" if avg_scores[best_rule] >= 0.80 else "medium" if avg_scores[best_rule] >= 0.50 else "low"
        result["auto_rule_summary"] = f"Auto: {best_rule}, joint score {avg_scores[best_rule]:.2f} ({confidence})"
        if set((result.get("ang_params") or {}).keys()) != set(RULE_METADATA[best_rule]["params"]):
            result["ang_params"] = _default_ang_params_for_rule(
                best_rule,
                primary_angular if primary_angular is not None else np.asarray(primary["z"])[:, ix],
                primary_angles if primary_angles is not None else angles,
                gamma,
            )
    result["selected_dataset_index"] = selected_dataset_index
    return result


def estimate_peak_defaults(
    x=None,
    angles=None,
    intensity=None,
    *,
    ix=None,
    iy=None,
    rule_name="D2h_B1g",
    config_mode="parallel",
    datasets=None,
    selected_dataset_index=0,
    x_value=None,
    angle_value=None,
    preserve_x0=False,
    force_rule=False,
):
    """Fast, non-iterative defaults for adding a map-fit peak near the selected slice."""
    if datasets is not None:
        return estimate_peak_defaults_for_datasets(
            datasets,
            selected_dataset_index=selected_dataset_index,
            x_value=x_value,
            angle_value=angle_value,
            rule_name=rule_name,
            preserve_x0=preserve_x0,
            force_rule=force_rule,
        )
    x = np.asarray(x, dtype=float)
    angles = np.asarray(angles, dtype=float)
    z = np.asarray(intensity, dtype=float)
    if z.shape != (angles.size, x.size):
        if z.shape == (x.size, angles.size):
            z = z.T
        else:
            raise ValueError("Cannot estimate peak defaults from mismatched data.")
    if x.size == 0 or angles.size == 0:
        return {}

    ix0 = int(np.clip(ix if ix is not None else x.size // 2, 0, x.size - 1))
    iy0 = int(np.clip(iy if iy is not None else angles.size // 2, 0, angles.size - 1))
    spectral = np.asarray(z[iy0, :], dtype=float)
    finite = np.isfinite(spectral)
    if not np.any(finite):
        spectral = np.zeros_like(x)
        finite = np.ones_like(x, dtype=bool)

    radius = max(3, min(20, x.size // 10))
    lo = max(0, ix0 - radius)
    hi = min(x.size, ix0 + radius + 1)
    local = spectral[lo:hi]
    if np.any(np.isfinite(local)):
        local_rel = int(np.nanargmax(local))
        peak_idx = lo + local_rel
    else:
        peak_idx = ix0
    if preserve_x0:
        peak_idx = ix0

    baseline = float(np.nanpercentile(spectral[finite], 20.0)) if np.any(finite) else 0.0
    left_y = float(np.nanmedian(spectral[:max(1, x.size // 10)]))
    right_y = float(np.nanmedian(spectral[-max(1, x.size // 10):]))
    dx_total = float(x[-1] - x[0]) if x.size > 1 else 1.0
    slope = (right_y - left_y) / dx_total if abs(dx_total) > 1e-12 else 0.0
    peak_height = max(0.0, float(spectral[peak_idx] - baseline)) if np.isfinite(spectral[peak_idx]) else 0.0

    half = baseline + 0.5 * peak_height
    left_idx = peak_idx
    while left_idx > 0 and np.isfinite(spectral[left_idx]) and spectral[left_idx] > half:
        left_idx -= 1
    right_idx = peak_idx
    while right_idx < x.size - 1 and np.isfinite(spectral[right_idx]) and spectral[right_idx] > half:
        right_idx += 1
    if right_idx > left_idx:
        gamma = abs(float(x[right_idx] - x[left_idx])) / 2.0
    elif x.size > 1:
        gamma = abs(float(np.nanmedian(np.diff(np.sort(x))))) * 2.0
    else:
        gamma = 2.0
    gamma = float(np.clip(gamma if np.isfinite(gamma) and gamma > 0 else 2.0, 0.1, 50.0))

    angular = np.asarray(z[:, peak_idx], dtype=float)
    fallback_rule = rule_name if rule_name in RULE_METADATA else "D2h_B1g"
    force_rule = force_rule or not RULE_METADATA[fallback_rule].get("auto_select", True)
    best_rule, scored_ang_params, rule_scores, auto_summary = _score_symmetry_rules(
        angles,
        angular,
        gamma,
        config_mode=config_mode,
        fallback_rule=fallback_rule,
    )
    if force_rule:
        best_rule = fallback_rule
        scored_ang_params = _default_ang_params_for_rule(best_rule, angular, angles, gamma)
        auto_summary = f"Auto: {best_rule}, re-estimated"
    if np.any(np.isfinite(angular)):
        phi_idx = int(np.nanargmax(angular))
        phi = ((float(angles[phi_idx]) + 180.0) % 360.0) - 180.0
        angular_amp = max(peak_height, float(np.nanmax(angular) - np.nanpercentile(angular[np.isfinite(angular)], 20.0)))
    else:
        phi = 0.0
        angular_amp = peak_height
    amp_param = float(np.sqrt(max(angular_amp * max(gamma, 1e-9), 1e-12)))

    seed_rule = best_rule if force_rule else fallback_rule
    ang_params = {}
    for name in RULE_METADATA.get(seed_rule, RULE_METADATA["D2h_B1g"])["params"]:
        if name == "phi":
            ang_params[name] = [phi, -180.0, 180.0]
        elif name in {"a", "b", "d"}:
            val = amp_param
            if name == "b" and seed_rule == "D2h_Ag" and config_mode == "parallel":
                val = max(amp_param * 0.5, 1e-6)
            ang_params[name] = [val, -np.inf, np.inf]
    if scored_ang_params:
        ang_params = scored_ang_params
    best_score = rule_scores.get(best_rule)

    return {
        "x0": float(x[ix0] if preserve_x0 else x[peak_idx]),
        "gamma": gamma,
        "spec_params": {
            "x0": float(x[ix0] if preserve_x0 else x[peak_idx]),
            "gamma": gamma,
        },
        "background_offset": baseline,
        "background_slope": slope,
        "ang_params": ang_params,
        "best_rule": best_rule,
        "rule_scores": {name: float(score) for name, score in rule_scores.items() if np.isfinite(score)},
        "auto_rule_score": float(best_score) if best_score is not None and np.isfinite(best_score) else None,
        "auto_rule_summary": auto_summary,
    }

class MapFittingEngine:
    fit_schema_version = 4
    background_angle_reference = "pre_rotation_acquisition_angle"

    @staticmethod
    def _new_bg_params():
        return {
            "offset": [0.0, -np.inf, np.inf],
            "slope_x": [0.0, -np.inf, np.inf],
            "slope_theta": [0.0, -np.inf, np.inf],
            "amp_si": [0.0, 0.0, np.inf],
            "amp_b1g_peak": [0.0, -np.inf, np.inf],
        }

    @staticmethod
    def _new_si_bg_peak_params():
        return {
            "x0": [520.0, 0.0, 5000.0],
            "gamma": [2.0, 0.001, 100.0],
            "phi": [0.0, -180.0, 180.0],
        }

    def __init__(self):
        self.datasets = [
            {"label": "Parallel (XX)", "config": "parallel", "x": None, "ang": None, "background_ang": None, "z": None, "nickname": "XX"},
            {"label": "Cross (YX)",    "config": "cross",    "x": None, "ang": None, "background_ang": None, "z": None, "nickname": "YX"}
        ]
        self.peaks = []
        self.bg_params = [self._new_bg_params(), self._new_bg_params()]
        self.x_min_limit = -np.inf
        self.x_max_limit = np.inf
        self.unit = "cm-1"
        self.si_bg_unit = "cm-1"
        self.si_bg_mode = "none"
        self.si_bg_source_path = ""
        self.si_bg_payload = None
        self.si_bg_load_warning = ""
        self.si_bg_interp_xx = None
        self.si_bg_interp_yx = None
        self.si_bg_iso_xx = None
        self.si_bg_iso_yx = None
        self.si_bg_ang_xx = None
        self.si_bg_ang_yx = None
        self.si_bg_peak_source_center = None
        self.si_bg_peak_source_gamma = None
        self.si_bg_peak_ref_center = None
        self.si_bg_peak_params = self._new_si_bg_peak_params()
        self.row_fit_config = {}

    def active_dataset_indices(self):
        return [
            idx for idx, ds in enumerate(self.datasets)
            if ds["z"] is not None and ds["x"] is not None and ds["ang"] is not None
        ]

    def set_data(self, index, x, ang, z, nickname="Run", background_ang=None):
        display_angles = np.asarray(ang, dtype=float)
        background_angles = display_angles if background_ang is None else np.asarray(background_ang, dtype=float)
        if background_angles.shape != display_angles.shape:
            raise ValueError("Background acquisition angles must match the display angle axis.")
        self.datasets[index]["x"] = np.asarray(x, dtype=float)
        self.datasets[index]["ang"] = display_angles
        self.datasets[index]["background_ang"] = np.array(background_angles, copy=True)
        self.datasets[index]["z"] = np.asarray(z, dtype=float)
        self.datasets[index]["nickname"] = nickname
        if index == 0:
            self.x_min_limit = float(np.nanmin(self.datasets[index]["x"]))
            self.x_max_limit = float(np.nanmax(self.datasets[index]["x"]))

    @staticmethod
    def convert_between_units(value, from_unit, to_unit):
        if from_unit == to_unit:
            return value
        if from_unit == "cm-1" and to_unit == "meV":
            return value / 8.065544
        if from_unit == "meV" and to_unit == "cm-1":
            return value * 8.065544
        return value

    def _convert_bg_x(self, x_array):
        x_array = np.asarray(x_array, dtype=float)
        if self.unit == "meV" and self.si_bg_unit == "cm-1":
            return x_array * 8.065544
        if self.unit == "cm-1" and self.si_bg_unit == "meV":
            return x_array / 8.065544
        return np.array(x_array, copy=True)

    def refresh_si_peak_units(self):
        if self.si_bg_peak_source_center is not None:
            self.si_bg_peak_ref_center = self.convert_between_units(self.si_bg_peak_source_center, self.si_bg_unit, self.unit)
            self.si_bg_peak_params["x0"][0] = self.si_bg_peak_ref_center
        if self.si_bg_peak_source_gamma is not None:
            self.si_bg_peak_params["gamma"][0] = abs(self.convert_between_units(self.si_bg_peak_source_gamma, self.si_bg_unit, self.unit))

    def get_si_profile_shift(self):
        if self.si_bg_peak_ref_center is None:
            return 0.0
        return self.si_bg_peak_params["x0"][0] - self.si_bg_peak_ref_center

    def clear_si_bg(self):
        self.si_bg_mode = "none"
        self.si_bg_source_path = ""
        self.si_bg_payload = None
        self.si_bg_load_warning = ""
        self.si_bg_interp_xx = self.si_bg_interp_yx = None
        self.si_bg_iso_xx = self.si_bg_iso_yx = None
        self.si_bg_ang_xx = self.si_bg_ang_yx = None
        self.si_bg_peak_source_center = None
        self.si_bg_peak_source_gamma = None
        self.si_bg_peak_ref_center = None

    def load_si_bg(self, filepath=None, payload=None):
        data = None
        try:
            if payload is None:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = payload
            self.clear_si_bg()
            self.si_bg_payload = data
            self.si_bg_source_path = filepath or str(data.get("source_path", ""))
            self.si_bg_unit = data.get("unit", "cm-1")
            self.si_bg_mode = data.get("schema", "legacy")

            if self.si_bg_mode == "advanced_si_bg_v2":
                components = data.get("components", {})
                loaded_components = []
                missing_components = []
                self.si_bg_peak_params = self._new_si_bg_peak_params()
                for attr, key in (
                    ("si_bg_iso_xx", "isotropic_xx"),
                    ("si_bg_iso_yx", "isotropic_yx"),
                    ("si_bg_ang_xx", "angular_xx"),
                    ("si_bg_ang_yx", "angular_yx"),
                ):
                    if key in components:
                        s = components[key]
                        try:
                            setattr(self, attr, SafeBSpline(s["t"], s["c"], s["k"]))
                        except Exception as exc:
                            raise ValueError(f"Invalid advanced Si BG component '{key}': {exc}") from exc
                        loaded_components.append(key)
                    else:
                        missing_components.append(key)
                b1g_meta = data.get("b1g", {}) if isinstance(data.get("b1g", {}), dict) else {}
                phi = float(b1g_meta.get("phi_deg", self.si_bg_peak_params["phi"][0]))
                fit_phi = bool(b1g_meta.get("fit_phi", True))
                phi_bounds = b1g_meta.get("phi_bounds_deg", [-180.0, 180.0])
                try:
                    phi_lo, phi_hi = float(phi_bounds[0]), float(phi_bounds[1])
                except Exception:
                    phi_lo, phi_hi = -180.0, 180.0
                if fit_phi:
                    self.si_bg_peak_params["phi"] = [phi, phi_lo, phi_hi]
                else:
                    self.si_bg_peak_params["phi"] = [phi, phi, phi]
                peak_meta = data.get("removed_peak", {})
                center = peak_meta.get("shared_center", peak_meta.get("nominal_center", None))
                gamma = peak_meta.get("gamma_guess", None)
                if gamma is None and isinstance(peak_meta.get("profiles"), dict):
                    gamma_values = [
                        p.get("gamma") for p in peak_meta.get("profiles", {}).values()
                        if isinstance(p, dict) and p.get("gamma") is not None
                    ]
                    gamma = float(np.nanmedian(gamma_values)) if gamma_values else None
                if center is not None:
                    self.si_bg_peak_source_center = float(center)
                if gamma is not None:
                    self.si_bg_peak_source_gamma = float(gamma)
                self.refresh_si_peak_units()
                msg = (
                    f"Loaded advanced Si BG ({self.si_bg_unit}); "
                    f"components={','.join(loaded_components) if loaded_components else 'none'}; "
                    f"source_center={self.si_bg_peak_source_center}, source_gamma={self.si_bg_peak_source_gamma}; "
                    f"fit_center={self.si_bg_peak_params['x0'][0]:.6g} {self.unit}, "
                    f"fit_gamma={self.si_bg_peak_params['gamma'][0]:.6g} {self.unit}"
                )
                if missing_components:
                    msg += f"; missing={','.join(missing_components)}"
                return True, msg

            if "XX" in data:
                s = data["XX"]
                try:
                    self.si_bg_interp_xx = SafeBSpline(s["t"], s["c"], s["k"])
                except Exception as exc:
                    raise ValueError(f"Invalid legacy Si BG component 'XX': {exc}") from exc
            if "YX" in data:
                s = data["YX"]
                try:
                    self.si_bg_interp_yx = SafeBSpline(s["t"], s["c"], s["k"])
                except Exception as exc:
                    raise ValueError(f"Invalid legacy Si BG component 'YX': {exc}") from exc
            self.si_bg_mode = "legacy"
            return True, f"Loaded Si BG profile ({self.si_bg_unit})"
        except Exception as exc:
            schema = data.get("schema", "(unknown)") if isinstance(data, dict) else "(unreadable)"
            unit = data.get("unit", "(unknown)") if isinstance(data, dict) else "(unknown)"
            keys = ",".join(sorted(data.keys())) if isinstance(data, dict) else "(none)"
            self.clear_si_bg()
            self.si_bg_load_warning = (
                f"{type(exc).__name__}: {exc}\n"
                f"Si BG load context: path={filepath or '(embedded payload)'}, schema={schema}, unit={unit}, top_keys={keys}"
            )
            return False, self.si_bg_load_warning

    def evaluate_si_bg(self, x_array, config_mode):
        x_converted = self._convert_bg_x(x_array)
        if config_mode == "parallel" and self.si_bg_interp_xx is not None:
            return self.si_bg_interp_xx(x_converted)
        if config_mode != "parallel" and self.si_bg_interp_yx is not None:
            return self.si_bg_interp_yx(x_converted)
        return np.zeros_like(np.asarray(x_array, dtype=float))

    def evaluate_advanced_si_bg(
        self,
        x_array,
        theta_array,
        config_mode,
        scale_bg,
        peak_amp,
        peak_x0,
        peak_gamma,
        phi,
        normalization_theta=None,
    ):
        profile_shift = peak_x0 - self.si_bg_peak_ref_center if self.si_bg_peak_ref_center is not None else 0.0
        x_converted = self._convert_bg_x(np.asarray(x_array, dtype=float) - profile_shift)
        iso = np.zeros_like(x_converted, dtype=float)
        angular = np.zeros_like(x_converted, dtype=float)
        b1g_basis = normalized_b1g_basis(
            theta_array,
            config_mode,
            phi,
            normalization_theta=normalization_theta,
        )
        if config_mode == "parallel":
            if self.si_bg_iso_xx is not None:
                iso = self.si_bg_iso_xx(x_converted)
            if self.si_bg_ang_xx is not None:
                angular = self.si_bg_ang_xx(x_converted) * b1g_basis
        else:
            if self.si_bg_iso_yx is not None:
                iso = self.si_bg_iso_yx(x_converted)
            if self.si_bg_ang_yx is not None:
                angular = self.si_bg_ang_yx(x_converted) * b1g_basis
        b1g_peak = b1g_basis * lorentzian_normalized(x_array, peak_x0, peak_gamma)
        return scale_bg * (iso + angular) + peak_amp * b1g_peak

    def _bg_index_for_dataset(self, dataset_index):
        return 1 if int(dataset_index) == 1 else 0

    def _bg_for_dataset(self, dataset_index):
        return self.bg_params[self._bg_index_for_dataset(dataset_index)]

    def _base_param_count(self):
        return 13 if self.si_bg_mode == "advanced_si_bg_v2" else 8

    def add_peak(self, name=None, rule_name="D2h_B1g", center=None, estimates=None):
        estimates = estimates or {}
        rule_name = estimates.get("best_rule") or rule_name
        rule_name = rule_name if rule_name in RULE_METADATA else "D2h_B1g"
        auto_name = True if name is None or name == "New Peak" else (name == rule_name)
        name = rule_name if name is None or name == "New Peak" else name
        if center is None:
            active = self.active_dataset_indices()
            ds = self.datasets[active[0]] if active else self.datasets[0]
            center = np.nanmedian(ds["x"]) if ds["x"] is not None else 300.0
        center = float(estimates.get("x0", center))
        gamma_guess = float(estimates.get("gamma", 2.0))
        spec_params = {"x0": [center, 0.0, 5000.0], "gamma": [gamma_guess, 0.001, 100.0]}
        ang_params = {}
        for p in RULE_METADATA[rule_name]["params"]:
            ang_params[p] = [10.0, -np.inf, np.inf]
            if p == "phi":
                ang_params[p] = [0.0, -180.0, 180.0]
        if rule_name == "D6h_E2g(anisotropy)":
            ang_params["b"][0] = -8.0
        for key, value in (estimates.get("ang_params") or {}).items():
            if key in ang_params and isinstance(value, (list, tuple)) and len(value) >= 3:
                ang_params[key] = [float(value[0]), float(value[1]), float(value[2])]
        peak = {
            "name": name,
            "rule": rule_name,
            "spec_params": spec_params,
            "ang_params": ang_params,
            "auto_name": auto_name,
        }
        for key in ("auto_rule_score", "auto_rule_summary", "rule_scores"):
            if key in estimates:
                peak[key] = estimates[key]
        self.peaks.append(peak)
        bg_idx = self._bg_index_for_dataset(estimates.get("selected_dataset_index", 0))
        if "background_offset" in estimates:
            self.bg_params[bg_idx]["offset"][0] = float(estimates["background_offset"])
        if "background_slope" in estimates:
            self.bg_params[bg_idx]["slope_x"][0] = float(estimates["background_slope"])
        self.sort_and_rename_peaks()
        return next((p for p in self.peaks if p["spec_params"] is spec_params), self.peaks[-1])

    def rename_peak(self, index, name):
        if 0 <= index < len(self.peaks):
            clean = str(name).strip()
            if clean:
                self.peaks[index]["name"] = clean
                self.peaks[index]["auto_name"] = False

    def set_peak_rule(self, index, rule_name, estimates=None):
        if not (0 <= index < len(self.peaks)) or rule_name not in RULE_METADATA:
            return
        estimates = estimates or {}
        peak = self.peaks[index]
        current_x0 = peak.get("spec_params", {}).get("x0", [np.nan])[0]
        peak["rule"] = rule_name
        peak["ang_params"] = {pn: [10.0, -np.inf, np.inf] for pn in RULE_METADATA[rule_name]["params"]}
        if "phi" in peak["ang_params"]:
            peak["ang_params"]["phi"] = [0.0, -180.0, 180.0]
        if rule_name == "D6h_E2g(anisotropy)":
            peak["ang_params"]["b"][0] = -8.0
        for key, value in (estimates.get("ang_params") or {}).items():
            if key in peak["ang_params"] and isinstance(value, (list, tuple)) and len(value) >= 3:
                peak["ang_params"][key] = [float(value[0]), float(value[1]), float(value[2])]
        if "gamma" in estimates:
            peak["spec_params"]["gamma"][0] = float(estimates["gamma"])
        if np.isfinite(current_x0):
            peak["spec_params"]["x0"][0] = float(current_x0)
        peak["auto_rule_summary"] = estimates.get("auto_rule_summary", f"Auto: {rule_name}, re-estimated")
        peak["auto_rule_score"] = estimates.get("auto_rule_score")
        if "rule_scores" in estimates:
            peak["rule_scores"] = estimates["rule_scores"]
        if peak.get("auto_name", False):
            peak["name"] = rule_name
        self.sort_and_rename_peaks()

    def sort_and_rename_peaks(self):
        self.peaks.sort(key=lambda p: p.get("spec_params", {}).get("x0", [np.inf])[0])
        rule_groups = {}
        for peak in self.peaks:
            peak.setdefault("auto_name", peak.get("name") == peak.get("rule"))
            rule_groups.setdefault(peak.get("rule", ""), []).append(peak)
        for rule, rule_peaks in rule_groups.items():
            if len(rule_peaks) > 1:
                idx = 1
                for peak in rule_peaks:
                    name = str(peak.get("name", ""))
                    if peak.get("auto_name", False) or name == rule or re.match(rf"^{re.escape(rule)}\(\d+\)$", name):
                        peak["name"] = f"{rule}({idx})"
                        peak["auto_name"] = True
                        idx += 1
            else:
                peak = rule_peaks[0]
                name = str(peak.get("name", ""))
                if peak.get("auto_name", False) or re.match(rf"^{re.escape(rule)}\(\d+\)$", name):
                    peak["name"] = rule
                    peak["auto_name"] = True

    def get_mask(self, dataset_index):
        ds = self.datasets[dataset_index]
        if ds["x"] is None:
            return None
        return (ds["x"] >= self.x_min_limit) & (ds["x"] <= self.x_max_limit)

    def _oriented_dataset_arrays(self, dataset_index):
        ds = self.datasets[dataset_index]
        x = np.asarray(ds["x"], dtype=float)
        ang = np.asarray(ds["ang"], dtype=float)
        z = np.asarray(ds["z"], dtype=float)
        if z.shape != (ang.size, x.size):
            if z.shape == (x.size, ang.size):
                z = z.T
            else:
                raise ValueError(f"{ds['label']} data shape {z.shape} does not match angle/shift axes.")
        return x, ang, z

    def _background_angles_for_dataset(self, dataset_index):
        ds = self.datasets[dataset_index]
        display_angles = np.asarray(ds["ang"], dtype=float)
        values = ds.get("background_ang")
        if values is None:
            return np.array(display_angles, copy=True)
        background_angles = np.asarray(values, dtype=float)
        if background_angles.shape != display_angles.shape:
            raise ValueError(f"{ds['label']} background angles do not match its display angle axis.")
        return background_angles

    def _fit_points_for_dataset(self, dataset_index):
        """Return finite fit points with display and acquisition angles."""
        x, ang, z = self._oriented_dataset_arrays(dataset_index)
        background_ang = self._background_angles_for_dataset(dataset_index)
        x_mask = self.get_mask(dataset_index)
        if x_mask is None:
            return np.array([]), np.array([]), np.array([]), np.array([]), 0, 0
        x_mask = np.asarray(x_mask, dtype=bool)
        XX, YY = np.meshgrid(x, ang)
        _XX_BG, YY_BG = np.meshgrid(x, background_ang)
        selected = np.tile(x_mask, (ang.size, 1))
        finite = selected & np.isfinite(XX) & np.isfinite(YY) & np.isfinite(YY_BG) & np.isfinite(z)
        selected_count = int(np.count_nonzero(selected))
        fit_count = int(np.count_nonzero(finite))
        return XX[finite], YY[finite], YY_BG[finite], z[finite], selected_count, selected_count - fit_count

    @staticmethod
    def _finite_range_text(values):
        arr = np.asarray(values, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return "no finite values"
        return f"{float(np.nanmin(finite)):.6g}..{float(np.nanmax(finite)):.6g}"

    def _param_labels(self):
        labels = []
        if self.si_bg_mode == "advanced_si_bg_v2":
            for idx, role in enumerate(("XX", "YX")):
                labels.extend([
                    f"bg_{role}.offset",
                    f"bg_{role}.slope_x",
                    f"bg_{role}.slope_theta",
                    f"bg_{role}.amp_si_scale",
                    f"bg_{role}.amp_b1g_peak",
                ])
            labels.extend(["si_bg.x0", "si_bg.gamma", "si_bg.phi"])
        else:
            for role in ("XX", "YX"):
                labels.extend([
                    f"bg_{role}.offset",
                    f"bg_{role}.slope_x",
                    f"bg_{role}.slope_theta",
                    f"bg_{role}.amp_si",
                ])
        for peak_idx, peak in enumerate(self.peaks, start=1):
            name = str(peak.get("name", f"peak{peak_idx}"))
            labels.extend([f"peak_{peak_idx}:{name}.x0", f"peak_{peak_idx}:{name}.gamma"])
            for param_name in RULE_METADATA.get(peak.get("rule"), {}).get("params", []):
                labels.append(f"peak_{peak_idx}:{name}.{param_name}")
        return labels

    def _format_param_table(self, p0=None, lower=None, upper=None, *, max_rows=80):
        try:
            p0 = np.asarray(self.flatten_params(0) if p0 is None else p0, dtype=float)
            lower = np.asarray(self.flatten_params(1) if lower is None else lower, dtype=float)
            upper = np.asarray(self.flatten_params(2) if upper is None else upper, dtype=float)
        except Exception as exc:
            return [f"Parameter table unavailable: {type(exc).__name__}: {exc}"]
        labels = self._param_labels()
        rows = []
        for idx in range(max(len(p0), len(lower), len(upper), len(labels))):
            label = labels[idx] if idx < len(labels) else f"param_{idx + 1}"
            val = p0[idx] if idx < len(p0) else np.nan
            lo = lower[idx] if idx < len(lower) else np.nan
            hi = upper[idx] if idx < len(upper) else np.nan
            flags = []
            if not np.isfinite(val):
                flags.append("bad value")
            if np.isnan(lo) or np.isnan(hi):
                flags.append("NaN bound")
            if lo > hi:
                flags.append("min>max")
            if np.isfinite(val) and not np.isnan(lo) and not np.isnan(hi) and (val < lo or val > hi):
                flags.append("value outside bounds")
            if idx < max_rows:
                rows.append(f"  {idx + 1:02d} {label}: value={val:.8g}, min={lo:.8g}, max={hi:.8g}" + (f" [{', '.join(flags)}]" if flags else ""))
        if len(rows) == 0:
            rows.append("  no parameters")
        if max(len(p0), len(lower), len(upper), len(labels)) > max_rows:
            rows.append(f"  ... {max(len(p0), len(lower), len(upper), len(labels)) - max_rows} more parameter(s)")
        return rows

    def fit_diagnostics(self, p0=None, lower=None, upper=None):
        lines = [
            f"Engine: mode={self.si_bg_mode}, unit={self.unit}, active_datasets={self.active_dataset_indices()}, peaks={len(self.peaks)}",
        ]
        if self.si_bg_mode and self.si_bg_mode != "none":
            components = []
            if self.si_bg_mode == "advanced_si_bg_v2":
                for attr, label in (
                    ("si_bg_iso_xx", "isotropic_xx"),
                    ("si_bg_iso_yx", "isotropic_yx"),
                    ("si_bg_ang_xx", "angular_xx"),
                    ("si_bg_ang_yx", "angular_yx"),
                ):
                    obj = getattr(self, attr, None)
                    components.append(f"{label}={'yes' if obj is not None else 'no'}")
            else:
                components = [
                    f"XX={'yes' if self.si_bg_interp_xx is not None else 'no'}",
                    f"YX={'yes' if self.si_bg_interp_yx is not None else 'no'}",
                ]
            lines.append(
                "Si BG: "
                f"schema={self.si_bg_mode}, si_unit={self.si_bg_unit}, source={self.si_bg_source_path or '(embedded/unspecified)'}, "
                f"ref_center={self.si_bg_peak_ref_center}, source_center={self.si_bg_peak_source_center}, "
                f"source_gamma={self.si_bg_peak_source_gamma}, components={', '.join(components)}"
            )
            if self.si_bg_mode == "advanced_si_bg_v2":
                lines.extend(self._format_param_table(p0, lower, upper, max_rows=20)[:13])
        for ds_idx in self.active_dataset_indices():
            try:
                ds = self.datasets[ds_idx]
                x, ang, z = self._oriented_dataset_arrays(ds_idx)
                _xf, _tf, _btf, _zf, selected_count, skipped_count = self._fit_points_for_dataset(ds_idx)
                lines.append(
                    f"Dataset {ds_idx} {ds.get('label', '')}: config={ds.get('config')}, shape={z.shape}, "
                    f"x={self._finite_range_text(x)} finite={int(np.count_nonzero(np.isfinite(x)))}/{x.size}, "
                    f"angle={self._finite_range_text(ang)} finite={int(np.count_nonzero(np.isfinite(ang)))}/{ang.size}, "
                    f"z finite={int(np.count_nonzero(np.isfinite(z)))}/{z.size}, "
                    f"selected_points={selected_count}, skipped_nonfinite={skipped_count}"
                )
            except Exception as exc:
                lines.append(f"Dataset {ds_idx}: diagnostic unavailable: {type(exc).__name__}: {exc}")
        try:
            p0_arr = np.asarray(self.flatten_params(0) if p0 is None else p0, dtype=float)
            lo_arr = np.asarray(self.flatten_params(1) if lower is None else lower, dtype=float)
            hi_arr = np.asarray(self.flatten_params(2) if upper is None else upper, dtype=float)
            lines.append(
                f"Parameter counts: values={p0_arr.size}, lower={lo_arr.size}, upper={hi_arr.size}; "
                f"nonfinite_values={int(np.count_nonzero(~np.isfinite(p0_arr)))}, "
                f"nan_bounds={int(np.count_nonzero(np.isnan(lo_arr))) + int(np.count_nonzero(np.isnan(hi_arr)))}, "
                f"infinite_bounds={int(np.count_nonzero(np.isinf(lo_arr))) + int(np.count_nonzero(np.isinf(hi_arr)))}"
            )
            problem_rows = []
            labels = self._param_labels()
            n = max(p0_arr.size, lo_arr.size, hi_arr.size)
            for idx in range(n):
                val = p0_arr[idx] if idx < p0_arr.size else np.nan
                lo = lo_arr[idx] if idx < lo_arr.size else np.nan
                hi = hi_arr[idx] if idx < hi_arr.size else np.nan
                bad = (not np.isfinite(val)) or np.isnan(lo) or np.isnan(hi) or lo > hi or (np.isfinite(val) and val < lo) or (np.isfinite(val) and val > hi)
                if bad:
                    label = labels[idx] if idx < len(labels) else f"param_{idx + 1}"
                    problem_rows.append(f"  {idx + 1:02d} {label}: value={val}, min={lo}, max={hi}")
            if problem_rows:
                lines.append("Problem parameters:")
                lines.extend(problem_rows[:20])
                if len(problem_rows) > 20:
                    lines.append(f"  ... {len(problem_rows) - 20} more")
        except Exception as exc:
            lines.append(f"Parameter diagnostics unavailable: {type(exc).__name__}: {exc}")
        return "\n".join(lines)

    def _largest_param_text(self, params, *, limit=8):
        arr = np.asarray(params, dtype=float)
        labels = self._param_labels()
        finite = np.isfinite(arr)
        if not np.any(finite):
            return "no finite parameters"
        order = np.argsort(np.abs(arr[finite]))[::-1]
        finite_indices = np.where(finite)[0]
        parts = []
        for idx in finite_indices[order[:limit]]:
            label = labels[idx] if idx < len(labels) else f"param_{idx + 1}"
            parts.append(f"{label}={arr[idx]:.6g}")
        return ", ".join(parts)

    def _checked_model_output(self, values, params, context):
        arr = np.asarray(values, dtype=float)
        if np.all(np.isfinite(arr)):
            return arr
        finite = arr[np.isfinite(arr)]
        finite_range = "no finite output"
        if finite.size:
            finite_range = f"{float(np.nanmin(finite)):.6g}..{float(np.nanmax(finite)):.6g}"
        raise FloatingPointError(
            f"{context} produced non-finite model values "
            f"({int(np.count_nonzero(~np.isfinite(arr)))}/{arr.size}); "
            f"finite_output_range={finite_range}; "
            f"largest_parameters={self._largest_param_text(params)}"
        )

    def _format_fit_exception(self, exc, p0=None, lower=None, upper=None):
        tb = traceback.format_exc(limit=8).strip()
        return (
            f"{type(exc).__name__}: {exc}\n\n"
            "Diagnostics:\n"
            f"{self.fit_diagnostics(p0, lower, upper)}\n\n"
            "Traceback:\n"
            f"{tb}"
        )

    def _prepare_fit_parameters(self):
        p0 = np.asarray(self.flatten_params(0), dtype=float)
        lower = np.asarray(self.flatten_params(1), dtype=float)
        upper = np.asarray(self.flatten_params(2), dtype=float)
        if not (p0.shape == lower.shape == upper.shape):
            return None, None, None, "Fit parameter values and bounds have incompatible lengths."
        if np.any(~np.isfinite(p0)):
            bad = int(np.where(~np.isfinite(p0))[0][0]) + 1
            return None, None, None, f"Initial fit parameter #{bad} is NaN or Inf. Check the parameter grid."
        if np.any(np.isnan(lower)) or np.any(np.isnan(upper)):
            return None, None, None, "Fit bounds contain NaN. Check the parameter grid."
        if np.any(lower > upper):
            bad = int(np.where(lower > upper)[0][0]) + 1
            return None, None, None, f"Fit parameter #{bad} has Min greater than Max."
        outside = (p0 < lower) | (p0 > upper)
        if np.any(outside):
            bad = int(np.where(outside)[0][0]) + 1
            return None, None, None, f"Initial fit parameter #{bad} is outside its Min/Max bounds."
        return p0, lower, upper, None

    def _background_flat(
        self,
        dataset_index,
        x_flat,
        theta_flat,
        config_mode,
        params,
        background_theta_flat=None,
    ):
        background_theta = theta_flat if background_theta_flat is None else background_theta_flat
        bg_idx = self._bg_index_for_dataset(dataset_index)
        if self.si_bg_mode == "advanced_si_bg_v2":
            base = 0 if bg_idx == 0 else 5
            bg_off, bg_slope_x, bg_slope_th = params[base], params[base + 1], params[base + 2]
            scale_bg, peak_amp = params[base + 3], params[base + 4]
            intensity = SelectionRules.Linear_Background(x_flat, background_theta, bg_off, bg_slope_x, bg_slope_th)
            intensity += self.evaluate_advanced_si_bg(
                x_flat,
                theta_flat,
                config_mode,
                scale_bg,
                peak_amp,
                params[10],
                params[11],
                params[12],
                normalization_theta=self.datasets[dataset_index].get("ang"),
            )
            return intensity
        base = 0 if bg_idx == 0 else 4
        bg_off, bg_slope_x, bg_slope_th, amp_si = params[base], params[base + 1], params[base + 2], params[base + 3]
        intensity = SelectionRules.Linear_Background(x_flat, background_theta, bg_off, bg_slope_x, bg_slope_th)
        intensity += amp_si * self.evaluate_si_bg(x_flat, config_mode)
        return intensity

    def _calc_single_config(
        self,
        x_flat,
        theta_flat,
        config_mode,
        params,
        dataset_index=0,
        background_theta_flat=None,
    ):
        intensity = self._background_flat(
            dataset_index,
            x_flat,
            theta_flat,
            config_mode,
            params,
            background_theta_flat=background_theta_flat,
        )
        idx = self._base_param_count()
        for peak in self.peaks:
            x0, gamma = params[idx], params[idx + 1]
            idx += 2
            rule_def = RULE_METADATA[peak["rule"]]
            n_ang = len(rule_def["params"])
            ang_p = params[idx:idx + n_ang]
            idx += n_ang
            I_val = rule_def["func"](theta_flat, config_mode, *ang_p)
            area = I_val / (np.abs(gamma) + 1e-9)
            intensity += area * lorentzian_normalized(x_flat, x0, gamma)
        return intensity

    def reconstruct(self, dataset_index):
        ds = self.datasets[dataset_index]
        if ds["z"] is None:
            return np.zeros((10, 10))
        XX, YY = np.meshgrid(ds["x"], ds["ang"])
        _XX_BG, YY_BG = np.meshgrid(ds["x"], self._background_angles_for_dataset(dataset_index))
        flat_params = self.flatten_params(which_val=0)
        z_flat = self._calc_single_config(
            XX.ravel(),
            YY.ravel(),
            ds["config"],
            flat_params,
            dataset_index,
            background_theta_flat=YY_BG.ravel(),
        )
        return z_flat.reshape(ds["z"].shape)

    def _joint_model_func(self, xy_tuple, *params):
        x1, th1, bg_th1, x2, th2, bg_th2 = xy_tuple
        z1 = self._calc_single_config(
            x1, th1, self.datasets[0]["config"], params, 0, background_theta_flat=bg_th1
        )
        z2 = self._calc_single_config(
            x2, th2, self.datasets[1]["config"], params, 1, background_theta_flat=bg_th2
        )
        return self._checked_model_output(np.concatenate([z1, z2]), params, "Joint model")

    def flatten_params(self, which_val=0):
        p = []
        if self.si_bg_mode == "advanced_si_bg_v2":
            for bg in self.bg_params:
                p.extend([bg["offset"][which_val], bg["slope_x"][which_val], bg["slope_theta"][which_val], bg["amp_si"][which_val], bg["amp_b1g_peak"][which_val]])
            p.extend([self.si_bg_peak_params["x0"][which_val], self.si_bg_peak_params["gamma"][which_val], self.si_bg_peak_params["phi"][which_val]])
        else:
            for bg in self.bg_params:
                p.extend([bg["offset"][which_val], bg["slope_x"][which_val], bg["slope_theta"][which_val], bg["amp_si"][which_val]])
        for peak in self.peaks:
            p.extend([peak["spec_params"]["x0"][which_val], peak["spec_params"]["gamma"][which_val]])
            for name in RULE_METADATA[peak["rule"]]["params"]:
                p.append(peak["ang_params"][name][which_val])
        return p

    def update_params_from_fit(self, popt):
        if self.si_bg_mode == "advanced_si_bg_v2":
            idx = 0
            for bg in self.bg_params:
                bg["offset"][0], bg["slope_x"][0], bg["slope_theta"][0], bg["amp_si"][0], bg["amp_b1g_peak"][0] = popt[idx:idx + 5]
                idx += 5
            self.si_bg_peak_params["x0"][0], self.si_bg_peak_params["gamma"][0], self.si_bg_peak_params["phi"][0] = popt[idx:idx + 3]
            idx += 3
        else:
            idx = 0
            for bg in self.bg_params:
                bg["offset"][0], bg["slope_x"][0], bg["slope_theta"][0], bg["amp_si"][0] = popt[idx:idx + 4]
                idx += 4
        for peak in self.peaks:
            peak["spec_params"]["x0"][0], peak["spec_params"]["gamma"][0] = popt[idx], popt[idx + 1]
            idx += 2
            for name in RULE_METADATA[peak["rule"]]["params"]:
                peak["ang_params"][name][0] = popt[idx]
                idx += 1

    def run_optimization(self, fixed_parameter_labels=None):
        self.last_fit_result = None
        active = [(idx, ds) for idx, ds in enumerate(self.datasets) if ds["z"] is not None and ds["x"] is not None and ds["ang"] is not None]
        if not active:
            return False, "Data missing"
        p0, lower, upper, param_error = self._prepare_fit_parameters()
        if param_error:
            return False, f"{param_error}\n\nDiagnostics:\n{self.fit_diagnostics(p0, lower, upper)}"
        labels = self._param_labels()
        fixed_labels = {str(label) for label in (fixed_parameter_labels or [])}
        unknown_fixed = fixed_labels.difference(labels)
        if unknown_fixed:
            return False, f"Unknown fixed parameter label(s): {', '.join(sorted(unknown_fixed))}"
        fixed_mask = np.asarray([label in fixed_labels for label in labels], dtype=bool)
        variable_indices = np.flatnonzero(~fixed_mask)
        if variable_indices.size == 0:
            return False, "At least one global fit parameter must remain free."

        def expand_params(variable_params):
            full = np.asarray(p0, dtype=float).copy()
            full[variable_indices] = np.asarray(variable_params, dtype=float)
            return full

        variable_p0 = p0[variable_indices]
        variable_lower = lower[variable_indices]
        variable_upper = upper[variable_indices]
        try:
            if len(active) == 1:
                ds_idx, ds = active[0]
                x_fit, th_fit, bg_th_fit, z_fit, selected_count, _skipped_count = self._fit_points_for_dataset(ds_idx)
                if len(z_fit) == 0:
                    if selected_count == 0:
                        return False, "No data points in selected X range."
                    return False, "No finite data points in selected X range."
                def single_model(xy_tuple, *variable_params):
                    params = expand_params(variable_params)
                    x_vals, th_vals, bg_th_vals = xy_tuple
                    values = self._calc_single_config(
                        x_vals,
                        th_vals,
                        ds["config"],
                        params,
                        ds_idx,
                        background_theta_flat=bg_th_vals,
                    )
                    return self._checked_model_output(values, params, f"Dataset {ds_idx} {ds['label']} model")
                single_model((x_fit, th_fit, bg_th_fit), *variable_p0)
                variable_popt, variable_pcov = curve_fit(
                    single_model,
                    (x_fit, th_fit, bg_th_fit),
                    z_fit,
                    p0=variable_p0,
                    bounds=(variable_lower, variable_upper),
                    maxfev=5000,
                )
            else:
                x1_fit, th1_fit, bg_th1_fit, z1_fit, selected1, _skipped1 = self._fit_points_for_dataset(0)
                x2_fit, th2_fit, bg_th2_fit, z2_fit, selected2, _skipped2 = self._fit_points_for_dataset(1)
                if len(z1_fit) == 0 or len(z2_fit) == 0:
                    if selected1 == 0 or selected2 == 0:
                        return False, "No data points in selected X range for one or both datasets."
                    return False, "No finite data points in selected X range for one or both datasets."
                z_combined = np.concatenate([z1_fit, z2_fit])
                joint_coordinates = np.vstack([
                    np.concatenate([x1_fit, x2_fit]),
                    np.concatenate([th1_fit, th2_fit]),
                    np.concatenate([bg_th1_fit, bg_th2_fit]),
                    np.concatenate([
                        np.zeros(x1_fit.size, dtype=float),
                        np.ones(x2_fit.size, dtype=float),
                    ]),
                ])

                def joint_model(packed_coordinates, *variable_params):
                    params = expand_params(variable_params)
                    x_values, theta_values, bg_theta_values, dataset_values = packed_coordinates
                    first = dataset_values < 0.5
                    second = ~first
                    output = np.empty(x_values.size, dtype=float)
                    output[first] = self._calc_single_config(
                        x_values[first],
                        theta_values[first],
                        self.datasets[0]["config"],
                        params,
                        0,
                        background_theta_flat=bg_theta_values[first],
                    )
                    output[second] = self._calc_single_config(
                        x_values[second],
                        theta_values[second],
                        self.datasets[1]["config"],
                        params,
                        1,
                        background_theta_flat=bg_theta_values[second],
                    )
                    return self._checked_model_output(output, params, "Joint model")

                joint_model(joint_coordinates, *variable_p0)
                variable_popt, variable_pcov = curve_fit(
                    joint_model,
                    joint_coordinates,
                    z_combined,
                    p0=variable_p0,
                    bounds=(variable_lower, variable_upper),
                    maxfev=5000,
                )
            popt = expand_params(variable_popt)
            pcov = np.zeros((p0.size, p0.size), dtype=float)
            pcov[np.ix_(variable_indices, variable_indices)] = np.asarray(variable_pcov, dtype=float)
            residuals = []
            for ds_idx, _ds in active:
                x_fit, th_fit, bg_th_fit, z_fit, _selected, _skipped = self._fit_points_for_dataset(ds_idx)
                fitted = self._calc_single_config(
                    x_fit,
                    th_fit,
                    self.datasets[ds_idx]["config"],
                    popt,
                    ds_idx,
                    background_theta_flat=bg_th_fit,
                )
                residuals.append(np.asarray(z_fit, dtype=float) - np.asarray(fitted, dtype=float))
            residual = np.concatenate(residuals) if residuals else np.asarray([], dtype=float)
            lower_arr = np.asarray(lower, dtype=float)
            upper_arr = np.asarray(upper, dtype=float)
            popt_arr = np.asarray(popt, dtype=float)
            bound_tolerance = 1e-6 * (1.0 + np.abs(popt_arr))
            at_bounds = (
                (np.isfinite(lower_arr) & (np.abs(popt_arr - lower_arr) <= bound_tolerance))
                | (np.isfinite(upper_arr) & (np.abs(popt_arr - upper_arr) <= bound_tolerance))
            ) & ~fixed_mask
            self.last_fit_result = {
                "params": popt_arr.copy(),
                "covariance": np.asarray(pcov, dtype=float).copy(),
                "labels": list(labels),
                "initial": np.asarray(p0, dtype=float).copy(),
                "lower": lower_arr.copy(),
                "upper": upper_arr.copy(),
                "n_observations": int(residual.size),
                "n_parameters": int(variable_indices.size),
                "n_fixed_parameters": int(np.count_nonzero(fixed_mask)),
                "degrees_of_freedom": int(residual.size - variable_indices.size),
                "residual_sum_squares": float(np.sum(residual * residual)),
                "residual_rmse": float(np.sqrt(np.mean(residual * residual))) if residual.size else np.nan,
                "at_bounds": [labels[i] if i < len(labels) else f"param_{i + 1}" for i in np.where(at_bounds)[0]],
                "fixed_parameter_labels": [label for label in labels if label in fixed_labels],
            }
            self.update_params_from_fit(popt)
            self.last_fit_result["peaks"] = copy.deepcopy(self.peaks)
            self.sort_and_rename_peaks()
            return True, "Success"
        except Exception as e:
            return False, self._format_fit_exception(e, p0, lower, upper)

    def row_fit_parameter_definitions(self, dataset_index):
        definitions = [
            {"key": "BG_Const", "label": "Background constant", "min": -np.inf, "max": np.inf, "default_fixed": True},
            {"key": "BG_Slope_X", "label": "Background slope X", "min": -np.inf, "max": np.inf, "default_fixed": True},
            {"key": "Amp_Si", "label": "Si background amplitude", "min": 0.0, "max": np.inf, "default_fixed": True},
        ]
        for i, peak in enumerate(self.peaks):
            prefix = f"P{i+1}_{peak['name']}"
            hard_gamma_min = float(peak["spec_params"]["gamma"][1])
            hard_gamma_max = float(peak["spec_params"]["gamma"][2])
            global_gamma = abs(float(peak["spec_params"]["gamma"][0]))
            gamma_min = max(hard_gamma_min, 0.5 * global_gamma)
            gamma_max = min(hard_gamma_max, 1.5 * global_gamma)
            if not np.isfinite(global_gamma) or global_gamma <= 0 or gamma_min > gamma_max:
                gamma_min, gamma_max = hard_gamma_min, hard_gamma_max
            definitions.extend([
                {"key": f"{prefix}_Area", "label": f"{peak['name']} area", "min": 0.0, "max": np.inf},
                {
                    "key": f"{prefix}_Gamma",
                    "label": f"{peak['name']} gamma",
                    "min": gamma_min,
                    "max": gamma_max,
                    "legacy_min": hard_gamma_min,
                    "legacy_max": hard_gamma_max,
                    "default_fixed": True,
                },
            ])
        return definitions

    @staticmethod
    def _row_fit_number(value, allow_infinite=False):
        text = str(value).strip().lower()
        if text in {"inf", "+inf", "infinity", "+infinity"}: return np.inf
        if text in {"-inf", "-infinity"}: return -np.inf
        value = float(value)
        if not allow_infinite and not np.isfinite(value):
            raise ValueError("Initial values must be finite numbers or 'global'.")
        return value

    def normalized_row_fit_config(self, config=None):
        source_config = self.row_fit_config if config is None else config
        source_config = source_config if isinstance(source_config, dict) else {}
        try:
            source_version = int(source_config.get("version", 0))
        except Exception:
            source_version = 0
        source = source_config.get("datasets", source_config)
        result = {"version": 3, "datasets": {}}
        for ds_idx in self.active_dataset_indices():
            raw = source.get(str(ds_idx), source.get(ds_idx, {}))
            raw = raw if isinstance(raw, dict) else {}
            rules = {}
            for definition in self.row_fit_parameter_definitions(ds_idx):
                key = definition["key"]
                item = raw.get(key, {}) if isinstance(raw.get(key, {}), dict) else {}
                initial = item.get("initial", "global")
                if isinstance(initial, str) and initial.strip().lower() == "global": initial = "global"
                else: initial = self._row_fit_number(initial)
                lower_source = item.get("min", definition["min"])
                upper_source = item.get("max", definition["max"])
                if key.endswith("_Gamma") and source_version < 2 and initial == "global" and item:
                    old_lower = self._row_fit_number(lower_source, True)
                    old_upper = self._row_fit_number(upper_source, True)
                    legacy_lower = float(definition.get("legacy_min", definition["min"]))
                    legacy_upper = float(definition.get("legacy_max", definition["max"]))
                    lower_matches = old_lower == legacy_lower or (np.isfinite(old_lower) and np.isfinite(legacy_lower) and np.isclose(old_lower, legacy_lower))
                    upper_matches = old_upper == legacy_upper or (np.isfinite(old_upper) and np.isfinite(legacy_upper) and np.isclose(old_upper, legacy_upper))
                    if lower_matches and upper_matches:
                        lower_source = definition["min"]
                        upper_source = definition["max"]
                lower = self._row_fit_number(lower_source, True)
                upper = self._row_fit_number(upper_source, True)
                if np.isnan(lower) or np.isnan(upper) or lower > upper:
                    raise ValueError(f"Invalid bounds for {definition['label']}.")
                if initial != "global" and not lower <= initial <= upper:
                    raise ValueError(f"Initial value for {definition['label']} is outside its bounds.")
                default_fixed = bool(definition.get("default_fixed", False))
                rules[key] = {
                    "initial": initial,
                    "min": float(lower),
                    "max": float(upper),
                    "fixed": bool(item["fixed"]) if "fixed" in item else default_fixed,
                }
            result["datasets"][str(ds_idx)] = rules
        return result

    def set_row_fit_config(self, config):
        self.row_fit_config = self.normalized_row_fit_config(config)
        return self.row_fit_config

    def _row_fit_global_values(self, ds_idx, angle, background_angle):
        bg = self._bg_for_dataset(ds_idx)
        values = {"BG_Const": bg["offset"][0] + bg["slope_theta"][0] * background_angle, "BG_Slope_X": bg["slope_x"][0], "Amp_Si": bg["amp_si"][0]}
        for i, peak in enumerate(self.peaks):
            prefix = f"P{i+1}_{peak['name']}"
            gamma = float(peak["spec_params"]["gamma"][0])
            rule = RULE_METADATA[peak["rule"]]
            angular = [peak["ang_params"][name][0] for name in rule["params"]]
            intensity = rule["func"](np.array([angle]), self.datasets[ds_idx]["config"], *angular)[0]
            values[f"{prefix}_Area"] = max(0.0, float(intensity) / (abs(gamma) + 1e-9))
            values[f"{prefix}_Gamma"] = gamma
        return values

    def annotate_row_fit_anomalies(self, results, threshold=2.0):
        """Attach peak-area anomalies relative to the frozen global fit."""
        threshold = float(threshold)
        if not np.isfinite(threshold) or threshold <= 1.0:
            raise ValueError("The row-fit anomaly threshold must be greater than 1.")
        global_tables = {int(table["dataset_idx"]): table for table in self.global_fit_trace_tables()}
        anomaly_count = 0
        for position, result in enumerate(results or []):
            dataset_index = int(result.get("dataset_index", position))
            global_table = global_tables.get(dataset_index)
            if global_table is None:
                continue
            headers = [str(value) for value in (result.get("headers") or [])]
            rows = result.get("rows_params") or result.get("params") or []
            global_headers = [str(value) for value in (global_table.get("headers") or [])]
            global_rows = global_table.get("rows") or []
            statuses = result.setdefault("row_status", [])
            while len(statuses) < len(rows):
                statuses.append({"success": True, "message": "", "at_bounds": []})
            for row_index, status in enumerate(statuses):
                if not isinstance(status, dict):
                    statuses[row_index] = status = {"success": bool(status), "message": "", "at_bounds": []}
                status["anomalies"] = []

            for column, header in enumerate(headers):
                if not header.endswith("_Area") or header not in global_headers:
                    continue
                global_column = global_headers.index(header)
                global_series = []
                for global_row in global_rows:
                    try:
                        global_series.append(float(global_row[global_column]))
                    except Exception:
                        global_series.append(np.nan)
                finite_global = np.abs(np.asarray(global_series, dtype=float))
                finite_global = finite_global[np.isfinite(finite_global)]
                global_scale = float(np.nanmax(finite_global)) if finite_global.size else 0.0
                zero_floor = max(global_scale * 1e-8, 1e-12)
                node_anomaly_floor = max(global_scale * 0.05, threshold * zero_floor)
                try:
                    peak_index = max(0, int(header.split("_", 1)[0][1:]) - 1)
                except Exception:
                    peak_index = -1
                peak_name = header
                if peak_index >= 0:
                    prefix = f"P{peak_index + 1}_"
                    peak_name = header[len(prefix):-len("_Area")] if header.startswith(prefix) else header

                for row_index, row in enumerate(rows):
                    if row_index >= len(global_rows):
                        continue
                    try:
                        row_area = float(row[column])
                        global_area = float(global_rows[row_index][global_column])
                    except Exception:
                        continue
                    if not np.isfinite(row_area) or not np.isfinite(global_area) or row_area < 0:
                        continue
                    global_reference = abs(global_area)
                    ratio = None
                    is_anomaly = False
                    if global_reference > zero_floor:
                        ratio = row_area / global_reference
                        is_anomaly = ratio > threshold
                    elif row_area > node_anomaly_floor:
                        is_anomaly = True
                    if not is_anomaly:
                        continue
                    statuses[row_index]["anomalies"].append({
                        "parameter": header,
                        "peak_index": peak_index,
                        "peak_name": peak_name,
                        "row_area": row_area,
                        "global_area": global_area,
                        "ratio": float(ratio) if ratio is not None and np.isfinite(ratio) else None,
                        "global_near_zero": bool(global_reference <= zero_floor),
                        "threshold": threshold,
                    })
                    anomaly_count += 1
            result["anomaly_threshold"] = threshold
        return anomaly_count

    def validate_row_by_row(self, row_fit_config=None, only_rows=None, existing_results=None):
        active = self.active_dataset_indices()
        if not active: return False, "No Data", None
        try: config = self.normalized_row_fit_config(row_fit_config)
        except Exception as exc: return False, f"Invalid row-fit settings: {exc}", None
        self.row_fit_config = config
        selected = None if only_rows is None else {int(k): {int(v) for v in values} for k, values in only_rows.items()}
        existing = {int(item.get("dataset_index", active[pos])): item for pos, item in enumerate(existing_results or [])}
        validation_results, succeeded, total = [], 0, 0
        for ds_idx in active:
            ds, bg = self.datasets[ds_idx], self._bg_for_dataset(ds_idx)
            x_full, angles, z_full = self._oriented_dataset_arrays(ds_idx)
            bg_angles = self._background_angles_for_dataset(ds_idx)
            mask = np.asarray(self.get_mask(ds_idx), dtype=bool) & np.isfinite(x_full)
            headers = ["Angle", "BG_Const", "BG_Slope_X", "Amp_Si"]
            if self.si_bg_mode == "advanced_si_bg_v2": headers.extend(["Amp_B1g_Peak", "B1g_Center", "B1g_Gamma", "B1g_Phi", "Profile_X_Shift"])
            for i, peak in enumerate(self.peaks): headers.extend([f"P{i+1}_{peak['name']}_Area", f"P{i+1}_{peak['name']}_Gamma", f"P{i+1}_{peak['name']}_Height"])
            definitions, rules = self.row_fit_parameter_definitions(ds_idx), config["datasets"][str(ds_idx)]
            old = existing.get(ds_idx, {})
            old_rows = old.get("rows_params") or old.get("params") or []
            old_rec, old_status = np.asarray(old.get("z_rec", [])), old.get("row_status") or []
            rows, rec, statuses = [], [], []
            for row_idx, angle in enumerate(angles):
                total += 1
                if selected is not None and row_idx not in selected.get(ds_idx, set()) and row_idx < len(old_rows) and old_rec.ndim == 2:
                    rows.append(list(old_rows[row_idx])); rec.append(np.array(old_rec[row_idx], copy=True))
                    status = dict(old_status[row_idx]) if row_idx < len(old_status) else {"success": True, "message": "Preserved"}
                    statuses.append(status); succeeded += int(bool(status.get("success", True))); continue
                z_row, background_angle = z_full[row_idx][mask], float(bg_angles[row_idx])
                finite = np.isfinite(x_full[mask]) & np.isfinite(z_row) & np.isfinite(angle) & np.isfinite(background_angle)
                x_fit, z_fit = x_full[mask][finite], z_row[finite]
                if not len(z_fit):
                    rows.append([angle, np.nan, np.nan, np.nan] + ([np.nan] * 5 if self.si_bg_mode == "advanced_si_bg_v2" else []) + [np.nan] * (3 * len(self.peaks)))
                    rec.append(np.zeros_like(x_full)); statuses.append({"success": False, "message": "No finite points", "n_points": 0, "rmse": np.nan, "at_bounds": []}); continue
                global_values = self._row_fit_global_values(ds_idx, float(angle), background_angle)
                full, variable, p0, lows, highs = [], [], [], [], []
                for index, definition in enumerate(definitions):
                    rule = rules[definition["key"]]
                    value = float(global_values[definition["key"]] if rule["initial"] == "global" else rule["initial"])
                    fixed = rule["fixed"] or rule["min"] == rule["max"]
                    if rule["min"] == rule["max"]: value = float(rule["min"])
                    if not fixed:
                        if np.isfinite(rule["min"]) and value <= rule["min"]: value = np.nextafter(rule["min"], rule["max"])
                        if np.isfinite(rule["max"]) and value >= rule["max"]: value = np.nextafter(rule["max"], rule["min"])
                        variable.append(index); p0.append(value); lows.append(rule["min"]); highs.append(rule["max"])
                    full.append(value)
                def model(x, values):
                    y = values[0] + values[1] * x
                    if self.si_bg_mode == "advanced_si_bg_v2":
                        y += self.evaluate_advanced_si_bg(
                            x,
                            np.full_like(x, angle),
                            ds["config"],
                            values[2],
                            bg["amp_b1g_peak"][0],
                            self.si_bg_peak_params["x0"][0],
                            self.si_bg_peak_params["gamma"][0],
                            self.si_bg_peak_params["phi"][0],
                            normalization_theta=angles,
                        )
                    else: y += values[2] * self.evaluate_si_bg(x, ds["config"])
                    pos = 3
                    for peak in self.peaks:
                        y += values[pos] * lorentzian_normalized(x, peak["spec_params"]["x0"][0], values[pos + 1]); pos += 2
                    return y
                def variable_model(x, *values):
                    merged = list(full)
                    for index, value in zip(variable, values): merged[index] = value
                    return model(x, merged)
                try:
                    fitted = list(full)
                    if variable:
                        optimized, _ = curve_fit(variable_model, x_fit, z_fit, p0=p0, bounds=(lows, highs), maxfev=5000)
                        for index, value in zip(variable, optimized): fitted[index] = float(value)
                    result_row = [angle, fitted[0], fitted[1], fitted[2]]
                    if self.si_bg_mode == "advanced_si_bg_v2": result_row.extend([bg["amp_b1g_peak"][0], self.si_bg_peak_params["x0"][0], self.si_bg_peak_params["gamma"][0], self.si_bg_peak_params["phi"][0], self.get_si_profile_shift()])
                    pos = 3
                    for peak in self.peaks:
                        area, gamma = fitted[pos], abs(fitted[pos + 1]); pos += 2
                        result_row.extend([area, gamma, area / (np.pi * gamma) if gamma > 0 else np.nan])
                    reconstruction = model(x_full, fitted); residual = model(x_fit, fitted) - z_fit
                    at_bounds = []
                    for index in variable:
                        rule, value = rules[definitions[index]["key"]], fitted[index]
                        tolerance = 1e-6 * (1 + abs(value))
                        if (np.isfinite(rule["min"]) and abs(value-rule["min"]) <= tolerance) or (np.isfinite(rule["max"]) and abs(value-rule["max"]) <= tolerance): at_bounds.append(definitions[index]["key"])
                    rows.append(result_row); rec.append(reconstruction); statuses.append({"success": True, "message": "Fit converged", "n_points": int(len(z_fit)), "rmse": float(np.sqrt(np.mean(residual**2))), "at_bounds": at_bounds}); succeeded += 1
                except Exception as exc:
                    result_row = [angle, np.nan, np.nan, np.nan] + ([np.nan] * 5 if self.si_bg_mode == "advanced_si_bg_v2" else []) + [np.nan] * (3 * len(self.peaks))
                    rows.append(result_row); rec.append(np.zeros_like(x_full)); statuses.append({"success": False, "message": str(exc), "n_points": int(len(z_fit)), "rmse": np.nan, "at_bounds": []})
            validation_results.append({"dataset_index": ds_idx, "label": ds["label"], "config": ds["config"], "x": ds["x"], "ang": ds["ang"], "z_raw": ds["z"], "z_rec": np.asarray(rec), "params": rows, "rows_params": rows, "headers": headers, "row_status": statuses, "row_fit_config": rules})
        self.annotate_row_fit_anomalies(validation_results, threshold=2.0)
        return True, f"Validation Complete: {succeeded}/{total} rows converged", validation_results

    def get_peak_reconstructions(self):
        results = []
        flat_params = self.flatten_params(which_val=0)
        for ds_idx in self.active_dataset_indices():
            ds = self.datasets[ds_idx]
            XX, YY = np.meshgrid(ds["x"], ds["ang"])
            _XX_BG, YY_BG = np.meshgrid(ds["x"], self._background_angles_for_dataset(ds_idx))
            x_flat, theta_flat = XX.ravel(), YY.ravel()
            z_bg = self._background_flat(
                ds_idx,
                x_flat,
                theta_flat,
                ds["config"],
                flat_params,
                background_theta_flat=YY_BG.ravel(),
            ).reshape(ds["z"].shape)
            results.append({"dataset_idx": ds_idx, "name": "Background", "matrix": z_bg})
            current_idx = self._base_param_count()
            for peak in self.peaks:
                x0, gamma = flat_params[current_idx], flat_params[current_idx + 1]
                current_idx += 2
                rule_def = RULE_METADATA[peak["rule"]]
                n_ang = len(rule_def["params"])
                ang_p = flat_params[current_idx:current_idx + n_ang]
                current_idx += n_ang
                area = rule_def["func"](theta_flat, ds["config"], *ang_p) / (np.abs(gamma) + 1e-9)
                matrix = (area * lorentzian_normalized(x_flat, x0, gamma)).reshape(ds["z"].shape)
                results.append({"dataset_idx": ds_idx, "name": peak["name"], "matrix": matrix})
        return results

    def get_background_subtracted(self, dataset_index):
        ds = self.datasets[dataset_index]
        if ds["z"] is None:
            return None
        bg = next((r["matrix"] for r in self.get_peak_reconstructions() if r["dataset_idx"] == dataset_index and r["name"] == "Background"), None)
        return None if bg is None else ds["z"] - bg

    def global_fit_trace_tables(self):
        tables = []
        for ds_idx in self.active_dataset_indices():
            ds = self.datasets[ds_idx]
            bg = self._bg_for_dataset(ds_idx)
            headers = ["Angle", "BG_Const", "BG_Slope_X", "Amp_Si"]
            if self.si_bg_mode == "advanced_si_bg_v2":
                headers.extend(["Amp_B1g_Peak", "B1g_Center", "B1g_Gamma", "B1g_Phi", "Profile_X_Shift"])
            for i, peak in enumerate(self.peaks):
                headers.extend([f"P{i+1}_{peak['name']}_Area", f"P{i+1}_{peak['name']}_Gamma", f"P{i+1}_{peak['name']}_Height"])
            rows = []
            background_angles = self._background_angles_for_dataset(ds_idx)
            for row_index, angle in enumerate(ds["ang"]):
                background_angle = float(background_angles[row_index])
                row = [
                    angle,
                    bg["offset"][0] + bg["slope_theta"][0] * background_angle,
                    bg["slope_x"][0],
                    bg["amp_si"][0],
                ]
                if self.si_bg_mode == "advanced_si_bg_v2":
                    row.extend([bg["amp_b1g_peak"][0], self.si_bg_peak_params["x0"][0], self.si_bg_peak_params["gamma"][0], self.si_bg_peak_params["phi"][0], self.get_si_profile_shift()])
                for peak in self.peaks:
                    gamma = peak["spec_params"]["gamma"][0]
                    rule_def = RULE_METADATA[peak["rule"]]
                    ang_p = [peak["ang_params"][pn][0] for pn in rule_def["params"]]
                    area = rule_def["func"](np.array([angle]), ds["config"], *ang_p)[0] / (np.abs(gamma) + 1e-9)
                    height = area / (np.pi * gamma) if gamma != 0 else np.nan
                    row.extend([area, gamma, height])
                rows.append(row)
            tables.append({"dataset_idx": ds_idx, "label": ds["label"], "headers": headers, "rows": rows})
        return tables

    def get_peak_polar_areas(self, dataset_index):
        ds = self.datasets[dataset_index]
        if ds["ang"] is None or ds["z"] is None:
            return None
        if not self.peaks:
            return np.zeros((len(ds["ang"]), 0))
        areas = np.zeros((len(ds["ang"]), len(self.peaks)))
        flat_params = self.flatten_params(which_val=0)
        idx = self._base_param_count()
        for p_idx, peak in enumerate(self.peaks):
            gamma = flat_params[idx + 1]
            idx += 2
            rule_def = RULE_METADATA[peak["rule"]]
            n_ang = len(rule_def["params"])
            ang_p = flat_params[idx:idx + n_ang]
            idx += n_ang
            areas[:, p_idx] = rule_def["func"](ds["ang"], ds["config"], *ang_p) / (np.abs(gamma) + 1e-9)
        return areas

    def _migrate_bg_params(self, bg_params):
        if isinstance(bg_params, list):
            out = [self._new_bg_params(), self._new_bg_params()]
            for idx, src in enumerate(bg_params[:2]):
                if isinstance(src, dict):
                    for key in out[idx]:
                        if key in src:
                            out[idx][key] = list(src[key])
            return out
        if isinstance(bg_params, dict):
            out = [self._new_bg_params(), self._new_bg_params()]
            offset = list(bg_params.get("offset", [0.0, -np.inf, np.inf]))
            slope = list(bg_params.get("slope", [0.0, -np.inf, np.inf]))
            for bg in out:
                bg["offset"] = offset.copy()
                bg["slope_theta"] = slope.copy()
            return out
        return [self._new_bg_params(), self._new_bg_params()]

    def to_dict(self):
        return {
            "fit_schema_version": self.fit_schema_version,
            "background_angle_reference": self.background_angle_reference,
            "peaks": self.peaks,
            "bg_params": self.bg_params,
            "x_min_limit": float(self.x_min_limit) if np.isfinite(self.x_min_limit) else -1e9,
            "x_max_limit": float(self.x_max_limit) if np.isfinite(self.x_max_limit) else 1e9,
            "unit": self.unit,
            "si_bg_mode": self.si_bg_mode,
            "si_bg_unit": self.si_bg_unit,
            "si_bg_source_path": self.si_bg_source_path,
            "si_bg_payload": self.si_bg_payload,
            "si_bg_peak_params": self.si_bg_peak_params,
            "si_bg_peak_source_center": self.si_bg_peak_source_center,
            "si_bg_peak_source_gamma": self.si_bg_peak_source_gamma,
            "row_fit_config": self.row_fit_config,
        }

    def from_dict(self, data):
        data = data or {}
        self.clear_si_bg()
        self.peaks = data.get("peaks", [])
        for peak in self.peaks:
            peak.setdefault("auto_name", peak.get("name") == peak.get("rule"))
            if peak.get("rule") not in RULE_METADATA:
                peak["rule"] = "D2h_B1g"
        self.bg_params = self._migrate_bg_params(data.get("bg_params", self.bg_params))
        self.x_min_limit = data.get("x_min_limit", -np.inf)
        self.x_max_limit = data.get("x_max_limit", np.inf)
        self.unit = data.get("unit", self.unit)
        self.row_fit_config = data.get("row_fit_config", {})
        saved_si_peak_params = data.get("si_bg_peak_params")
        if saved_si_peak_params:
            self.si_bg_peak_params = saved_si_peak_params
        self.si_bg_peak_source_center = data.get("si_bg_peak_source_center", self.si_bg_peak_source_center)
        self.si_bg_peak_source_gamma = data.get("si_bg_peak_source_gamma", self.si_bg_peak_source_gamma)
        payload = data.get("si_bg_payload")
        path = data.get("si_bg_source_path")
        if payload:
            self.load_si_bg(filepath=path, payload=payload)
        elif path and os.path.exists(path):
            self.load_si_bg(path)
        else:
            self.si_bg_mode = data.get("si_bg_mode", "none")
            if self.si_bg_mode not in {"none", ""}:
                warning = "Si BG profile data was not embedded and the source file is unavailable."
                self.clear_si_bg()
                self.si_bg_load_warning = warning
        if saved_si_peak_params:
            self.si_bg_peak_params = saved_si_peak_params
        self.sort_and_rename_peaks()

    @staticmethod
    def _format_param(v):
        return f"{v[0]} [{v[1]}, {v[2]}]"

    def export_parameters_text(self):
        lines = [
            "# Joint Fit Export",
            f"# Range: {self.x_min_limit} - {self.x_max_limit}",
            f"# Unit: {self.unit}",
            f"# Fit schema: {self.fit_schema_version}",
            "# Background angle reference: pre-rotation acquisition angle (raw 0 deg)",
        ]
        for idx, label in enumerate(("XX", "YX")):
            bg = self.bg_params[idx]
            lines.append(f"[Background {label}]")
            lines.append(f"Offset: {self._format_param(bg['offset'])}")
            lines.append(f"Slope_X: {self._format_param(bg['slope_x'])}")
            lines.append(f"Slope_Theta: {self._format_param(bg['slope_theta'])}")
            lines.append(f"Amp_Si: {self._format_param(bg['amp_si'])}")
            if self.si_bg_mode == "advanced_si_bg_v2":
                lines.append(f"Amp_B1g_Peak: {self._format_param(bg['amp_b1g_peak'])}")
            lines.append("")
        if self.si_bg_mode == "advanced_si_bg_v2":
            lines.extend([
                "[Advanced Si BG]",
                f"B1g_Center: {self._format_param(self.si_bg_peak_params['x0'])}",
                f"B1g_Gamma: {self._format_param(self.si_bg_peak_params['gamma'])}",
                f"B1g_Phi: {self._format_param(self.si_bg_peak_params['phi'])}",
                f"Profile_X_Shift: {self.get_si_profile_shift()}",
                "",
            ])
        for i, peak in enumerate(self.peaks):
            lines.append(f"[Peak {i + 1}: {peak['name']}]")
            lines.append(f"Rule: {peak['rule']}")
            lines.append(f"Center (x0): {self._format_param(peak['spec_params']['x0'])}")
            lines.append(f"Width (Gamma): {self._format_param(peak['spec_params']['gamma'])}")
            for key, value in peak["ang_params"].items():
                lines.append(f"{key}: {self._format_param(value)}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _parse_param_value(value):
        token = r"[-+]?(?:inf|nan|\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
        m = re.search(rf"({token})\s*\[\s*({token})\s*,\s*({token})\s*\]", value, flags=re.I)
        if m:
            return [float(m.group(1)), float(m.group(2)), float(m.group(3))]
        return [float(str(value).split()[0]), -np.inf, np.inf]

    def import_parameters_text(self, text):
        self.peaks = []
        current = None
        self.bg_params = [self._new_bg_params(), self._new_bg_params()]
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if "Range:" in line:
                    parts = line.split(":", 1)[-1].split("-")
                    if len(parts) == 2:
                        try:
                            self.x_min_limit, self.x_max_limit = float(parts[0]), float(parts[1])
                        except Exception:
                            pass
                if "Unit:" in line:
                    self.unit = line.split(":", 1)[-1].strip()
                continue
            if line == "[Background]":
                current = "BG_LEGACY"
                continue
            if line == "[Background XX]":
                current = "BG0"
                continue
            if line == "[Background YX]":
                current = "BG1"
                continue
            if line == "[Advanced Si BG]":
                current = "ADVBG"
                self.si_bg_mode = "advanced_si_bg_v2"
                continue
            m_peak = re.match(r"\[Peak \d+: (.*)\]", line)
            if m_peak:
                current = {"name": m_peak.group(1), "spec_params": {}, "ang_params": {}, "auto_name": False}
                self.peaks.append(current)
                continue
            if ":" not in line:
                continue
            key, raw = [part.strip() for part in line.split(":", 1)]
            if current == "BG_LEGACY":
                if key == "Offset":
                    vals = self._parse_param_value(raw)
                    self.bg_params[0]["offset"] = vals.copy()
                    self.bg_params[1]["offset"] = vals.copy()
                elif key == "Slope":
                    vals = self._parse_param_value(raw)
                    self.bg_params[0]["slope_theta"] = vals.copy()
                    self.bg_params[1]["slope_theta"] = vals.copy()
            elif current in {"BG0", "BG1"}:
                bg = self.bg_params[0 if current == "BG0" else 1]
                key_map = {"Offset": "offset", "Slope_X": "slope_x", "Slope_Theta": "slope_theta", "Amp_Si": "amp_si", "Amp_B1g_Peak": "amp_b1g_peak"}
                if key in key_map:
                    bg[key_map[key]] = self._parse_param_value(raw)
            elif current == "ADVBG":
                key_map = {"B1g_Center": "x0", "B1g_Gamma": "gamma", "B1g_Phi": "phi"}
                if key in key_map:
                    self.si_bg_peak_params[key_map[key]] = self._parse_param_value(raw)
            elif isinstance(current, dict):
                if key == "Rule":
                    current["rule"] = raw if raw in RULE_METADATA else "D2h_B1g"
                    current["ang_params"] = {pn: [10.0, -np.inf, np.inf] for pn in RULE_METADATA[current["rule"]]["params"]}
                    if "phi" in current["ang_params"]:
                        current["ang_params"]["phi"] = [0.0, -180.0, 180.0]
                else:
                    vals = self._parse_param_value(raw)
                    if key == "Center (x0)":
                        current["spec_params"]["x0"] = vals
                    elif key == "Width (Gamma)":
                        current["spec_params"]["gamma"] = vals
                    elif key in current.get("ang_params", {}):
                        current["ang_params"][key] = vals
        for peak in self.peaks:
            peak.setdefault("rule", "D2h_B1g")
            peak.setdefault("spec_params", {}).setdefault("x0", [300.0, 0.0, 5000.0])
            peak.setdefault("spec_params", {}).setdefault("gamma", [2.0, 0.001, 100.0])
            peak.setdefault("ang_params", {})
            for pn in RULE_METADATA[peak["rule"]]["params"]:
                peak["ang_params"].setdefault(pn, [0.0 if pn == "phi" else 10.0, -180.0 if pn == "phi" else -np.inf, 180.0 if pn == "phi" else np.inf])
            peak.setdefault("auto_name", peak.get("name") == peak.get("rule"))
        self.sort_and_rename_peaks()


# ============================================================================
# Stage-1 and Stage-2 API
# ============================================================================


def discover_merge(options: MergeDiscoverOptions) -> MergeDiscoverResult:
    files, pattern_hint, unique_xxxx, unique_yyyy, is_polarization_merge = detect_similar_files(options.seed_file)
    wavelength_nm, raw_angle_values, raw_intensity_matrix = build_raw_matrix_wavelength_axis(
        files,
        assume_xgrid_consistent=options.assume_xgrid_consistent,
        polarization_as_xxxx=is_polarization_merge,
    )
    # Parse raw_xxxx, raw_yyyy
    raw_xxxx = []
    raw_yyyy = []
    for p in files:
        x, y, _ = _parse_merge_xxxx_yyyy_from_path(p, polarization_as_xxxx=is_polarization_merge)
        raw_xxxx.append(x)
        raw_yyyy.append(y)
    # Build primitive matrix (averaged over yyyy for each xxxx)
    primitive_xxxx, primitive_angle_values, primitive_matrix, raw_rows_by_xxxx = build_primitive_matrix_by_xxxx(
        files,
        wavelength_nm,
        raw_angle_values,
        raw_intensity_matrix,
        polarization_as_xxxx=is_polarization_merge,
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
            raman_shift_cm1=cm1_from_ev(np.asarray(discover_result.wavelength_nm, dtype=float) / 1000.0),
            energy_ev=np.asarray(discover_result.wavelength_nm, dtype=float) / 1000.0,
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


def infer_merge_input_unit(path: str, default: str = "nm") -> str:
    """Infer the raw x-axis unit for merge input files from a filename."""
    base = os.path.basename(path or "").lower()
    if "mev" in base:
        return "meV"
    if "cm-1" in base or "cm^-1" in base or "cm1" in base or "wavenumber" in base or "raman_shift" in base:
        return "cm-1"
    return default


def _dark_subtracted_discover_result(disc: MergeDiscoverResult, dark_value: float) -> MergeDiscoverResult:
    dark = float(dark_value or 0.0)
    if dark == 0.0:
        return disc
    intensity = np.asarray(disc.intensity_matrix, dtype=float) - dark
    intensity[intensity < 0] = 0
    primitive = None
    if disc.primitive_matrix is not None:
        primitive = np.asarray(disc.primitive_matrix, dtype=float) - dark
        primitive[primitive < 0] = 0
    return replace(
        disc,
        intensity_matrix=intensity,
        primitive_matrix=primitive if primitive is not None else intensity,
    )


def merge_raw_to_run(
    seed_file: str,
    *,
    input_unit: str = "auto",
    manual_laser_nm: Optional[float] = None,
    dark_value: float = 0.0,
    cosmic_policy: str = "off",
    cosmic_indices: Optional[Sequence[int]] = None,
    nickname: Optional[str] = None,
) -> Tuple[object, Dict[str, object], Optional[Dict[str, object]]]:
    """Headless merge workflow used by the agent CLI.

    Returns ``(run, summary, cosmic_report)``. Importing data_structure here
    keeps the core analysis module usable without a hard GUI data-model import
    during module initialization.
    """
    from data_structure import Run, RunType, infer_intensity_unit, new_run_id, parse_filename

    input_unit = infer_merge_input_unit(seed_file) if input_unit in {None, "", "auto"} else str(input_unit)
    if input_unit not in {"nm", "cm-1", "meV"}:
        raise ValueError("input_unit must be auto, nm, cm-1, or meV.")
    cosmic_policy = str(cosmic_policy or "off").lower()
    if cosmic_policy not in {"off", "detect", "apply-auto", "apply-indices"}:
        raise ValueError("cosmic_policy must be off, detect, apply-auto, or apply-indices.")

    disc = discover_merge(MergeDiscoverOptions(seed_file=seed_file))
    active_disc = disc
    cosmic_report = None

    if cosmic_policy != "off":
        cosmic_options = MergeCosmicOptions(
            files=active_disc.files,
            pattern_hint=active_disc.pattern_hint,
            unique_xxxx=active_disc.unique_xxxx,
            unique_yyyy=active_disc.unique_yyyy,
            wavelength_nm=active_disc.wavelength_nm,
            angle_values=active_disc.angle_values,
            intensity_matrix=active_disc.intensity_matrix,
            title=active_disc.title,
            candidate_laser_nm=active_disc.candidate_laser_nm,
            raw_files=active_disc.raw_files,
            raw_xxxx=active_disc.raw_xxxx,
            raw_yyyy=active_disc.raw_yyyy,
            raw_angle_values=active_disc.raw_angle_values,
            raw_intensity_matrix=active_disc.raw_intensity_matrix,
            raw_rows_by_xxxx=active_disc.raw_rows_by_xxxx,
            primitive_xxxx=active_disc.primitive_xxxx,
            primitive_matrix=active_disc.primitive_matrix,
            cosmic_matrix=active_disc.cosmic_matrix,
            is_polarization_merge=active_disc.is_polarization_merge,
            dark_value=float(dark_value or 0.0),
        )
        cosmic_result = discover_cosmics(cosmic_options)
        cosmic_report = {
            "evidence": cosmic_result.evidence,
            "num_peaks": len(cosmic_result.peaks),
            "peaks": [
                {
                    "index": idx,
                    "row_index": int(pk.row_index),
                    "col_index": int(pk.col_index),
                    "xxxx": pk.xxxx,
                    "yyyy": pk.yyyy,
                    "angle_deg": pk.angle_deg,
                    "center_wavelength_nm": pk.center_wavelength_nm,
                    "intensity": pk.intensity,
                    "fwhm_nm": pk.fwhm_nm,
                    "is_confirmed_cosmic": bool(pk.is_confirmed_cosmic),
                    "test_results": pk.test_results,
                }
                for idx, pk in enumerate(cosmic_result.peaks)
            ],
        }
        if cosmic_policy in {"apply-auto", "apply-indices"}:
            if cosmic_policy == "apply-auto":
                remove_mask = [bool(pk.is_confirmed_cosmic) for pk in cosmic_result.peaks]
            else:
                selected = {int(v) for v in (cosmic_indices or [])}
                remove_mask = [idx in selected for idx in range(len(cosmic_result.peaks))]
            active_disc = apply_cosmic_removal(cosmic_result, remove_mask)
            cosmic_report["applied_indices"] = [idx for idx, flag in enumerate(remove_mask) if flag]

    preview_disc = _dark_subtracted_discover_result(active_disc, float(dark_value or 0.0))
    preview = preview_merge(
        preview_disc,
        MergePreviewOptions(
            seed_file=seed_file,
            files=list(preview_disc.raw_files or preview_disc.files),
            cosmic_enable=False,
            manual_laser_nm=manual_laser_nm,
            interactive_confirm=False,
            require_laser_nm=False,
            use_raman_x=input_unit != "nm",
            raman_x_mode=input_unit if input_unit in {"cm-1", "meV"} else "cm-1",
        ),
        prompt=None,
    )

    source_path = os.path.abspath(seed_file)
    metadata = parse_filename(source_path)
    metadata["merge_pattern_hint"] = disc.pattern_hint
    metadata["merge_unique_xxxx"] = disc.unique_xxxx
    metadata["merge_unique_yyyy"] = disc.unique_yyyy
    metadata["merged_files"] = disc.files
    metadata["raw_x_unit"] = input_unit
    if disc.is_polarization_merge:
        metadata["polarization_rows"] = list(disc.primitive_xxxx or disc.unique_xxxx)
        metadata["raw_y_unit"] = "polarization"
    if nickname:
        metadata["nickname"] = nickname
    elif disc.is_polarization_merge and metadata.get("sample"):
        metadata["nickname"] = f"{metadata.get('sample')} pol merge"

    intensity_unit = "au"
    if disc.raw_intensity_matrix is not None:
        intensity_unit = infer_intensity_unit(disc.raw_intensity_matrix)

    run = Run(
        id=new_run_id(prefix="run"),
        source_path=source_path,
        source_mtime=None,
        shift_cm1=preview.raman_shift_cm1,
        energy_eV=preview.energy_ev,
        intensity=None,
        intensity_2d=preview.intensity_matrix,
        angle_values=preview.angle_values,
        intensity_unit=intensity_unit,
        angle_unit="deg",
        metadata=metadata,
        run_type=RunType.RUN_2D,
        raw_table=None,
    )
    summary = {
        "run_id": run.id,
        "nickname": run.nickname,
        "seed_file": source_path,
        "input_unit": input_unit,
        "pattern_hint": disc.pattern_hint,
        "files": list(disc.files),
        "num_files": len(disc.files),
        "is_polarization_merge": bool(disc.is_polarization_merge),
        "primitive_xxxx": list(disc.primitive_xxxx),
        "unique_yyyy": list(disc.unique_yyyy),
        "shape": list(np.asarray(preview.intensity_matrix).shape),
        "raw_x_unit": metadata.get("raw_x_unit"),
        "polarization_rows": metadata.get("polarization_rows", []),
        "cosmic_policy": cosmic_policy,
    }
    return run, summary, cosmic_report


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
        is_polarization_merge=options.is_polarization_merge,
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
            polarization_as_xxxx=bool(getattr(discover_result, "is_polarization_merge", False)),
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
