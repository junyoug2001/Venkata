

"""
Core data model and basic I/O utilities for the Raman GUI.

This module is intentionally GUI-agnostic:

- Defines core data classes (Run / ViewState / ExperimentSet).
- Provides basic filename parsing and table loading.
- Implements simple unit conversions (nm → wavenumber, Raman shift cm^-1 / eV).
- Contains no wxPython or plotting code.
"""

# -------------------------
# Units and simple constants
# -------------------------

#: conversion factor: 1 nm = 1e-7 cm
CM_PER_NM: float = 1e-7

#: 1 cm^-1 = 1 / 8065.544 eV
EV_PER_CM1: float = 1.0 / 8065.544

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple, Union

import os
import re

import numpy as np
import pandas as pd


# -------------------------
# Utility helpers
# -------------------------

# Filename pattern adapted from the legacy raman_plotter_5.py script.
# This encodes the standard naming convention used in your lab, but
# keeps all parsing logic self-contained in this file.
FNAME_RE = re.compile(
    r'^'
    r'(?P<category>\d{8})_'
    r'(?P<sample>[^_]+)_'
    r'(?P<inttime>\d+sx\d+)_'
    r'(?P<grating>\d+g)_'
    r'(?P<slit>\d+mu)_'
    r'(?:(?P<cm_val>\d+cm-1)_)?'  # Optional field like 350cm-1
    r'(?P<laser_nm>\d+(?:,\d+)?nm)_'
    r'(?P<power>[^_]+)_'
    r'(?P<pol>(xx|xy|yx|yy|RL|LR|LL|RR))_'
    r'(?P<temp>\d+K)'
    r'(?P<angular>-8deg_(?P<angle_idx>\d{4})_(?P<rep_idx>\d{4}))?'
    r'(?P<tail>.*?)$'
)


def parse_filename(path: str) -> Dict[str, Any]:
    """
    Parse a Raman data filename into a structured metadata dictionary.

    This is adapted from the legacy raman_plotter_5.py logic and
    reflects the standard naming convention used in your files:

        YYYYMMDD_sample_inttime_grating_slit_laser_nm_power_pol_temp[-8deg_idx_rep]tail.ext

    If the filename does not match this pattern, a dictionary with
    all fields set to None (or sensible defaults) is returned.

    Parameters
    ----------
    path : str
        Full path to the data file.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:
        - "base": original basename (with extension)
        - "stem": basename without extension
        - "parts": underscore-separated tokens from the stem
        - "category", "sample", "inttime", "grating", "slit"
        - "laser_nm_str", "laser_nm" (float, if parsable)
        - "power", "pol"
        - "temp_str", "temp_K" (float, if parsable)
        - "has_angular", "angle_idx", "rep_idx", "tail"
    """
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    parts = stem.split("_")

    info: Dict[str, Any] = {
        "base": base,
        "stem": stem,
        "parts": parts,
        "category": None,
        "sample": None,
        "inttime": None,
        "grating": None,
        "slit": None,
        "cm_val": None,
        "laser_nm_str": None,
        "laser_nm": None,
        "power": None,
        "pol": None,
        "temp_str": None,
        "temp_K": None,
        "has_angular": False,
        "angle_idx": None,
        "rep_idx": None,
        "tail": None,
    }

    m = FNAME_RE.match(base)
    if not m:
        # Filename does not follow the strict pattern; return best-effort info.
        # We still keep stem/parts filled for manual inspection.
        return info

    gd = m.groupdict()

    info["category"] = gd.get("category")
    info["sample"] = gd.get("sample")
    info["inttime"] = gd.get("inttime")
    info["grating"] = gd.get("grating")
    info["slit"] = gd.get("slit")
    info["cm_val"] = gd.get("cm_val")
    info["laser_nm_str"] = gd.get("laser_nm")
    info["power"] = gd.get("power")
    info["pol"] = gd.get("pol")
    info["temp_str"] = gd.get("temp")
    info["has_angular"] = gd.get("angular") is not None
    info["angle_idx"] = gd.get("angle_idx")
    info["rep_idx"] = gd.get("rep_idx")
    info["tail"] = gd.get("tail")

    # Numeric conversions
    temp = gd.get("temp")
    if temp and temp.endswith("K"):
        try:
            info["temp_K"] = float(temp[:-1])
        except ValueError:
            pass

    laser_field = gd.get("laser_nm")
    if laser_field:
        # e.g. '532nm' or '532,0nm'
        nm_str = laser_field.replace("nm", "").replace(",", ".")
        try:
            info["laser_nm"] = float(nm_str)
        except ValueError:
            pass

    return info


def load_table(path: str) -> Union[Tuple[np.ndarray, np.ndarray],
                                   Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load a simple text / table file and return numeric arrays instead of a DataFrame.

    The behavior is intentionally minimal and aimed at your current data:

    - 1D data (spectra):
        * Assumed to have exactly two numeric columns.
        * Returns: (x, y)
          where x is x-axis array (e.g. wavelength or shift),
                y is intensity array.

    - 2D data (e.g. angular_matrix_meV.csv):
        * First column: y-axis values (e.g. angle in degrees).
        * Remaining columns: intensity matrix, column-wise.
        * Column labels (except the first) are treated as x-axis values when
          they can be parsed as floats; otherwise, a simple index grid is used.
        * Returns: (x_axis, y_axis, intensity_matrix)

    This function does not try to infer physical units; that is the job of
    the caller (e.g. converting meV → eV → cm^-1).
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in {".txt", ".dat"}:
        df = pd.read_csv(path, delim_whitespace=True, comment="#", header=None)
    elif ext == ".csv":
        df = pd.read_csv(path, comment="#")
    elif ext == ".tsv":
        df = pd.read_csv(path, sep="\t", comment="#")
    elif ext in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        # Fallback: let pandas try to infer
        df = pd.read_csv(path, comment="#")

    # Drop completely empty columns
    df = df.dropna(axis=1, how="all")

    ncols = df.shape[1]
    if ncols < 2:
        raise ValueError(f"File {path} must have at least 2 columns, got {ncols}.")

    # 1D: exactly two columns → (x, y)
    if ncols == 2:
        x = df.iloc[:, 0].to_numpy(dtype=float)
        y = df.iloc[:, 1].to_numpy(dtype=float)
        return x, y

    # 2D: first column = y-axis, remaining columns = intensity (x-grid in header)
    y_axis = df.iloc[:, 0].to_numpy(dtype=float)
    data = df.iloc[:, 1:]

    # Try to parse column labels as floats for x-axis
    try:
        x_axis = np.array([float(str(c)) for c in data.columns], dtype=float)
    except Exception:
        x_axis = np.arange(data.shape[1], dtype=float)

    intensity = data.to_numpy(dtype=float)

    return x_axis, y_axis, intensity


def nm_to_wavenumber(nm: np.ndarray) -> np.ndarray:
    """
    Convert wavelength in nm to absolute wavenumber in cm^-1.

    Note: this is *not* the Raman shift; it is simply 1 / lambda in cm.
    Raman shift requires a reference laser wavelength.
    """
    nm = np.asarray(nm, dtype=float)
    lam_cm = nm * CM_PER_NM
    return 1.0 / lam_cm


def raman_shift_cm1(laser_nm: float, wl_nm: np.ndarray) -> np.ndarray:
    """
    Compute Raman shift (in cm^-1) given a laser wavelength (nm)
    and an array of scattered wavelengths (nm).

    shift = 1/lambda_laser(cm) - 1/lambda_scattered(cm)
    """
    wl_nm = np.asarray(wl_nm, dtype=float)
    lam0_cm = laser_nm * CM_PER_NM
    lam_cm = wl_nm * CM_PER_NM
    return (1.0 / lam0_cm) - (1.0 / lam_cm)


def shift_to_eV(shift_cm: np.ndarray) -> np.ndarray:
    """
    Convert Raman shift in cm^-1 to energy in eV.
    """
    shift_cm = np.asarray(shift_cm, dtype=float)
    return shift_cm * EV_PER_CM1


def infer_intensity_unit(y: np.ndarray) -> str:
    """
    Decide whether an array of intensities is effectively integer-like
    (=> 'count') or not (=> 'au').

    - Integer dtypes are treated as 'count'.
    - Float dtypes are checked against rounding with a small tolerance.
    """
    y = np.asarray(y)
    if y.size == 0:
        return "au"

    if np.issubdtype(y.dtype, np.integer):
        return "count"

    y_float = y.astype(float)
    rounded = np.rint(y_float)
    if np.allclose(y_float, rounded, rtol=0.0, atol=1e-9):
        return "count"
    return "au"


# -------------------------
# Core data classes
# -------------------------


@dataclass
class Run:
    """
    One Raman dataset.

    A Run may represent:
    - A single 1D spectrum (shift vs intensity).
    - A 2D map (e.g. angle × shift) if `intensity_2d` and `angle_values`
      are populated.

    This class is deliberately generic and independent of any specific
    CLI workflow. More specialized metadata can be added later via the
    `metadata` dictionary.
    """

    # Identity / origin
    id: str
    source_path: str
    source_mtime: Optional[float] = None  # os.path.getmtime, if known

    # Axes
    wl_nm: Optional[np.ndarray] = None        # raw wavelength axis (nm)
    shift_cm1: Optional[np.ndarray] = None    # Raman shift in cm^-1
    energy_eV: Optional[np.ndarray] = None    # Raman shift in eV

    # Data
    intensity: Optional[np.ndarray] = None        # 1D spectrum
    intensity_2d: Optional[np.ndarray] = None     # 2D map (e.g. angle × shift)
    angle_values: Optional[np.ndarray] = None     # angle axis for 2D data, in degrees

    # Units and additional metadata
    intensity_unit: str = "au"    # 'count' or 'au'
    angle_unit: str = "deg"       # 'deg' or 'rad'
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Optional copy of the raw loaded table
    raw_table: Optional[pd.DataFrame] = None

    # --- label helpers ---

    @property
    def nickname(self) -> str:
        """
        Human-friendly name for this run.

        The primary storage is metadata["nickname"]. If this key is missing
        or empty, we fall back to:
        - metadata["sample"] if available, otherwise
        - the internal id.

        This keeps the core data model independent from any particular GUI
        while still providing a stable, human-readable label.
        """
        md = self.metadata or {}
        name = md.get("nickname")
        if isinstance(name, str) and name.strip():
            return name.strip()

        sample = md.get("sample")
        if isinstance(sample, str) and sample.strip():
            return sample.strip()

        return self.id

    @nickname.setter
    def nickname(self, value: str) -> None:
        """
        Set the human-friendly nickname for this run.

        The value is stored in metadata["nickname"] as a stripped string.
        """
        if not isinstance(self.metadata, dict):
            self.metadata = {}
        self.metadata["nickname"] = str(value).strip()

    # --- convenience ---

    @property
    def is_2d(self) -> bool:
        """Return True if this Run stores 2D data."""
        return self.intensity_2d is not None and self.angle_values is not None

    @property
    def n_points(self) -> int:
        """Number of points in the primary 1D axis, if available."""
        if self.shift_cm1 is not None:
            return int(self.shift_cm1.size)
        if self.wl_nm is not None:
            return int(self.wl_nm.size)
        if self.intensity is not None:
            return int(self.intensity.size)
        return 0

    def to_dict_summary(self) -> Dict[str, Any]:
        """
        Lightweight JSON-friendly summary (no full arrays).

        Useful for experiment summaries and quick inspection without
        serializing all numeric data.
        """
        return {
            "id": self.id,
            "source_path": self.source_path,
            "source_mtime": self.source_mtime,
            "intensity_unit": self.intensity_unit,
            "angle_unit": self.angle_unit,
            "shapes": {
                "wl_nm": None if self.wl_nm is None else list(self.wl_nm.shape),
                "shift_cm1": None if self.shift_cm1 is None else list(self.shift_cm1.shape),
                "energy_eV": None if self.energy_eV is None else list(self.energy_eV.shape),
                "intensity": None if self.intensity is None else list(self.intensity.shape),
                "intensity_2d": None if self.intensity_2d is None else list(self.intensity_2d.shape),
                "angle_values": None if self.angle_values is None else list(self.angle_values.shape),
            },
            "metadata_keys": list(self.metadata.keys()),
        }

    def export_csv(self, base_filepath: Optional[str] = None, output_dir: Optional[str] = None):
        """
        Export the Run data to two CSV files, one with Raman shift in cm-1
        and another in meV.

        If `base_filepath` is provided, it is used. Otherwise, a filename is
        generated from metadata and saved in `output_dir` (if provided) or
        the run's source directory.
        """
        if not self.is_2d:
            raise ValueError("Export to CSV is only supported for 2D runs.")

        if self.shift_cm1 is None or self.angle_values is None or self.intensity_2d is None:
            raise ValueError("Run is missing data for CSV export.")

        if base_filepath is None:
            # Determine the directory to save in
            if output_dir:
                save_dir = output_dir
            else:
                save_dir = os.path.dirname(self.source_path) if self.source_path else '.'
            
            # Rebuild filename from individual metadata components
            parts = []
            for key in ["category", "sample", "inttime", "grating", "slit", "cm_val", "laser_nm_str", "power", "pol", "temp_str"]:
                val = self.metadata.get(key)
                if val:
                    parts.append(str(val))
            
            if self.metadata.get("has_angular"):
                parts.append("angular_matrix")
            
            tail = self.metadata.get("tail")
            if tail and not self.metadata.get("has_angular"):
                parts.append(str(tail))

            base_name = "_".join(parts)
            
            if 'merged_files' in self.metadata:
                base_name += "_merged"
            
            base_filepath = os.path.join(save_dir, base_name)

        # --- cm-1 export ---
        df_cm1 = pd.DataFrame(
            data=self.intensity_2d,
            index=self.angle_values,
            columns=self.shift_cm1,
        )
        df_cm1.index.name = "angle_deg"
        df_cm1.columns.name = "Raman shift (cm-1)"
        
        base, ext = os.path.splitext(base_filepath)
        if not ext:
            ext = ".csv"
        cm1_path = f"{base}_cm-1{ext}"
        
        df_cm1.to_csv(cm1_path)

        # --- meV export ---
        if self.energy_eV is not None:
            energy_mev = self.energy_eV * 1000.0
            df_mev = pd.DataFrame(
                data=self.intensity_2d,
                index=self.angle_values,
                columns=energy_mev,
            )
            df_mev.index.name = "angle_deg"
            df_mev.columns.name = "Energy (meV)"
            
            mev_path = f"{base}_meV{ext}"
            df_mev.to_csv(mev_path)

    # --- construction helpers ---

    @classmethod
    def from_file(cls, path: str) -> "Run":
        """
        Build a Run from a data file using load_table and parse_filename.

        Logic:
        - Use load_table(path):
            * (x, y)  -> treat as 1D spectrum.
            * (x, y, Z) -> treat as 2D map (e.g. angular_matrix_meV.csv).
        - For 2D case:
            * x is assumed to be in meV (column labels).
            * y is assumed to be angle in degrees.
            * Converts meV -> eV -> cm^-1 for the x-axis.
        - For 1D case:
            * Only stores raw x and intensity in metadata for now; detailed
              axis semantics will be handled later in the analysis layer.
        """
        arrays = load_table(path)
        info = parse_filename(path)
        metadata = dict(info)

        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = None

        # 1D: (x, y)
        if isinstance(arrays, tuple) and len(arrays) == 2:
            x, y = arrays
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)

            metadata.setdefault("raw_dim", "1d")
            metadata.setdefault("raw_x_unit", "unknown")
            metadata.setdefault("raw_y_unit", "intensity")

            return cls(
                id=new_run_id(),
                source_path=os.path.abspath(path),
                source_mtime=mtime,
                wl_nm=None,
                shift_cm1=None,
                energy_eV=None,
                intensity=y,
                intensity_2d=None,
                angle_values=None,
                intensity_unit=infer_intensity_unit(y),
                angle_unit="deg",
                metadata=metadata,
                raw_table=None,
            )

        # 2D: (x_axis, y_axis, matrix)
        if isinstance(arrays, tuple) and len(arrays) == 3:
            x_raw, y_axis, intensity = arrays
            x_raw = np.asarray(x_raw, dtype=float)
            y_axis = np.asarray(y_axis, dtype=float)
            intensity = np.asarray(intensity, dtype=float)

            # Interpret x_raw as meV for now
            energies_eV = x_raw / 1000.0
            shift_cm1 = energies_eV / EV_PER_CM1

            metadata.setdefault("raw_dim", "2d")
            metadata.setdefault("raw_x_unit", "meV")
            metadata.setdefault("raw_y_unit", "deg")

            return cls(
                id=new_run_id(),
                source_path=os.path.abspath(path),
                source_mtime=mtime,
                wl_nm=None,
                shift_cm1=shift_cm1,
                energy_eV=energies_eV,
                intensity=None,
                intensity_2d=intensity,
                angle_values=y_axis,
                intensity_unit=infer_intensity_unit(intensity),
                angle_unit="deg",
                metadata=metadata,
                raw_table=None,
            )

        raise ValueError("Unsupported data shape returned by load_table(path).")


@dataclass
class ViewState:
    """
    Visualization state for a single GUI 'View' tab.

    This does not depend on any particular plotting backend. It just
    tells the GUI which runs to display and which options to use.
    """

    id: str
    title: str = "View"

    # Which runs are displayed in this view (0–2 for now)
    run_ids: List[str] = field(default_factory=list)

    # Display options
    x_axis: str = "shift_cm1"   # "shift_cm1" or "energy_eV"
    normalize: bool = False
    show_legend: bool = False

    # Curve-fit placeholders (to be filled by analysis / GUI)
    active_fit_id: Optional[str] = None
    fit_params: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_runs(self) -> int:
        return len(self.run_ids)


@dataclass
class ExperimentSet:
    """
    Collection of Runs + Views + experiment-level metadata.

    This is the main object that the GUI will manipulate and that can
    be saved/loaded as a whole (JSON for light summaries, HDF5 for full
    numeric arrays).

    The ExperimentSet is also responsible for assigning and managing
    human-friendly run nicknames (Run01, Run02, ...) via a simple
    monotonically increasing counter.
    """

    id: str
    runs: Dict[str, Run] = field(default_factory=dict)
    views: Dict[str, ViewState] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    #: Next index used for assigning default run nicknames, e.g. "Run01".
    next_run_index: int = 1

    # ---- Run management ----
    def new_run_nickname(self) -> str:
        """
        Generate the next default run nickname of the form "RunNN".
        """
        nickname = f"Run{self.next_run_index:02d}"
        self.next_run_index += 1
        return nickname

    def add_run(self, run: Run) -> None:
        """
        Add a Run to the experiment.

        If the run does not yet have a nickname in its metadata, assign
        a default one of the form "RunNN" using this experiment's
        monotonically increasing index.
        """
        if not isinstance(run.metadata, dict):
            run.metadata = {}

        # Only assign a nickname if one is not already present
        nickname = run.metadata.get("nickname")
        if not (isinstance(nickname, str) and nickname.strip()):
            run.nickname = self.new_run_nickname()

        self.runs[run.id] = run

    def remove_run(self, run_id: str) -> None:
        if run_id in self.runs:
            del self.runs[run_id]
        # Detach from any views that reference this run
        for v in self.views.values():
            if run_id in v.run_ids:
                v.run_ids = [rid for rid in v.run_ids if rid != run_id]

    def get_run(self, run_id: str) -> Optional[Run]:
        return self.runs.get(run_id)

    def get_run_nickname(self, run_id: str) -> str:
        """
        Return the nickname for the given run id.

        Falls back to the run id itself if the run is missing or if
        no nickname is defined.
        """
        run = self.runs.get(run_id)
        if run is None:
            return run_id
        return run.nickname

    def set_run_nickname(self, run_id: str, name: str) -> None:
        """
        Set or update the nickname for the given run id.

        If the run does not exist, this function is a no-op.
        """
        run = self.runs.get(run_id)
        if run is None:
            return
        run.nickname = name

    # ---- View management ----

    def add_view(self, view: ViewState) -> None:
        self.views[view.id] = view

    def remove_view(self, id: int) -> None:
        pass

    def get_view(self, view_id: str) -> Optional[ViewState]:
        return self.views.get(view_id)

    # ---- Serialization helpers ----

    def to_json_dict(self) -> Dict[str, Any]:
        """
        Lightweight JSON representation of the experiment.

        This is not meant to hold full numeric arrays, only metadata and
        shape information. Use export_hdf5 for full numeric storage.
        """
        return {
            "id": self.id,
            "metadata": self.metadata,
            "runs": {rid: r.to_dict_summary() for rid, r in self.runs.items()},
            "views": {vid: asdict(v) for vid, v in self.views.items()},
            "next_run_index": self.next_run_index,
        }

    def export_json(self, path: str) -> None:
        """Export summary metadata to a JSON file."""
        import json

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json_dict(), f, indent=2, ensure_ascii=False)

    def export_hdf5(self, path: str) -> None:
        """
        Export the experiment (including numeric arrays) to an HDF5 file.

        HDF5 layout (high-level sketch):
        - attrs:
            - experiment_id = self.id
        - metadata/json : experiment-level metadata as JSON string
        - runs/<run_id>/... : all arrays + scalar attrs for each Run
        - views/json : JSON string with serialized ViewState dicts
        """
        import h5py
        import json

        with h5py.File(path, "w") as h5:
            h5.attrs["experiment_id"] = self.id

            # Experiment-level metadata as JSON
            md_grp = h5.create_group("metadata")
            md_grp.attrs["json"] = json.dumps(self.metadata, ensure_ascii=False)
            md_grp.attrs["next_run_index"] = self.next_run_index

            # Runs
            runs_grp = h5.create_group("runs")
            for rid, r in self.runs.items():
                rg = runs_grp.create_group(rid)

                # Basic scalar attrs
                rg.attrs["source_path"] = r.source_path
                rg.attrs["source_mtime"] = r.source_mtime if r.source_mtime is not None else 0.0
                rg.attrs["intensity_unit"] = r.intensity_unit
                rg.attrs["angle_unit"] = r.angle_unit
                rg.attrs["metadata_json"] = json.dumps(r.metadata, ensure_ascii=False)

                # Numeric arrays
                if r.wl_nm is not None:
                    rg.create_dataset("wl_nm", data=r.wl_nm)
                if r.shift_cm1 is not None:
                    rg.create_dataset("shift_cm1", data=r.shift_cm1)
                if r.energy_eV is not None:
                    rg.create_dataset("energy_eV", data=r.energy_eV)
                if r.intensity is not None:
                    rg.create_dataset("intensity", data=r.intensity)
                if r.intensity_2d is not None:
                    rg.create_dataset("intensity_2d", data=r.intensity_2d)
                if r.angle_values is not None:
                    rg.create_dataset("angle_values", data=r.angle_values)

            # Views: store as a JSON string for simplicity
            views_grp = h5.create_group("views")
            views_grp.attrs["json"] = json.dumps(
                {vid: asdict(v) for vid, v in self.views.items()},
                ensure_ascii=False,
            )


# -------------------------
# ID helper functions
# -------------------------


def new_run_id(prefix: str = "run") -> str:
    """Generate a simple time-based run id."""
    import time
    return f"{prefix}_{int(time.time() * 1000)}"


def new_view_id(prefix: str = "view") -> str:
    """Generate a simple time-based view id."""
    import time
    return f"{prefix}_{int(time.time() * 1000)}"


def new_experiment_id(prefix: str = "exp") -> str:
    """Generate a simple time-based experiment id."""
    import time
    return f"{prefix}_{int(time.time() * 1000)}"


# -------------------------
# Quick test helper (manual use only)
# -------------------------




def test_with_files(filename1: str, filename2: str) -> None:
    """
    Quick manual test for data_structure.py.

    - Creates an ExperimentSet.
    - Parses the two filenames and builds Run objects with metadata.
    - Prints a summary of each Run.
    - Produces simple 2D matplotlib plots:
        * x-axis: Raman shift in cm^-1 (bottom)
        * top x-axis: Raman shift in eV
        * y-axis: angle (deg)
        * color: intensity
    """
    import matplotlib.pyplot as plt

    exp = ExperimentSet(id=new_experiment_id())

    run1 = Run.from_file(filename1)
    run2 = Run.from_file(filename2)

    exp.add_run(run1)
    exp.add_run(run2)

    # Print metadata summaries
    for i, run in enumerate((run1, run2), start=1):
        md = run.metadata
        print(f"Run {i}:")
        print(f"  id           : {run.id}")
        print(f"  source_path  : {run.source_path}")
        print(f"  category     : {md.get('category')}")
        print(f"  sample       : {md.get('sample')}")
        print(f"  inttime      : {md.get('inttime')}")
        print(f"  grating      : {md.get('grating')}")
        print(f"  slit         : {md.get('slit')}")
        print(f"  laser_nm_str : {md.get('laser_nm_str')}")
        print(f"  laser_nm     : {md.get('laser_nm')}")
        print(f"  power        : {md.get('power')}")
        print(f"  pol          : {md.get('pol')}")
        print(f"  temp_str     : {md.get('temp_str') or md.get('temp')}")
        print(f"  temp_K       : {md.get('temp_K')}")
        print(f"  has_angular  : {md.get('has_angular')}")
        print(f"  angle_idx    : {md.get('angle_idx')}")
        print(f"  rep_idx      : {md.get('rep_idx')}")
        print(f"  tail         : {md.get('tail')}")
        print(f"  angle_values : {None if run.angle_values is None else run.angle_values.shape}")
        print(f"  shift_cm1    : {None if run.shift_cm1 is None else run.shift_cm1.shape}")
        print(f"  energy_eV    : {None if run.energy_eV is None else run.energy_eV.shape}")
        print(f"  intensity_2d : {None if run.intensity_2d is None else run.intensity_2d.shape}")
        print(f"  intensity_unit: {run.intensity_unit}")
        print("")

    # Prepare plots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    runs = [run1, run2]
    titles = [
        f"Run 1: {os.path.basename(run1.source_path)}",
        f"Run 2: {os.path.basename(run2.source_path)}",
    ]

    for ax, run, title in zip(axes, runs, titles):
        if run.intensity_2d is None or run.shift_cm1 is None or run.angle_values is None:
            ax.text(0.5, 0.5, "No 2D data", ha="center", va="center")
            ax.set_title(title)
            continue

        x_cm1 = np.asarray(run.shift_cm1, dtype=float)
        y_angle = np.asarray(run.angle_values, dtype=float)
        Z = np.asarray(run.intensity_2d, dtype=float)

        # Ensure orientation matches (angle, shift)
        if Z.shape == (y_angle.size, x_cm1.size):
            Z_plot = Z
        elif Z.shape == (x_cm1.size, y_angle.size):
            Z_plot = Z.T
        else:
            # Fallback: let imshow handle it; axes will be index-based
            Z_plot = Z
            x_cm1 = np.arange(Z_plot.shape[1], dtype=float)
            y_angle = np.arange(Z_plot.shape[0], dtype=float)

        im = ax.pcolormesh(x_cm1, y_angle, Z_plot, shading="auto")
        ax.set_xlabel("Raman shift (cm$^{-1}$)")
        ax.set_title(title)
        if ax is axes[0]:
            ax.set_ylabel("Angle (deg)")

        cb = fig.colorbar(im, ax=ax)
        cb.set_label("Intensity (arb. units)")

        # Top x-axis in eV
        def cm1_to_eV(x):
            return x * EV_PER_CM1

        def eV_to_cm1(e):
            return e / EV_PER_CM1

        ax_top = ax.secondary_xaxis("top", functions=(cm1_to_eV, eV_to_cm1))
        ax_top.set_xlabel("Raman shift (eV)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        test_with_files(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python data_structure.py file1 file2")
        print("This will parse the filenames, build two 2D Runs,")
        print("print their metadata, and show simple 2D maps.")
